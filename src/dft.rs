//! Exact DFTs of every length that `F_p` supports, and DFTs on finite
//! abelian groups.
//!
//! A primitive `n`-th root of unity exists in `F_p` exactly when `n` divides
//! `p − 1 = 2^32 · 3 · 5 · 17 · 257 · 65537`. For such `n`, [`Dft`] computes
//! `y_k = Σ_j x_j ω^{jk}` with `ω = Fp::root_of_unity(n)` in `O(n log n)`.
//! It uses the Good–Thomas prime-factor map over the coprime factors
//! `{2^j, 3, 5, 17, 257, 65537}`, the tensor decomposition the paper uses for
//! its auxiliary lengths. The `2^j` axis uses [`Ntt`] and `3`, `5` use direct
//! evaluation. `17`, `257` and `65537` are Fermat primes, so Rader's
//! reduction turns them into cyclic convolutions of power-of-two length
//! `q − 1`, done with exact-length NTTs.

use alloc::vec;
use alloc::vec::Vec;

use crate::field::{Fp, P};
use crate::ntt::Ntt;
use crate::par;

const ODD_PRIMES: [u64; 5] = [3, 5, 17, 257, 65537];

/// Whether `F_p` has a primitive `n`-th root of unity.
pub fn is_supported_len(n: usize) -> bool {
    n > 0 && (P - 1) % n as u64 == 0
}

/// A one-dimensional transform applied to contiguous fibres, with caller scratch.
trait Fibre: Sync {
    fn apply(&self, x: &mut [Fp], scratch: &mut Vec<Fp>);
}

#[derive(Clone, Debug)]
enum Axis {
    Trivial,
    Ntt(Ntt),
    /// Row-major `q × q` matrix `w^{jk}`.
    Direct(Vec<Fp>, usize),
    Rader(Rader),
}

/// Rader's prime-length DFT: `X_0 = Σ x`, `X_{g^{-m}} = x_0 + (a ⊛ b)_m`
/// with `a_j = x_{g^j}`, `b_t = w^{g^{-t}}`, a cyclic convolution of length `q − 1`.
#[derive(Clone, Debug)]
struct Rader {
    q: usize,
    /// `g^j mod q`.
    gin: Vec<u32>,
    /// `g^{-m} mod q`.
    gout: Vec<u32>,
    ntt: Ntt,
    /// Bit-reversed spectrum of `b`, pre-scaled by `1/(q−1)`.
    b_hat: Vec<Fp>,
}

impl Rader {
    fn new(q: usize, w: Fp) -> Self {
        let l = q - 1;
        assert!(l.is_power_of_two());
        let g = (2..q as u64).find(|&g| is_generator(g, q as u64)).unwrap();
        let ginv = pow_mod(g, q as u64 - 2, q as u64);
        let mut gin = Vec::with_capacity(l);
        let mut gout = Vec::with_capacity(l);
        let (mut a, mut b) = (1u64, 1u64);
        for _ in 0..l {
            gin.push(a as u32);
            gout.push(b as u32);
            a = a * g % q as u64;
            b = b * ginv % q as u64;
        }
        let ntt = Ntt::new(l).with_threads(1);
        let inv_l = Fp::new(l as u64).inv();
        let mut b_hat: Vec<Fp> = gout.iter().map(|&e| w.pow(e as u64) * inv_l).collect();
        ntt.forward_bitrev(&mut b_hat);
        Rader { q, gin, gout, ntt, b_hat }
    }

    fn apply(&self, x: &mut [Fp], scratch: &mut Vec<Fp>) {
        let l = self.q - 1;
        scratch.clear();
        scratch.extend(self.gin.iter().map(|&i| x[i as usize]));
        let x0 = x[0];
        let total: Fp = x.iter().sum();
        self.ntt.forward_bitrev(scratch);
        for (s, b) in scratch.iter_mut().zip(&self.b_hat) {
            *s *= *b;
        }
        self.ntt.inverse_from_bitrev_unscaled(scratch);
        x[0] = total;
        for m in 0..l {
            x[self.gout[m] as usize] = x0 + scratch[m];
        }
    }
}

fn pow_mod(mut b: u64, mut e: u64, m: u64) -> u64 {
    let mut r = 1u64;
    b %= m;
    while e > 0 {
        if e & 1 == 1 {
            r = r * b % m;
        }
        b = b * b % m;
        e >>= 1;
    }
    r
}

/// For Fermat primes `q`, `g` generates `(Z/q)^*` iff `g^{(q−1)/2} ≠ 1`.
fn is_generator(g: u64, q: u64) -> bool {
    pow_mod(g, (q - 1) / 2, q) != 1
}

impl Axis {
    fn new(len: usize, root: Fp) -> Self {
        if len == 1 {
            Axis::Trivial
        } else if len.is_power_of_two() {
            Axis::Ntt(Ntt::with_root(len, root).with_threads(1))
        } else if len <= 5 {
            let mut m = Vec::with_capacity(len * len);
            for j in 0..len {
                let wj = root.pow(j as u64);
                let mut w = Fp::ONE;
                for _ in 0..len {
                    m.push(w);
                    w *= wj;
                }
            }
            Axis::Direct(m, len)
        } else {
            Axis::Rader(Rader::new(len, root))
        }
    }
}

impl Fibre for Axis {
    fn apply(&self, buf: &mut [Fp], scratch: &mut Vec<Fp>) {
        match self {
            Axis::Trivial => {}
            Axis::Ntt(p) => p.forward(buf),
            Axis::Direct(m, q) => {
                scratch.clear();
                for k in 0..*q {
                    let row = &m[k * q..(k + 1) * q];
                    scratch.push(row.iter().zip(buf.iter()).map(|(a, b)| *a * *b).sum());
                }
                buf.copy_from_slice(scratch);
            }
            Axis::Rader(r) => r.apply(buf, scratch),
        }
    }
}

/// Apply per-axis transforms to a row-major array with the given `dims`
/// (last axis fastest), using up to `threads` threads.
fn apply_axes<T: Fibre>(data: &mut [Fp], dims: &[usize], axes: &[T], threads: usize) {
    let total = data.len();
    let mut stride = 1;
    for ax in (0..dims.len()).rev() {
        let a = dims[ax];
        if a > 1 {
            let axis = &axes[ax];
            if stride == 1 {
                // Contiguous fibres: hand whole groups of fibres to each task.
                let per = (total / threads.max(1)).max(a).div_ceil(a) * a;
                par::for_each_chunk(data, per, threads, |group| {
                    let mut scratch = Vec::new();
                    for fib in group.chunks_exact_mut(a) {
                        axis.apply(fib, &mut scratch);
                    }
                });
            } else {
                strided_axis(data, a, stride, axis, threads);
            }
        }
        stride *= a;
    }
}

/// Transform all fibres `data[base + off + k·stride]`, `k < a`.
fn strided_axis<T: Fibre>(data: &mut [Fp], a: usize, stride: usize, axis: &T, threads: usize) {
    let block = a * stride;
    let blocks = data.len() / block;
    let run_block = |blk: &mut [Fp]| {
        let mut fibre = vec![Fp::ZERO; a];
        let mut scratch = Vec::new();
        for off in 0..stride {
            for k in 0..a {
                fibre[k] = blk[off + k * stride];
            }
            axis.apply(&mut fibre, &mut scratch);
            for k in 0..a {
                blk[off + k * stride] = fibre[k];
            }
        }
    };
    if threads <= 1 || blocks >= threads {
        par::for_each_chunk(data, block, threads.min(blocks), run_block);
        return;
    }
    // Few large blocks: split the offsets of each block across threads.
    // Workers read the block, transform into private buffers, then we scatter.
    #[cfg(feature = "std")]
    for blk in data.chunks_exact_mut(block) {
        let per = stride.div_ceil(threads);
        let view: &[Fp] = blk;
        let results: Vec<Vec<Fp>> = std::thread::scope(|s| {
            let handles: Vec<_> = (0..stride)
                .step_by(per)
                .map(|start| {
                    s.spawn(move || {
                        let end = (start + per).min(stride);
                        let mut out = Vec::with_capacity((end - start) * a);
                        let mut fibre = vec![Fp::ZERO; a];
                        let mut scratch = Vec::new();
                        for off in start..end {
                            for k in 0..a {
                                fibre[k] = view[off + k * stride];
                            }
                            axis.apply(&mut fibre, &mut scratch);
                            out.extend_from_slice(&fibre);
                        }
                        out
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let mut off = 0;
        for out in results {
            for fib in out.chunks_exact(a) {
                for k in 0..a {
                    blk[off + k * stride] = fib[k];
                }
                off += 1;
            }
        }
    }
    #[cfg(not(feature = "std"))]
    par::for_each_chunk(data, block, 1, run_block);
}

/// A planned DFT of length `n` (`n` must divide `p − 1`).
#[derive(Clone, Debug)]
pub struct Dft {
    n: usize,
    factors: Vec<usize>,
    axes: Vec<Axis>,
    /// Good–Thomas input map: grid position → input index.
    in_map: Vec<u32>,
    /// Good–Thomas output map: grid position → output index.
    out_map: Vec<u32>,
    inv_n: Fp,
    threads: usize,
}

impl Dft {
    /// # Panics
    /// If `n` does not divide `p − 1` (no primitive `n`-th root in `F_p`).
    pub fn new(n: usize) -> Self {
        assert!(is_supported_len(n), "no primitive {n}-th root of unity in F_p (n must divide p-1)");
        let mut factors = Vec::new();
        let two = 1usize << n.trailing_zeros();
        if two > 1 {
            factors.push(two);
        }
        let mut rest = n / two;
        for &q in &ODD_PRIMES {
            if rest as u64 % q == 0 {
                factors.push(q as usize);
                rest /= q as usize;
            }
        }
        debug_assert_eq!(rest, 1);
        let omega = Fp::root_of_unity(n as u64);
        // Axis i uses ω_i = ω^{M_i}, M_i = n / n_i.
        let axes = factors.iter().map(|&f| Axis::new(f, omega.pow((n / f) as u64))).collect();

        // Input:  index = Σ i_a M_a mod n.  Output: index = Σ k_a M_a (M_a^{-1} mod n_a) mod n.
        let (in_map, out_map) = if factors.len() > 1 {
            let coef_in: Vec<u64> = factors.iter().map(|&f| (n / f) as u64).collect();
            let coef_out: Vec<u64> = factors
                .iter()
                .map(|&f| {
                    let m = (n / f) as u64;
                    m * inv_mod(m % f as u64, f as u64) % n as u64
                })
                .collect();
            let mut in_map = vec![0u32; n];
            let mut out_map = vec![0u32; n];
            let mut digits = vec![0usize; factors.len()];
            for pos in 0..n {
                let (mut a, mut b) = (0u64, 0u64);
                for (i, &d) in digits.iter().enumerate() {
                    a = (a + d as u64 * coef_in[i]) % n as u64;
                    b = (b + d as u64 * coef_out[i]) % n as u64;
                }
                in_map[pos] = a as u32;
                out_map[pos] = b as u32;
                for i in (0..factors.len()).rev() {
                    digits[i] += 1;
                    if digits[i] < factors[i] {
                        break;
                    }
                    digits[i] = 0;
                }
            }
            (in_map, out_map)
        } else {
            (Vec::new(), Vec::new())
        };
        Dft { n, factors, axes, in_map, out_map, inv_n: Fp::new(n as u64).inv(), threads: 0 }
    }

    /// Set the number of threads (`0` = automatic, the default; `1` = single-threaded).
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.threads = threads;
        self
    }

    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// The coprime factors used by Good–Thomas.
    pub fn factors(&self) -> &[usize] {
        &self.factors
    }

    /// `y_k = Σ_j x_j ω^{jk}`, `ω = Fp::root_of_unity(n)`.
    pub fn forward(&self, x: &[Fp]) -> Vec<Fp> {
        let mut y = x.to_vec();
        self.forward_in_place(&mut y);
        y
    }

    /// `x_j = n^{-1} Σ_k y_k ω^{-jk}`.
    pub fn inverse(&self, y: &[Fp]) -> Vec<Fp> {
        let mut x = y.to_vec();
        self.inverse_in_place(&mut x);
        x
    }

    /// In-place forward transform. Uses one internal buffer of `n` elements
    /// when `n` has more than one coprime factor.
    pub fn forward_in_place(&self, x: &mut [Fp]) {
        let mut scratch = Vec::new();
        self.run(x, &mut scratch, par::threads_for(self.n, self.threads));
    }

    /// In-place inverse transform.
    pub fn inverse_in_place(&self, x: &mut [Fp]) {
        let t = par::threads_for(self.n, self.threads);
        let mut scratch = Vec::new();
        self.run_inverse(x, &mut scratch, t);
    }

    /// Forward-transform every length-`n` row of `data`, rows in parallel.
    pub fn forward_batch(&self, data: &mut [Fp]) {
        let t = self.batch_threads(data);
        par::for_each_chunk(data, self.rows_per_task(data, t), t, |rows| {
            let mut s = Vec::new();
            rows.chunks_exact_mut(self.n).for_each(|r| self.run(r, &mut s, 1));
        });
    }

    /// Inverse-transform every length-`n` row of `data`, rows in parallel.
    pub fn inverse_batch(&self, data: &mut [Fp]) {
        let t = self.batch_threads(data);
        par::for_each_chunk(data, self.rows_per_task(data, t), t, |rows| {
            let mut s = Vec::new();
            rows.chunks_exact_mut(self.n).for_each(|r| self.run_inverse(r, &mut s, 1));
        });
    }

    fn batch_threads(&self, data: &[Fp]) -> usize {
        assert!(data.len() % self.n == 0, "batch length must be a multiple of n");
        par::threads_for(data.len(), self.threads).min((data.len() / self.n).max(1))
    }

    fn rows_per_task(&self, data: &[Fp], t: usize) -> usize {
        (data.len() / self.n).div_ceil(t.max(1)).max(1) * self.n
    }

    fn run(&self, x: &mut [Fp], scratch: &mut Vec<Fp>, threads: usize) {
        assert_eq!(x.len(), self.n, "input length mismatch");
        if self.factors.len() <= 1 {
            apply_axes(x, &self.factors, &self.axes, threads);
            return;
        }
        scratch.clear();
        scratch.extend(self.in_map.iter().map(|&i| x[i as usize]));
        apply_axes(scratch, &self.factors, &self.axes, threads);
        for (g, &o) in scratch.iter().zip(&self.out_map) {
            x[o as usize] = *g;
        }
    }

    fn run_inverse(&self, x: &mut [Fp], scratch: &mut Vec<Fp>, threads: usize) {
        // DFT_{ω^{-1}}(y)[j] = DFT_ω(y)[−j mod n]
        self.run(x, scratch, threads);
        x[1..].reverse();
        let inv = self.inv_n;
        x.iter_mut().for_each(|v| *v *= inv);
    }
}

impl Fibre for Dft {
    fn apply(&self, x: &mut [Fp], scratch: &mut Vec<Fp>) {
        self.run(x, scratch, 1);
    }
}

/// Inverse direction of a [`Dft`] as a fibre transform.
struct InvDft<'a>(&'a Dft);

impl Fibre for InvDft<'_> {
    fn apply(&self, x: &mut [Fp], scratch: &mut Vec<Fp>) {
        self.0.run_inverse(x, scratch, 1);
    }
}

fn inv_mod(a: u64, m: u64) -> u64 {
    if m == 1 {
        return 0;
    }
    let (mut r0, mut r1) = (a as i128, m as i128);
    let (mut s0, mut s1) = (1i128, 0i128);
    while r1 != 0 {
        let q = r0 / r1;
        (r0, r1) = (r1, r0 - q * r1);
        (s0, s1) = (s1, s0 - q * s1);
    }
    assert_eq!(r0, 1, "not coprime");
    s0.rem_euclid(m as i128) as u64
}

/// DFT on `G = Z_{a_1} × … × Z_{a_r}` (row-major, last axis fastest):
/// `X̂(ξ) = Σ_y x(y) Π_i ω_{a_i}^{ξ_i y_i}`, with `ω_a = Fp::root_of_unity(a)`.
#[derive(Clone, Debug)]
pub struct GroupDft {
    dims: Vec<usize>,
    plans: Vec<Dft>,
    threads: usize,
}

impl GroupDft {
    /// # Panics
    /// If some `a_i` does not divide `p − 1`.
    pub fn new(dims: &[usize]) -> Self {
        GroupDft { dims: dims.to_vec(), plans: dims.iter().map(|&a| Dft::new(a)).collect(), threads: 0 }
    }

    /// Set the number of threads (`0` = automatic, the default).
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.threads = threads;
        self
    }

    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    pub fn len(&self) -> usize {
        self.dims.iter().product()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn forward(&self, x: &[Fp]) -> Vec<Fp> {
        let mut y = x.to_vec();
        self.forward_in_place(&mut y);
        y
    }

    pub fn inverse(&self, x: &[Fp]) -> Vec<Fp> {
        let mut y = x.to_vec();
        self.inverse_in_place(&mut y);
        y
    }

    pub fn forward_in_place(&self, x: &mut [Fp]) {
        assert_eq!(x.len(), self.len(), "input length mismatch");
        let t = par::threads_for(x.len(), self.threads);
        apply_axes(x, &self.dims, &self.plans, t);
    }

    pub fn inverse_in_place(&self, x: &mut [Fp]) {
        assert_eq!(x.len(), self.len(), "input length mismatch");
        let t = par::threads_for(x.len(), self.threads);
        let inv: Vec<InvDft<'_>> = self.plans.iter().map(InvDft).collect();
        apply_axes(x, &self.dims, &inv, t);
    }
}
