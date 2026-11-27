//! Exact DFTs of every length that `F_p` supports, and DFTs on finite
//! abelian groups.
//!
//! A primitive `n`-th root of unity exists in `F_p` exactly when `n` divides
//! `p − 1 = 2^32 · 3 · 5 · 17 · 257 · 65537`. For such `n`, [`Dft`] computes
//! `y_k = Σ_j x_j ω^{jk}` with `ω = Fp::root_of_unity(n)` in `O(n log n)`.
//! It uses the Good–Thomas prime-factor map over the coprime factors
//! `{2^j, 3, 5, 17, 257, 65537}`, the same tensor decomposition the paper uses
//! for its auxiliary lengths. The `2^j` axis uses [`Ntt`], small primes use
//! direct evaluation, and `257`/`65537` use chirp-z.

use crate::chirp::ChirpZ;
use crate::field::{Fp, P};
use crate::ntt::Ntt;

const ODD_PRIMES: [u64; 5] = [3, 5, 17, 257, 65537];

/// Whether `F_p` has a primitive `n`-th root of unity.
pub fn is_supported_len(n: usize) -> bool {
    n > 0 && (P - 1) % n as u64 == 0
}

#[derive(Clone, Debug)]
enum Axis {
    Trivial,
    Ntt(Ntt),
    /// Row-major `q × q` matrix `w^{jk}`.
    Direct(Vec<Fp>, usize),
    Chirp(ChirpZ),
}

impl Axis {
    fn new(len: usize, root: Fp) -> Self {
        if len == 1 {
            Axis::Trivial
        } else if len.is_power_of_two() {
            Axis::Ntt(Ntt::with_root(len, root))
        } else if len <= 17 {
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
            Axis::Chirp(ChirpZ::new(len, len, root))
        }
    }

    fn apply(&self, buf: &mut [Fp], tmp: &mut [Fp]) {
        match self {
            Axis::Trivial => {}
            Axis::Ntt(p) => p.forward(buf),
            Axis::Direct(m, q) => {
                for k in 0..*q {
                    let row = &m[k * q..(k + 1) * q];
                    tmp[k] = row.iter().zip(buf.iter()).fold(Fp::ZERO, |s, (a, b)| s + *a * *b);
                }
                buf.copy_from_slice(&tmp[..*q]);
            }
            Axis::Chirp(c) => {
                let y = c.apply(buf);
                buf.copy_from_slice(&y);
            }
        }
    }
}

/// Apply per-axis transforms to a row-major array with the given `dims`.
fn apply_axes(data: &mut [Fp], dims: &[usize], axes: &[Axis]) {
    let max = dims.iter().copied().max().unwrap_or(1);
    let mut fibre = vec![Fp::ZERO; max];
    let mut tmp = vec![Fp::ZERO; max];
    let mut stride = 1;
    for ax in (0..dims.len()).rev() {
        let a = dims[ax];
        if a > 1 {
            let block = a * stride;
            for base in (0..data.len()).step_by(block) {
                for off in 0..stride {
                    let start = base + off;
                    if stride == 1 {
                        axes[ax].apply(&mut data[start..start + a], &mut tmp);
                    } else {
                        for k in 0..a {
                            fibre[k] = data[start + k * stride];
                        }
                        axes[ax].apply(&mut fibre[..a], &mut tmp);
                        for k in 0..a {
                            data[start + k * stride] = fibre[k];
                        }
                    }
                }
            }
        }
        stride *= a;
    }
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
            // increment row-major digits (last axis fastest)
            for i in (0..factors.len()).rev() {
                digits[i] += 1;
                if digits[i] < factors[i] {
                    break;
                }
                digits[i] = 0;
            }
        }
        Dft { n, factors, axes, in_map, out_map, inv_n: Fp::new(n as u64).inv() }
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
        assert_eq!(x.len(), self.n, "input length mismatch");
        if self.factors.len() == 1 {
            let mut y = x.to_vec();
            apply_axes(&mut y, &self.factors, &self.axes);
            return y;
        }
        let mut grid: Vec<Fp> = self.in_map.iter().map(|&i| x[i as usize]).collect();
        apply_axes(&mut grid, &self.factors, &self.axes);
        let mut y = vec![Fp::ZERO; self.n];
        for (g, &o) in grid.iter().zip(&self.out_map) {
            y[o as usize] = *g;
        }
        y
    }

    /// `x_j = n^{-1} Σ_k y_k ω^{-jk}`.
    pub fn inverse(&self, y: &[Fp]) -> Vec<Fp> {
        // DFT_{ω^{-1}}(y)[j] = DFT_ω(y)[−j mod n]
        let f = self.forward(y);
        let n = self.n;
        (0..n).map(|j| f[(n - j) % n] * self.inv_n).collect()
    }
}

fn inv_mod(a: u64, m: u64) -> u64 {
    if m == 1 {
        return 0;
    }
    // extended Euclid
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
}

impl GroupDft {
    /// # Panics
    /// If some `a_i` does not divide `p − 1`.
    pub fn new(dims: &[usize]) -> Self {
        GroupDft { dims: dims.to_vec(), plans: dims.iter().map(|&a| Dft::new(a)).collect() }
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

    fn run(&self, x: &[Fp], inverse: bool) -> Vec<Fp> {
        assert_eq!(x.len(), self.len(), "input length mismatch");
        let mut data = x.to_vec();
        let mut stride = 1;
        let mut fibre = Vec::new();
        for ax in (0..self.dims.len()).rev() {
            let a = self.dims[ax];
            let plan = &self.plans[ax];
            fibre.resize(a, Fp::ZERO);
            for base in (0..data.len()).step_by(a * stride) {
                for off in 0..stride {
                    for k in 0..a {
                        fibre[k] = data[base + off + k * stride];
                    }
                    let out = if inverse { plan.inverse(&fibre) } else { plan.forward(&fibre) };
                    for k in 0..a {
                        data[base + off + k * stride] = out[k];
                    }
                }
            }
            stride *= a;
        }
        data
    }

    pub fn forward(&self, x: &[Fp]) -> Vec<Fp> {
        self.run(x, false)
    }

    pub fn inverse(&self, x: &[Fp]) -> Vec<Fp> {
        self.run(x, true)
    }
}
