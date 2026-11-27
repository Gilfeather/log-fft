//! Power-of-two number-theoretic transforms over the Goldilocks field.
//!
//! The forward transform is a recursive decimation-in-frequency (Gentleman–Sande)
//! pass from natural order to bit-reversed order. The inverse is a recursive
//! decimation-in-time (Cooley–Tukey) pass from bit-reversed order back to
//! natural order. Twiddles are stored contiguously per level. Recursion keeps
//! sub-transforms in cache, and large transforms split across threads (`std`
//! feature). Convolution-style callers use [`Ntt::forward_bitrev`] and
//! [`Ntt::inverse_from_bitrev`] directly, which skips both bit-reversal
//! permutations.

use alloc::vec;
use alloc::vec::Vec;

use crate::field::Fp;
use crate::par;

/// Below this length the recursion switches to an iterative loop.
const ITER_BLOCK: usize = 1 << 10;

/// A planned length-`n` transform (`n` a power of two, `n ≤ 2^32`):
/// `y_k = Σ_j x_j ω^{jk}` for a fixed primitive `n`-th root `ω`.
#[derive(Clone, Debug)]
pub struct Ntt {
    n: usize,
    log_n: u32,
    root: Fp,
    /// Level tables: for each `h = 1, 2, 4, …, n/2`, entries `w_{2h}^k`
    /// (`k < h`, `w_{2h} = ω^{n/2h}`) are stored at `tw[h−1 .. 2h−1]`.
    tw: Vec<Fp>,
    /// The same tables for `ω^{-1}`.
    itw: Vec<Fp>,
    inv_n: Fp,
    threads: usize,
}

fn level_tables(n: usize, root: Fp) -> Vec<Fp> {
    let half = n / 2;
    if half == 0 {
        return Vec::new();
    }
    let mut big = Vec::with_capacity(half);
    let mut w = Fp::ONE;
    for _ in 0..half {
        big.push(w);
        w *= root;
    }
    let mut tw = vec![Fp::ZERO; n - 1];
    let mut h = 1;
    while h <= half {
        let step = half / h; // w_{2h} = ω^{n/2h} = ω^{step}
        for k in 0..h {
            tw[h - 1 + k] = big[k * step];
        }
        h *= 2;
    }
    tw
}

impl Ntt {
    /// Plan with the crate's canonical root `ω = Fp::root_of_unity(n)`.
    pub fn new(n: usize) -> Self {
        Self::with_root(n, Fp::root_of_unity(n as u64))
    }

    /// Plan with a caller-supplied primitive `n`-th root of unity.
    ///
    /// # Panics
    /// If `n` is not a power of two, exceeds `2^32`, or `root` is not a
    /// primitive `n`-th root.
    pub fn with_root(n: usize, root: Fp) -> Self {
        assert!(n.is_power_of_two(), "Ntt length must be a power of two, got {n}");
        assert!(n as u64 <= 1 << 32, "Ntt length exceeds 2^32");
        assert_eq!(root.pow(n as u64), Fp::ONE, "root^n != 1");
        if n > 1 {
            assert_ne!(root.pow((n / 2) as u64), Fp::ONE, "root is not primitive");
        }
        Ntt {
            n,
            log_n: n.trailing_zeros(),
            root,
            tw: level_tables(n, root),
            itw: level_tables(n, root.inv()),
            inv_n: Fp::new(n as u64).inv(),
            threads: 0,
        }
    }

    /// Set the number of threads (`0` = automatic, the default; `1` = single-threaded).
    /// Has no effect without the `std` feature.
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

    /// The root `ω` this plan uses.
    pub fn root(&self) -> Fp {
        self.root
    }

    fn check(&self, x: &[Fp]) {
        assert_eq!(x.len(), self.n, "input length {} != plan length {}", x.len(), self.n);
    }

    /// In-place forward transform, natural order in and out.
    pub fn forward(&self, x: &mut [Fp]) {
        self.fwd(x, par::threads_for(self.n, self.threads));
    }

    /// In-place inverse transform, natural order in and out (includes `1/n`).
    pub fn inverse(&self, x: &mut [Fp]) {
        self.inv(x, par::threads_for(self.n, self.threads), true);
    }

    /// In-place inverse transform without the `1/n` scaling.
    pub fn inverse_unscaled(&self, x: &mut [Fp]) {
        self.inv(x, par::threads_for(self.n, self.threads), false);
    }

    /// Forward transform leaving the output in **bit-reversed** order.
    /// Pair it with [`Ntt::inverse_from_bitrev`] for convolutions.
    pub fn forward_bitrev(&self, x: &mut [Fp]) {
        self.fwd_bitrev(x, par::threads_for(self.n, self.threads));
    }

    /// Inverse transform of a **bit-reversed** spectrum, natural-order output, includes `1/n`.
    pub fn inverse_from_bitrev(&self, x: &mut [Fp]) {
        self.inv_from_bitrev(x, par::threads_for(self.n, self.threads), true);
    }

    /// As [`Ntt::inverse_from_bitrev`], without the `1/n` scaling.
    pub fn inverse_from_bitrev_unscaled(&self, x: &mut [Fp]) {
        self.inv_from_bitrev(x, par::threads_for(self.n, self.threads), false);
    }

    /// Forward-transform every length-`n` row of `data` (`data.len()` must be a
    /// multiple of `n`). Rows are processed in parallel.
    pub fn forward_batch(&self, data: &mut [Fp]) {
        let t = self.batch_threads(data);
        par::for_each_chunk(data, self.n, t, |row| self.fwd(row, 1));
    }

    /// Inverse-transform every length-`n` row of `data`.
    pub fn inverse_batch(&self, data: &mut [Fp]) {
        let t = self.batch_threads(data);
        par::for_each_chunk(data, self.n, t, |row| self.inv(row, 1, true));
    }

    fn batch_threads(&self, data: &[Fp]) -> usize {
        assert!(data.len() % self.n == 0, "batch length must be a multiple of n");
        par::threads_for(data.len(), self.threads).min((data.len() / self.n).max(1))
    }

    fn fwd(&self, x: &mut [Fp], t: usize) {
        self.fwd_bitrev(x, t);
        bit_reverse(x, self.log_n);
    }

    fn inv(&self, x: &mut [Fp], t: usize, scale: bool) {
        self.check(x);
        bit_reverse(x, self.log_n);
        self.inv_from_bitrev(x, t, scale);
    }

    fn fwd_bitrev(&self, x: &mut [Fp], t: usize) {
        self.check(x);
        if self.n > 1 {
            dif(x, &self.tw, t);
        }
    }

    fn inv_from_bitrev(&self, x: &mut [Fp], t: usize, scale: bool) {
        self.check(x);
        if self.n > 1 {
            dit(x, &self.itw, t);
        }
        if scale {
            let inv = self.inv_n;
            par::for_each_chunk(x, ITER_BLOCK * 16, t, |c| c.iter_mut().for_each(|v| *v *= inv));
        }
    }
}

#[inline(always)]
fn dif_butterflies(lo: &mut [Fp], hi: &mut [Fp], w: &[Fp]) {
    for ((a, b), &wk) in lo.iter_mut().zip(hi.iter_mut()).zip(w) {
        let (u, v) = (*a, *b);
        *a = u + v;
        *b = (u - v) * wk;
    }
}

#[inline(always)]
fn dit_butterflies(lo: &mut [Fp], hi: &mut [Fp], w: &[Fp]) {
    for ((a, b), &wk) in lo.iter_mut().zip(hi.iter_mut()).zip(w) {
        let u = *a;
        let v = *b * wk;
        *a = u + v;
        *b = u - v;
    }
}

fn dif_iter(x: &mut [Fp], tw: &[Fp]) {
    let n = x.len();
    let mut h = n / 2;
    while h > 1 {
        let w = &tw[h - 1..2 * h - 1];
        for chunk in x.chunks_exact_mut(2 * h) {
            let (lo, hi) = chunk.split_at_mut(h);
            dif_butterflies(lo, hi, w);
        }
        h /= 2;
    }
    for pair in x.chunks_exact_mut(2) {
        let (u, v) = (pair[0], pair[1]);
        pair[0] = u + v;
        pair[1] = u - v;
    }
}

fn dit_iter(x: &mut [Fp], tw: &[Fp]) {
    for pair in x.chunks_exact_mut(2) {
        let (u, v) = (pair[0], pair[1]);
        pair[0] = u + v;
        pair[1] = u - v;
    }
    let n = x.len();
    let mut h = 2;
    while h < n {
        let w = &tw[h - 1..2 * h - 1];
        for chunk in x.chunks_exact_mut(2 * h) {
            let (lo, hi) = chunk.split_at_mut(h);
            dit_butterflies(lo, hi, w);
        }
        h *= 2;
    }
}

/// Split one butterfly stage across `threads`.
fn stage<F: Fn(&mut [Fp], &mut [Fp], &[Fp]) + Sync>(lo: &mut [Fp], hi: &mut [Fp], w: &[Fp], threads: usize, f: F) {
    if threads <= 1 {
        f(lo, hi, w);
        return;
    }
    #[cfg(feature = "std")]
    {
        let per = lo.len().div_ceil(threads);
        std::thread::scope(|s| {
            for ((l, h), ww) in lo.chunks_mut(per).zip(hi.chunks_mut(per)).zip(w.chunks(per)) {
                let f = &f;
                s.spawn(move || f(l, h, ww));
            }
        });
    }
    #[cfg(not(feature = "std"))]
    f(lo, hi, w);
}

fn dif(x: &mut [Fp], tw: &[Fp], threads: usize) {
    let n = x.len();
    if n <= ITER_BLOCK {
        dif_iter(x, tw);
        return;
    }
    let h = n / 2;
    let w = &tw[h - 1..2 * h - 1];
    let (lo, hi) = x.split_at_mut(h);
    stage(lo, hi, w, threads, dif_butterflies);
    let (ta, tb) = (threads / 2, threads - threads / 2);
    par::join(threads, || dif(lo, tw, ta.max(1)), || dif(hi, tw, tb.max(1)));
}

fn dit(x: &mut [Fp], tw: &[Fp], threads: usize) {
    let n = x.len();
    if n <= ITER_BLOCK {
        dit_iter(x, tw);
        return;
    }
    let h = n / 2;
    let (lo, hi) = x.split_at_mut(h);
    let (ta, tb) = (threads / 2, threads - threads / 2);
    par::join(threads, || dit(lo, tw, ta.max(1)), || dit(hi, tw, tb.max(1)));
    stage(lo, hi, &tw[h - 1..2 * h - 1], threads, dit_butterflies);
}

/// Permute `x` (length `2^log_n`) into bit-reversed index order.
pub fn bit_reverse(x: &mut [Fp], log_n: u32) {
    if log_n == 0 {
        return;
    }
    let shift = usize::BITS - log_n;
    for i in 0..x.len() {
        let j = i.reverse_bits() >> shift;
        if i < j {
            x.swap(i, j);
        }
    }
}

/// Exact linear convolution `c_k = Σ_{i+j=k} a_i b_j` in `F_p`.
///
/// Result length `a.len() + b.len() − 1` (empty if either input is empty).
pub fn convolve(a: &[Fp], b: &[Fp]) -> Vec<Fp> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let out = a.len() + b.len() - 1;
    if a.len().min(b.len()) <= 32 {
        let mut c = vec![Fp::ZERO; out];
        for (i, &ai) in a.iter().enumerate() {
            for (j, &bj) in b.iter().enumerate() {
                c[i + j] += ai * bj;
            }
        }
        return c;
    }
    let n = out.next_power_of_two();
    let plan = Ntt::new(n);
    let mut fa = a.to_vec();
    fa.resize(n, Fp::ZERO);
    let mut fb = b.to_vec();
    fb.resize(n, Fp::ZERO);
    plan.forward_bitrev(&mut fa);
    plan.forward_bitrev(&mut fb);
    for (x, y) in fa.iter_mut().zip(&fb) {
        *x *= *y;
    }
    plan.inverse_from_bitrev(&mut fa);
    fa.truncate(out);
    fa
}

/// `f^{-1} mod X^n` by Newton iteration, `O(n log n)`. Needs `f[0] ≠ 0`.
pub fn series_inverse(f: &[Fp], n: usize) -> Vec<Fp> {
    assert!(!f.is_empty() && f[0] != Fp::ZERO, "series_inverse needs f(0) != 0");
    let mut g = vec![f[0].inv()];
    while g.len() < n {
        let m = (2 * g.len()).min(n);
        // g ← g (2 − f g) mod X^m
        let fm: Vec<Fp> = f.iter().take(m).copied().collect();
        let mut fg = convolve(&fm, &g);
        fg.truncate(m);
        for v in fg.iter_mut() {
            *v = -*v;
        }
        fg[0] += Fp::new(2);
        let mut ng = convolve(&g, &fg);
        ng.truncate(m);
        g = ng;
    }
    g.truncate(n);
    g
}
