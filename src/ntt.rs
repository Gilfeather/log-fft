//! Power-of-two number-theoretic transforms over the Goldilocks field.

use crate::field::Fp;

/// A planned length-`n` transform (`n` a power of two, `n ≤ 2^32`):
/// `y_k = Σ_j x_j ω^{jk}` for a fixed primitive `n`-th root `ω`.
#[derive(Clone, Debug)]
pub struct Ntt {
    n: usize,
    log_n: u32,
    root: Fp,
    /// `ω^i` for `i < n/2`.
    twiddles: Vec<Fp>,
    /// `ω^{-i}` for `i < n/2`.
    inv_twiddles: Vec<Fp>,
    inv_n: Fp,
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
        assert!(n <= 1 << 32, "Ntt length exceeds 2^32");
        assert_eq!(root.pow(n as u64), Fp::ONE, "root^n != 1");
        if n > 1 {
            assert_ne!(root.pow((n / 2) as u64), Fp::ONE, "root is not primitive");
        }
        let half = n / 2;
        let mut twiddles = Vec::with_capacity(half);
        let mut w = Fp::ONE;
        for _ in 0..half {
            twiddles.push(w);
            w *= root;
        }
        let inv_twiddles = twiddles
            .iter()
            .enumerate()
            .map(|(i, _)| if i == 0 { Fp::ONE } else { twiddles[half - i] * -Fp::ONE })
            .collect::<Vec<_>>();
        // ω^{-i} = ω^{n-i} = −ω^{n/2-i}  (for 0 < i < n/2)
        Ntt {
            n,
            log_n: n.trailing_zeros(),
            root,
            twiddles,
            inv_twiddles,
            inv_n: Fp::new(n as u64).inv(),
        }
    }

    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    pub fn root(&self) -> Fp {
        self.root
    }

    /// In-place forward transform.
    pub fn forward(&self, x: &mut [Fp]) {
        self.run(x, &self.twiddles);
    }

    /// In-place inverse transform (includes the `1/n` scaling).
    pub fn inverse(&self, x: &mut [Fp]) {
        self.run(x, &self.inv_twiddles);
        for v in x.iter_mut() {
            *v *= self.inv_n;
        }
    }

    /// In-place inverse transform without the `1/n` scaling.
    pub fn inverse_unscaled(&self, x: &mut [Fp]) {
        self.run(x, &self.inv_twiddles);
    }

    fn run(&self, x: &mut [Fp], tw: &[Fp]) {
        let n = self.n;
        assert_eq!(x.len(), n, "input length {} != plan length {n}", x.len());
        if n <= 1 {
            return;
        }
        bit_reverse(x, self.log_n);
        let mut len = 2;
        while len <= n {
            let half = len / 2;
            let step = n / len;
            for chunk in x.chunks_exact_mut(len) {
                let (lo, hi) = chunk.split_at_mut(half);
                for k in 0..half {
                    let v = hi[k] * tw[k * step];
                    let u = lo[k];
                    lo[k] = u + v;
                    hi[k] = u - v;
                }
            }
            len <<= 1;
        }
    }
}

fn bit_reverse(x: &mut [Fp], log_n: u32) {
    let n = x.len();
    for i in 0..n {
        let j = i.reverse_bits() >> (usize::BITS - log_n);
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
    plan.forward(&mut fa);
    plan.forward(&mut fb);
    for (x, y) in fa.iter_mut().zip(&fb) {
        *x *= *y;
    }
    plan.inverse(&mut fa);
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
