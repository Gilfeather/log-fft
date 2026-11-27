//! Geometric evaluation (chirp-z) and its inverse.
//!
//! [`ChirpZ`] evaluates `y_k = Σ_{j<n} x_j q^{jk}` for `k < m` and any
//! nonzero `q`, in `O((n+m) log(n+m))`. It uses the square-root-free
//! identity `jk = C(j+k,2) − C(j,2) − C(k,2)`, so no `√q` is needed.
//!
//! [`InverseChirpZ`] solves `V_n(q) x = y` (interpolation at the nodes
//! `q^0, …, q^{n−1}`, which must be distinct). It applies the paper's Newton
//! factorisation (Lemma up:newton-lemma)
//! `V^{-1} = D_H^{-1} · T^{-T} · M^{-1} · T^{-1} · D_H^{-1}`. Here `T` is lower
//! triangular Toeplitz with entries `1/H_k`, `H_k = Π_{i≤k}(1 − q^i)`, and
//! both Toeplitz factors are fast convolutions.

use alloc::vec;
use alloc::vec::Vec;

use crate::field::Fp;
use crate::ntt::{series_inverse, Ntt};

/// Planned chirp-z transform `y_k = Σ_{j<n} x_j q^{jk}`, `k < m`.
#[derive(Clone, Debug)]
pub struct ChirpZ {
    n: usize,
    m: usize,
    /// `q^{−C(j,2)}` for `j < max(n, m)`.
    pre: Vec<Fp>,
    /// Spectrum of `c_t = q^{C(t,2)}`, `t < n + m − 1`, zero-padded.
    c_hat: Vec<Fp>,
    ntt: Ntt,
}

/// `q^{C(t,2)}` for `t < len`.
fn chirp_powers(q: Fp, len: usize) -> Vec<Fp> {
    let mut out = Vec::with_capacity(len);
    let mut cur = Fp::ONE; // q^{C(t,2)}
    let mut qt = Fp::ONE; // q^t
    for _ in 0..len {
        out.push(cur);
        cur *= qt;
        qt *= q;
    }
    out
}

impl ChirpZ {
    /// Plan for `n` inputs, `m` outputs, ratio `q ≠ 0`.
    pub fn new(n: usize, m: usize, q: Fp) -> Self {
        assert!(q != Fp::ZERO, "chirp-z ratio must be nonzero");
        assert!(n > 0 && m > 0, "chirp-z needs n, m > 0");
        let len = n + m - 1;
        // Cyclic length ≥ n + m − 1 keeps outputs n−1 … n+m−2 free of wraparound.
        let size = len.next_power_of_two();
        let ntt = Ntt::new(size);
        let mut c_hat = chirp_powers(q, len);
        c_hat.resize(size, Fp::ZERO);
        ntt.forward_bitrev(&mut c_hat);
        let pre = chirp_powers(q.inv(), n.max(m));
        ChirpZ { n, m, pre, c_hat, ntt }
    }

    pub fn input_len(&self) -> usize {
        self.n
    }

    pub fn output_len(&self) -> usize {
        self.m
    }

    /// Evaluate. `x.len()` must equal `n`.
    pub fn apply(&self, x: &[Fp]) -> Vec<Fp> {
        assert_eq!(x.len(), self.n);
        let size = self.ntt.len();
        // a_j = x_j q^{−C(j,2)}, reversed: ar[n−1−j] = a_j.
        let mut buf = vec![Fp::ZERO; size];
        for (j, &xj) in x.iter().enumerate() {
            buf[self.n - 1 - j] = xj * self.pre[j];
        }
        self.ntt.forward_bitrev(&mut buf);
        for (b, c) in buf.iter_mut().zip(&self.c_hat) {
            *b *= *c;
        }
        self.ntt.inverse_from_bitrev(&mut buf);
        (0..self.m).map(|k| buf[self.n - 1 + k] * self.pre[k]).collect()
    }
}

/// Planned inverse of the square geometric Vandermonde `V_n(q) = (q^{jk})`.
#[derive(Clone, Debug)]
pub struct InverseChirpZ {
    n: usize,
    /// `1/H_i`.
    inv_h: Vec<Fp>,
    /// `1/M_j = (−1)^j q^{−C(j,2)} H_j`.
    inv_mid: Vec<Fp>,
    /// Spectrum of `s = (Σ_k t_k X^k)^{-1} mod X^n`, `t_k = 1/H_k`.
    s_hat: Vec<Fp>,
    ntt: Ntt,
}

impl InverseChirpZ {
    /// # Panics
    /// If two nodes `q^i, q^j` (`i < j < n`) coincide, i.e. `q^k = 1` for some `0 < k < n`.
    pub fn new(n: usize, q: Fp) -> Self {
        assert!(n > 0);
        assert!(q != Fp::ZERO, "ratio must be nonzero");
        let mut h = vec![Fp::ONE; n];
        let mut qk = Fp::ONE;
        for i in 1..n {
            qk *= q;
            let f = Fp::ONE - qk;
            assert!(f != Fp::ZERO, "nodes not distinct: q^{i} = 1");
            h[i] = h[i - 1] * f;
        }
        let inv_h = batch_inverse(&h);
        let cinv = chirp_powers(q.inv(), n);
        let inv_mid = (0..n).map(|j| Fp::sign(j as i64) * cinv[j] * h[j]).collect();
        let s = series_inverse(&inv_h, n);
        let size = (2 * n - 1).next_power_of_two();
        let ntt = Ntt::new(size);
        let mut s_hat = s;
        s_hat.resize(size, Fp::ZERO);
        ntt.forward_bitrev(&mut s_hat);
        InverseChirpZ { n, inv_h, inv_mid, s_hat, ntt }
    }

    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// `(s · x) mod X^n`, i.e. the lower Toeplitz `T^{-1}`.
    fn lower(&self, x: &[Fp]) -> Vec<Fp> {
        let mut buf = x.to_vec();
        buf.resize(self.ntt.len(), Fp::ZERO);
        self.ntt.forward_bitrev(&mut buf);
        for (b, s) in buf.iter_mut().zip(&self.s_hat) {
            *b *= *s;
        }
        self.ntt.inverse_from_bitrev(&mut buf);
        buf.truncate(self.n);
        buf
    }

    /// Solve `V_n(q) x = y`.
    pub fn apply(&self, y: &[Fp]) -> Vec<Fp> {
        assert_eq!(y.len(), self.n);
        let v: Vec<Fp> = y.iter().zip(&self.inv_h).map(|(a, b)| *a * *b).collect();
        let mut v = self.lower(&v);
        v.iter_mut().zip(&self.inv_mid).for_each(|(a, b)| *a *= *b);
        // T^{-T} = R T^{-1} R
        v.reverse();
        let mut v = self.lower(&v);
        v.reverse();
        v.iter_mut().zip(&self.inv_h).for_each(|(a, b)| *a *= *b);
        v
    }
}

/// Montgomery batch inversion. All inputs must be nonzero.
pub fn batch_inverse(xs: &[Fp]) -> Vec<Fp> {
    let mut prefix = Vec::with_capacity(xs.len());
    let mut acc = Fp::ONE;
    for &x in xs {
        assert!(x != Fp::ZERO, "batch_inverse of zero");
        prefix.push(acc);
        acc *= x;
    }
    let mut inv = acc.inv();
    let mut out = vec![Fp::ZERO; xs.len()];
    for i in (0..xs.len()).rev() {
        out[i] = prefix[i] * inv;
        inv *= xs[i];
    }
    out
}
