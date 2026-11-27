//! The concrete commuting family used for verification.
//!
//! Vectors are functions `x : F_2^m → F_p`, stored as slices of length `2^m`.
//! `Q_j` is translation by the `j`-th basis vector: `(Q_j x)(y) = x(y ⊕ e_j)`.
//! The `Q_j` are commuting involutions. The Walsh–Hadamard transform `S`
//! diagonalises all of them at once. At frequency `ν`, `Q_j` has
//! eigenvalue `(−1)^{ν_j}`, so `k_j(ν) = ν_j`.
//!
//! The target is `F = Q_1 ⋯ Q_m`, translation by the all-ones vector.
//! Its multiplier at `ν` is `z(ν) = (−1)^{|ν|}`.
//!
//! Applied directly, `F` is just a permutation, so it costs nothing. The point
//! here is to check that the paper's predictor-and-repair pipeline rebuilds
//! `F` exactly, using only linear combinations of single `Q_j` applications
//! plus a small spectral table. In the paper's recursion, each `Q_j` is an
//! expensive call: a sub-transform along one axis.

use crate::field::Fp;
use std::cell::Cell;

/// Counts the operations that the paper's cost model charges for.
#[derive(Debug, Default)]
pub struct OpCounter {
    /// Single-row applications of some `Q_j`.
    pub q_calls: Cell<u64>,
    /// Full row conversions by `S` or `S^{-1}` (spectral table).
    pub spectral_rows: Cell<u64>,
}

impl OpCounter {
    pub fn q(&self) -> u64 {
        self.q_calls.get()
    }
    pub fn spectral(&self) -> u64 {
        self.spectral_rows.get()
    }
    fn bump_q(&self) {
        self.q_calls.set(self.q_calls.get() + 1);
    }
    fn bump_spectral(&self) {
        self.spectral_rows.set(self.spectral_rows.get() + 1);
    }
}

/// `Q_j x`: translation by `e_j`.
pub fn apply_q(j: usize, x: &[Fp], ctr: &OpCounter) -> Vec<Fp> {
    ctr.bump_q();
    let bit = 1usize << j;
    (0..x.len()).map(|y| x[y ^ bit]).collect()
}

/// `F x`, the reference answer: translation by the all-ones vector.
pub fn apply_f_reference(m: usize, x: &[Fp]) -> Vec<Fp> {
    let mask = (1usize << m) - 1;
    (0..x.len()).map(|y| x[y ^ mask]).collect()
}

/// In-place unnormalised Walsh–Hadamard transform `S`.
pub fn wht_in_place(x: &mut [Fp]) {
    let n = x.len();
    assert!(n.is_power_of_two());
    let mut h = 1;
    while h < n {
        for i in (0..n).step_by(2 * h) {
            for k in i..i + h {
                let (a, b) = (x[k], x[k + h]);
                x[k] = a + b;
                x[k + h] = a - b;
            }
        }
        h *= 2;
    }
}

/// Forward conversion `S x`.
pub fn to_spectral(x: &[Fp], ctr: &OpCounter) -> Vec<Fp> {
    ctr.bump_spectral();
    let mut y = x.to_vec();
    wht_in_place(&mut y);
    y
}

/// Inverse conversion `S^{-1} x = S x / 2^m`.
pub fn from_spectral(x: &[Fp], ctr: &OpCounter) -> Vec<Fp> {
    ctr.bump_spectral();
    let mut y = x.to_vec();
    wht_in_place(&mut y);
    let inv_n = Fp::new(x.len() as u64).inv();
    for v in &mut y {
        *v *= inv_n;
    }
    y
}

/// `z(ν) = (−1)^{|ν|}`, the multiplier of `F`.
pub fn target_multiplier(nu: usize) -> Fp {
    Fp::sign(nu.count_ones() as i64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q_diagonalised_by_wht() {
        let m = 4;
        let ctr = OpCounter::default();
        let x: Vec<Fp> = (0..1 << m).map(|i| Fp::new(i * i + 3)).collect();
        let xs = to_spectral(&x, &ctr);
        for j in 0..m {
            let qs = to_spectral(&apply_q(j, &x, &ctr), &ctr);
            for nu in 0..1usize << m {
                let ev = Fp::sign(((nu >> j) & 1) as i64);
                assert_eq!(qs[nu], ev * xs[nu]);
            }
        }
        assert_eq!(from_spectral(&xs, &ctr), x);
    }
}
