//! Newton factorisation of geometric Vandermonde matrices (Lemma up:newton-lemma).
//!
//! `V = V_a(q) = (q^{jk})_{0≤j,k<a}` with distinct nodes `q^k`. Then
//!
//! ```text
//! L_{ij} = (−1)^j q^{C(j,2)} H_i / H_{i−j}   (0 ≤ j ≤ i < a),   H_i = Π_{k=1}^{i} (1 − q^k),
//! Δ = diag(L_ii),     V = L Δ^{-1} Lᵀ.
//! ```
//!
//! Factoring the entries gives `L = D_H · T · D_s`, where
//! `D_H = diag(H_i)`, `D_s = diag((−1)^j q^{C(j,2)})`, and `T` is the lower
//! triangular Toeplitz matrix with coefficients `1/H_k`. Hence
//!
//! ```text
//! V = D_H · T · M · Tᵀ · D_H,    M = D_s² Δ^{-1} = diag((−1)^j q^{C(j,2)} / H_j).
//! ```
//!
//! So `V` and `V^{-1}` are bounded products of invertible diagonals and
//! invertible triangular Toeplitz matrices.

use crate::field::Fp;

/// Precomputed factors of `V_a(q)`.
#[derive(Clone, Debug)]
pub struct NewtonFactors {
    pub a: usize,
    pub q: Fp,
    /// `H_i`, `i < a`.
    pub h: Vec<Fp>,
    /// Toeplitz coefficients `1/H_k`.
    pub t: Vec<Fp>,
    /// Middle diagonal `M_j = (−1)^j q^{C(j,2)} / H_j`.
    pub mid: Vec<Fp>,
}

impl NewtonFactors {
    pub fn new(a: usize, q: Fp) -> Self {
        let mut h = vec![Fp::ONE; a];
        let mut qk = Fp::ONE;
        for i in 1..a {
            qk *= q;
            h[i] = h[i - 1] * (Fp::ONE - qk);
            assert!(h[i] != Fp::ZERO, "nodes q^k not distinct (q^{i} = 1)");
        }
        let t: Vec<Fp> = h.iter().map(|v| v.inv()).collect();
        let mid = (0..a)
            .map(|j| Fp::sign(j as i64) * q.pow((j * j.saturating_sub(1) / 2) as u64) * t[j])
            .collect();
        NewtonFactors { a, q, h, t, mid }
    }

    /// The dense lower factor `L` (for tests).
    pub fn l_matrix(&self) -> Vec<Vec<Fp>> {
        let a = self.a;
        (0..a)
            .map(|i| {
                (0..a)
                    .map(|j| {
                        if j > i {
                            Fp::ZERO
                        } else {
                            Fp::sign(j as i64)
                                * self.q.pow((j * j.saturating_sub(1) / 2) as u64)
                                * self.h[i]
                                * self.t[i - j]
                        }
                    })
                    .collect()
            })
            .collect()
    }
}

/// Dense `V_a(q)`.
pub fn vandermonde(a: usize, q: Fp) -> Vec<Vec<Fp>> {
    (0..a).map(|j| (0..a).map(|k| q.pow((j * k) as u64)).collect()).collect()
}

/// `L = V P` where column `j` of `P` holds the coefficients of `Π_{s<j}(X − q^s)`.
pub fn l_from_newton_basis(a: usize, q: Fp) -> Vec<Vec<Fp>> {
    let mut p = vec![vec![Fp::ZERO; a]; a]; // p[row][col]
    let mut poly = vec![Fp::ONE];
    for j in 0..a {
        for (r, c) in poly.iter().enumerate() {
            p[r][j] = *c;
        }
        // poly *= (X − q^j)
        let root = q.pow(j as u64);
        let mut next = vec![Fp::ZERO; poly.len() + 1];
        for (r, c) in poly.iter().enumerate() {
            next[r + 1] += *c;
            next[r] -= root * *c;
        }
        poly = next;
    }
    let v = vandermonde(a, q);
    (0..a)
        .map(|i| (0..a).map(|j| (0..a).fold(Fp::ZERO, |s, k| s + v[i][k] * p[k][j])).collect())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l_formula_and_ldl_identity() {
        for (a, q) in [(2, Fp::new(5)), (5, Fp::new(3)), (8, Fp::root_of_unity(8)), (17, Fp::root_of_unity(17)), (12, Fp::new(7))] {
            let nf = NewtonFactors::new(a, q);
            let l = nf.l_matrix();
            assert_eq!(l, l_from_newton_basis(a, q), "L formula a={a}");
            let v = vandermonde(a, q);
            for i in 0..a {
                for k in 0..a {
                    let s = (0..a).fold(Fp::ZERO, |s, j| s + l[i][j] * l[j][j].inv() * l[k][j]);
                    assert_eq!(s, v[i][k], "V = LΔ⁻¹Lᵀ at ({i},{k}), a={a}");
                }
            }
        }
    }
}
