//! Matrix-root trick (Lemma matrix-root).
//!
//! Take a polynomial `p = Σ_{i=1}^{h} Π_{r=0}^{b−1} λ_{ir}` with `b` factors per
//! term. Set
//!
//! ```text
//! S_i e_r = λ_{ir} e_{r+1 mod b},      Z e_r = ξ^r e_r   (ξ a primitive b-th root),
//! L = Σ_i Z^{⊗(i−1)} ⊗ S_i ⊗ I^{⊗(h−i)}.
//! ```
//!
//! Then `L^b = p · I`. So `p` can be evaluated with `b` applications of the
//! linear pencil `L`, instead of expanding the product. In the paper the
//! `λ_{ir}` are affine in the queries `Q_j`. Here they are scalars, which is
//! the eigenvalue-wise statement.
//!
//! Why it works: `Z S_i = ξ S_i Z`, so the summands `ξ`-commute. The
//! `ξ`-binomial coefficients `[b choose k]_ξ` vanish for `0 < k < b`, which
//! leaves `L^b = Σ_i (summand_i)^b = Σ_i (Π_r λ_{ir}) I`.

use crate::field::Fp;

/// Dense square matrix over `F_p`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mat {
    pub n: usize,
    pub a: Vec<Fp>,
}

impl Mat {
    pub fn zero(n: usize) -> Self {
        Mat { n, a: vec![Fp::ZERO; n * n] }
    }
    pub fn identity(n: usize) -> Self {
        let mut m = Self::zero(n);
        for i in 0..n {
            m.a[i * n + i] = Fp::ONE;
        }
        m
    }
    pub fn get(&self, i: usize, j: usize) -> Fp {
        self.a[i * self.n + j]
    }
    pub fn mul(&self, o: &Mat) -> Mat {
        let n = self.n;
        let mut c = Mat::zero(n);
        for i in 0..n {
            for k in 0..n {
                let x = self.get(i, k);
                if x == Fp::ZERO {
                    continue;
                }
                for j in 0..n {
                    c.a[i * n + j] += x * o.get(k, j);
                }
            }
        }
        c
    }
    pub fn add(&self, o: &Mat) -> Mat {
        Mat { n: self.n, a: self.a.iter().zip(&o.a).map(|(x, y)| *x + *y).collect() }
    }
    pub fn scale(&self, s: Fp) -> Mat {
        Mat { n: self.n, a: self.a.iter().map(|x| *x * s).collect() }
    }
    pub fn kron(&self, o: &Mat) -> Mat {
        let n = self.n * o.n;
        let mut c = Mat::zero(n);
        for i in 0..self.n {
            for j in 0..self.n {
                let x = self.get(i, j);
                if x == Fp::ZERO {
                    continue;
                }
                for k in 0..o.n {
                    for l in 0..o.n {
                        c.a[(i * o.n + k) * n + j * o.n + l] = x * o.get(k, l);
                    }
                }
            }
        }
        c
    }
    pub fn pow(&self, e: usize) -> Mat {
        (0..e).fold(Mat::identity(self.n), |acc, _| acc.mul(self))
    }
}

/// Build `L` from `lambdas[i][r]` (`h` terms, `b` factors each).
pub fn build_pencil(lambdas: &[Vec<Fp>]) -> Mat {
    let h = lambdas.len();
    let b = lambdas[0].len();
    assert!(lambdas.iter().all(|l| l.len() == b));
    let xi = Fp::root_of_unity(b as u64);
    let mut z = Mat::zero(b);
    for r in 0..b {
        z.a[r * b + r] = xi.pow(r as u64);
    }
    let mut l = Mat::zero(b.pow(h as u32));
    for (i, lam) in lambdas.iter().enumerate() {
        let mut s = Mat::zero(b);
        for r in 0..b {
            // S e_r = λ_r e_{r+1}: column r, row r+1.
            s.a[((r + 1) % b) * b + r] = lam[r];
        }
        let mut term = Mat::identity(1);
        for _ in 0..i {
            term = term.kron(&z);
        }
        term = term.kron(&s);
        for _ in i + 1..h {
            term = term.kron(&Mat::identity(b));
        }
        l = l.add(&term);
    }
    l
}

/// `p = Σ_i Π_r λ_{ir}`.
pub fn polynomial_value(lambdas: &[Vec<Fp>]) -> Fp {
    lambdas
        .iter()
        .map(|l| l.iter().fold(Fp::ONE, |a, &x| a * x))
        .fold(Fp::ZERO, |a, x| a + x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pencil_power_is_scalar() {
        for (h, b) in [(1, 2), (2, 2), (3, 4), (2, 8)] {
            let lambdas: Vec<Vec<Fp>> = (0..h)
                .map(|i| (0..b).map(|r| Fp::new((31 * i + 7 * r + 3) as u64)).collect())
                .collect();
            let l = build_pencil(&lambdas);
            let p = polynomial_value(&lambdas);
            assert_eq!(l.pow(b), Mat::identity(l.n).scale(p), "h={h} b={b}");
        }
    }
}
