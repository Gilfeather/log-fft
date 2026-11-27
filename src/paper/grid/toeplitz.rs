//! Triangular Toeplitz actions (Lemma up:toeplitz, layers.tex).
//!
//! A lower triangular Toeplitz matrix of side `a` with first column `h` is
//! multiplication by `h(X)` in `K[X]/(X^a)`. The paper realises it
//! *in place* (scratch registers come back unchanged) using only:
//!
//! * shear additions `y ← y ± A x` with `A` Toeplitz. Each is one cyclic
//!   convolution of length `b ≥ 2a`, computed by `F_b = V_b(u)`.
//! * diagonal scalings, including `D_u = diag(1, u, …, u^{a−1})`.
//! * a signed interchange.
//!
//! Construction: normalise `h(0) = 1` and solve `f(uX) = h(X) f(X) mod X^a`
//! with `f(0) = 1`. Let `S` be multiplication by `f`. Then
//! `T = D_u S D_u^{-1} S^{-1}`.
//! `B = diag(S, S^{-1}) = [[I,S],[0,I]]·[[I,0],[−S^{-1},I]]·[[I,S],[0,I]]·[[0,−I],[I,0]]`,
//! and `(D_u ⊕ I) B (D_u^{-1} ⊕ I) B^{-1} = T ⊕ I`.

use super::smooth::SmoothDft;
use crate::field::Fp;

/// `(f · x) mod X^a`, computed directly in `O(a²)`. Used as the reference.
pub fn mul_trunc_direct(f: &[Fp], x: &[Fp]) -> Vec<Fp> {
    let a = x.len();
    let mut y = vec![Fp::ZERO; a];
    for i in 0..a {
        for k in 0..=i.min(f.len().saturating_sub(1)) {
            y[i] += f[k] * x[i - k];
        }
    }
    y
}

/// Inverse power series `f^{-1} mod X^a`. Needs `f(0) ≠ 0`.
pub fn series_inv(f: &[Fp], a: usize) -> Vec<Fp> {
    let inv0 = f[0].inv();
    let mut g = vec![Fp::ZERO; a];
    g[0] = inv0;
    for i in 1..a {
        let mut s = Fp::ZERO;
        for k in 1..=i.min(f.len() - 1) {
            s += f[k] * g[i - k];
        }
        g[i] = -s * inv0;
    }
    g
}

/// Convolution engine: Toeplitz shears `y ← y + (f·x mod X^a)` via `F_b`.
#[derive(Clone, Debug)]
pub struct ShearEngine {
    pub a: usize,
    pub dft: SmoothDft,
    /// Number of length-`b` transforms performed (forward and inverse).
    pub transforms: std::cell::Cell<u64>,
}

impl ShearEngine {
    pub fn new(a: usize, b: usize) -> Self {
        assert!(b >= 2 * a, "need b ≥ 2a (got a={a}, b={b})");
        ShearEngine { a, dft: SmoothDft::new(b), transforms: Default::default() }
    }

    /// `(f·x) mod X^a`, computed through one cyclic convolution of length `b`.
    pub fn mul_trunc(&self, f: &[Fp], x: &[Fp]) -> Vec<Fp> {
        let (a, b) = (self.a, self.dft.b);
        let mut fp = vec![Fp::ZERO; b];
        let mut xp = vec![Fp::ZERO; b];
        fp[..a.min(f.len())].copy_from_slice(&f[..a.min(f.len())]);
        xp[..a].copy_from_slice(x);
        self.transforms.set(self.transforms.get() + 3);
        self.dft.cyclic_convolve(&fp, &xp)[..a].to_vec()
    }

    /// Shear `y ← y + sign·(f·x mod X^a)`.
    fn shear(&self, y: &mut [Fp], f: &[Fp], x: &[Fp], sign: Fp) {
        let t = self.mul_trunc(f, x);
        for (yi, ti) in y.iter_mut().zip(t) {
            *yi += sign * ti;
        }
    }
}

/// Coefficients prepared once per Toeplitz factor ("coefficient preparation").
#[derive(Clone, Debug)]
pub struct ToeplitzPlan {
    pub a: usize,
    /// `h(0)`, factored out so that the normalised `h` has constant term 1.
    pub scale: Fp,
    /// `f` with `f(uX) = h(X) f(X) mod X^a`, `f(0) = 1`.
    pub f: Vec<Fp>,
    /// `f^{-1} mod X^a`.
    pub f_inv: Vec<Fp>,
    /// `u^i` for `i < a`.
    pub du: Vec<Fp>,
}

impl ToeplitzPlan {
    /// `h` is the first column (`h.len() ≤ a`, `h(0) ≠ 0`). `u` is the root of `F_b`.
    pub fn new(h: &[Fp], a: usize, u: Fp) -> Self {
        assert!(h[0] != Fp::ZERO, "triangular Toeplitz must be invertible");
        let scale = h[0];
        let inv = scale.inv();
        let hn: Vec<Fp> = (0..a).map(|i| h.get(i).copied().unwrap_or(Fp::ZERO) * inv).collect();
        let du: Vec<Fp> = (0..a).map(|i| u.pow(i as u64)).collect();
        let mut f = vec![Fp::ZERO; a];
        f[0] = Fp::ONE;
        for i in 1..a {
            // u^i f_i = Σ_{k=0}^{i} h_k f_{i−k}  ⇒  f_i (u^i − 1) = Σ_{k≥1} h_k f_{i−k}
            let mut s = Fp::ZERO;
            for k in 1..=i {
                s += hn[k] * f[i - k];
            }
            let den = du[i] - Fp::ONE;
            assert!(den != Fp::ZERO, "u^i = 1 for some 1 ≤ i < a; need ord(u) ≥ a");
            f[i] = s * den.inv();
        }
        let f_inv = series_inv(&f, a);
        ToeplitzPlan { a, scale, f, f_inv, du }
    }
}

/// `B = diag(S, S^{-1})` on registers `(x, y)`, using three shears and a signed interchange.
fn apply_b(eng: &ShearEngine, plan: &ToeplitzPlan, x: &mut Vec<Fp>, y: &mut Vec<Fp>) {
    let one = Fp::ONE;
    // Right to left: [[0,−I],[I,0]] : (x, y) → (−y, x)
    let nx: Vec<Fp> = y.iter().map(|v| -*v).collect();
    *y = std::mem::replace(x, nx);
    eng.shear(x, &plan.f, y, one); //      x ← x + S y
    eng.shear(y, &plan.f_inv, x, -one); // y ← y − S^{-1} x
    eng.shear(x, &plan.f, y, one); //      x ← x + S y
}

/// `B^{-1} = diag(S^{-1}, S)`: undo the factors of [`apply_b`] in reverse order.
fn apply_b_inv(eng: &ShearEngine, plan: &ToeplitzPlan, x: &mut Vec<Fp>, y: &mut Vec<Fp>) {
    let one = Fp::ONE;
    eng.shear(x, &plan.f, y, -one);
    eng.shear(y, &plan.f_inv, x, one);
    eng.shear(x, &plan.f, y, -one);
    // inverse of (x, y) → (−y, x) is (x, y) → (y, −x)
    let ny: Vec<Fp> = x.iter().map(|v| -*v).collect();
    *x = std::mem::replace(y, ny);
}

/// Apply the lower triangular Toeplitz `T` with first column `h` to `x`,
/// in place, as in the paper. `scratch` must come back unchanged, and the
/// function returns `T x`. Arbitrary scratch contents are allowed, because
/// the construction restores them.
pub fn lower_toeplitz_paper(
    eng: &ShearEngine,
    plan: &ToeplitzPlan,
    x: &[Fp],
    scratch: &mut Vec<Fp>,
) -> Vec<Fp> {
    let mut r1 = x.to_vec();
    // (D_u ⊕ I) B (D_u^{-1} ⊕ I) B^{-1}, applied right to left.
    apply_b_inv(eng, plan, &mut r1, scratch);
    let du_inv: Vec<Fp> = plan.du.iter().map(|v| v.inv()).collect();
    for (v, d) in r1.iter_mut().zip(&du_inv) {
        *v *= *d;
    }
    apply_b(eng, plan, &mut r1, scratch);
    for (v, d) in r1.iter_mut().zip(&plan.du) {
        *v *= *d;
    }
    // Restore h(0).
    for v in &mut r1 {
        *v *= plan.scale;
    }
    r1
}

/// Upper triangular Toeplitz `U = R L R` (`R` = reversal), with `L` having first column `h`.
pub fn upper_toeplitz_paper(
    eng: &ShearEngine,
    plan: &ToeplitzPlan,
    x: &[Fp],
    scratch: &mut Vec<Fp>,
) -> Vec<Fp> {
    let rx: Vec<Fp> = x.iter().rev().copied().collect();
    let mut y = lower_toeplitz_paper(eng, plan, &rx, scratch);
    y.reverse();
    y
}

/// Reference: upper triangular Toeplitz with first row `h`.
pub fn upper_toeplitz_direct(h: &[Fp], x: &[Fp]) -> Vec<Fp> {
    let rx: Vec<Fp> = x.iter().rev().copied().collect();
    let mut y = mul_trunc_direct(h, &rx);
    y.reverse();
    y
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paper::grid::smooth::choose_b;

    fn seq(a: usize, s: u64) -> Vec<Fp> {
        (0..a as u64).map(|i| Fp::new(i * i * s + 3 * i + s)).collect()
    }

    #[test]
    fn conv_engine_matches_direct() {
        for a in [1, 2, 5, 17, 40, 100] {
            let eng = ShearEngine::new(a, choose_b(a));
            let (f, x) = (seq(a, 7), seq(a, 11));
            assert_eq!(eng.mul_trunc(&f, &x), mul_trunc_direct(&f, &x));
        }
    }

    #[test]
    fn paper_construction_exact_and_restores_scratch() {
        for a in [1, 2, 3, 8, 17, 64, 129] {
            let eng = ShearEngine::new(a, choose_b(a));
            let h = seq(a, 5);
            let plan = ToeplitzPlan::new(&h, a, eng.dft.root);
            // f(uX) = h(X) f(X) / h(0)
            let fu: Vec<Fp> = plan.f.iter().zip(&plan.du).map(|(c, d)| *c * *d).collect();
            let hn: Vec<Fp> = h.iter().map(|v| *v * plan.scale.inv()).collect();
            assert_eq!(fu, mul_trunc_direct(&hn, &plan.f));

            let x = seq(a, 13);
            let original_scratch = seq(a, 99);
            let mut scratch = original_scratch.clone();
            let y = lower_toeplitz_paper(&eng, &plan, &x, &mut scratch);
            assert_eq!(y, mul_trunc_direct(&h, &x), "a={a}");
            assert_eq!(scratch, original_scratch, "scratch not restored, a={a}");

            let yu = upper_toeplitz_paper(&eng, &plan, &x, &mut scratch);
            assert_eq!(yu, upper_toeplitz_direct(&h, &x));
            assert_eq!(scratch, original_scratch);
        }
    }
}
