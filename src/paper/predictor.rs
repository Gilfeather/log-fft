//! Count predictor `P_β` (Lemma count-predict, binary case `ℓ = 2`, `ζ = −1`).
//!
//! For a shift `β ∈ F_2^m`, at frequency `ν` let `b_j = ν_j ⊕ β_j` and
//! `c = Σ_j b_j`. Then `(−1)^{c − |β|} = z(ν)`. The predictor is
//!
//! ```text
//! P_β = (−1)^{u − |β|} · Σ_{h=0}^{D} (−2)^h · L_h,
//! L_h = [T^h] (1+T)^{−u} · Π_j (1 + T·π_j),
//! π_j = I_{j,1} = (1 − (−1)^{β_j} Q_j) / 2.
//! ```
//!
//! `π_j` is the spectral projector onto `{b_j = 1}`. Since `π_j` has
//! eigenvalue `b_j`, `L_h` has eigenvalue `binom(c − u, h)`. The truncated
//! binomial sum `Σ_{h≤D} (−2)^h binom(c−u, h)` equals `(1−2)^{c−u}` exactly
//! when `0 ≤ c − u ≤ D`, that is, on the window. There `P_β` has eigenvalue
//! `z(ν)`. Outside the window it gives some other integer: an exception.
//!
//! Simplification: here `Π_j (1 + Tπ_j)` is expanded by a direct dynamic
//! programme, which costs `m·D` `Q`-calls per row. The paper instead uses the
//! matrix-root trick (see [`crate::paper::matrix_root`]) plus routing, so that one
//! predictor application is `O(D)` pencil stages. This module checks that
//! the predictor is correct, not how much it costs.

use crate::field::{binom, Fp};
use crate::paper::ops::{apply_q, OpCounter};
use crate::paper::params::Params;

/// Coefficients `(−1)^{u−|β|} (−2)^h binom(−u, h−i)` mixing `e_i(π)` into `P_β`,
/// returned as a vector over `i = 0..=D`.
fn mixing_coeffs(params: &Params, beta: u32) -> Vec<Fp> {
    let d = params.degree();
    let pre = Fp::sign(params.u - beta.count_ones() as i64);
    let mut w = vec![Fp::ZERO; d + 1];
    let mut pow = Fp::ONE; // (−2)^h
    for h in 0..=d {
        for i in 0..=h {
            w[i] += pow * binom(-params.u, h - i);
        }
        pow *= Fp::from_i64(-2);
    }
    w.iter().map(|&c| c * pre).collect()
}

/// Apply `P_β` to one row `x` using single `Q_j` calls.
pub fn apply_predictor(params: &Params, beta: u32, x: &[Fp], ctr: &OpCounter) -> Vec<Fp> {
    let d = params.degree();
    let n = x.len();
    let half = Fp::new(2).inv();
    // e[i] accumulates e_i(π_1, …, π_j) · x, truncated at degree D.
    let mut e: Vec<Vec<Fp>> = vec![vec![Fp::ZERO; n]; d + 1];
    e[0] = x.to_vec();
    for j in 0..params.m {
        let sgn = Fp::sign(((beta >> j) & 1) as i64);
        let top = d.min(j + 1);
        for i in (1..=top).rev() {
            // e[i] += π_j e[i−1],  π_j y = (y − sgn·Q_j y)/2
            let qy = apply_q(j, &e[i - 1], ctr);
            for t in 0..n {
                let v = (e[i - 1][t] - sgn * qy[t]) * half;
                e[i][t] += v;
            }
        }
    }
    let w = mixing_coeffs(params, beta);
    let mut out = vec![Fp::ZERO; n];
    for (i, ei) in e.iter().enumerate() {
        for t in 0..n {
            out[t] += w[i] * ei[t];
        }
    }
    out
}

/// The scalar eigenvalue `p_β(ν)` of `P_β` at frequency `ν`.
pub fn predictor_value(params: &Params, beta: u32, nu: usize) -> Fp {
    let c = ((nu as u32 ^ beta) & mask(params.m)).count_ones() as i64;
    count_value(params, beta, c)
}

/// Predictor value as a function of the count `c`.
pub fn count_value(params: &Params, beta: u32, c: i64) -> Fp {
    let d = params.degree();
    let pre = Fp::sign(params.u - beta.count_ones() as i64);
    let mut s = Fp::ZERO;
    let mut pow = Fp::ONE;
    for h in 0..=d {
        s += pow * binom(c - params.u, h);
        pow *= Fp::from_i64(-2);
    }
    pre * s
}

/// Count `c = |ν ⊕ β|` restricted to `m` bits.
pub fn count(m: usize, beta: u32, nu: usize) -> i64 {
    ((nu as u32 ^ beta) & mask(m)).count_ones() as i64
}

fn mask(m: usize) -> u32 {
    if m >= 32 {
        u32::MAX
    } else {
        (1u32 << m) - 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paper::ops::{target_multiplier, to_spectral};

    #[test]
    fn exact_on_window_wrong_outside() {
        let m = 9;
        let params = Params::with_window(m, 3, 6);
        for beta in 0..1u32 << m {
            for nu in 0..1usize << m {
                let c = count(m, beta, nu);
                let p = predictor_value(&params, beta, nu);
                if params.in_window(c) {
                    assert_eq!(p, target_multiplier(nu));
                } else {
                    assert_ne!(p, target_multiplier(nu), "β={beta} ν={nu} c={c}");
                }
            }
        }
    }

    #[test]
    fn paper_m3_example() {
        // The paper's illustration: on 1 ≤ c ≤ 2 the predictor is (−1)^{|β|}(2c−3).
        let params = Params::with_window(3, 1, 2);
        for beta in 0..8u32 {
            for c in 1..=2 {
                let expect = Fp::sign(beta.count_ones() as i64) * Fp::from_i64(2 * c - 3);
                assert_eq!(count_value(&params, beta, c), expect);
            }
        }
    }

    #[test]
    fn operator_matches_spectral_values() {
        let m = 6;
        let params = Params::with_half_width(m, 1.0);
        let ctr = OpCounter::default();
        let x: Vec<Fp> = (0..1u64 << m).map(|i| Fp::new(i * 7919 + 13)).collect();
        let xs = to_spectral(&x, &ctr);
        for beta in [0u32, 5, 0b101101, 63] {
            let px = to_spectral(&apply_predictor(&params, beta, &x, &ctr), &ctr);
            for nu in 0..1usize << m {
                assert_eq!(px[nu], predictor_value(&params, beta, nu) * xs[nu]);
            }
        }
    }
}
