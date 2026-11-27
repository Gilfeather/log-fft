//! Exact repair of a predicted batch (Lemma hermite-repair, Theorem thm:repair).
//!
//! We have `W` input rows `X_1 … X_W`. Row `i` is given shift `β_i`, and we
//! want `F X_i` for every row. Predicting each row with `P_i = P_{β_i}` is
//! right at every frequency except the exceptions
//! `J(ν) = { i : p_i(ν) ≠ z(ν) }`, and `|J(ν)| ≤ s`.
//!
//! Exactness is recovered as follows:
//!
//! 1. Vandermonde check matrix `H` (`s × W`), `H_{t,i} = x_i^t`, with distinct
//!    nodes `x_i`. Any `|J| ≤ s` columns of `H` have a left inverse made from
//!    their first `|J|` rows. This is the generalised Reed–Solomon
//!    parity-check structure.
//! 2. Hermite polynomial `g(y) = (3y − y³)/2`. It satisfies `g(±1) = ±1` and
//!    `g'(±1) = 0`, so `g(y) − z = (y − z)² · D_z(y)` with
//!    `D_z(y) = −(y + 2z)/2`.
//! 3. Per frequency, `e_i = (p_i − z)·X̂_i` and `He = H·(PX)^ − z·H·X̂`.
//!    Solving gives `e_J`. Choose `a ∈ F^s` with `(Hᵀa)_i = D_z(p_i)·e_i`
//!    for `i ∈ J`, and set `Ẑ = Hᵀa`.
//! 4. Output `g(P)X − P·Ẑ + F·Ẑ`. The per-row error at `ν` is
//!    `(p_i − z)(D_z(p_i)e_i − Ẑ_i)`. The first factor vanishes off `J` and
//!    the second vanishes on `J`, so the output is exact.
//!
//! Cost: `ℓ + 2 = 4` predictor applications (`PX`, `P²X`, `P³X`, `PẐ`).
//! The spectral table uses `2s` input rows (`HX` and `H·PX`) and `2s` output
//! rows (`a` and `F a`, since `F Hᵀ a = Hᵀ F a`). That is `4s` full row
//! conversions, matching Theorem thm:repair. The other steps are pencil
//! operations: row mixing by `H` or `Hᵀ`, plus `Q_j` calls.

use crate::field::{solve, Fp};
use crate::ops::{from_spectral, target_multiplier, to_spectral, OpCounter};
use crate::params::Params;
use crate::predictor::{apply_predictor, predictor_value};

/// Result of a repaired batch.
#[derive(Debug)]
pub struct RepairOutput {
    /// `F X_i` for every input row.
    pub rows: Vec<Vec<Fp>>,
    /// Exception budget `s` actually used.
    pub s: usize,
    /// Largest `|J(ν)|` over all frequencies.
    pub max_exceptions: usize,
}

/// The paper's exception budget `⌊W / (2m³)⌋`.
pub fn paper_exception_budget(w: usize, m: usize) -> usize {
    w / (2 * m * m * m)
}

/// Smallest admissible `s` for the given shifts: `max_ν |J(ν)|`.
pub fn required_budget(params: &Params, shifts: &[u32]) -> usize {
    (0..1usize << params.m)
        .map(|nu| exception_set(params, shifts, nu).len())
        .max()
        .unwrap_or(0)
}

fn exception_set(params: &Params, shifts: &[u32], nu: usize) -> Vec<usize> {
    let z = target_multiplier(nu);
    shifts
        .iter()
        .enumerate()
        .filter(|&(_, &b)| predictor_value(params, b, nu) != z)
        .map(|(i, _)| i)
        .collect()
}

fn node(i: usize) -> Fp {
    Fp::new(i as u64 + 1)
}

/// `Σ_i coef(t, i) · rows[i]` for `t = 0..out` (row mixing, a pencil operation).
fn mix<F: Fn(usize, usize) -> Fp>(rows: &[Vec<Fp>], out: usize, coef: F) -> Vec<Vec<Fp>> {
    let n = rows.first().map_or(0, |r| r.len());
    (0..out)
        .map(|t| {
            let mut acc = vec![Fp::ZERO; n];
            for (i, r) in rows.iter().enumerate() {
                let c = coef(t, i);
                if c != Fp::ZERO {
                    for k in 0..n {
                        acc[k] += c * r[k];
                    }
                }
            }
            acc
        })
        .collect()
}

/// Compute `F X_i` for all rows, exactly, by prediction plus repair.
///
/// `s = None` uses the smallest admissible budget. Passing `Some(s)` with
/// `s` smaller than [`required_budget`] panics, because the Vandermonde
/// argument then fails.
pub fn apply_f_repaired(
    params: &Params,
    xs: &[Vec<Fp>],
    shifts: &[u32],
    s: Option<usize>,
    ctr: &OpCounter,
) -> RepairOutput {
    let m = params.m;
    let n = 1usize << m;
    let w = xs.len();
    assert_eq!(shifts.len(), w);
    assert!(xs.iter().all(|r| r.len() == n));

    let max_exc = required_budget(params, shifts);
    let s = s.unwrap_or(max_exc);
    assert!(s >= max_exc, "budget s={s} below max |J|={max_exc}");

    // Predictor applications 1–3: PX, P²X, P³X.
    let px: Vec<_> = (0..w).map(|i| apply_predictor(params, shifts[i], &xs[i], ctr)).collect();
    let p2x: Vec<_> = (0..w).map(|i| apply_predictor(params, shifts[i], &px[i], ctr)).collect();
    let p3x: Vec<_> = (0..w).map(|i| apply_predictor(params, shifts[i], &p2x[i], ctr)).collect();

    let (a_rows, fa_rows) = if s == 0 {
        (Vec::new(), Vec::new())
    } else {
        // Check rows HX and H·PX (pencil mixing), then forward conversions.
        let hx = mix(xs, s, |t, i| node(i).pow(t as u64));
        let hpx = mix(&px, s, |t, i| node(i).pow(t as u64));
        let hx_s: Vec<_> = hx.iter().map(|r| to_spectral(r, ctr)).collect();
        let hpx_s: Vec<_> = hpx.iter().map(|r| to_spectral(r, ctr)).collect();

        // Spectral table: one linear map per frequency, 2s rows to 2s rows.
        let mut a_s = vec![vec![Fp::ZERO; n]; s];
        let mut fa_s = vec![vec![Fp::ZERO; n]; s];
        let half = Fp::new(2).inv();
        for nu in 0..n {
            let z = target_multiplier(nu);
            let jset = exception_set(params, shifts, nu);
            let k = jset.len();
            if k == 0 {
                continue;
            }
            // He_t = (H·PX)^_t − z·(HX)^_t, t < k.
            let he: Vec<Fp> = (0..k).map(|t| hpx_s[t][nu] - z * hx_s[t][nu]).collect();
            // Σ_c x_{J_c}^t e_c = He_t  ⇒  e_J.
            let vt: Vec<Vec<Fp>> = (0..k)
                .map(|t| jset.iter().map(|&i| node(i).pow(t as u64)).collect())
                .collect();
            let e = solve(vt, he);
            // Target (Hᵀa)_i = D_z(p_i)·e_i on J, D_z(p) = −(p + 2z)/2.
            let rhs: Vec<Fp> = jset
                .iter()
                .zip(&e)
                .map(|(&i, &ei)| {
                    let p = predictor_value(params, shifts[i], nu);
                    -(p + Fp::new(2) * z) * half * ei
                })
                .collect();
            // Σ_t a_t x_i^t = rhs_i for i ∈ J, a supported on t < k.
            let vj: Vec<Vec<Fp>> = jset
                .iter()
                .map(|&i| (0..k).map(|t| node(i).pow(t as u64)).collect())
                .collect();
            let a = solve(vj, rhs);
            for t in 0..k {
                a_s[t][nu] = a[t];
                fa_s[t][nu] = z * a[t];
            }
        }
        (
            a_s.iter().map(|r| from_spectral(r, ctr)).collect::<Vec<_>>(),
            fa_s.iter().map(|r| from_spectral(r, ctr)).collect::<Vec<_>>(),
        )
    };

    // Ẑ = Hᵀa, FẐ = Hᵀ(Fa)   (pencil mixing).
    let zhat = if s == 0 {
        vec![vec![Fp::ZERO; n]; w]
    } else {
        mix(&a_rows, w, |i, t| node(i).pow(t as u64))
    };
    let fzhat = if s == 0 {
        vec![vec![Fp::ZERO; n]; w]
    } else {
        mix(&fa_rows, w, |i, t| node(i).pow(t as u64))
    };

    // Predictor application 4: PẐ. Then assemble g(P)X − PẐ + FẐ.
    let half = Fp::new(2).inv();
    let three = Fp::new(3);
    let rows = (0..w)
        .map(|i| {
            let pz = if s == 0 {
                vec![Fp::ZERO; n]
            } else {
                apply_predictor(params, shifts[i], &zhat[i], ctr)
            };
            (0..n)
                .map(|k| (three * px[i][k] - p3x[i][k]) * half - pz[k] + fzhat[i][k])
                .collect()
        })
        .collect();

    RepairOutput {
        rows,
        s,
        max_exceptions: max_exc,
    }
}

/// Shifts where every `β ∈ F_2^m` is used equally often (`W` a multiple of `2^m`).
pub fn balanced_shifts(m: usize, w: usize) -> Vec<u32> {
    let n = 1usize << m;
    assert!(w % n == 0, "W must be a multiple of 2^m for balanced shifts");
    (0..w).map(|i| (i % n) as u32).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::apply_f_reference;

    fn rows(m: usize, w: usize, seed: u64) -> Vec<Vec<Fp>> {
        let mut st = seed;
        (0..w)
            .map(|_| {
                (0..1usize << m)
                    .map(|_| {
                        st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                        Fp::new(st >> 1)
                    })
                    .collect()
            })
            .collect()
    }

    fn check(params: Params, w: usize, seed: u64) -> RepairOutput {
        let ctr = OpCounter::default();
        let xs = rows(params.m, w, seed);
        let shifts = balanced_shifts(params.m, w);
        let out = apply_f_repaired(&params, &xs, &shifts, None, &ctr);
        for (x, y) in xs.iter().zip(&out.rows) {
            assert_eq!(y, &apply_f_reference(params.m, x));
        }
        assert_eq!(ctr.spectral() as usize, 4 * out.s);
        out
    }

    #[test]
    fn exact_without_exceptions() {
        let out = check(Params::paper(5), 32, 1);
        assert_eq!(out.s, 0);
    }

    #[test]
    fn exact_with_exceptions() {
        for (m, b) in [(4, 0.5), (6, 1.0), (7, 1.0), (8, 1.5)] {
            let p = Params::with_half_width(m, b);
            assert!(p.window_bites());
            let out = check(p, 1 << m, 42 + m as u64);
            assert!(out.s > 0, "expected exceptions for m={m}");
        }
    }

    #[test]
    #[should_panic(expected = "budget")]
    fn too_small_budget_rejected() {
        let p = Params::with_half_width(6, 1.0);
        let ctr = OpCounter::default();
        let xs = rows(6, 64, 3);
        apply_f_repaired(&p, &xs, &balanced_shifts(6, 64), Some(1), &ctr);
    }
}
