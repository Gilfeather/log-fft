//! Window parameters of the count predictor (repair.tex, binary case `ℓ = 2`).
//!
//! The paper sets
//! * `r` = the least integer with `2^r ≥ 2m`
//! * `μ = m/2`, `B = ⌈√(2 m r)⌉`
//! * window `[u, v]` with `u = ⌊μ − B⌋`, `v = ⌈μ + B⌉`
//! * predictor degree `D = v − u`
//! * exception budget `s = ⌊W / (2 m³)⌋`
//!
//! Hoeffding's inequality gives `Pr_β[c ∉ [u,v]] ≤ 2·exp(−2B²/m) ≤ 1/(2m³)`.

/// Window `[u, v]` for `m` binary queries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Params {
    /// Number of commuting involutions `Q_1 … Q_m`.
    pub m: usize,
    /// Lower end of the window (may be negative).
    pub u: i64,
    /// Upper end of the window (may exceed `m`).
    pub v: i64,
}

impl Params {
    /// The parameters exactly as chosen in the paper.
    pub fn paper(m: usize) -> Self {
        assert!(m >= 1);
        let b = paper_half_width(m);
        Self::with_half_width(m, b as f64)
    }

    /// A window `[⌊m/2 − b⌋, ⌈m/2 + b⌉]` with a chosen half-width `b`.
    ///
    /// For small `m` the paper's window covers all of `[0, m]`, so there are
    /// no exceptions and the repair step has nothing to do. A narrower window
    /// makes exceptions appear, which tests the repair step.
    pub fn with_half_width(m: usize, b: f64) -> Self {
        let mu = m as f64 / 2.0;
        Params {
            m,
            u: (mu - b).floor() as i64,
            v: (mu + b).ceil() as i64,
        }
    }

    /// An explicit window `[u, v]`.
    pub fn with_window(m: usize, u: i64, v: i64) -> Self {
        assert!(u <= v);
        Params { m, u, v }
    }

    /// Predictor degree `D = v − u`.
    pub fn degree(&self) -> usize {
        (self.v - self.u) as usize
    }

    /// True when the window is strictly narrower than the full count range
    /// `[0, m]`, that is, when the predictor really has degree below `m`.
    pub fn window_bites(&self) -> bool {
        self.degree() < self.m
    }

    /// `true` when count `c` lies inside the window.
    pub fn in_window(&self, c: i64) -> bool {
        self.u <= c && c <= self.v
    }

    /// Exact fraction of shifts `β ∈ F_2^m` whose count falls outside the
    /// window. It is the same for every frequency.
    pub fn exception_fraction(&self) -> f64 {
        // Work in log space so large m neither overflows nor underflows.
        let m = self.m as u32;
        let ln2m = m as f64 * core::f64::consts::LN_2;
        let mut out = 0f64;
        let mut ln_binom = 0f64;
        for c in 0..=m {
            if !self.in_window(c as i64) {
                out += (ln_binom - ln2m).exp();
            }
            if c < m {
                ln_binom += ((m - c) as f64).ln() - ((c + 1) as f64).ln();
            }
        }
        out
    }

    /// The paper's Hoeffding bound `2·exp(−2B²/m)` on the exception fraction,
    /// with `B = paper_half_width(m)`.
    pub fn paper_hoeffding_bound(m: usize) -> f64 {
        let b = paper_half_width(m) as f64;
        2.0 * (-2.0 * b * b / m as f64).exp()
    }
}

/// `r` = least integer with `2^r ≥ 2m`.
pub fn paper_r(m: usize) -> u32 {
    let mut r = 0;
    while (1usize << r) < 2 * m {
        r += 1;
    }
    r
}

/// `B = ⌈√(2 m r)⌉` (binary case).
pub fn paper_half_width(m: usize) -> u64 {
    let r = paper_r(m) as f64;
    (2.0 * m as f64 * r).sqrt().ceil() as u64
}

/// Smallest `m` for which the paper's window is narrower than `[0, m]`.
pub fn paper_threshold_m() -> usize {
    (1..).find(|&m| Params::paper(m).window_bites()).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_m_window_covers_everything() {
        // m = 3 gives [u, v] = [-4, 7]: the report's sanity value.
        let p = Params::paper(3);
        assert_eq!((p.u, p.v), (-4, 7));
        assert!(!p.window_bites());
        assert_eq!(p.exception_fraction(), 0.0);
    }

    #[test]
    fn hoeffding_bound_holds() {
        for m in 1..400 {
            let p = Params::paper(m);
            assert!(p.exception_fraction() <= Params::paper_hoeffding_bound(m) + 1e-15);
            assert!(Params::paper_hoeffding_bound(m) <= 1.0 / (2.0 * (m as f64).powi(3)) + 1e-15);
        }
    }

    #[test]
    fn threshold_exists() {
        let t = paper_threshold_m();
        assert!(Params::paper(t).window_bites());
        assert!(!Params::paper(t - 1).window_bites());
    }
}
