//! # logfourier
//!
//! An exact, executable check of the main building blocks in
//! *"The logarithmic exponent of finite Fourier transforms"* (OpenAI preprint,
//! 2026-11-24, <https://github.com/openai/math>). The paper proves
//! `N (log N)^{1/2+o(1)}` linear-circuit upper bounds and a matching
//! `N √(log N)` lower bound for exact finite Fourier transforms.
//!
//! **This is not a fast FFT.** The paper's upper bound is asymptotic, with no
//! explicit constants. Its predictor window only becomes narrower than
//! `[0, m]` once `m` is large (see [`params::paper_threshold_m`]). Each level
//! of the recursion then needs about `2(m + D)` axes. For real FFTs, use
//! `rustfft` or a similar crate.
//!
//! What this crate checks, all exactly in `F_p` (`p = 2^64 − 2^32 + 1`):
//!
//! * [`predictor`]: the count predictor `P_β` gives exactly `z = Π_j q_j` on
//!   the window `c ∈ [u, v]` (Lemma count-predict, binary case).
//! * [`repair`]: a Vandermonde (generalised Reed–Solomon) check combined with
//!   the Hermite polynomial `g(y) = (3y − y³)/2` turns a batch of predictions
//!   with at most `s` exceptions per frequency into exact output. It uses
//!   `ℓ + 2 = 4` predictor applications and `4s` spectral row conversions
//!   (Lemma hermite-repair, Theorem thm:repair).
//! * [`matrix_root`]: `L^b = p · I` for the `ξ`-commuting pencil
//!   (Lemma matrix-root).
//! * [`params`]: the paper's window parameters and the Hoeffding bound on
//!   exceptions.
//!
//! Not implemented: routing (route.tex), the recursion that drives the
//! exponent toward 1/2 (kernel.tex), and the extension to cyclic groups and
//! general fields (grids.tex). The predictor here is expanded directly, not
//! through the matrix-root plus routing construction. So the `Q`-call counts
//! reported by [`ops::OpCounter`] do **not** reflect the paper's savings.

pub mod field;
pub mod matrix_root;
pub mod ops;
pub mod params;
pub mod predictor;
pub mod repair;

pub use field::Fp;
pub use params::Params;
