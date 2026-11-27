//! Executable building blocks from *"The logarithmic exponent of finite
//! Fourier transforms"* (OpenAI preprint, 2026-11-24,
//! <https://github.com/openai/math>).
//!
//! The paper proves `N (log N)^{1/2+o(1)}` linear-circuit bounds. Its
//! recursion only engages above sizes far beyond any computer, and the paper
//! itself uses direct transforms for bounded sizes. So the fast transforms
//! in this crate ([`crate::ntt`], [`crate::dft`], [`crate::chirp`]) are the
//! direct path. This module implements the paper's lemmas as exact, tested
//! code:
//!
//! * [`predictor`]: count predictor `P_β` (Lemma count-predict, `ℓ = 2`)
//! * [`repair`]: Vandermonde check + Hermite `g(y) = (3y − y³)/2` repair
//!   (Lemma hermite-repair, Theorem thm:repair)
//! * [`matrix_root`]: `L^b = p·I` for the `ξ`-commuting pencil (Lemma matrix-root)
//! * [`params`]: window `u, v, D`, exception budget, Hoeffding bound
//! * [`ops`]: commuting involutions on `F_2^m` used as the test setting
//! * [`grid`]: Newton factorisation `V = LΔ⁻¹Lᵀ`, the in-place Toeplitz
//!   construction `T = D_u S D_u⁻¹ S⁻¹` with shears, and fibre-wise group DFTs
//!   (layers.tex, grids.tex). It is written for clarity, not speed. The
//!   fast inverse chirp-z in [`crate::chirp`] uses the same Newton
//!   factorisation.

pub mod grid;
pub mod matrix_root;
pub mod ops;
pub mod params;
pub mod predictor;
pub mod repair;

pub use params::Params;
