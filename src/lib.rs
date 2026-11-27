//! # log-fft
//!
//! Exact Fourier transforms over the Goldilocks field
//! `F_p`, `p = 2^64 − 2^32 + 1`. The crate also includes executable building
//! blocks of the construction in *"The logarithmic exponent of finite
//! Fourier transforms"* (OpenAI preprint, 2026-11-24,
//! <https://github.com/openai/math>).
//!
//! ## Transforms
//!
//! * [`Ntt`]: power-of-two NTT (`n ≤ 2^32`), in place, with batch and
//!   bit-reversed (permutation-free) entry points.
//! * [`Dft`]: DFT of **every** length `F_p` supports (`n | p − 1`), using
//!   Good–Thomas over the coprime factors `2^j, 3, 5, 17, 257, 65537`, with
//!   Rader's reduction for the Fermat primes `17, 257, 65537`.
//! * [`ChirpZ`]: `y_k = Σ_j x_j q^{jk}` for any nonzero `q` and any sizes,
//!   without needing `√q`.
//! * [`InverseChirpZ`]: interpolation at geometric nodes `q^0 … q^{n−1}`.
//!   It uses the paper's Newton factorisation `V = D_H · T · M · Tᵀ · D_H`,
//!   so it runs in `O(n log n)`.
//! * [`GroupDft`]: DFT on `Z_{a_1} × … × Z_{a_r}`.
//! * [`convolve`], [`series_inverse`]: exact polynomial multiplication and
//!   power-series inversion.
//!
//! Everything is exact modular arithmetic, with no rounding. Large transforms
//! run on scoped threads by default (`with_threads(1)` turns this off). With
//! `default-features = false` the crate is `no_std` + `alloc`.
//!
//! ```
//! use log_fft::{Dft, Fp};
//! let dft = Dft::new(255); // 255 = 3·5·17
//! let x: Vec<Fp> = (0..255).map(Fp::new).collect();
//! let y = dft.forward(&x);
//! assert_eq!(dft.inverse(&y), x);
//! ```
//!
//! ## About the paper
//!
//! The paper proves `N(log N)^{1/2+o(1)}` circuit bounds. Its constants are
//! non-explicit, and its recursion is defined to use direct transforms for
//! bounded sizes. From its own cost formulas, any saving needs arrays
//! beyond about `2^500` entries. So the transforms above are the path the
//! paper's algorithm takes at every size a computer can hold. The paper's
//! lemmas themselves (predictor, Hermite repair, matrix root, Newton/Toeplitz
//! reduction) are implemented and tested in [`paper`].

#![cfg_attr(not(feature = "std"), no_std)]
// Index-heavy numeric code reads more clearly with explicit index loops.
#![allow(clippy::needless_range_loop)]

extern crate alloc;

pub mod chirp;
pub mod dft;
pub mod field;
pub mod ntt;
#[cfg(feature = "std")]
pub mod paper;
mod par;
#[cfg(kani)]
mod proofs;

pub use chirp::{ChirpZ, InverseChirpZ};
pub use dft::{is_supported_len, Dft, GroupDft};
pub use field::Fp;
pub use ntt::{convolve, series_inverse, Ntt};

#[cfg(doctest)]
#[doc = include_str!("../README.md")]
pub struct ReadmeDoctests;
