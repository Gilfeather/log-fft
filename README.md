# log-fft

Exact Fourier transforms over the Goldilocks field `F_p`, `p = 2^64 − 2^32 + 1`.
It also includes executable building blocks of the construction in
**"The logarithmic exponent of finite Fourier transforms"** (OpenAI preprint,
2026-11-24, [openai/math](https://github.com/openai/math)).

All arithmetic is exact modular arithmetic, with no floating point and no
dependencies.

## Transforms

| Type | What it computes | Cost |
|---|---|---|
| `Ntt` | power-of-two NTT, `n ≤ 2^32`, in place; batch and bit-reversed entry points | `O(n log n)` |
| `Dft` | DFT of **every** length `F_p` supports (`n` divides `p − 1 = 2^32·3·5·17·257·65537`), via Good–Thomas + Rader | `O(n log n)` |
| `ChirpZ` | `y_k = Σ_j x_j q^{jk}`, any nonzero `q`, any `n` inputs and `m` outputs (no `√q` needed) | `O((n+m) log(n+m))` |
| `InverseChirpZ` | interpolation at geometric nodes `q^0 … q^{n−1}`, via the paper's Newton factorisation | `O(n log n)` |
| `GroupDft` | DFT on `Z_{a_1} × … × Z_{a_r}` | `O(N log N)` |
| `convolve`, `series_inverse` | exact polynomial product and power-series inverse | `O(n log n)` |

```rust
use log_fft::{ChirpZ, Dft, Fp, InverseChirpZ, Ntt};

let dft = Dft::new(255); // 255 = 3·5·17
let x: Vec<Fp> = (0..255).map(Fp::new).collect();
assert_eq!(dft.inverse(&dft.forward(&x)), x);

let q = Fp::new(3);
let y = ChirpZ::new(1000, 1000, q).apply(&x.repeat(4)[..1000]);
let back = InverseChirpZ::new(1000, q).apply(&y);
assert_eq!(back, x.repeat(4)[..1000].to_vec());

// Zero-copy on raw u64 data (all values must be < p):
let mut raw: Vec<u64> = (0..1024).collect();
Ntt::new(1024).forward(Fp::from_canonical_slice_mut(&mut raw).unwrap());
```

**API notes**

- **In place and batched:** `Dft::{forward_in_place, inverse_in_place, forward_batch, inverse_batch}`, `Ntt::{forward_batch, inverse_batch}`, `GroupDft::{forward_in_place, inverse_in_place}`.
- **Convolution building blocks:** `Ntt::forward_bitrev` and `Ntt::inverse_from_bitrev` skip both bit-reversal permutations.
- **Threads:** large transforms (≥ 2^18 elements) use scoped threads automatically. Use `.with_threads(1)` to force a single thread, or `.with_threads(t)` to pick a count.
- **`Fp`:** `From<u64|u32|i64|bool>`, `u64::from(fp)`, `Fp::from_canonical`, `Display`, `/`, `Sum`, and `Product`. Zero-copy slice views are available via `Fp::as_u64_slice` and `Fp::from_canonical_slice{,_mut}`.

## Features

- `std` (default) enables threads and the `paper` module (which uses `f64` math).
- `default-features = false` gives `no_std` + `alloc`, single-threaded. This is checked on `thumbv7em-none-eabihf`.

There are no dependencies.

## Formal verification (Kani)

`cargo kani` checks these claims for **every** input in the domain, not a
sample. Kani also rules out panics, overflow and out-of-bounds access on
those paths:

| Harness | Claim |
|---|---|
| `reduce128_is_mod_p` | the Goldilocks reduction returns `x mod p` for all 2^128 inputs |
| `mul_is_reduced_product` | `a * b` is that reduction of the exact 128-bit product |
| `add_is_mod_p`, `sub_is_mod_p`, `neg_is_mod_p` | `+`, `−`, negation are exact mod `p` on canonical inputs |
| `new_reduces_and_from_canonical_rejects`, `from_i64_is_mod_p` | constructors reduce correctly; `from_canonical` accepts exactly `v < p` |
| `canonical_slice_view` | the zero-copy `&[u64] → &[Fp]` view accepts exactly canonical data and keeps the values |
| `bit_reverse_involution` | the bit-reversal permutation is correct and self-inverse (length 8) |

The proof of `reduce128` avoids 128-bit division. It uses `2^64 ≡ 2^32−1`
and `2^96 ≡ −1 (mod p)`, which the compiler checks with `const` assertions,
and shows the result differs from `x` by a small multiple of `p`. A
deliberately broken reduction is rejected by the same harness.

Not formally verified: the transforms themselves (`Ntt`, `Dft`, …). A
symbolic proof even for `n = 2` did not finish within 25 minutes. Those are
covered by the exhaustive and randomised tests against `O(n²)` references.

## Performance

`cargo run --release --example bench` on Apple silicon (12 cores). Times are in ms, and the machine was under load, so treat them as rough:

| Transform | n | v0.1.1 | v0.1.2, 1 thread | v0.1.2, auto threads |
|---|---|---|---|---|
| `Ntt` | 2^20 | 21.8 | 9.6 | 4.0 |
| `Ntt` | 2^22 | 83.0 | 43 | 17 |
| `Dft` (4096·3·5·17) | 1,044,480 | 39.2 | 27.6 | 10.1 |
| `Dft` (3·5·65537) | 983,055 | 116.6 | 24.0 | 7.6 |
| `ChirpZ` | 10^6 | 77.9 | — | 9.5 |
| `InverseChirpZ` | 10^6 | 156.4 | — | 21.7 |

## About the paper, and what this crate does with it

The paper proves that exact finite Fourier transforms have linear circuits
of size `N (log N)^{1/2+o(1)}`, and that `N √(log N)` gates are necessary.
The result is asymptotic:

* The constants and parameters (`m`, `L`, `q`, `r`) are only stated to be
  "large enough", and the recursion is defined to use **direct transforms
  for bounded sizes**.
* From the paper's own cost formulas (crate author's estimate, not a figure
  in the paper), one recursion level only saves anything once a single
  array has more than about `2^500` entries. Reaching the paper's exponent
  needs far more.

So at every size a computer can hold, the paper's algorithm runs the direct
path, and that path is what the transforms above implement. The paper's
lemmas are implemented as exact, tested code in `log_fft::paper`:

| Module | Paper | Checked |
|---|---|---|
| `paper::predictor` | Lemma count-predict | `P_β` returns exactly `z` on the window, and is wrong outside it |
| `paper::repair` | Lemma hermite-repair, Thm thm:repair | Vandermonde (GRS) checks + `g(y)=(3y−y³)/2` give exact output; 4 predictor applications, `4s` spectral rows |
| `paper::matrix_root` | Lemma matrix-root | `L^b = p·I` |
| `paper::params` | repair.tex | window `u, v, D`, Hoeffding bound |
| `paper::grid` | Lemmas up:newton-lemma, up:toeplitz, up:fibers | `V = LΔ⁻¹Lᵀ`; in-place `T = D_u S D_u⁻¹ S⁻¹` from shears with scratch restored; group DFTs |

Not implemented: the routing and kernel recursion (kernel.tex), and the
jet/deformation convolution over fields without roots of unity (packing.tex).
These could be added later as an opt-in path, but by the estimate above they
would not be faster at any feasible size.

`cargo run --release --example paper_demo` prints the paper's parameters and
runs the repair and grid checks.

## License

MIT OR Apache-2.0
