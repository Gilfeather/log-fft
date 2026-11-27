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
| `Ntt` | power-of-two NTT, `n ≤ 2^32`, in place | `O(n log n)` |
| `Dft` | DFT of **every** length `F_p` supports (`n` divides `p − 1 = 2^32·3·5·17·257·65537`), via Good–Thomas | `O(n log n)` |
| `ChirpZ` | `y_k = Σ_j x_j q^{jk}`, any nonzero `q`, any `n` inputs and `m` outputs (no `√q` needed) | `O((n+m) log(n+m))` |
| `InverseChirpZ` | interpolation at geometric nodes `q^0 … q^{n−1}`, via the paper's Newton factorisation | `O(n log n)` |
| `GroupDft` | DFT on `Z_{a_1} × … × Z_{a_r}` | `O(N log N)` |
| `convolve`, `series_inverse` | exact polynomial product and power-series inverse | `O(n log n)` |

```rust
use log_fft::{ChirpZ, Dft, Fp, InverseChirpZ};

let dft = Dft::new(255); // 255 = 3·5·17
let x: Vec<Fp> = (0..255).map(Fp::new).collect();
assert_eq!(dft.inverse(&dft.forward(&x)), x);

let q = Fp::new(3);
let y = ChirpZ::new(1000, 1000, q).apply(&x.repeat(4)[..1000]);
let back = InverseChirpZ::new(1000, q).apply(&y);
assert_eq!(back, x.repeat(4)[..1000].to_vec());
```

Rough single-threaded timings (`cargo run --release --example bench`, Apple silicon):

| Transform | n | ms |
|---|---|---|
| `Ntt` | 2^20 | ~21 |
| `Dft` (4096·3·5·17) | 1,044,480 | ~39 |
| `ChirpZ` | 10^6 | ~78 |
| `InverseChirpZ` | 10^6 | ~154 |

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
