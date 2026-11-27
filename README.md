# logfourier

An exact, executable check of the main building blocks in OpenAI's preprint
**"The logarithmic exponent of finite Fourier transforms"** (2026-11-24,
[openai/math](https://github.com/openai/math)). The paper proves
`N (log N)^{1/2+o(1)}` upper bounds and an `N √(log N)` lower bound for exact
finite Fourier transforms, counted in 2-input linear gates.

> **This is not a fast FFT.** The result is asymptotic and its constants are
> not explicit. With the paper's own parameters, the predictor window is first
> narrower than `[0, m]` at `m = 60`. One recursion level then needs about
> 234 axes, so the method cannot win below roughly `N ≈ 2^234`. That figure
> is this crate's estimate, not the paper's. For real workloads use `rustfft`.

## What is checked (exactly, in `F_p`, `p = 2^64 − 2^32 + 1`)

| Module | Paper | What it checks |
|---|---|---|
| `predictor` | Lemma count-predict | `P_β` returns exactly `z` on the window `c ∈ [u,v]` and is wrong outside it |
| `repair` | Lemma hermite-repair / Thm thm:repair | Vandermonde check plus `g(y)=(3y−y³)/2` repair gives exact output, using 4 predictor applications and `4s` spectral row conversions |
| `matrix_root` | Lemma matrix-root | `L^b = p·I` for the `ξ`-commuting pencil |
| `params` | repair.tex | window `u, v, D`, exception budget, Hoeffding bound |
| `grid::newton` | Lemma up:newton-lemma | `L_ij = (−1)^j q^{C(j,2)} H_i/H_{i−j}` equals `VP`, and `V = LΔ⁻¹Lᵀ` |
| `grid::toeplitz` | Lemma up:toeplitz | `T = D_u S D_u⁻¹ S⁻¹` with `B = diag(S,S⁻¹)` built from 3 shears and an interchange; result exact, scratch restored |
| `grid::smooth` | grids.tex step 5 | Good–Thomas `F_b` for the auxiliary length, `2a ≤ b < 6a` |
| `grid` | Thm up:grid / thm:main | cyclic DFT `V_a(ω)`, geometric evaluation `V_a(q)`, their inverses, and DFTs on `Z_{a1}×…×Z_{ar}` match the naive `O(N²)` results |

Test setting: `Q_j` is translation by `e_j` on `F_2^m` (commuting
involutions diagonalised by the WHT), and the target is
`F = Q_1⋯Q_m`.

## Not implemented

- Routing (route.tex) and the recursion that drives the exponent toward 1/2 (kernel.tex)
- In grids.tex: grouping axes up to `N^{1/24}`, extension-field packing `E = K(ζ_M)`, and shear-layer cost accounting. Good–Thomas factors are drawn from primes dividing `p − 1` (3, 5, 17, 257), not the paper's primes in `[t, t²]`. `paper_b_a` computes the paper's choice for reference.
- The predictor is expanded directly, without the matrix-root plus routing
  construction, so `OpCounter`'s `Q`-call counts do **not** reflect the
  paper's savings.

## Run

```bash
cargo test --release
cargo run --release --example demo
```
