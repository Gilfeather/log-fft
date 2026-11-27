//! `cargo run --release --example demo`
//!
//! 1. Prints the paper's window parameters and where they start to bite.
//! 2. Checks prediction + Hermite/Vandermonde repair against the reference.

use log_fft::paper::ops::{apply_f_reference, OpCounter};
use log_fft::paper::params::{paper_half_width, paper_r, paper_threshold_m};
use log_fft::paper::predictor::apply_predictor;
use log_fft::paper::repair::{apply_f_repaired, balanced_shifts, paper_exception_budget};
use log_fft::{paper::Params, Fp};

fn main() {
    println!("== Paper parameters (binary case) ==");
    println!("{:>5} {:>3} {:>4} {:>6} {:>6} {:>4} {:>8} {:>12}", "m", "r", "B", "u", "v", "D", "D<m?", "exc.frac");
    for m in [3, 8, 16, 32, 48, 64, 65, 80, 96, 128, 256, 1024] {
        let p = Params::paper(m);
        println!(
            "{:>5} {:>3} {:>4} {:>6} {:>6} {:>4} {:>8} {:>12.3e}",
            m,
            paper_r(m),
            paper_half_width(m),
            p.u,
            p.v,
            p.degree(),
            p.window_bites(),
            p.exception_fraction()
        );
    }
    let t = paper_threshold_m();
    let d = Params::paper(t).degree();
    let axes = 2 * (t + d - 1);
    println!(
        "\nWindow first bites at m = {t} (D = {d}). One kernel level then uses about A = 2(m+D-1) = {axes} axes,"
    );
    println!("so a single level needs N >= 2^{axes} (rough lower estimate, not a figure from the paper).");

    println!("\n== Exact repair check ==");
    for (m, b) in [(6usize, 1.0f64), (8, 1.5), (10, 2.0)] {
        let params = Params::with_half_width(m, b);
        let w = 1usize << m;
        let shifts = balanced_shifts(m, w);
        let xs: Vec<Vec<Fp>> = (0..w)
            .map(|i| (0..1u64 << m).map(|k| Fp::new((i as u64 + 1) * 1_000_003 + k * k)).collect())
            .collect();

        // Prediction alone (no repair) is wrong at the exceptions.
        let ctr0 = OpCounter::default();
        let wrong_rows = (0..w)
            .filter(|&i| apply_predictor(&params, shifts[i], &xs[i], &ctr0) != apply_f_reference(m, &xs[i]))
            .count();

        let ctr = OpCounter::default();
        let out = apply_f_repaired(&params, &xs, &shifts, None, &ctr);
        let exact = xs.iter().zip(&out.rows).all(|(x, y)| *y == apply_f_reference(m, x));
        println!(
            "m={m:>2} window=[{},{}] D={} W={w}: prediction-only wrong rows {wrong_rows}/{w}, \
             s={} (paper budget W/2m^3 = {}), repaired exact = {exact}, \
             Q-calls={} (naive {}), spectral rows={}",
            params.u,
            params.v,
            params.degree(),
            out.s,
            paper_exception_budget(w, m),
            ctr.q(),
            w * m,
            ctr.spectral()
        );
        assert!(exact);
    }
    println!("\nNote: Q-calls are high because the predictor is expanded directly");
    println!("(no matrix-root/routing). This checks correctness, not the paper's cost savings.");

    println!("\n== Cyclic / abelian-group DFT via Newton + in-place Toeplitz (grids.tex) ==");
    use log_fft::paper::grid::{naive_group_dft, newton::vandermonde, AxisPlan, GroupDft};
    let cases: Vec<(usize, Option<u64>)> = vec![(17, None), (255, None), (1020, None), (4096, None), (1000, Some(3)), (777, Some(5))];
    for (a, q) in cases {
        let (plan, label) = match q {
            None => (AxisPlan::dft(a), "DFT".to_string()),
            Some(q) => (AxisPlan::geometric(a, Fp::new(q)), format!("V_a(q={q})")),
        };
        let x: Vec<Fp> = (0..a as u64).map(|i| Fp::new(i * 2654435761 + 1)).collect();
        let mut scratch = vec![Fp::ZERO; a];
        let y = plan.forward(&x, &mut scratch);
        let v = vandermonde(a, plan.q);
        let expect: Vec<Fp> = (0..a).map(|j| (0..a).fold(Fp::ZERO, |s, k| s + v[j][k] * x[k])).collect();
        let ok = y == expect;
        let back = plan.inverse(&y, &mut scratch) == x;
        let restored = scratch.iter().all(|v| *v == Fp::ZERO);
        println!(
            "{label:>11} a={a:>5}  b={:>5} factors={:<14}  forward exact={ok}  inverse exact={back}  scratch restored={restored}",
            plan.engine.dft.b,
            format!("{:?}", plan.engine.dft.factors),
        );
        assert!(ok && back && restored);
    }
    let dims = [3usize, 5, 17, 4];
    let g = GroupDft::new(&dims);
    let x: Vec<Fp> = (0..g.len() as u64).map(|i| Fp::new(i * i + 9)).collect();
    let exact = g.forward(&x) == naive_group_dft(&dims, &x);
    println!("G = Z3 x Z5 x Z17 x Z4 (N={}): exact = {exact}", g.len());
    assert!(exact);
}
