//! `cargo run --release --example bench`: rough timings (single-threaded).
use log_fft::{ChirpZ, Dft, Fp, InverseChirpZ, Ntt};
use std::time::Instant;

fn time<F: FnMut()>(mut f: F) -> f64 {
    let reps = 5;
    let t = Instant::now();
    for _ in 0..reps {
        f();
    }
    t.elapsed().as_secs_f64() / reps as f64 * 1e3
}

fn main() {
    println!("{:<28} {:>10} {:>12}", "transform", "n", "ms / call");
    for log in [16, 20, 22] {
        let n = 1usize << log;
        let plan = Ntt::new(n);
        let mut x: Vec<Fp> = (0..n as u64).map(Fp::new).collect();
        println!("{:<28} {:>10} {:>12.2}", "Ntt (2^k)", n, time(|| plan.forward(&mut x)));
    }
    for n in [65535usize, 255 * 4096, 65537 * 15] {
        let plan = Dft::new(n);
        let x: Vec<Fp> = (0..n as u64).map(Fp::new).collect();
        println!("{:<28} {:>10} {:>12.2}", format!("Dft {:?}", plan.factors()), n, time(|| { plan.forward(&x); }));
    }
    for n in [100_000usize, 1_000_000] {
        let q = Fp::new(3);
        let x: Vec<Fp> = (0..n as u64).map(Fp::new).collect();
        let c = ChirpZ::new(n, n, q);
        println!("{:<28} {:>10} {:>12.2}", "ChirpZ (q=3)", n, time(|| { c.apply(&x); }));
        let ic = InverseChirpZ::new(n, q);
        println!("{:<28} {:>10} {:>12.2}", "InverseChirpZ (q=3)", n, time(|| { ic.apply(&x); }));
    }
}
