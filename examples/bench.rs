//! `cargo run --release --example bench`: rough timings, single-threaded vs automatic threads.
use log_fft::{ChirpZ, Dft, Fp, InverseChirpZ, Ntt};
use std::time::Instant;

fn time<F: FnMut()>(mut f: F) -> f64 {
    f(); // warm up
    let reps = 5;
    let t = Instant::now();
    for _ in 0..reps {
        f();
    }
    t.elapsed().as_secs_f64() / reps as f64 * 1e3
}

fn main() {
    println!("{:<26} {:>10} {:>10} {:>10}", "transform", "n", "1 thread", "auto");
    for log in [16, 20, 22] {
        let n = 1usize << log;
        let mut x: Vec<Fp> = (0..n as u64).map(Fp::new).collect();
        let (p1, pa) = (Ntt::new(n).with_threads(1), Ntt::new(n));
        let a = time(|| p1.forward(&mut x));
        let b = time(|| pa.forward(&mut x));
        println!("{:<26} {:>10} {:>10.2} {:>10.2}", "Ntt", n, a, b);
    }
    for n in [65535usize, 255 * 4096, 65537 * 15] {
        let x: Vec<Fp> = (0..n as u64).map(Fp::new).collect();
        let (p1, pa) = (Dft::new(n).with_threads(1), Dft::new(n));
        let a = time(|| { p1.forward(&x); });
        let b = time(|| { pa.forward(&x); });
        println!("{:<26} {:>10} {:>10.2} {:>10.2}", format!("Dft {:?}", pa.factors()), n, a, b);
    }
    for n in [100_000usize, 1_000_000] {
        let q = Fp::new(3);
        let x: Vec<Fp> = (0..n as u64).map(Fp::new).collect();
        let c = ChirpZ::new(n, n, q);
        println!("{:<26} {:>10} {:>10} {:>10.2}", "ChirpZ (q=3)", n, "-", time(|| { c.apply(&x); }));
        let ic = InverseChirpZ::new(n, q);
        println!("{:<26} {:>10} {:>10} {:>10.2}", "InverseChirpZ (q=3)", n, "-", time(|| { ic.apply(&x); }));
    }
}
