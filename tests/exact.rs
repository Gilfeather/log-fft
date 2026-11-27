//! Exactness tests against naive `O(n²)` references, exhaustive sweeps over
//! supported lengths, random sweeps, edge cases, and spot checks at large sizes.

#![allow(clippy::needless_range_loop)]
use log_fft::field::P;
use log_fft::{convolve, is_supported_len, series_inverse, ChirpZ, Dft, Fp, GroupDft, InverseChirpZ, Ntt};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        // splitmix64
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn fp(&mut self) -> Fp {
        // Bias toward edge values sometimes.
        match self.next() % 16 {
            0 => Fp::ZERO,
            1 => Fp::ONE,
            2 => Fp::new(P - 1),
            3 => Fp::new(self.next() % 8),
            _ => Fp::new(self.next()),
        }
    }
    fn vec(&mut self, n: usize) -> Vec<Fp> {
        (0..n).map(|_| self.fp()).collect()
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

fn naive_eval(x: &[Fp], q: Fp, m: usize) -> Vec<Fp> {
    (0..m)
        .map(|k| {
            let qk = q.pow(k as u64);
            let mut acc = Fp::ZERO;
            for &xj in x.iter().rev() {
                acc = acc * qk + xj; // Horner
            }
            acc
        })
        .collect()
}

fn one_output(x: &[Fp], q: Fp, k: usize) -> Fp {
    let qk = q.pow(k as u64);
    x.iter().rev().fold(Fp::ZERO, |acc, &xj| acc * qk + xj)
}

fn divisors_of_p_minus_1(limit: usize) -> Vec<usize> {
    let mut out = Vec::new();
    for mask in 0..32u32 {
        let odd: u64 = [3u64, 5, 17, 257, 65537]
            .iter()
            .enumerate()
            .filter(|(i, _)| mask >> i & 1 == 1)
            .map(|(_, q)| *q)
            .product();
        for j in 0..=32 {
            let n = odd << j;
            if n as usize <= limit {
                out.push(n as usize);
            }
        }
    }
    out.sort();
    out
}

// ---------------------------------------------------------------- field

#[test]
fn field_ring_axioms_random() {
    let mut r = Rng(1);
    for _ in 0..200_000 {
        let (a, b, c) = (r.fp(), r.fp(), r.fp());
        assert_eq!((a + b) + c, a + (b + c));
        assert_eq!((a * b) * c, a * (b * c));
        assert_eq!(a * (b + c), a * b + a * c);
        assert_eq!(a - b + b, a);
        assert_eq!(a + (-a), Fp::ZERO);
        if a != Fp::ZERO {
            assert_eq!(a * a.inv(), Fp::ONE);
        }
        let reference = (a.value() as u128 * b.value() as u128 % P as u128) as u64;
        assert_eq!((a * b).value(), reference);
    }
}

#[test]
fn roots_of_unity_have_exact_order() {
    for n in divisors_of_p_minus_1(usize::MAX) {
        let w = Fp::root_of_unity(n as u64);
        assert_eq!(w.pow(n as u64), Fp::ONE, "n={n}");
        for q in [2u64, 3, 5, 17, 257, 65537] {
            if n as u64 % q == 0 {
                assert_ne!(w.pow(n as u64 / q), Fp::ONE, "ω_{n} not primitive (factor {q})");
            }
        }
    }
}

// ---------------------------------------------------------------- NTT

#[test]
fn ntt_matches_naive_all_small_powers() {
    let mut r = Rng(2);
    for log in 0..=11 {
        let n = 1usize << log;
        let plan = Ntt::new(n);
        for _ in 0..3 {
            let x = r.vec(n);
            let mut y = x.clone();
            plan.forward(&mut y);
            assert_eq!(y, naive_eval(&x, plan.root(), n), "n={n}");
            plan.inverse(&mut y);
            assert_eq!(y, x, "roundtrip n={n}");
        }
    }
}

#[test]
fn ntt_large_spot_checks() {
    let mut r = Rng(3);
    for log in [16, 20, 22] {
        let n = 1usize << log;
        let plan = Ntt::new(n);
        let x = r.vec(n);
        let mut y = x.clone();
        plan.forward(&mut y);
        for _ in 0..8 {
            let k = r.below(n);
            assert_eq!(y[k], one_output(&x, plan.root(), k), "n={n} k={k}");
        }
        plan.inverse(&mut y);
        assert_eq!(y, x);
    }
}

#[test]
#[should_panic(expected = "not primitive")]
fn ntt_rejects_non_primitive_root() {
    Ntt::with_root(8, Fp::root_of_unity(4));
}

#[test]
#[should_panic(expected = "power of two")]
fn ntt_rejects_non_power_of_two() {
    Ntt::new(12);
}

// ---------------------------------------------------------------- DFT

#[test]
fn dft_every_supported_length_up_to_2048_matches_naive() {
    let mut r = Rng(4);
    let lens = divisors_of_p_minus_1(2048);
    assert!(lens.len() > 50);
    for n in lens {
        let plan = Dft::new(n);
        let x = r.vec(n);
        let y = plan.forward(&x);
        assert_eq!(y, naive_eval(&x, Fp::root_of_unity(n as u64), n), "n={n} factors={:?}", plan.factors());
        assert_eq!(plan.inverse(&y), x, "roundtrip n={n}");
    }
}

#[test]
fn dft_large_supported_lengths_spot_checks() {
    let mut r = Rng(5);
    for n in [65537usize, 257 * 255, 65537 * 3, 1 << 18, (1 << 10) * 255, 257 * 17 * 5 * 3 * 8] {
        assert!(is_supported_len(n));
        let plan = Dft::new(n);
        let x = r.vec(n);
        let y = plan.forward(&x);
        let w = Fp::root_of_unity(n as u64);
        for k in [0, 1, n - 1, r.below(n), r.below(n)] {
            assert_eq!(y[k], one_output(&x, w, k), "n={n} k={k}");
        }
        assert_eq!(plan.inverse(&y), x, "roundtrip n={n}");
    }
}

#[test]
fn dft_agrees_with_ntt_and_chirp() {
    let mut r = Rng(6);
    for n in [16usize, 1024] {
        let x = r.vec(n);
        let mut y = x.clone();
        Ntt::new(n).forward(&mut y);
        assert_eq!(Dft::new(n).forward(&x), y);
    }
    for n in [255usize, 771] {
        let x = r.vec(n);
        let c = ChirpZ::new(n, n, Fp::root_of_unity(n as u64)).apply(&x);
        assert_eq!(Dft::new(n).forward(&x), c);
    }
}

#[test]
fn dft_is_linear_and_shift_property_holds() {
    let mut r = Rng(7);
    let n = 255 * 4;
    let plan = Dft::new(n);
    let (x, z) = (r.vec(n), r.vec(n));
    let a = r.fp();
    let lhs = plan.forward(&x.iter().zip(&z).map(|(u, v)| a * *u + *v).collect::<Vec<_>>());
    let (fx, fz) = (plan.forward(&x), plan.forward(&z));
    let rhs: Vec<Fp> = fx.iter().zip(&fz).map(|(u, v)| a * *u + *v).collect();
    assert_eq!(lhs, rhs);
    // Cyclic shift by s multiplies coefficient k by ω^{-sk}.
    let s = 37;
    let shifted: Vec<Fp> = (0..n).map(|j| x[(j + s) % n]).collect();
    let fs = plan.forward(&shifted);
    let w = Fp::root_of_unity(n as u64);
    for k in 0..n {
        assert_eq!(fs[k], fx[k] * w.pow(((n - s) * k % n) as u64));
    }
}

#[test]
#[should_panic(expected = "no primitive")]
fn dft_rejects_unsupported_length() {
    Dft::new(7);
}

// ---------------------------------------------------------------- chirp-z

#[test]
fn chirp_random_sizes_and_ratios() {
    let mut r = Rng(8);
    let special = [Fp::ONE, -Fp::ONE, Fp::new(2), Fp::root_of_unity(3), Fp::root_of_unity(64), Fp::new(P - 2)];
    for it in 0..400 {
        let n = 1 + r.below(120);
        let m = 1 + r.below(120);
        let q = if it % 4 == 0 { special[r.below(special.len())] } else { loop { let q = r.fp(); if q != Fp::ZERO { break q; } } };
        let x = r.vec(n);
        assert_eq!(ChirpZ::new(n, m, q).apply(&x), naive_eval(&x, q, m), "n={n} m={m} q={q:?}");
    }
}

#[test]
fn chirp_large_spot_checks() {
    let mut r = Rng(9);
    let (n, m) = (100_003usize, 70_001usize);
    let q = Fp::new(0x1234_5678_9ABC);
    let x = r.vec(n);
    let y = ChirpZ::new(n, m, q).apply(&x);
    for k in [0, 1, m - 1, r.below(m), r.below(m), r.below(m)] {
        assert_eq!(y[k], one_output(&x, q, k), "k={k}");
    }
}

#[test]
fn inverse_chirp_roundtrip_random() {
    let mut r = Rng(10);
    for _ in 0..300 {
        let n = 1 + r.below(300);
        let q = loop {
            let q = r.fp();
            if q != Fp::ZERO && (1..n).all(|k| q.pow(k as u64) != Fp::ONE) {
                break q;
            }
        };
        let x = r.vec(n);
        let y = ChirpZ::new(n, n, q).apply(&x);
        assert_eq!(InverseChirpZ::new(n, q).apply(&y), x, "n={n} q={q:?}");
    }
}

#[test]
fn inverse_chirp_is_true_inverse_of_dft() {
    let mut r = Rng(11);
    for n in [3usize, 17, 255, 4096, 65537] {
        let w = Fp::root_of_unity(n as u64);
        let x = r.vec(n);
        let y = Dft::new(n).forward(&x);
        assert_eq!(InverseChirpZ::new(n, w).apply(&y), x, "n={n}");
    }
}

#[test]
fn inverse_chirp_large() {
    let mut r = Rng(12);
    let n = 200_000;
    let q = Fp::new(7);
    let x = r.vec(n);
    let y = ChirpZ::new(n, n, q).apply(&x);
    assert_eq!(InverseChirpZ::new(n, q).apply(&y), x);
}

#[test]
#[should_panic(expected = "not distinct")]
fn inverse_chirp_rejects_repeated_nodes() {
    InverseChirpZ::new(5, Fp::root_of_unity(4));
}

// ---------------------------------------------------------------- group DFT

#[test]
fn group_dft_matches_naive() {
    let mut r = Rng(13);
    for dims in [vec![1], vec![4], vec![3, 5], vec![2, 3, 5], vec![17, 4], vec![8, 15], vec![3, 3, 3], vec![5, 1, 4], vec![2, 2, 2, 2, 2]] {
        let g = GroupDft::new(&dims);
        let x = r.vec(g.len());
        let y = g.forward(&x);
        let roots: Vec<Fp> = dims.iter().map(|&a| Fp::root_of_unity(a as u64)).collect();
        let digits = |mut i: usize| {
            let mut d = vec![0; dims.len()];
            for a in (0..dims.len()).rev() {
                d[a] = i % dims[a];
                i /= dims[a];
            }
            d
        };
        for xi in 0..g.len() {
            let dx = digits(xi);
            let mut acc = Fp::ZERO;
            for yi in 0..g.len() {
                let dy = digits(yi);
                let w = (0..dims.len()).fold(Fp::ONE, |w, a| w * roots[a].pow((dx[a] * dy[a]) as u64));
                acc += w * x[yi];
            }
            assert_eq!(y[xi], acc, "dims={dims:?} xi={xi}");
        }
        assert_eq!(g.inverse(&y), x);
    }
}

// ---------------------------------------------------------------- convolution

#[test]
fn convolve_matches_schoolbook() {
    let mut r = Rng(14);
    for _ in 0..300 {
        let (la, lb) = (r.below(150), r.below(150));
        let (a, b) = (r.vec(la), r.vec(lb));
        let mut c = if la == 0 || lb == 0 { Vec::new() } else { vec![Fp::ZERO; la + lb - 1] };
        if la > 0 && lb > 0 {
            for i in 0..la {
                for j in 0..lb {
                    c[i + j] += a[i] * b[j];
                }
            }
        }
        assert_eq!(convolve(&a, &b), c, "la={la} lb={lb}");
    }
}

#[test]
fn series_inverse_is_inverse() {
    let mut r = Rng(15);
    for n in [1usize, 2, 3, 7, 64, 100, 1000, 5000] {
        let mut f = r.vec(n);
        if f[0] == Fp::ZERO {
            f[0] = Fp::ONE;
        }
        let g = series_inverse(&f, n);
        let mut fg = convolve(&f, &g);
        fg.truncate(n);
        let mut one = vec![Fp::ZERO; n];
        one[0] = Fp::ONE;
        assert_eq!(fg, one, "n={n}");
    }
}
