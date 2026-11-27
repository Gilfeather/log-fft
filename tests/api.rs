//! Tests for the v0.1.2 API surface: thread-count independence, bit-reversed
//! entry points, batch and in-place forms, and `Fp` conversions.

use log_fft::field::P;
use log_fft::ntt::bit_reverse;
use log_fft::{Dft, Fp, GroupDft, Ntt};

fn vec(n: usize, seed: u64) -> Vec<Fp> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            Fp::new(s ^ (s >> 29))
        })
        .collect()
}

#[test]
fn ntt_result_independent_of_threads() {
    for log in [10, 17, 18, 20, 21] {
        let n = 1usize << log;
        let x = vec(n, log as u64);
        let mut a = x.clone();
        Ntt::new(n).with_threads(1).forward(&mut a);
        for t in [0, 2, 3, 7, 16] {
            let mut b = x.clone();
            Ntt::new(n).with_threads(t).forward(&mut b);
            assert_eq!(a, b, "n={n} threads={t}");
            Ntt::new(n).with_threads(t).inverse(&mut b);
            assert_eq!(b, x, "inverse n={n} threads={t}");
        }
    }
}

#[test]
fn bitrev_entry_points() {
    for log in [0, 1, 2, 5, 11, 19] {
        let n = 1usize << log;
        let plan = Ntt::new(n);
        let x = vec(n, 100 + log as u64);
        let mut nat = x.clone();
        plan.forward(&mut nat);
        let mut br = x.clone();
        plan.forward_bitrev(&mut br);
        bit_reverse(&mut br, log);
        assert_eq!(br, nat, "forward_bitrev n={n}");
        let mut y = x.clone();
        plan.forward_bitrev(&mut y);
        plan.inverse_from_bitrev(&mut y);
        assert_eq!(y, x, "bitrev roundtrip n={n}");
        let mut u = x.clone();
        plan.forward(&mut u);
        plan.inverse_unscaled(&mut u);
        let nn = Fp::new(n as u64);
        assert_eq!(u, x.iter().map(|v| *v * nn).collect::<Vec<_>>(), "unscaled n={n}");
    }
}

#[test]
fn ntt_batch_matches_rows() {
    let n = 256;
    let rows = 300;
    let data = vec(n * rows, 7);
    let plan = Ntt::new(n);
    let mut batch = data.clone();
    plan.forward_batch(&mut batch);
    for (r, chunk) in data.chunks(n).enumerate() {
        let mut row = chunk.to_vec();
        plan.forward(&mut row);
        assert_eq!(&batch[r * n..(r + 1) * n], &row[..]);
    }
    plan.inverse_batch(&mut batch);
    assert_eq!(batch, data);
}

#[test]
fn dft_result_independent_of_threads_and_form() {
    // 983055 = 3·5·65537: the outer axis has a single block, which exercises the
    // split-offset parallel path. 1044480 = 4096·3·5·17 exercises Rader(17) on a
    // strided axis.
    for n in [983_055usize, 1_044_480, 65_535, 255 * 1024] {
        let x = vec(n, n as u64);
        let single = Dft::new(n).with_threads(1).forward(&x);
        for t in [0, 2, 5, 12] {
            let plan = Dft::new(n).with_threads(t);
            assert_eq!(plan.forward(&x), single, "n={n} t={t}");
            let mut ip = x.clone();
            plan.forward_in_place(&mut ip);
            assert_eq!(ip, single);
            plan.inverse_in_place(&mut ip);
            assert_eq!(ip, x, "inverse n={n} t={t}");
        }
    }
}

#[test]
fn dft_batch_matches_rows() {
    for n in [255usize, 257, 17 * 64, 1] {
        let rows = 97;
        let data = vec(n * rows, n as u64 + 1);
        let plan = Dft::new(n);
        let mut batch = data.clone();
        plan.forward_batch(&mut batch);
        for (r, chunk) in data.chunks(n).enumerate() {
            assert_eq!(&batch[r * n..(r + 1) * n], &plan.forward(chunk)[..], "n={n} row={r}");
        }
        plan.inverse_batch(&mut batch);
        assert_eq!(batch, data);
    }
}

#[test]
fn group_dft_large_parallel_consistent() {
    let dims = [17usize, 4096, 15];
    let x = vec(dims.iter().product(), 3);
    let a = GroupDft::new(&dims).with_threads(1).forward(&x);
    let b = GroupDft::new(&dims).forward(&x);
    assert_eq!(a, b);
    // Separable check: equals applying Dft along each axis by hand.
    let mut c = x.clone();
    let (d0, d1, d2) = (dims[0], dims[1], dims[2]);
    let p2 = Dft::new(d2);
    for row in c.chunks_mut(d2) {
        p2.forward_in_place(row);
    }
    let p1 = Dft::new(d1);
    for i in 0..d0 {
        for k in 0..d2 {
            let fib: Vec<Fp> = (0..d1).map(|j| c[(i * d1 + j) * d2 + k]).collect();
            let y = p1.forward(&fib);
            for j in 0..d1 {
                c[(i * d1 + j) * d2 + k] = y[j];
            }
        }
    }
    let p0 = Dft::new(d0);
    for jk in 0..d1 * d2 {
        let fib: Vec<Fp> = (0..d0).map(|i| c[i * d1 * d2 + jk]).collect();
        let y = p0.forward(&fib);
        for i in 0..d0 {
            c[i * d1 * d2 + jk] = y[i];
        }
    }
    assert_eq!(a, c);
    assert_eq!(GroupDft::new(&dims).inverse(&a), x);
}

#[test]
fn fp_conversions_and_operators() {
    assert_eq!(Fp::from(5u64), Fp::new(5));
    assert_eq!(Fp::from(P), Fp::ZERO);
    assert_eq!(Fp::from(7u32), Fp::new(7));
    assert_eq!(Fp::from(-1i64), Fp::new(P - 1));
    assert_eq!(Fp::from(true), Fp::ONE);
    assert_eq!(u64::from(Fp::new(42)), 42);
    assert_eq!(Fp::from_canonical(P - 1), Some(Fp::new(P - 1)));
    assert_eq!(Fp::from_canonical(P), None);
    assert_eq!(format!("{}", Fp::new(123)), "123");
    let xs: Vec<Fp> = (1..=10u64).map(Fp::new).collect();
    assert_eq!(xs.iter().sum::<Fp>(), Fp::new(55));
    assert_eq!(xs.iter().copied().product::<Fp>(), Fp::new(3_628_800));
    assert_eq!(Fp::new(10) / Fp::new(4) * Fp::new(4), Fp::new(10));
    let mut d = Fp::new(9);
    d /= Fp::new(3);
    assert_eq!(d, Fp::new(3));

    let mut raw = vec![1u64, 2, P - 1, 0];
    {
        let view = Fp::from_canonical_slice_mut(&mut raw).unwrap();
        Ntt::new(4).forward(view);
    }
    let mut expect = vec![Fp::new(1), Fp::new(2), Fp::new(P - 1), Fp::ZERO];
    Ntt::new(4).forward(&mut expect);
    assert_eq!(raw, Fp::as_u64_slice(&expect));
    assert!(raw.iter().all(|&v| v < P));
    assert!(Fp::from_canonical_slice(&[0, P]).is_none());
    assert_eq!(Fp::from_canonical_slice(&[3, 4]).unwrap(), &[Fp::new(3), Fp::new(4)]);
}

#[test]
#[should_panic(expected = "inverse of zero")]
fn division_by_zero_panics() {
    let _ = Fp::ONE / Fp::ZERO;
}
