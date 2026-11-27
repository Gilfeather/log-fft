//! Kani proof harnesses (`cargo kani`). Each harness checks its claim for
//! **every** input in the stated domain, not a sample. Kani also checks for
//! panics, arithmetic overflow and out-of-bounds access along every path.

use crate::field::{reduce128, Fp, P};

fn any_fp() -> Fp {
    let v: u64 = kani::any();
    kani::assume(v < P);
    Fp::from_canonical(v).unwrap()
}

// ------------------------------------------------------------------ field

/// The Goldilocks reduction is correct for all 2^128 inputs.
///
/// Write `x = lo + 2^64·h₀ + 2^96·h₁` (`h₀, h₁ < 2^32`). Since
/// `2^64 ≡ 2^32 − 1` and `2^96 ≡ −1 (mod p)`, which are compile-time
/// assertions in `field.rs`, we have `x ≡ v := lo + (2^32−1)·h₀ − h₁`.
/// The harness proves `r < p` and `v − r ∈ {−p, 0, p, 2p}`, so `r ≡ x (mod p)`.
/// Together these give `r = x mod p`. No 128-bit division is involved.
#[kani::proof]
fn reduce128_is_mod_p() {
    let x: u128 = kani::any();
    let r = reduce128(x);
    assert!(r < P);
    let lo = x as u64;
    let hi = (x >> 64) as u64;
    let (h0, h1) = (hi & 0xFFFF_FFFF, hi >> 32);
    assert!(x == lo as u128 + ((h0 as u128) << 64) + ((h1 as u128) << 96));
    let v = lo as i128 + 0xFFFF_FFFFi128 * h0 as i128 - h1 as i128;
    let d = v - r as i128;
    let p = P as i128;
    assert!(d == -p || d == 0 || d == p || d == 2 * p);
}

#[kani::proof]
fn add_is_mod_p() {
    let (a, b) = (any_fp(), any_fp());
    let s = a.value() as u128 + b.value() as u128; // < 2p
    let expect = if s >= P as u128 { s - P as u128 } else { s };
    assert!((a + b).value() as u128 == expect);
}

#[kani::proof]
fn sub_is_mod_p() {
    let (a, b) = (any_fp(), any_fp());
    let (x, y) = (a.value() as u128, b.value() as u128);
    let expect = if x >= y { x - y } else { x + P as u128 - y };
    assert!((a - b).value() as u128 == expect);
}

#[kani::proof]
fn neg_is_mod_p() {
    let a = any_fp();
    let r = (-a).value();
    assert!(r < P);
    let s = r as u128 + a.value() as u128;
    assert!(s == 0 || s == P as u128);
}

/// `Mul` is exactly `reduce128` of the exact 128-bit product. With
/// `reduce128_is_mod_p`, this gives `a·b mod p` for all canonical inputs.
#[kani::proof]
fn mul_is_reduced_product() {
    let (a, b) = (any_fp(), any_fp());
    assert!((a * b).value() == reduce128(a.value() as u128 * b.value() as u128));
}

#[kani::proof]
fn new_reduces_and_from_canonical_rejects() {
    let v: u64 = kani::any();
    let expect = if v >= P { v - P } else { v }; // u64 < 2p
    assert!(Fp::new(v).value() == expect);
    match Fp::from_canonical(v) {
        Some(x) => {
            assert!(v < P);
            assert!(x.value() == v);
        }
        None => assert!(v >= P),
    }
}

#[kani::proof]
fn from_i64_is_mod_p() {
    let v: i64 = kani::any();
    let r = Fp::from_i64(v).value();
    assert!(r < P);
    let d = r as i128 - v as i128;
    let p = P as i128;
    assert!(d == -p || d == 0 || d == p);
}

// ------------------------------------------------------------------ slices

/// The zero-copy view accepts exactly the canonical arrays and preserves values.
#[kani::proof]
fn canonical_slice_view() {
    let raw: [u64; 3] = kani::any();
    match Fp::from_canonical_slice(&raw) {
        Some(view) => {
            for i in 0..3 {
                assert!(raw[i] < P);
                assert_eq!(view[i].value(), raw[i]);
            }
        }
        None => assert!(raw.iter().any(|&v| v >= P)),
    }
}

// ------------------------------------------------------------------ NTT

/// Bit reversal is an involution for every array of length 8.
#[kani::proof]
#[kani::unwind(9)]
fn bit_reverse_involution() {
    let raw: [u64; 8] = kani::any();
    let mut x = raw.map(Fp::new);
    let orig = x;
    crate::ntt::bit_reverse(&mut x, 3);
    for i in 0..8usize {
        let j = i.reverse_bits() >> (usize::BITS - 3);
        assert!(x[i] == orig[j]);
    }
    crate::ntt::bit_reverse(&mut x, 3);
    assert!(x == orig);
}
