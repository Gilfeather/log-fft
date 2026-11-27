//! Exact arithmetic in the Goldilocks prime field `F_p`, `p = 2^64 - 2^32 + 1`.
//!
//! The identities in the paper hold over `Z[1/2]`. Reducing modulo an odd
//! prime is a ring homomorphism, so an identity that holds over `Z[1/2]`
//! also holds in `F_p`. Every check in this crate is therefore exact, with
//! no floating-point rounding.

use core::fmt;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// The Goldilocks prime `2^64 - 2^32 + 1`.
pub const P: u64 = 0xFFFF_FFFF_0000_0001;

/// A multiplicative generator of `F_p^*`.
pub const GENERATOR: u64 = 7;

/// An element of `F_p`, always stored reduced in `[0, p)`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Fp(u64);

impl Fp {
    pub const ZERO: Fp = Fp(0);
    pub const ONE: Fp = Fp(1);

    pub fn new(v: u64) -> Self {
        Fp(v % P)
    }

    pub fn from_i64(v: i64) -> Self {
        if v >= 0 {
            Fp::new(v as u64)
        } else {
            -Fp::new(v.unsigned_abs())
        }
    }

    pub fn value(self) -> u64 {
        self.0
    }

    pub fn pow(self, mut e: u64) -> Self {
        let mut base = self;
        let mut acc = Fp::ONE;
        while e > 0 {
            if e & 1 == 1 {
                acc *= base;
            }
            base *= base;
            e >>= 1;
        }
        acc
    }

    /// Multiplicative inverse. Panics on zero.
    pub fn inv(self) -> Self {
        assert!(self.0 != 0, "inverse of zero");
        self.pow(P - 2)
    }

    /// `(-1)^e` for any integer `e`.
    pub fn sign(e: i64) -> Self {
        if e.rem_euclid(2) == 0 {
            Fp::ONE
        } else {
            -Fp::ONE
        }
    }

    /// A primitive `n`-th root of unity. `n` must divide `p - 1`.
    pub fn root_of_unity(n: u64) -> Self {
        assert!(n > 0 && (P - 1) % n == 0, "no primitive {n}-th root in F_p");
        Fp(GENERATOR).pow((P - 1) / n)
    }
}

impl fmt::Debug for Fp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Show small negatives as negatives for readability.
        if self.0 > P / 2 {
            write!(f, "-{}", P - self.0)
        } else {
            write!(f, "{}", self.0)
        }
    }
}

impl Add for Fp {
    type Output = Fp;
    fn add(self, rhs: Fp) -> Fp {
        let (s, carry) = self.0.overflowing_add(rhs.0);
        if carry || s >= P {
            Fp(s.wrapping_sub(P))
        } else {
            Fp(s)
        }
    }
}

impl Sub for Fp {
    type Output = Fp;
    fn sub(self, rhs: Fp) -> Fp {
        if self.0 >= rhs.0 {
            Fp(self.0 - rhs.0)
        } else {
            Fp(self.0.wrapping_sub(rhs.0).wrapping_add(P))
        }
    }
}

impl Neg for Fp {
    type Output = Fp;
    fn neg(self) -> Fp {
        Fp::ZERO - self
    }
}

/// `2^64 mod p = 2^32 − 1`.
const EPSILON: u64 = 0xFFFF_FFFF;

/// Reduce a 128-bit value modulo `p`, using `2^64 ≡ 2^32 − 1` and `2^96 ≡ −1`.
#[inline(always)]
pub(crate) fn reduce128(x: u128) -> u64 {
    let lo = x as u64;
    let hi = (x >> 64) as u64;
    let hi_hi = hi >> 32;
    let hi_lo = hi & EPSILON;
    // lo − hi_hi·2^96 ≡ lo − hi_hi
    let (mut t0, borrow) = lo.overflowing_sub(hi_hi);
    if borrow {
        t0 = t0.wrapping_sub(EPSILON); // add 2^64 ≡ subtract (2^64 mod p) back
    }
    // + hi_lo·2^64 ≡ hi_lo·(2^32 − 1)
    let t1 = hi_lo * EPSILON;
    let (mut r, carry) = t0.overflowing_add(t1);
    if carry {
        r = r.wrapping_add(EPSILON);
    }
    if r >= P { r - P } else { r }
}

impl Mul for Fp {
    type Output = Fp;
    #[inline(always)]
    fn mul(self, rhs: Fp) -> Fp {
        Fp(reduce128(self.0 as u128 * rhs.0 as u128))
    }
}

impl AddAssign for Fp {
    fn add_assign(&mut self, rhs: Fp) {
        *self = *self + rhs;
    }
}
impl SubAssign for Fp {
    fn sub_assign(&mut self, rhs: Fp) {
        *self = *self - rhs;
    }
}
impl MulAssign for Fp {
    fn mul_assign(&mut self, rhs: Fp) {
        *self = *self * rhs;
    }
}

/// Generalised binomial coefficient `binom(top, k)` for any integer `top`.
pub fn binom(top: i64, k: usize) -> Fp {
    let mut num = Fp::ONE;
    let mut den = Fp::ONE;
    for t in 0..k as i64 {
        num *= Fp::from_i64(top - t);
        den *= Fp::from_i64(t + 1);
    }
    num * den.inv()
}

/// Solve the square system `a · x = rhs` by Gaussian elimination.
/// Panics if `a` is singular.
pub fn solve(mut a: Vec<Vec<Fp>>, mut rhs: Vec<Fp>) -> Vec<Fp> {
    let n = rhs.len();
    for col in 0..n {
        let piv = (col..n)
            .find(|&r| a[r][col] != Fp::ZERO)
            .expect("singular system");
        a.swap(col, piv);
        rhs.swap(col, piv);
        let inv = a[col][col].inv();
        for c in col..n {
            a[col][c] *= inv;
        }
        rhs[col] *= inv;
        for r in 0..n {
            if r != col && a[r][col] != Fp::ZERO {
                let f = a[r][col];
                for c in col..n {
                    let t = a[col][c];
                    a[r][c] -= f * t;
                }
                let t = rhs[col];
                rhs[r] -= f * t;
            }
        }
    }
    rhs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arithmetic() {
        let a = Fp::new(P - 1);
        assert_eq!(a + Fp::ONE, Fp::ZERO);
        assert_eq!(Fp::from_i64(-3) * Fp::from_i64(-5), Fp::new(15));
        let x = Fp::new(123_456_789);
        assert_eq!(x * x.inv(), Fp::ONE);
        let w = Fp::root_of_unity(1 << 20);
        assert_eq!(w.pow(1 << 20), Fp::ONE);
        assert_ne!(w.pow(1 << 19), Fp::ONE);
    }

    #[test]
    fn fast_reduction_matches_modulo() {
        let edges = [0u64, 1, 2, EPSILON, EPSILON + 1, 1 << 32, P - 1, P - 2, P / 2, (1 << 63) + 12345];
        for &a in &edges {
            for &b in &edges {
                let a = a % P;
                let b = b % P;
                let x = a as u128 * b as u128;
                assert_eq!(reduce128(x), (x % P as u128) as u64, "{a} * {b}");
            }
        }
        // Arbitrary 128-bit inputs, not only products.
        let mut s = 0x9E37_79B9_7F4A_7C15u128;
        for _ in 0..2_000_000 {
            s = s.wrapping_mul(0x2360_ED05_1FC6_5DA4_4385_DF64_9FCC_F645).wrapping_add(1);
            assert_eq!(reduce128(s), (s % P as u128) as u64, "x = {s}");
        }
        for x in [u128::MAX, u128::MAX - 1, (P as u128) << 64, ((P as u128) << 64) - 1, 1u128 << 127] {
            assert_eq!(reduce128(x), (x % P as u128) as u64, "x = {x}");
        }
    }

    #[test]
    fn generalised_binomial() {
        assert_eq!(binom(5, 2), Fp::new(10));
        assert_eq!(binom(-3, 2), Fp::new(6)); // (-3)(-4)/2
        assert_eq!(binom(2, 5), Fp::ZERO);
    }
}
