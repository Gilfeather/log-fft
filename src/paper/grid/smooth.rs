//! Auxiliary length-`b` transforms `F_b = V_b(u)` (grids.tex, step 5).
//!
//! The paper takes `b_a = P·ℓ^j` with pairwise coprime factors, and writes
//! `F_{b_a}` with Good's prime-factor factorisation as a tensor product of
//! short transforms, up to CRT permutations. Here `b` is a divisor of
//! `p − 1 = 2^32·3·5·17·257·65537` of the form `2^j · Π(subset of {3,5,17,257})`.
//! `F_b` is computed by Good–Thomas over those coprime factors, using radix-2
//! for the `2^j` part and direct evaluation for the odd primes.
//!
//! Deviation: the paper's `P` is a product of primes in `[t, t²]` adjoined
//! via an extension field `K(ζ_M)`. Goldilocks only contains roots of order
//! dividing `p − 1`, so this crate draws factors from the primes that
//! divide `p − 1`. [`paper_b_a`] computes the paper's choice for reference.

use crate::field::{Fp, P};

const ODD_PRIMES: [usize; 4] = [3, 5, 17, 257];

/// The least admissible `b ≥ 2a` that divides `p − 1` and has the smooth form above.
pub fn choose_b(a: usize) -> usize {
    let target = 2 * a;
    let mut best = usize::MAX;
    for mask in 0..1u32 << ODD_PRIMES.len() {
        let odd: usize = (0..ODD_PRIMES.len())
            .filter(|i| mask >> i & 1 == 1)
            .map(|i| ODD_PRIMES[i])
            .product();
        let mut b = odd;
        while b < target {
            b *= 2;
        }
        if b < best && b.trailing_zeros() <= 32 {
            best = b;
        }
    }
    debug_assert!((P - 1) % best as u64 == 0);
    best
}

/// The paper's `b_a`: `t = ⌈log₂ N⌉`, `P` = longest prefix product of the
/// odd primes in `[t, t²]` with `P < 2a`, then `b_a = P·2^j`, the least such
/// value `≥ 2a`. (`ℓ = 2`; characteristic 0 or large is assumed.)
pub fn paper_b_a(a: usize, n: usize) -> usize {
    let t = (n as f64).log2().ceil().max(1.0) as usize;
    let mut prod = 1usize;
    for q in t.max(3)..=t * t {
        if q % 2 == 1 && is_prime(q) {
            if prod * q >= 2 * a {
                break;
            }
            prod *= q;
        }
    }
    let mut b = prod;
    while b < 2 * a {
        b *= 2;
    }
    b
}

fn is_prime(n: usize) -> bool {
    n >= 2 && (2..).take_while(|d| d * d <= n).all(|d| n % d != 0)
}

/// `F_b = V_b(u)` with `u` a fixed primitive `b`-th root, via Good–Thomas.
#[derive(Clone, Debug)]
pub struct SmoothDft {
    pub b: usize,
    /// Pairwise coprime factors: `2^j` (if `j > 0`) followed by odd primes.
    pub factors: Vec<usize>,
    /// The primitive `b`-th root `u`.
    pub root: Fp,
}

impl SmoothDft {
    pub fn new(b: usize) -> Self {
        assert!((P - 1) % b as u64 == 0, "b = {b} does not divide p − 1");
        let mut factors = Vec::new();
        let two = 1usize << b.trailing_zeros();
        if two > 1 {
            factors.push(two);
        }
        let mut rest = b / two;
        for q in ODD_PRIMES.iter().chain(&[65537]) {
            if rest % q == 0 {
                factors.push(*q);
                rest /= q;
            }
        }
        assert_eq!(rest, 1, "b = {b} has a repeated odd factor");
        SmoothDft { b, factors, root: Fp::root_of_unity(b as u64) }
    }

    /// `y_k = Σ_n x_n u^{nk}`.
    pub fn forward(&self, x: &[Fp]) -> Vec<Fp> {
        assert_eq!(x.len(), self.b);
        pfa(x, &self.factors, self.root)
    }

    /// `x_n = b^{-1} Σ_k y_k u^{-nk}`.
    pub fn inverse(&self, y: &[Fp]) -> Vec<Fp> {
        let mut x = pfa(y, &self.factors, self.root.inv());
        let inv_b = Fp::new(self.b as u64).inv();
        for v in &mut x {
            *v *= inv_b;
        }
        x
    }

    /// Cyclic convolution of two length-`b` sequences.
    pub fn cyclic_convolve(&self, f: &[Fp], g: &[Fp]) -> Vec<Fp> {
        let fs = self.forward(f);
        let gs = self.forward(g);
        self.inverse(&fs.iter().zip(&gs).map(|(a, b)| *a * *b).collect::<Vec<_>>())
    }
}

/// Good–Thomas prime-factor algorithm with root `w` of order `Π factors`.
fn pfa(x: &[Fp], factors: &[usize], w: Fp) -> Vec<Fp> {
    let n = x.len();
    if factors.len() <= 1 {
        return if n.is_power_of_two() { radix2(x, w) } else { direct(x, w) };
    }
    let n1 = factors[0];
    let n2 = n / n1;
    // Input map n = (n2·i1 + n1·i2) mod n; output by CRT.
    let w1 = w.pow(n2 as u64); // order n1
    let w2 = w.pow(n1 as u64); // order n2
    let mut grid = vec![Fp::ZERO; n];
    for i1 in 0..n1 {
        for i2 in 0..n2 {
            grid[i1 * n2 + i2] = x[(n2 * i1 + n1 * i2) % n];
        }
    }
    // Rows: transforms of length n2 (recursively over the remaining factors).
    for i1 in 0..n1 {
        let row = pfa(&grid[i1 * n2..(i1 + 1) * n2], &factors[1..], w2);
        grid[i1 * n2..(i1 + 1) * n2].copy_from_slice(&row);
    }
    // Columns: transforms of length n1.
    for k2 in 0..n2 {
        let col: Vec<Fp> = (0..n1).map(|i1| grid[i1 * n2 + k2]).collect();
        let col = pfa(&col, &factors[..1], w1);
        for (k1, v) in col.into_iter().enumerate() {
            grid[k1 * n2 + k2] = v;
        }
    }
    // CRT output map: k ≡ k1 (mod n1), k ≡ k2 (mod n2).
    let e1 = n2 * inv_mod(n2 % n1, n1) % n; // ≡ 1 mod n1, 0 mod n2
    let e2 = n1 * inv_mod(n1 % n2, n2) % n;
    let mut out = vec![Fp::ZERO; n];
    for k1 in 0..n1 {
        for k2 in 0..n2 {
            out[(k1 * e1 + k2 * e2) % n] = grid[k1 * n2 + k2];
        }
    }
    out
}

fn inv_mod(a: usize, m: usize) -> usize {
    if m == 1 {
        return 0;
    }
    (1..m).find(|&x| a * x % m == 1).expect("not coprime")
}

fn direct(x: &[Fp], w: Fp) -> Vec<Fp> {
    let n = x.len();
    (0..n)
        .map(|k| {
            let wk = w.pow(k as u64);
            let mut acc = Fp::ZERO;
            let mut pw = Fp::ONE;
            for v in x {
                acc += *v * pw;
                pw *= wk;
            }
            acc
        })
        .collect()
}

fn radix2(x: &[Fp], w: Fp) -> Vec<Fp> {
    let n = x.len();
    let bits = n.trailing_zeros();
    let mut a: Vec<Fp> = (0..n)
        .map(|i| x[if bits == 0 { 0 } else { i.reverse_bits() >> (usize::BITS - bits) }])
        .collect();
    let mut len = 2;
    while len <= n {
        let wl = w.pow((n / len) as u64);
        for start in (0..n).step_by(len) {
            let mut t = Fp::ONE;
            for k in 0..len / 2 {
                let (u, v) = (a[start + k], a[start + k + len / 2] * t);
                a[start + k] = u + v;
                a[start + k + len / 2] = u - v;
                t *= wl;
            }
        }
        len *= 2;
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pfa_matches_direct() {
        for b in [2, 3, 8, 15, 48, 85, 120, 255, 1020, 4352] {
            let d = SmoothDft::new(b);
            let x: Vec<Fp> = (0..b as u64).map(|i| Fp::new(i * i * 31 + 7)).collect();
            assert_eq!(d.forward(&x), direct(&x, d.root), "b={b}");
            assert_eq!(d.inverse(&d.forward(&x)), x);
        }
    }

    #[test]
    fn b_choices_in_range() {
        for a in 2..3000 {
            let b = choose_b(a);
            assert!(2 * a <= b && b < 6 * a, "a={a} b={b}");
            let bp = paper_b_a(a, 1 << 20);
            assert!(2 * a <= bp && bp < 6 * a, "paper a={a} b={bp}");
        }
    }
}
