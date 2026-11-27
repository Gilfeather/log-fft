//! Cyclic and finite-abelian-group DFTs through the grids.tex route.
//!
//! Each axis transform `V_a(q)` (the DFT when `q` is a primitive `a`-th
//! root) is written by the Newton lemma as
//! `D_H · T · M · Tᵀ · D_H`, and its inverse as
//! `D_H^{-1} · T^{-T} · M^{-1} · T^{-1} · D_H^{-1}`.
//! Each triangular Toeplitz factor is applied in place using shears plus
//! `D_u`, where every shear is one cyclic convolution of an auxiliary
//! length `b ≥ 2a` (Lemma up:toeplitz). `F_b` itself is a Good–Thomas tensor
//! product of short transforms. A transform on `G = Π Z_{a_i}` is the tensor
//! product `⊗ V_{a_i}(q_i)`, applied fibre by fibre (Lemma up:fibers).
//!
//! Not implemented: grouping axes up to `N^{1/24}`, the extension-field
//! packing `E = K(ζ_M)`, and the shear-layer cost accounting. In the paper,
//! the `√log` saving comes from combining this route with the kernel
//! recursion, which is not implemented either. So this module checks
//! *correctness* of the reduction. It is an `O(N log N)` route, like Bluestein.

pub mod newton;
pub mod smooth;
pub mod toeplitz;

use crate::field::Fp;
use newton::NewtonFactors;
use smooth::choose_b;
use toeplitz::{lower_toeplitz_paper, series_inv, upper_toeplitz_paper, ShearEngine, ToeplitzPlan};

/// A planned `V_a(q)` (forward) and `V_a(q)^{-1}` (inverse) on one axis.
#[derive(Clone, Debug)]
pub struct AxisPlan {
    pub a: usize,
    pub q: Fp,
    pub newton: NewtonFactors,
    pub engine: ShearEngine,
    t_plan: ToeplitzPlan,
    tinv_plan: ToeplitzPlan,
}

impl AxisPlan {
    /// `V_a(q)` for any `q` whose powers `q^0 … q^{a−1}` are distinct.
    pub fn geometric(a: usize, q: Fp) -> Self {
        let newton = NewtonFactors::new(a, q);
        let engine = ShearEngine::new(a, choose_b(a));
        let u = engine.dft.root;
        let t_plan = ToeplitzPlan::new(&newton.t, a, u);
        let tinv_plan = ToeplitzPlan::new(&series_inv(&newton.t, a), a, u);
        AxisPlan { a, q, newton, engine, t_plan, tinv_plan }
    }

    /// The DFT of length `a`: `q` = a primitive `a`-th root of unity.
    pub fn dft(a: usize) -> Self {
        Self::geometric(a, Fp::root_of_unity(a as u64))
    }

    /// `V x`, where `V_{jk} = q^{jk}`.
    pub fn forward(&self, x: &[Fp], scratch: &mut Vec<Fp>) -> Vec<Fp> {
        let nf = &self.newton;
        let mut y: Vec<Fp> = x.iter().zip(&nf.h).map(|(a, b)| *a * *b).collect();
        y = upper_toeplitz_paper(&self.engine, &self.t_plan, &y, scratch);
        y.iter_mut().zip(&nf.mid).for_each(|(a, b)| *a *= *b);
        y = lower_toeplitz_paper(&self.engine, &self.t_plan, &y, scratch);
        y.iter_mut().zip(&nf.h).for_each(|(a, b)| *a *= *b);
        y
    }

    /// `V^{-1} x`.
    pub fn inverse(&self, x: &[Fp], scratch: &mut Vec<Fp>) -> Vec<Fp> {
        let nf = &self.newton;
        let mut y: Vec<Fp> = x.iter().zip(&nf.t).map(|(a, b)| *a * *b).collect(); // D_H^{-1}
        y = lower_toeplitz_paper(&self.engine, &self.tinv_plan, &y, scratch);
        y.iter_mut().zip(&nf.mid).for_each(|(a, b)| *a *= b.inv());
        y = upper_toeplitz_paper(&self.engine, &self.tinv_plan, &y, scratch);
        y.iter_mut().zip(&nf.t).for_each(|(a, b)| *a *= *b);
        y
    }
}

/// Transform on `G = Z_{a_1} × … × Z_{a_r}`, stored row-major (last axis fastest).
#[derive(Clone, Debug)]
pub struct GroupDft {
    pub dims: Vec<usize>,
    pub axes: Vec<AxisPlan>,
}

impl GroupDft {
    /// Character transform with `q_i` a primitive `a_i`-th root on each axis.
    pub fn new(dims: &[usize]) -> Self {
        GroupDft { dims: dims.to_vec(), axes: dims.iter().map(|&a| AxisPlan::dft(a)).collect() }
    }

    pub fn len(&self) -> usize {
        self.dims.iter().product()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn apply(&self, x: &[Fp], inverse: bool) -> Vec<Fp> {
        assert_eq!(x.len(), self.len());
        let mut data = x.to_vec();
        let mut stride = 1;
        for ax in (0..self.dims.len()).rev() {
            let a = self.dims[ax];
            let plan = &self.axes[ax];
            // One scratch register per axis, reused by all fibres (Lemma up:fibers).
            let mut scratch = vec![Fp::ZERO; a];
            let block = a * stride;
            for base in (0..data.len()).step_by(block) {
                for off in 0..stride {
                    let fibre: Vec<Fp> = (0..a).map(|k| data[base + off + k * stride]).collect();
                    let out = if inverse {
                        plan.inverse(&fibre, &mut scratch)
                    } else {
                        plan.forward(&fibre, &mut scratch)
                    };
                    for (k, v) in out.into_iter().enumerate() {
                        data[base + off + k * stride] = v;
                    }
                }
            }
            debug_assert!(scratch.iter().all(|v| *v == Fp::ZERO), "scratch not restored");
            stride *= a;
        }
        data
    }

    /// `X̂(ξ) = Σ_y x(y) Π_i q_i^{ξ_i y_i}`.
    pub fn forward(&self, x: &[Fp]) -> Vec<Fp> {
        self.apply(x, false)
    }

    pub fn inverse(&self, x: &[Fp]) -> Vec<Fp> {
        self.apply(x, true)
    }
}

/// `O(N²)` reference character transform.
pub fn naive_group_dft(dims: &[usize], x: &[Fp]) -> Vec<Fp> {
    let n: usize = dims.iter().product();
    let roots: Vec<Fp> = dims.iter().map(|&a| Fp::root_of_unity(a as u64)).collect();
    let digits = |mut idx: usize| {
        let mut d = vec![0usize; dims.len()];
        for i in (0..dims.len()).rev() {
            d[i] = idx % dims[i];
            idx /= dims[i];
        }
        d
    };
    (0..n)
        .map(|xi| {
            let dx = digits(xi);
            (0..n).fold(Fp::ZERO, |acc, yi| {
                let dy = digits(yi);
                let w = (0..dims.len())
                    .fold(Fp::ONE, |w, i| w * roots[i].pow(((dx[i] * dy[i]) % dims[i]) as u64));
                acc + w * x[yi]
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use newton::vandermonde;

    fn seq(n: usize, s: u64) -> Vec<Fp> {
        (0..n as u64).map(|i| Fp::new(i.wrapping_mul(2654435761) ^ s)).collect()
    }

    #[test]
    fn axis_dft_and_geometric() {
        let cases: Vec<(usize, Fp)> = vec![
            (2, Fp::root_of_unity(2)),
            (3, Fp::root_of_unity(3)),
            (15, Fp::root_of_unity(15)),
            (17, Fp::root_of_unity(17)),
            (64, Fp::root_of_unity(64)),
            (85, Fp::root_of_unity(85)),
            (255, Fp::root_of_unity(255)),
            (10, Fp::new(3)), // geometric evaluation, not a root of unity
        ];
        for (a, q) in cases {
            let plan = AxisPlan::geometric(a, q);
            let v = vandermonde(a, q);
            let x = seq(a, 77);
            let mut scratch = vec![Fp::ZERO; a];
            let y = plan.forward(&x, &mut scratch);
            let expect: Vec<Fp> = (0..a).map(|j| (0..a).fold(Fp::ZERO, |s, k| s + v[j][k] * x[k])).collect();
            assert_eq!(y, expect, "forward a={a}");
            assert_eq!(plan.inverse(&y, &mut scratch), x, "inverse a={a}");
            assert!(scratch.iter().all(|v| *v == Fp::ZERO));
        }
    }

    #[test]
    fn abelian_group_dft() {
        for dims in [vec![12], vec![3, 5], vec![4, 17], vec![2, 3, 5], vec![8, 15]] {
            let g = GroupDft::new(&dims);
            let x = seq(g.len(), 5);
            let y = g.forward(&x);
            assert_eq!(y, naive_group_dft(&dims, &x), "dims={dims:?}");
            assert_eq!(g.inverse(&y), x);
        }
    }
}
