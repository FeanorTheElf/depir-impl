use feanor_math::algorithms;
use feanor_math::integer::IntegerRingStore;
use feanor_math::pid::EuclideanRingStore;
use feanor_math::rings::zn::zn_rns;
use feanor_math::integer::*;

use std::cmp::max;

use crate::globals::*;

pub mod base;
pub mod cyclotomic_rns;
pub mod tensor;
pub mod cyclotomic_cc;

pub type RNSBase = zn_rns::Zn<Zn, BigIntRing>;

pub fn required_root_of_unity_order(m: usize, log2_N: usize) -> usize {
    let bluestein_log2 = StaticRing::<i64>::RING.abs_log2_ceil(&(m as i64)).unwrap() + 1;
    return m << max(bluestein_log2, log2_N + 1);
}

pub fn sample_primes_arithmetic_progression(a: Int, n: Int, min_bits: usize) -> impl Clone + Iterator<Item = Int> {
    let start: Int = ZZ.euclidean_div(ZZ.power_of_two(min_bits), &n);
    return (start..).map(move |i| i * n + a).filter(move |p| algorithms::miller_rabin::is_prime(ZZ, &(*p as Int), 8)).filter(move |p| *p >= ZZ.power_of_two(min_bits))
}