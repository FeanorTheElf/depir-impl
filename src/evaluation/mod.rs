use feanor_math::homomorphism::*;
use feanor_math::vector::vec_fn::{VectorFn, IntoVectorFn};
use feanor_math::ring::*;
use feanor_math::rings::zn::ZnRingStore;

use crate::globals::{Int, ZZ, Zn, TopLevelModulus};
use crate::params::{PolyParams, AllModuli};

use eval::FastPolyEvaluator;

///
/// Yet another implementation of `Zn`, this one uses as little space as possible
/// to represent its elements.
/// 
pub mod zn;

///
/// Compute the CRT-shortest-lift-map `Z/q <-> Z/(p1 p2 ... pr) ~ Z/p1 x ... x Z/pr` fast.
/// 
pub mod prime_components;

///
/// The actual evaluation using the datastructure.
/// 
pub mod eval;

///
/// Fast asynchronous reads from an SSD drive on Windows.
/// Note that this currently does not use multithreading,
/// so the only multithreading is at the very top in [`crate::Server`].
/// 
pub mod async_read;

///
/// An evaluation datastructure component, stored in RAM
/// 
pub mod ram_preevaluations;

///
/// An evaluation datastructure component, stored on Disk
/// 
pub mod disk_preevaluations;

///
/// Encapsulates the fast evaluation using the datastructure
/// 
pub struct Evaluator<'a> {
    homs: Vec<CanHom<Zn, &'a TopLevelModulus>>,
    poly_params: PolyParams,
    evaluator_m3: Option<FastPolyEvaluator<'a, 3>>,
    evaluator_m4: Option<FastPolyEvaluator<'a, 4>>,
    evaluator_m5: Option<FastPolyEvaluator<'a, 5>>,
    evaluator_m6: Option<FastPolyEvaluator<'a, 6>>
}

impl<'a> Evaluator<'a> {

    pub fn new(primes: &'a AllModuli, poly_params: PolyParams, poly: &[Int]) -> Self {
        let homs = primes.level_2_primes.iter().map(|Fp| Fp.into_can_hom(Zn::new(*Fp.modulus() as u64)).ok().unwrap()).collect::<Vec<_>>();
        match poly_params.m() {
            3 => Evaluator {
                poly_params,
                evaluator_m3: Some(FastPolyEvaluator::new(primes, poly, poly_params, ZZ)),
                evaluator_m4: None,
                evaluator_m5: None,
                evaluator_m6: None,
                homs: homs
            },
            4 => Evaluator {
                poly_params,
                evaluator_m3: None,
                evaluator_m4: Some(FastPolyEvaluator::new(primes, poly, poly_params, ZZ)),
                evaluator_m5: None,
                evaluator_m6: None,
                homs: homs
            },
            5 => Evaluator {
                poly_params,
                evaluator_m3: None,
                evaluator_m4: None,
                evaluator_m5: Some(FastPolyEvaluator::new(primes, poly, poly_params, ZZ)),
                evaluator_m6: None,
                homs: homs
            },
            6 => Evaluator {
                poly_params,
                evaluator_m3: None,
                evaluator_m4: None,
                evaluator_m5: None,
                evaluator_m6: Some(FastPolyEvaluator::new(primes, poly, poly_params, ZZ)),
                homs: homs
            },
            _ => panic!("unsupported m")
        }
    }

    pub fn evaluate_many(&self, prime_index: usize, points: &[El<Zn>], out: &mut [El<TopLevelModulus>]) {
        assert!(points.len() % self.poly_params.m() == 0);
        let hom = &self.homs[prime_index];
        match self.poly_params.m() {
            3 => self.evaluator_m3.as_ref().unwrap().eval_many_level_2(prime_index, points.into_fn().map(|x| hom.map(x)), |i, x| out[i] = x),
            4 => self.evaluator_m4.as_ref().unwrap().eval_many_level_2(prime_index, points.into_fn().map(|x| hom.map(x)), |i, x| out[i] = x),
            5 => self.evaluator_m5.as_ref().unwrap().eval_many_level_2(prime_index, points.into_fn().map(|x| hom.map(x)), |i, x| out[i] = x),
            6 => self.evaluator_m6.as_ref().unwrap().eval_many_level_2(prime_index, points.into_fn().map(|x| hom.map(x)), |i, x| out[i] = x),
            _ => panic!()
        }
    }
}