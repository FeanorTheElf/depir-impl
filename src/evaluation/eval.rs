use std::cell::RefCell;

use feanor_math::homomorphism::*;
use feanor_math::integer::*;
use feanor_math::primitive_int::{StaticRing, StaticRingBase};
use feanor_math::rings::zn::{ZnRingStore, ZnRing};
use feanor_math::ring::*;
use feanor_math::vector::vec_fn::*;

use crate::globals::{Level1Modulus, Level2Modulus, Level0ModulusRam, Level0ModulusDisk};
use crate::params::AllModuli;
use crate::params::PolyEvaluationParams;
use crate::params::PolyParams;

use super::disk_preevaluations::PrimeFieldDiskPreevaluations;
use super::async_read::perform_reads_async;
use super::ram_preevaluations::PrimeFieldRAMPreevaluations;
use super::prime_components::*;

pub type Level0PrimeDataRam<'a, const m: usize> = PrimeFieldRAMPreevaluations<'a, m>;
pub type Level0PrimeDataDisk<'a, const m: usize> = PrimeFieldDiskPreevaluations<'a, m>;

pub type Level1PrimeData<'a> = PrimeDecomposition<'a, &'a Level1Modulus, Level0ModulusRam, Level0ModulusDisk, Level0ModulusDisk, StaticRing<i32>>;
pub type Level2PrimeData<'a> = PrimeDecomposition<'a, &'a Level2Modulus, Level0ModulusRam, Level0ModulusDisk, Level1Modulus, StaticRing<i32>>;

static EMPTY: [Level0ModulusDisk; 0] = [];

pub struct FastPolyEvaluator<'a, const m: usize> {
    level_0_data_ram: Vec<Level0PrimeDataRam<'a, m>>, 
    level_0_data_disk: Vec<Level0PrimeDataDisk<'a, m>>,
    level_1_data: Vec<Level1PrimeData<'a>>,
    level_2_data: Vec<Level2PrimeData<'a>>,
}

impl<'a, const m: usize> FastPolyEvaluator<'a, m> {

    fn create_level_1_data(ram_primes: &'a [Level0ModulusRam], disk_primes: &'a [Level0ModulusDisk], lvl1_primes: &'a [Level1Modulus], poly_params: PolyParams) -> Vec<Level1PrimeData<'a>> {
        let mut level_1_data: Vec<Level1PrimeData<'a>> = Vec::new();
        for Fp in lvl1_primes {
            let (ram_primes, disk_primes) = PolyEvaluationParams::on_level_1(poly_params, *Fp).reduce_ram_disk(ram_primes, disk_primes);

            let max_prime = ram_primes.iter().map(|Fp| *Fp.modulus() as i64).chain(disk_primes.iter().map(|Fp| *Fp.modulus() as i64)).max().unwrap();
            let gamma = (ram_primes.len() + disk_primes.len()) * max_prime as usize;

            let conv = Level1PrimeData::new(
                Fp, 
                (ram_primes, disk_primes, &EMPTY), 
                StaticRing::<i32>::RING, 
                gamma as i32
            );
            level_1_data.push(conv);
        }
        return level_1_data;
    }

    fn create_level_2_data(ram_primes: &'a [Level0ModulusRam], disk_primes: &'a [Level0ModulusDisk], lvl1_primes: &'a [Level1Modulus], lvl2_primes: &'a [Level2Modulus], poly_params: PolyParams) -> Vec<Level2PrimeData<'a>> {
        let mut level_2_data: Vec<Level2PrimeData<'a>> = Vec::new();
        for Fp in lvl2_primes {
            let (ram_primes, disk_primes, lvl1_primes) = PolyEvaluationParams::on_level_2(poly_params, *Fp).reduce_ram_disk_lvl1(ram_primes, disk_primes, lvl1_primes);

            let max_prime = ram_primes.iter().map(|Fp| *Fp.modulus() as i64)
                .chain(disk_primes.iter().map(|Fp| *Fp.modulus() as i64))
                .chain(lvl1_primes.iter().map(|Fp| *Fp.modulus() as i64)).max().unwrap();

            let gamma = (ram_primes.len() + disk_primes.len() + lvl1_primes.len()) * max_prime as usize;

            let conv = Level2PrimeData::new(
                Fp, 
                (ram_primes, disk_primes, lvl1_primes), 
                StaticRing::<i32>::RING, 
                gamma as i32
            );
            level_2_data.push(conv);
        }
        return level_2_data;
    }

    pub fn new<I>(primes: &'a AllModuli, poly: &[El<I>], poly_params: PolyParams, ZZ: I) -> Self
        where I: IntegerRingStore,
            I::Type: IntegerRing,
            El<I>: Clone,
            <Level0ModulusRam as RingStore>::Type: CanHomFrom<I::Type>,
            <Level0ModulusDisk as RingStore>::Type: CanHomFrom<I::Type>
    {        
        let level_1_data = Self::create_level_1_data(&primes.level_0_ram_primes, &primes.level_0_disk_primes, &primes.level_1_primes, poly_params);
        let level_2_data = Self::create_level_2_data(&primes.level_0_ram_primes, &primes.level_0_disk_primes, &primes.level_1_primes, &primes.level_2_primes, poly_params);

        let mut level_0_data_ram: Vec<PrimeFieldRAMPreevaluations<'_, m>> = Vec::new();
        for Fp in &primes.level_0_ram_primes {
            level_0_data_ram.push(PrimeFieldRAMPreevaluations::new(Fp, poly_params.d(), poly, &ZZ));
        }

        let mut level_0_data_disk: Vec<PrimeFieldDiskPreevaluations<'_, m>> = Vec::new();
        for Fp in &primes.level_0_disk_primes {
            level_0_data_disk.push(PrimeFieldDiskPreevaluations::new(Fp, poly_params.d(), poly, &ZZ));
        }

        FastPolyEvaluator { level_0_data_ram, level_0_data_disk, level_1_data, level_2_data }
    }

    fn get_ram_data(&self, parent_len0: usize, supply_index: usize) -> &Level0PrimeDataRam<m> {
        let ram_data_index = self.level_0_data_ram.len() - parent_len0 + supply_index;
        return &self.level_0_data_ram[ram_data_index];
    }

    fn get_disk_data(&self, parent_len1: usize, supply_index: usize) -> &Level0PrimeDataDisk<m> {
        let disk_data_index = self.level_0_data_disk.len() - parent_len1 + supply_index;
        return &self.level_0_data_disk[disk_data_index];
    }

    fn get_level_1_data_index(&self, parent_len2: usize, supply_index: usize) -> usize {
        self.level_1_data.len() - parent_len2 + supply_index
    }

    #[inline(never)]
    fn process_ram<'env, 'b, V, R, T2, T3>(&'env self, x: V, prime_data: &PrimeDecomposition<'b, R, Level0ModulusRam, T2, T3, StaticRing<i32>>, composers: &RefCell<Vec<PrimeComposer<'b, R, Level0ModulusRam, T2, T3, StaticRing<i32>>>>)
        where V: VectorFn<El<R>>,
            R: Clone,
            R: ZnRingStore, R::Type: ZnRing + CanHomFrom<StaticRingBase<i32>> + CanHomFrom<BigIntRingBase>,
            T2: ZnRingStore, T2::Type: ZnRing + CanHomFrom<StaticRingBase<i32>> + CanHomFrom<BigIntRingBase>,
            T3: ZnRingStore, T3::Type: ZnRing + CanHomFrom<StaticRingBase<i32>> + CanHomFrom<BigIntRingBase>
    {
        // we perform prefetching, as the hardware prefetcher will not be able to predict our access pattern;
        // this is the number of loop iterations that we prefetch elements in advance of being used
        const LOOKAHEAD: usize = 4;

        assert!(x.len() % m == 0);
        assert!(prime_data.len0() >= LOOKAHEAD);
        let point_count = x.len() / m;

        let mut composer_borrow = composers.borrow_mut();

        if LOOKAHEAD == 0 {

            // basically, we just want to execute this code here
            for k in 0..point_count {
                let in_point = || core::array::from_fn(|l| x.at(k * m + l));
                for j in 0..prime_data.len0() {
                    let point = prime_data.direct_decompose0_multiple(in_point(), j);
                    composer_borrow[k].supply0(self.get_ram_data(prime_data.len0(), j).lookup(point), j);
                }
            }

        } else {

            // ... but possibly use prefetching
            for k in 0..point_count {
                let in_point = || core::array::from_fn(|l| x.at(k * m + l));
                let mut i = 0;
                let mut index_cycle = [0; LOOKAHEAD];
    
                for j in 0..LOOKAHEAD {
                    index_cycle[j] = self.get_ram_data(prime_data.len0(), j).get_index(prime_data.direct_decompose0_multiple(in_point(), j));
                }
        
                for j in 0..(prime_data.len0() - LOOKAHEAD) {
                    composer_borrow[k].supply0(self.get_ram_data(prime_data.len0(), j).lookup_index(index_cycle[i]), j);
                    
                    let point = prime_data.direct_decompose0_multiple(in_point(), j + LOOKAHEAD);
                    index_cycle[i] = self.get_ram_data(prime_data.len0(), j + LOOKAHEAD).get_index(point);
                    self.get_ram_data(prime_data.len0(), j + LOOKAHEAD).prefetch(index_cycle[i]);
                    i = (i + 1) % LOOKAHEAD;
                }
    
                for j in (prime_data.len0() - LOOKAHEAD)..prime_data.len0() {
                    composer_borrow[k].supply0(self.get_ram_data(prime_data.len0(), j).lookup_index(index_cycle[i]), j);
                    i = (i + 1) % LOOKAHEAD;
                }
            }
        }
    }

    #[inline(never)]
    fn process_disk<'env, 'b, V, R, T1, T3>(&'env self, x: V, prime_data: &PrimeDecomposition<'b, R, T1, Level0ModulusDisk, T3, StaticRing<i32>>, composers: &RefCell<Vec<PrimeComposer<'b, R, T1, Level0ModulusDisk, T3, StaticRing<i32>>>>)
        where V: VectorFn<El<R>>,
            R: Clone,
            R: ZnRingStore, R::Type: ZnRing + CanHomFrom<StaticRingBase<i32>> + CanHomFrom<BigIntRingBase>,
            T1: ZnRingStore, T1::Type: ZnRing + CanHomFrom<StaticRingBase<i32>> + CanHomFrom<BigIntRingBase>,
            T3: ZnRingStore, T3::Type: ZnRing + CanHomFrom<StaticRingBase<i32>> + CanHomFrom<BigIntRingBase>
    {
        assert!(x.len() % m == 0);
        let point_count = x.len() / m;

        for j in 0..prime_data.len1() {
            perform_reads_async(|read_context| {

                println!("Reading {}", self.get_disk_data(prime_data.len1(), j).get_filename());
                let composers_ref = &composers;
                let mut file_reader = read_context.open_file(
                    self.get_disk_data(prime_data.len1(), j).get_filename(), 
                    self.get_disk_data(prime_data.len1(), j).lookup_callback(move |el, k| composers_ref.borrow_mut()[k].supply1(el, j))
                );
                for k in 0..point_count {
                    let point = prime_data.direct_decompose1_multiple(core::array::from_fn(|l| x.at(k * m + l)), j);
                    file_reader.submit(self.get_disk_data(prime_data.len1(), j).lookup_index(point));
                }
            });
        }
    }

    #[inline(never)]
    pub(super) fn eval_many_level_1<'env, V, F>(&'env self, i: usize, x: V, mut result: F)
        where V: VectorFn<El<Level1Modulus>>,
            F: FnMut(usize, El<Level1Modulus>)
    {
        assert!(x.len() % m == 0);
        let point_count = x.len() / m;
        let prime_data = &self.level_1_data[i];
        let composers = RefCell::new(Iterator::map(0..point_count, |_| prime_data.start_compose()).collect::<Vec<_>>());
        
        self.process_ram(&x, &prime_data, &composers);
        self.process_disk(&x, &prime_data, &composers);
        
        for (i, res) in composers.into_inner().into_iter().enumerate() {
            result(i, res.finish())
        }
    }

    #[inline(never)]
    pub(super) fn eval_many_level_2<'env, V, F>(&'env self, i: usize, x: V, mut result: F)
        where V: VectorFn<El<Level2Modulus>>,
            F: FnMut(usize, El<Level2Modulus>)
    {
        assert!(x.len() % m == 0);
        let point_count = x.len() / m;
        let prime_data = &self.level_2_data[i];

        let composers = RefCell::new(Iterator::map(0..point_count, |_| prime_data.start_compose()).collect::<Vec<_>>());

        self.process_ram(&x, &prime_data, &composers);
        self.process_disk(&x, &prime_data, &composers);

        // and then the data from level 1
        for j in 0..prime_data.len2() {

            let composers_ref = &composers;
            self.eval_many_level_1(
                self.get_level_1_data_index(prime_data.len2(), j), 
                (&x).map(|c| prime_data.direct_decompose2(c, j)), 
                |k, res| composers_ref.borrow_mut()[k].supply2(res, j)
            );
        }

        for (i, res) in composers.into_inner().into_iter().enumerate() {
            result(i, res.finish())
        }
    }
}

#[cfg(test)]
use crate::rings::sample_primes_arithmetic_progression;
#[cfg(test)]
use feanor_math::assert_el_eq;
#[cfg(test)]
use crate::interpolate::evaluate_poly;

#[test]
fn test_fast_poly_eval_level_1() {
    let GF = |p: i64| <Level0ModulusRam as RingStore>::Type::new(p as u32);

    // 1 + 2x^2 + y + xy + x^2y + 2y^2 + 2y^3
    let poly_params = PolyParams::new(2, 2, 3);
    let poly = [1, 0, 2, 0, 1, 1, 1, 2, 0, 2];
    let d = 3;
    let target = <Level1Modulus as RingStore>::Type::new(60013);
    let primes = AllModuli {
        level_0_ram_primes: sample_primes_arithmetic_progression(1, 2, 1).take(20).map(GF).collect::<Vec<_>>(),
        level_0_disk_primes: vec![],
        level_1_primes: vec![target],
        level_2_primes: vec![]
    };
    let evaluator: FastPolyEvaluator<'_, 2> = FastPolyEvaluator::new(&primes, &poly, poly_params, StaticRing::<i32>::RING);
    for (x, y) in [(0, 0), (1, 0), (0, 1), (0, 50000), (1, 50000), (0, 50001), (1, 50001), (50000, 50000), (50001, 50000), (50000, 50001), (50001, 50001)] {
        let point = [target.int_hom().map(x), target.int_hom().map(y)];
        let mut result = [primes.level_1_primes[0].zero()];
        evaluator.eval_many_level_1(0, point.into_fn(), |i, x| result[i] = x);
        assert_el_eq!(&target, &evaluate_poly((&poly[..]).into_fn().map(|x| target.int_hom().map(x)), d, &point[..], &target), &result[0]);
    }
}

#[test]
fn test_fast_poly_eval_level_2() {
    let GF = |p: i64| <Level0ModulusRam as RingStore>::Type::new(p as u32);

    // 1 + 2x^2 + y + xy + x^2y + 2y^2 + 2y^3
    let poly_params = PolyParams::new(2, 2, 3);
    let poly = [1, 0, 2, 0, 1, 1, 1, 2, 0, 2];
    let d = 3;
    let target = <Level2Modulus as RingStore>::Type::new(1048583);
    let primes = AllModuli {
        level_0_ram_primes: sample_primes_arithmetic_progression(1, 2, 1).take(20).map(GF).collect::<Vec<_>>(),
        level_0_disk_primes: vec![],
        level_1_primes: vec![],
        level_2_primes: vec![target]
    };
    let evaluator: FastPolyEvaluator<'_, 2> = FastPolyEvaluator::new(&primes, &poly, poly_params, StaticRing::<i32>::RING);
    for (x, y) in [(0, 0), (1, 0), (0, 1), (0, 500000), (1, 500000), (0, 500001), (1, 500001), (500000, 500000), (500001, 500000), (500000, 500001), (500001, 500001)] {
        let point = [target.int_hom().map(x), target.int_hom().map(y)];
        let mut result = [primes.level_2_primes[0].zero()];
        evaluator.eval_many_level_2(0, point.into_fn(), |i, x| result[i] = x);
        assert_el_eq!(&target, &evaluate_poly((&poly[..]).into_fn().map(|x| target.int_hom().map(x)), d, &point[..], &target), &result[0]);
    }
}

#[test]
fn test_fast_poly_eval_level_mixed() {
    let GF = |p: i64| <Level0ModulusRam as RingStore>::Type::new(p as u32);

    // 1 + 2x^2 + y + xy + x^2y + 2y^2 + 2y^3
    let poly_params = PolyParams::new(2, 2, 3);
    let poly = [1, 0, 2, 0, 1, 1, 1, 2, 0, 2];
    let d = 3;
    let level_0_primes = sample_primes_arithmetic_progression(1, 2, 1).take(10).map(GF).collect::<Vec<_>>();
    let level_1_primes = vec![
        <Level1Modulus as RingStore>::Type::new(37),
        <Level1Modulus as RingStore>::Type::new(41),
        <Level1Modulus as RingStore>::Type::new(43),
        <Level1Modulus as RingStore>::Type::new(47),
        <Level1Modulus as RingStore>::Type::new(53),
        <Level1Modulus as RingStore>::Type::new(59)
    ];
    let target = <Level2Modulus as RingStore>::Type::new(1048583);
    let primes = AllModuli {
        level_0_ram_primes: level_0_primes,
        level_0_disk_primes: vec![],
        level_1_primes: level_1_primes,
        level_2_primes: vec![target]
    };
    let evaluator: FastPolyEvaluator<'_, 2> = FastPolyEvaluator::new(&primes, &poly, poly_params, StaticRing::<i32>::RING);
    for (x, y) in [(0, 0), (1, 0), (0, 1), (0, 500000), (1, 500000), (0, 500001), (1, 500001), (500000, 500000), (500001, 500000), (500000, 500001), (500001, 500001)] {
        let point = [target.int_hom().map(x), target.int_hom().map(y)];
        let mut result = [primes.level_2_primes[0].zero()];
        evaluator.eval_many_level_2(0, point.into_fn(), |i, x| result[i] = x );
        assert_el_eq!(&target, &evaluate_poly((&poly[..]).into_fn().map(|x| target.int_hom().map(x)), d, &point[..], &target), &result[0]);
    }
}