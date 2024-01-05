use std::cmp::{min, max};
use std::collections::HashSet;

use feanor_math::integer::{IntegerRingStore, int_cast, BigIntRing};
use feanor_math::mempool::DefaultMemoryProvider;
use feanor_math::primitive_int::StaticRing;
use feanor_math::rings::finite::FiniteRingStore;
use feanor_math::{ring::*, default_memory_provider};
use feanor_math::rings::zn::{ZnRingStore, zn_42, zn_rns};

use crate::evaluation::disk_preevaluations::estimate_used_disk_space;
use crate::interpolate::InterpolationMatrix;
use crate::{CompressedEl, CiphertextSeed, CiphertextRing, MainRing, PlaintextRing};
use crate::globals::{Level1Modulus, Level2Modulus, Level0ModulusRam, Level0ModulusDisk, CompressLevel0RamEl, CompressLevel0DiskEl, TopLevelModulus};
use crate::rings::{sample_primes_arithmetic_progression, required_root_of_unity_order, tensor, cyclotomic_cc};
use crate::{globals::{ZZbig, binomial, ZZ}, evaluation::ram_preevaluations::estimate_used_ram};

#[derive(Hash, PartialEq, Eq, Clone, Copy)]
pub struct HEParams {
    plaintext_modulus: i64,
    log2_ring_degree: usize,
    ciphertext_len: usize,
    ciphertext_moduli_bitlength: usize,
    ciphertext_moduli_count: usize,
    reply_ciphertext_moduli_count: usize
}

impl HEParams {

    pub fn new(plaintext_modulus: i64, log2_ring_degree: usize, ciphertext_len: usize, ciphertext_moduli_bitlength: usize, ciphertext_moduli_count: usize, reply_ciphertext_moduli_count: usize) -> Self {
        Self { plaintext_modulus, log2_ring_degree, ciphertext_len, ciphertext_moduli_bitlength, ciphertext_moduli_count, reply_ciphertext_moduli_count }
    }

    pub fn ciphertext_primes(self) -> impl Clone + Iterator<Item = i64> {
        sample_primes_arithmetic_progression(1, required_root_of_unity_order(self.ciphertext_len, self.log2_ring_degree) as i64, self.ciphertext_moduli_bitlength)
            .take(self.ciphertext_moduli_count)
    }

    pub fn reply_ciphertext_moudulus_bits(self, ciphertext_moduli: &[TopLevelModulus]) -> f64 {
        ciphertext_moduli.iter().take(self.reply_ciphertext_moduli_count).map(|Fp| (*Fp.modulus() as f64).log(2.)).sum::<f64>()
    }

    pub fn reply_ciphertext_size(self) -> usize {
        std::mem::size_of::<CompressedEl>() * self.reply_ciphertext_moduli_count * (1 << self.log2_ring_degree) + std::mem::size_of::<CiphertextSeed>()
    }

    pub fn ciphertext_modulus_bits(self, ciphertext_modulus: &[TopLevelModulus]) -> f64 {
        ciphertext_modulus.iter().map(|Fp| (*Fp.modulus() as f64).log(2.)).sum::<f64>()
    }

    pub fn print(self, ciphertext_modulus: &[TopLevelModulus]) {
        println!("    Security:               RLWE({}, {}, Binom(4))", 1 << self.log2_ring_degree, self.ciphertext_modulus_bits(ciphertext_modulus).round());
        println!("    Plaintext modulus:      {}", self.plaintext_modulus);
        println!("    Ciphertext modulus:     {} bits", self.ciphertext_modulus_bits(ciphertext_modulus).round());
        println!("    Ring degree:            {}", 1 << self.log2_ring_degree);
        println!("    Ciphertext len:         {}", self.ciphertext_len);
    }

    pub fn reply_error_bits(self) -> f64 {
        0.
    }

    pub fn error_bits(self, degree: usize) -> f64 {
        // ring expansion factor
        let delta = (1 << self.log2_ring_degree) as f64;
        // since chi = binom(4)
        let e0_error_log = (self.plaintext_modulus as f64).log2() + 2.;
        (e0_error_log + delta.log2() + 1.) * degree as f64
    }

    pub fn ciphertext_size(self) -> usize {
        std::mem::size_of::<CompressedEl>() * self.ciphertext_moduli_count * (1 << self.log2_ring_degree) + std::mem::size_of::<CiphertextSeed>()
    }
}

#[derive(Hash, PartialEq, Eq, Clone, Copy)]
pub struct DEPIRParams {
    db_poly_params: PolyParams,
    he_params: HEParams,
    lvl1_primes_bound: usize,
    ram_bytes_bound: usize,
    ram_prime_bound: usize
}

pub struct AllPrimes {
    pub level_2_primes: Vec<i64>,
    pub level_1_primes: Vec<i64>,
    pub level_0_disk_primes: Vec<i64>,
    pub level_0_ram_primes: Vec<i64>
}

pub struct AllModuli {
    pub level_2_primes: Vec<Level2Modulus>,
    pub level_1_primes: Vec<Level1Modulus>,
    pub level_0_disk_primes: Vec<Level0ModulusDisk>,
    pub level_0_ram_primes: Vec<Level0ModulusRam>
}

impl AllModuli {
    pub fn new(primes: AllPrimes) -> Self {
        return AllModuli {
            level_2_primes: primes.level_2_primes.iter().map(|p| <Level2Modulus as RingStore>::Type::new(*p as u64)).collect(),
            level_1_primes: primes.level_1_primes.iter().map(|p| <Level1Modulus as RingStore>::Type::new(*p as u32)).collect(),
            level_0_disk_primes: primes.level_0_disk_primes.iter().map(|p| <Level0ModulusDisk as RingStore>::Type::new(*p as u32)).collect(),
            level_0_ram_primes: primes.level_0_ram_primes.iter().map(|p| <Level0ModulusRam as RingStore>::Type::new(*p as u32)).collect(),
        };
    }
}

impl DEPIRParams {
    
    pub fn new(db_poly_params: PolyParams, he_params: HEParams, lvl1_primes_bound: usize, ram_bytes_bound: usize, ram_prime_bound: usize) -> Self {
        Self { db_poly_params, he_params, lvl1_primes_bound, ram_bytes_bound, ram_prime_bound }
    }

    pub fn create_primes(self) -> AllPrimes {
        let level_2_primes = self.he_params.ciphertext_primes();
        for p in level_2_primes.clone() {
            assert!(p - 1 <= u32::MAX as i64);
        }

        let required_evaluations = level_2_primes.clone().map(|p| PolyEvaluationParams::new(self.db_poly_params, p));
        let reduction_primes = sample_primes_arithmetic_progression(1, 2, 1).take(1000).collect::<Vec<_>>();

        let reduction_primes_ref = &reduction_primes[..];
        let level_0_1_primes = required_evaluations.flat_map(move |eval| eval.reduce_shortest_lift(reduction_primes_ref.into_iter().copied())).collect::<HashSet<_>>();
        assert!(level_0_1_primes.iter().all(|eval| eval.poly_params() == self.db_poly_params));

        let level_0_primes = level_0_1_primes.iter().flat_map(move |eval| eval.reduce_shortest_lift(reduction_primes_ref.into_iter().copied())).collect::<HashSet<_>>();
        assert!(level_0_1_primes.iter().all(|eval| eval.poly_params() == self.db_poly_params));

        let mut level_0_1_primes = level_0_1_primes.into_iter().map(|eval| eval.char()).collect::<Vec<_>>();
        level_0_1_primes.sort();

        let level_0_primes_len = max(level_0_primes.len(), level_0_1_primes.len() - self.lvl1_primes_bound);
        
        let ram_primes_len = get_ram_prime_number(level_0_1_primes[..level_0_primes_len].iter().copied(), self.db_poly_params.m(), self.ram_bytes_bound, self.ram_prime_bound);
        let level_0_primes_ram = &level_0_1_primes[..ram_primes_len];
        let level_0_primes_disk = &level_0_1_primes[ram_primes_len..level_0_primes_len];
        let level_1_primes = &level_0_1_primes[level_0_primes_len..];

        return AllPrimes {
            level_2_primes: level_2_primes.collect(),
            level_1_primes: level_1_primes.iter().copied().collect(),
            level_0_disk_primes: level_0_primes_disk.iter().copied().collect(),
            level_0_ram_primes: level_0_primes_ram.iter().copied().collect()
        }
    }

    pub fn disk_accesses(&self, primes: &AllModuli) -> usize {
        self.he_params.ciphertext_len * (1 << self.he_params.log2_ring_degree) * self.db_poly_params.disk_accesses(primes)
    }

    pub fn ram_accesses(&self, primes: &AllModuli) -> usize {
        self.he_params.ciphertext_len * (1 << self.he_params.log2_ring_degree) * self.db_poly_params.ram_accesses(primes)
    }

    pub fn ram_memory(&self, primes: &AllModuli) -> usize {
        self.db_poly_params.ram_memory(primes)
    }

    pub fn disk_memory(&self, primes: &AllModuli) -> usize {
        self.db_poly_params.disk_memory(primes)
    }

    pub fn create_scalar_plaintext_ring(&self) -> zn_42::Zn {
        zn_42::Zn::new(self.he_params.plaintext_modulus as u64)
    }

    pub fn create_plaintext_ring(&self) -> PlaintextRing {
        cyclotomic_cc::Pow2CyclotomicRingBase::new(self.create_scalar_plaintext_ring(), self.he_params.log2_ring_degree)
    }

    pub fn create_scalar_ciphertext_ring(&self, primes: &[TopLevelModulus]) -> zn_rns::Zn<zn_42::Zn, BigIntRing, DefaultMemoryProvider> {
        zn_rns::Zn::new(primes.iter().map(|Fp| zn_42::Zn::new(*Fp.modulus() as u64)).collect(), ZZbig, default_memory_provider!())
    }

    pub fn create_reply_ciphertext_ring(&self, primes: &[TopLevelModulus]) -> MainRing {
        self.create_ciphertext_ring(primes).base_ring().get_ring().reduce_rns_base(self.he_params.reply_ciphertext_moduli_count)
    }

    pub fn create_ciphertext_ring(&self, primes: &[TopLevelModulus]) -> CiphertextRing {
        tensor::RNSCyclotomicTensorRingBase::new(self.create_scalar_ciphertext_ring(primes), self.he_params.log2_ring_degree, self.he_params.ciphertext_len)
    }

    pub fn d(&self) -> usize {
        self.db_poly_params.d()
    }

    pub fn m(&self) -> usize {
        self.db_poly_params.m()
    }

    pub fn N(&self) -> usize {
        self.db_poly_params.max_monomials()
    }

    pub fn ciphertext_moduli_count(&self) -> usize {
        self.he_params.ciphertext_moduli_count
    }

    pub fn reply_ciphertext_moduli_count(&self) -> usize {
        self.he_params.reply_ciphertext_moduli_count
    }

    pub fn poly_params(&self) -> PolyParams {
        self.db_poly_params
    }
    
    pub fn validate(self, top_level_primes: &[TopLevelModulus]) {
        assert!(self.db_poly_params.variable_number >= 3 && self.db_poly_params.variable_number <= 6);
        assert!(self.db_poly_params.total_degree < self.he_params.plaintext_modulus as usize);
        assert!(self.db_poly_params.total_degree < self.he_params.ciphertext_len);
        assert!(self.he_params.plaintext_modulus >= 2);
        assert!(self.he_params.ciphertext_len >= 1);
        assert!(self.he_params.ciphertext_len % 2 == 1);
        assert!(self.he_params.ciphertext_moduli_count > 0);
        assert!(self.db_poly_params.poly_inf_norm >= self.he_params.plaintext_modulus / 2);
        // if let Some(relin_params) = self.relin_params {
        //     assert!(self.he_params.ciphertext_moduli_count >= relin_params.relin_ciphertext_moduli_count);
        //     assert!(ZZbig.is_geq(&ZZbig.pow(ZZbig.power_of_two(relin_params.relin_decomposition_basis_bits), relin_params.relin_decomposition_len - 1), &ZZbig.prod(top_level_primes.iter().map(|Fp| int_cast(*Fp.modulus(), &ZZbig, &ZZ)))));
        // }
        assert_eq!(top_level_primes.len(), self.he_params.ciphertext_moduli_count);
    }

    pub fn print(self, primes: &AllModuli) {
        println!("  N (database size):      {}", self.db_poly_params.max_monomials());
        println!("  HE params");
        self.he_params.print(&primes.level_2_primes);
        println!("  Worst-case error:       {} bits", self.he_params.error_bits(self.db_poly_params.d()).round());
        println!("  Reply modulus:          {} bits", self.he_params.reply_ciphertext_moudulus_bits(&primes.level_2_primes).round());
        println!("  Worst-case reply noise: {} bits", self.he_params.reply_error_bits().round());
        println!("  d (degree of poly):     {}", self.db_poly_params.d());
        println!("  m (number of vars):     {}", self.db_poly_params.m());
        println!("  Level 0 ram primes:     {}, largest {}", primes.level_0_ram_primes.len(), primes.level_0_ram_primes.last().map(|Fp| *Fp.modulus()).unwrap());
        println!("  Level 0 disk primes:    {}, largest {}", primes.level_0_disk_primes.len(), primes.level_0_disk_primes.last().map(|Fp| *Fp.modulus()).unwrap_or(0));
        println!("  Level 1 primes:         {}, largest {} ", primes.level_1_primes.len(), primes.level_1_primes.last().map(|Fp| *Fp.modulus() as i64).unwrap_or(0));
        println!("  Level 2 primes:         {}, largest {} ", primes.level_2_primes.len(), *primes.level_2_primes.last().unwrap().modulus());
        println!("  Query size:             {} MB", self.he_params.ciphertext_size() as f64 * self.db_poly_params.m() as f64 / (1 << 20) as f64);
        println!("  Reply size:             {} MB", self.he_params.reply_ciphertext_size() as f64 / (1 << 20) as f64);
        println!("  Server Ram memory:      {} GB", (self.ram_memory(primes) as f64 / (1 << 30) as f64).ceil());
        println!("  Server Disk memory:     {} GB", (self.disk_memory(primes) as f64 / (1 << 30) as f64).ceil());
        println!("  Database accesses:      {} random reads of size {} B to RAM", self.ram_accesses(primes), std::mem::size_of::<CompressLevel0RamEl>());
        println!("                          {} random reads of size {} B to disk", self.disk_accesses(primes), std::mem::size_of::<CompressLevel0DiskEl>());
        println!("  Estimated runtime:      {} s", STANDARD_PCIE_SSD_ENV.estimate_running_time(self, primes));
    }

    pub fn create_point_grid(self) -> InterpolationMatrix<zn_42::Zn, Vec<El<zn_42::Zn>>> {
        let ring = self.create_scalar_plaintext_ring();
        InterpolationMatrix::new(
            Iterator::map(0..self.db_poly_params.m(), |_| ring.elements().take(self.db_poly_params.d() + 1)
                .collect::<Vec<_>>()), 
                ring.clone()
        )
    }
}

#[derive(Hash, PartialEq, Eq, Clone, Copy)]
pub struct PolyParams {
    variable_number: usize,
    poly_inf_norm: i64,
    total_degree: usize
}

impl PolyParams {

    pub fn new(variable_number: usize, poly_inf_norm: i64, total_degree: usize) -> Self {
        Self { variable_number, poly_inf_norm, total_degree }
    }

    pub fn d(&self) -> usize {
        self.total_degree
    }

    pub fn m(&self) -> usize {
        self.variable_number
    }

    pub fn max_monomials(&self) -> usize {
        binomial(self.d() + self.m(), self.m())
    }

    fn poly_eval_bound_log2(&self, char: i64) -> f64 {
        let input_bound = int_cast((char - 1) / 2 + 1, &ZZbig, &ZZ);
        ZZbig.abs_log2_ceil(
            &ZZbig.sum((0..=self.d()).map(|k: usize| {
                let degree_k_part = ZZbig.prod([
                    int_cast(binomial(k + self.m() - 1, k) as i64, &ZZbig, &ZZ),
                    int_cast(self.poly_inf_norm, &ZZbig, &ZZ),
                    ZZbig.pow(ZZbig.clone_el(&input_bound), k)
                ].into_iter());
                return degree_k_part;
            }))
        ).unwrap() as f64 - POLY_EVAL_BOUND_HEURISTIC_LOG2
    }

    pub fn ram_memory(self, primes: &AllModuli) -> usize {
        let level_0_primes_ram = &primes.level_0_ram_primes;
        int_cast(
            ZZbig.sum(level_0_primes_ram.iter().map(
                |Fp| ZZbig.coerce(&StaticRing::<i64>::RING, estimate_used_ram(*Fp.modulus() as i64, self.variable_number) as i64)
            )), &ZZ, &ZZbig
        ) as usize
    }

    pub fn disk_memory(self, primes: &AllModuli) -> usize {
        let level_0_primes_disk = &primes.level_0_disk_primes;
        int_cast(
            ZZbig.sum(level_0_primes_disk.iter().map(
                |Fp| ZZbig.coerce(&StaticRing::<i64>::RING, estimate_used_disk_space(*Fp.modulus() as i64, self.variable_number) as i64)
            )), &ZZ, &ZZbig
        ) as usize
    }

    pub fn ram_accesses(self, primes: &AllModuli) -> usize {
        let mut result = 0;
        
        for Fp2 in &primes.level_2_primes {
            let (ram, _, lvl1) = PolyEvaluationParams::on_level_2(self, *Fp2).reduce_ram_disk_lvl1(&primes.level_0_ram_primes, &primes.level_0_disk_primes, &primes.level_1_primes);
            result += ram.len();
            for Fp1 in lvl1 {
                result += PolyEvaluationParams::on_level_1(self, *Fp1).reduce_ram_disk(&primes.level_0_ram_primes, &primes.level_0_disk_primes).0.len();
            }
        }

        return result;
    }

    pub fn disk_accesses(self, primes: &AllModuli) -> usize {
        let mut result = 0;
        
        for Fp2 in &primes.level_2_primes {
            let (_, disk, lvl1) = PolyEvaluationParams::on_level_2(self, *Fp2).reduce_ram_disk_lvl1(&primes.level_0_ram_primes, &primes.level_0_disk_primes, &primes.level_1_primes);
            result += disk.len();
            for Fp1 in lvl1 {
                result += PolyEvaluationParams::on_level_1(self, *Fp1).reduce_ram_disk(&primes.level_0_ram_primes, &primes.level_0_disk_primes).1.len();
            }
        }

        return result;
    }
}

#[derive(Hash, PartialEq, Eq, Clone, Copy)]
pub struct PolyEvaluationParams {
    poly: PolyParams,
    characteristic: i64
}

const POLY_EVAL_BOUND_HEURISTIC_LOG2: f64 = 0.;

impl PolyEvaluationParams {

    pub fn new(poly: PolyParams, characteristic: i64) -> Self {
        Self { poly, characteristic }
    }

    pub fn on_level_1(poly: PolyParams, ring: Level1Modulus) -> Self {
        Self::new(poly, *ring.modulus() as i64)
    }

    pub fn on_level_2(poly: PolyParams, ring: Level2Modulus) -> Self {
        Self::new(poly, *ring.modulus() as i64)
    }

    pub fn poly_params(&self) -> PolyParams {
        self.poly
    }

    pub fn char(&self) -> i64 {
        self.characteristic
    }

    pub fn poly_eval_bound_log2(&self) -> f64 {
        self.poly_params().poly_eval_bound_log2(self.char())
    }

    pub fn reduce_shortest_lift<'a, I>(self, reduction_primes: I) -> impl 'a + Iterator<Item = Self>
        where I: 'a + Clone + Iterator<Item = i64>
    {
        // size bound for the result of the polynomial evaluation
        let bound = self.poly_eval_bound_log2();
        // `* 4 = * 2 * 2` because we need to distinguish +/- and we need a slack factor for `PrimeDecomposition::compose` to work
        let required_bound = bound + 2.;
        assert!(required_bound > 0.);

        assert!(reduction_primes.clone().map(|p| (p as f64).log2()).sum::<f64>() >= required_bound);

        let result = reduction_primes.chain([i64::MIN].into_iter()).scan(0., move |current, p| {
            if *current < required_bound {
                assert!(p != i64::MIN);
                *current += (p as f64).log2();
                Some(Self::new(self.poly_params(), p))
            } else {
                None
            }
        });
        return result;
    }

    pub fn reduce_ram_disk<'a>(&self, ram_primes: &'a [Level0ModulusRam], disk_primes: &'a [Level0ModulusDisk]) -> (&'a [Level0ModulusRam], &'a [Level0ModulusDisk]) {
        let prime_count = self.reduce_shortest_lift(ram_primes.iter().rev().map(|Fp| *Fp.modulus() as i64).chain(disk_primes.iter().rev().map(|Fp| *Fp.modulus() as i64))).count();
        let ram_prime_count = min(ram_primes.len(), prime_count);
        let disk_prime_count = prime_count - ram_prime_count;
        return (&ram_primes[(ram_primes.len() - ram_prime_count)..], &disk_primes[(disk_primes.len() - disk_prime_count)..]);
    }

    pub fn reduce_ram_disk_lvl1<'a>(&self, ram_primes: &'a [Level0ModulusRam], disk_primes: &'a [Level0ModulusDisk], lvl1_primes: &'a [Level1Modulus]) -> (&'a [Level0ModulusRam], &'a [Level0ModulusDisk], &'a [Level1Modulus]) {
        let prime_count = self.reduce_shortest_lift(ram_primes.iter().rev().map(|Fp| *Fp.modulus() as i64)
            .chain(disk_primes.iter().rev().map(|Fp| *Fp.modulus() as i64))
            .chain(lvl1_primes.iter().rev().map(|Fp| *Fp.modulus() as i64))).count();
        let ram_prime_count = min(ram_primes.len(), prime_count);
        let disk_prime_count = min(disk_primes.len(), prime_count - ram_prime_count);
        let lvl1_prime_count = prime_count - disk_prime_count - ram_prime_count;
        return (&ram_primes[(ram_primes.len() - ram_prime_count)..], &disk_primes[(disk_primes.len() - disk_prime_count)..], &lvl1_primes[(lvl1_primes.len() - lvl1_prime_count)..]);
    }
}

///
/// Counts the number of primes of the given list for whom we can store the associated database
/// in RAM without exceeding the RAM limit.
/// 
pub fn get_ram_prime_number<I>(primes: I, m: usize, ram_bytes: usize, ram_prime_size_bound: usize) -> usize
    where I: Iterator<Item = i64>
{
    let mut currently_used_ram = 0;
    let mut i = 0;
    for p in primes {
        let new_used_ram = currently_used_ram + estimate_used_ram(p, m);
        if new_used_ram <= ram_bytes && p <= ram_prime_size_bound as i64 {
            currently_used_ram = new_used_ram;
            i += 1;
        } else {
            return i;
        }
    }
    return i;
}

#[derive(Debug, Clone, Copy)]
pub struct PerformanceEnvironment {
    ram_read_iops: usize,
    disk_read_iops: usize
}

pub const STANDARD_PCIE_SSD_ENV: PerformanceEnvironment = PerformanceEnvironment { ram_read_iops: 40000000, disk_read_iops: 1000000 };

impl PerformanceEnvironment {

    pub fn estimate_running_time(self, params: DEPIRParams, primes: &AllModuli) -> f64 {
        return params.ram_accesses(primes) as f64 / self.ram_read_iops as f64 + params.disk_accesses(primes) as f64 /  self.disk_read_iops as f64;
    }
}

///
/// Parameters for executing tests - very small and insecure
/// 
#[allow(unused)]
pub const TEST_PARAMS: DEPIRParams = DEPIRParams {
    db_poly_params: PolyParams { variable_number: 3, poly_inf_norm: 32768, total_degree: 4 },
    he_params: HEParams { plaintext_modulus: 65537, log2_ring_degree: 8, ciphertext_len: 11, ciphertext_moduli_bitlength: 20, ciphertext_moduli_count: 5, reply_ciphertext_moduli_count: 3 },
    lvl1_primes_bound: 4,
    ram_bytes_bound: 256 << 20,
    ram_prime_bound: 256
};

///
/// Parameters for quick performance testing - small, but big enough to see effect of optimizations
/// 
#[allow(unused)]
pub const BENCH_PARAMS: DEPIRParams = DEPIRParams {
    db_poly_params: PolyParams { variable_number: 4, poly_inf_norm: 32768, total_degree: 18 },
    he_params: HEParams { plaintext_modulus: 65537, log2_ring_degree: 15, ciphertext_len: 19, ciphertext_moduli_bitlength: 20, ciphertext_moduli_count: 20, reply_ciphertext_moduli_count: 8 },
    lvl1_primes_bound: 32,
    ram_bytes_bound: 10 << 30,
    ram_prime_bound: 256
};

///
/// Parameters I adjust to test whatever I want at the moment
/// 
#[allow(unused)]
pub const EXPERIMENT_PARAMS: DEPIRParams = DEPIRParams {
    db_poly_params: PolyParams { variable_number: 4, poly_inf_norm: 32768, total_degree: 26 },
    he_params: HEParams { plaintext_modulus: 31, log2_ring_degree: 15, ciphertext_len: 27, ciphertext_moduli_bitlength: 20, ciphertext_moduli_count: 16, reply_ciphertext_moduli_count: 5 },
    lvl1_primes_bound: 0,
    ram_prime_bound: 10 << 30,
    ram_bytes_bound: 256
};

///
/// Large parameter set that just fits onto SSD
/// 
#[allow(unused)]
pub const SSD_PERF_PARAMS: DEPIRParams = DEPIRParams {
    db_poly_params: PolyParams { variable_number: 4, poly_inf_norm: 32768, total_degree: 30 },
    he_params: HEParams { plaintext_modulus: 65537, log2_ring_degree: 15, ciphertext_len: 31, ciphertext_moduli_bitlength: 20, ciphertext_moduli_count: 29, reply_ciphertext_moduli_count: 14 },
    lvl1_primes_bound: 30,
    ram_bytes_bound: 64 << 30,
    ram_prime_bound: 256
};

///
/// Crazy parameters that will never run
/// 
#[allow(unused)]
pub const EXPERIMENT_PERF_PARAMS: DEPIRParams = DEPIRParams {
    db_poly_params: PolyParams { variable_number: 6, poly_inf_norm: 32768, total_degree: 68 },
    he_params: HEParams { plaintext_modulus: 65537, log2_ring_degree: 16, ciphertext_len: 69, ciphertext_moduli_bitlength: 15, ciphertext_moduli_count: 60, reply_ciphertext_moduli_count: 28 },
    lvl1_primes_bound: usize::MAX,
    ram_bytes_bound: 256 << 30,
    ram_prime_bound: 256
};
