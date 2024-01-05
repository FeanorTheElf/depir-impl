use std::path::Path;

use feanor_math::integer::*;
use feanor_math::homomorphism::CanHomFrom;
use feanor_math::primitive_int::StaticRing;
use feanor_math::ring::*;
use feanor_math::rings::zn::ZnRingStore;

use crate::globals::{poly_hash, PATH, Level0ModulusDisk, CompressLevel0DiskEl};

pub struct PrimeFieldDiskPreevaluations<'a, const m: usize> {
    Fp: &'a Level0ModulusDisk,
    filename: String
}

pub fn estimate_used_disk_space(prime: i64, m: usize) -> usize {
    StaticRing::<i64>::RING.pow(prime, m) as usize * std::mem::size_of::<CompressLevel0DiskEl>()
}

impl<'a, const m: usize> PrimeFieldDiskPreevaluations<'a, m> {
    
    pub fn new<I>(Fp: &'a Level0ModulusDisk, _d: usize, poly: &[El<I>], ZZ: I) -> Self
        where I: IntegerRingStore,
            I::Type: IntegerRing,
            El<I>: Clone,
            <Level0ModulusDisk as RingStore>::Type: CanHomFrom<I::Type>
    {
        let filename = Self::filename(Fp, poly, &ZZ);
        assert!(Path::new(filename.as_str()).exists());
        Self { Fp, filename }
    }

    fn filename<I>(Fp: &Level0ModulusDisk, poly: &[El<I>], ZZ: I) -> String
        where I: IntegerRingStore,
            I::Type: IntegerRing
    {
        format!("{}poly_db_{}_{}", PATH, *Fp.modulus(), poly_hash(poly, ZZ))
    }
    
    fn compute_index(Fp: &Level0ModulusDisk, x: [El<Level0ModulusDisk>; m]) -> usize {
        let mut result = Fp.smallest_positive_lift(x[0]) as usize;
        for i in 1..m {
            result = result * *Fp.modulus() as usize + Fp.smallest_positive_lift(x[i]) as usize;
        }
        return result;
    }

    pub fn get_filename(&self) -> &str {
        &self.filename
    }

    pub fn lookup_index(&self, point: [El<Level0ModulusDisk>; m]) -> u64 {
        Self::compute_index(self.Fp, point) as u64
    }

    pub fn lookup_callback<'b, F>(&'b self, mut f: F) -> impl 'b + FnMut(CompressLevel0DiskEl, usize)
        where F: FnMut(El<Level0ModulusDisk>, usize) + 'b
    {
        move |compressed_el, index| f(self.Fp.get_ring().uncompress_el(compressed_el), index)
    }
}

#[test]
fn test_compute_index() {
    let Fp = <Level0ModulusDisk as RingStore>::Type::new(11);
    assert_eq!(1, PrimeFieldDiskPreevaluations::compute_index(&Fp, [0, 0, 0, 1]));
    assert_eq!(2, PrimeFieldDiskPreevaluations::compute_index(&Fp, [0, 0, 0, 2]));
    assert_eq!(3, PrimeFieldDiskPreevaluations::compute_index(&Fp, [0, 0, 0, 3]));
    assert_eq!(4, PrimeFieldDiskPreevaluations::compute_index(&Fp, [0, 0, 0, 4]));
    assert_eq!(15, PrimeFieldDiskPreevaluations::compute_index(&Fp, [0, 0, 1, 4]));
}