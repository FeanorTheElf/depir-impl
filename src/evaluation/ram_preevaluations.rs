use std::borrow::Borrow;
use std::fs::File;
use std::io::{BufReader, Read, BufWriter, Write};
use std::path::Path;

use feanor_math::homomorphism::*;
use feanor_math::vector::VectorView;
use feanor_math::vector::subvector::SubvectorFn;
use feanor_math::vector::vec_fn::VectorFn;
use feanor_math::{integer::*, vector::subvector::Subvector};
use feanor_math::primitive_int::StaticRing;
use feanor_math::{ring::*, assert_el_eq};
use feanor_math::rings::zn::ZnRingStore;
use feanor_math::rings::finite::*;

use crate::globals::{poly_hash, readwrite, PATH, CompressLevel0RamEl, Level0ModulusRam, timed, CompressLevel0DiskEl};
use crate::iters::{clone_array, multi_cartesian_product};
use crate::interpolate::evaluate_poly;

pub struct PrimeFieldRAMPreevaluations<'a, const m: usize> {
    data: Vec<CompressLevel0RamEl>,
    Fp: &'a Level0ModulusRam
}

pub fn estimate_used_ram(prime: i64, m: usize) -> usize {
    StaticRing::<i64>::RING.pow(prime, m) as usize * std::mem::size_of::<CompressLevel0RamEl>()
}

impl<'a, const m: usize> PrimeFieldRAMPreevaluations<'a, m> {

    pub fn create_new<I>(Fp: &'a Level0ModulusRam, d: usize, poly: &[El<I>], ZZ: I) -> Self
        where I: IntegerRingStore,
            I::Type: IntegerRing,
            <Level0ModulusRam as RingStore>::Type: CanHomFrom<I::Type>
    {
        let mut db = Vec::new();
        db.resize(StaticRing::<i64>::RING.pow(*Fp.modulus() as i64, m) as usize, 0);
        let hom = Fp.can_hom(&ZZ).unwrap();
        timed(&format!("PrimeDatabase::new for {}", Fp.modulus()), || {
            for point in multi_cartesian_product(Iterator::map(0..m, |_| Fp.elements()), clone_array::<_, m>) {
                db[Self::compute_index(Fp, point)] = Fp.get_ring().compress_el::<CompressLevel0RamEl>(evaluate_poly(poly.as_el_fn(&ZZ).map(|x| hom.map(x)), d, Subvector::new(point), Fp));
            }
            return PrimeFieldRAMPreevaluations {
                data: db,
                Fp: Fp
            }
        })
    }
    
    fn read(Fp: &'a Level0ModulusRam, _d: usize, mut reader: BufReader<File>) -> Self {
        let len = StaticRing::<i64>::RING.pow(*Fp.modulus() as i64, m) as usize;
        let mut db = Vec::with_capacity(len);
        timed(&format!("PrimeDatabase::read for {}", Fp.modulus()), || {
            let mut value = [0; std::mem::size_of::<CompressLevel0DiskEl>()];
            for _ in 0..len {
                reader.read_exact(&mut value).unwrap();
                db.push(Fp.get_ring().compress_el(Fp.get_ring().uncompress_el(readwrite::read_CompressLevel0El(&value))));
            }
            assert!(reader.bytes().next().is_none());
            return PrimeFieldRAMPreevaluations {
                data: db,
                Fp: Fp
            }
        })
    }

    fn write(&self, mut writer: BufWriter<File>) {
        let len = self.data.len();
        timed(&format!("PrimeDatabase::write for {}", self.Fp.modulus()), || {
            for i in 0..len {
                writer.write(readwrite::write_CompressLevel0El(self.Fp.get_ring().compress_el(self.Fp.get_ring().uncompress_el(self.data[i]))).borrow()).unwrap();
            }
        })
    }
    
    pub fn new<I>(Fp: &'a Level0ModulusRam, d: usize, poly: &[El<I>], ZZ: I) -> Self
        where I: IntegerRingStore,
            I::Type: IntegerRing,
            <Level0ModulusRam as RingStore>::Type: CanHomFrom<I::Type>
    {
        let filename = Self::filename(Fp, poly, &ZZ);
        let path = Path::new(filename.as_str());
        if path.exists() {
            let file = File::open(path).unwrap();
            let reader = BufReader::new(file);
            let result = Self::read(Fp, d, reader);
            assert_el_eq!(Fp, &evaluate_poly(SubvectorFn::new(poly.as_el_fn(&ZZ).map(|x| Fp.coerce(&ZZ, x))), d, Subvector::new([Fp.one(); m]), Fp), &result.lookup([Fp.one(); m]));
            result
        } else {
            let result = Self::create_new(Fp, d, poly, ZZ);
            let file = File::create(path).unwrap();
            let writer = BufWriter::new(file);
            result.write(writer);
            result
        }
    }

    fn filename<I>(Fp: &Level0ModulusRam, poly: &[El<I>], ZZ: I) -> String
        where I: IntegerRingStore,
            I::Type: IntegerRing
    {
        format!("{}poly_db_{}_{}", PATH, *Fp.modulus(), poly_hash(poly, ZZ))
    }
    
    fn compute_index(Fp: &Level0ModulusRam, x: [El<Level0ModulusRam>; m]) -> usize {
        let mut result = Fp.smallest_positive_lift(x[0]) as usize;
        for i in 1..m {
            result = result * *Fp.modulus() as usize + Fp.smallest_positive_lift(x[i]) as usize;
        }
        return result;
    }

    pub fn lookup(&self, point: [El<Level0ModulusRam>; m]) -> El<Level0ModulusRam> {
        self.lookup_index(self.get_index(point))
    }

    pub fn prefetch(&self, index: usize) {
        unsafe {
            std::intrinsics::prefetch_read_data(self.data.as_ptr().offset(index as isize), 0)
        }
    }

    pub fn lookup_index(&self, index: usize) -> El<Level0ModulusRam> {
        self.Fp.get_ring().uncompress_el(self.data[index])
    }

    pub fn get_index(&self, point: [El<Level0ModulusRam>; m]) -> usize {
        Self::compute_index(self.Fp, point)
    }
}

#[test]
fn test_compute_index() {
    let Fp = <Level0ModulusRam as RingStore>::Type::new(11);
    assert_eq!(1, PrimeFieldRAMPreevaluations::compute_index(&Fp, [0, 0, 0, 1]));
    assert_eq!(2, PrimeFieldRAMPreevaluations::compute_index(&Fp, [0, 0, 0, 2]));
    assert_eq!(3, PrimeFieldRAMPreevaluations::compute_index(&Fp, [0, 0, 0, 3]));
    assert_eq!(4, PrimeFieldRAMPreevaluations::compute_index(&Fp, [0, 0, 0, 4]));
    assert_eq!(15, PrimeFieldRAMPreevaluations::compute_index(&Fp, [0, 0, 1, 4]));
}