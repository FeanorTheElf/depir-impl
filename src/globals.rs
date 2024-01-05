use std::cmp::min;
use std::hash::Hasher;
use std::time::Instant;

use base64::Engine;
use feanor_math::integer::{IntegerRingStore, IntegerRing};
use feanor_math::integer::*;
pub use feanor_math::primitive_int::*;
pub use feanor_math::ring::*;
pub use feanor_math::rings::zn::zn_42;

use crate::evaluation::zn;

pub const ZZ: StaticRing<Int> = StaticRing::<Int>::RING;
pub const ZZbig: BigIntRing = BigIntRing::RING;

pub type ZZType = StaticRing<Int>;
pub type Int = i64;
pub type Zn = zn_42::Zn;

pub type Level0ModulusRam = zn::Zp16bit;
pub type CompressLevel0RamEl = u8;
pub type Level0ModulusDisk = zn::Zp16bit;
pub type CompressLevel0DiskEl = u16;

pub type Level1Modulus = zn::Zp16bit;
pub type Level2Modulus = zn::Zp42bit;

pub type TopLevelModulus = Level2Modulus;

///
/// The interpolated polynomial and the evaluation datastructure are
/// stored / read from here.
/// 
pub const PATH: &'static str = "E:\\";

pub fn dummy() -> El<Zn> {
    Zn::new(2).zero()
}

pub fn binomial(n: usize, mut k: usize) -> usize {
    if k > n {
        0
    } else {
        k = min(k, n - k);
        ((n - k + 1)..=n).product::<usize>() / (1..=k).product::<usize>()
    }
}

pub fn timed<T, F: FnOnce() -> T>(name: &str, f: F) -> T {
    let start = Instant::now();
    let result = f();
    let duration = Instant::now() - start;
    if duration.as_millis() > 500 {
        println!("{} done in {} ms", name, duration.as_millis());
    }
    return result;
}

pub fn poly_hash<I: IntegerRingStore>(poly: &[El<I>], ring: I) -> String
    where I::Type: IntegerRing
{
    
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    for c in poly {
        ring.hash(c, &mut hasher);
    }
    let hash = hasher.finish();
    let hash_bytes = [(hash & 0xFF) as u8, ((hash >> 8) & 0xFF) as u8, ((hash >> 16) & 0xFF) as u8, (hash >> 24) as u8];
    return base64::engine::general_purpose::STANDARD_NO_PAD.encode(&hash_bytes);
}

pub mod readwrite {
    use std::borrow::Borrow;

    use super::{Int, CompressLevel0DiskEl};

    pub fn write_CompressLevel0El(x: CompressLevel0DiskEl) -> impl Borrow<[u8]> {
        [(x & 0xFF) as u8, (x >> 8) as u8]
    }
    
    pub fn read_CompressLevel0El(x: &[u8]) -> CompressLevel0DiskEl {
        assert_eq!(x.len(), std::mem::size_of::<CompressLevel0DiskEl>());
        assert_eq!(2, std::mem::size_of::<CompressLevel0DiskEl>());
        ((x[1] as u16) << 8) | (x[0] as u16)
    }
    
    pub fn write_Int(x: Int) -> impl Borrow<[u8]> {
        x.to_le_bytes()
    }
    
    pub fn read_Int(x: &[u8]) -> Int {
        assert_eq!(x.len(), std::mem::size_of::<Int>());
        assert_eq!(8, std::mem::size_of::<Int>());
        Int::from_le_bytes([x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]])
    }
}
