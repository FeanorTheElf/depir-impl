#![allow(non_camel_case_types)]

//
// Temporarily copied over from bfv-rust, until that becomes a
// proper library
// 

use feanor_math::{rings::zn::{ZnRing, ZnRingStore}, vector::*, ring::*};

pub mod lift;
pub mod bgv_rescale;

pub trait RNSOperation {

    type Ring: ZnRingStore<Type = Self::RingType>;
    type RingType: ?Sized + ZnRing;
    type InRings<'a>: 'a + VectorView<Self::Ring>
        where Self: 'a;
    type OutRings<'a>: 'a + VectorView<Self::Ring>
        where Self: 'a;
    type Result<'a>: 'a + VectorView<El<Self::Ring>>
        where Self: 'a;

    fn input_rings<'a>(&'a self) -> Self::InRings<'a>;

    fn output_rings<'a>(&'a self) -> Self::OutRings<'a>;

    fn apply<'a, V,>(&'a self, input: V) -> Self::Result<'a>
        where V: ExactSizeIterator<Item = El<Self::Ring>>;
}
