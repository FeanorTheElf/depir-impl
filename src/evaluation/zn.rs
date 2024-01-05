use std::{marker::PhantomData, ops::{Mul, Shr, Sub, Shl, Div, Add}, cmp::max};

use feanor_math::ring::*;
use feanor_math::pid::*;
use feanor_math::homomorphism::*;
use feanor_math::rings::zn;
use feanor_math::rings::finite::FiniteRing;
use feanor_math::divisibility::{DivisibilityRing, DivisibilityRingStore};
use feanor_math::rings::zn::*;
use feanor_math::integer::*;
use feanor_math::primitive_int::{StaticRing, StaticRingBase, PrimitiveInt};
use feanor_math::integer::{IntegerRing, IntegerRingStore};

pub trait PrimitiveUnsigned: Sized + Copy + Mul<Output = Self> + Add<Output = Self> + Ord + Shr<usize, Output = Self> + Shl<usize, Output = Self> + From<u8> + Sub<Output = Self> + Div<Output = Self> + std::fmt::Display + TryFrom<u128> + Into<u128> + TryInto<Self::Signed> {

    type Signed: PrimitiveInt;

    fn bits() -> usize;

    fn max_value() -> usize {
        (1 << Self::bits()) - 1
    }

    fn as_signed_ref<'a>(&'a self) -> &'a Self::Signed;
}

impl PrimitiveUnsigned for u8 {

    type Signed = i8;

    fn bits() -> usize { Self::BITS as usize }

    fn as_signed_ref<'a>(&'a self) -> &'a Self::Signed {
        assert!(<Self as TryInto<Self::Signed>>::try_into(*self).is_ok());
        unsafe { std::mem::transmute(self) }
    }
}

impl PrimitiveUnsigned for u16 {

    type Signed = i16;

    fn bits() -> usize { Self::BITS as usize }

    fn as_signed_ref<'a>(&'a self) -> &'a Self::Signed {
        assert!(<Self as TryInto<Self::Signed>>::try_into(*self).is_ok());
        unsafe { std::mem::transmute(self) }
    }
}

impl PrimitiveUnsigned for u32 {

    type Signed = i32;

    fn bits() -> usize { Self::BITS as usize }

    fn as_signed_ref<'a>(&'a self) -> &'a Self::Signed {
        assert!(<Self as TryInto<Self::Signed>>::try_into(*self).is_ok());
        unsafe { std::mem::transmute(self) }
    }
}

impl PrimitiveUnsigned for u64 {

    type Signed = i64;

    fn bits() -> usize { Self::BITS as usize }
    
    fn as_signed_ref<'a>(&'a self) -> &'a Self::Signed {
        assert!(<Self as TryInto<Self::Signed>>::try_into(*self).is_ok());
        unsafe { std::mem::transmute(self) }
    }
}

impl PrimitiveUnsigned for u128 {

    type Signed = i128;

    fn bits() -> usize { Self::BITS as usize }
    
    fn as_signed_ref<'a>(&'a self) -> &'a Self::Signed {
        assert!(<Self as TryInto<Self::Signed>>::try_into(*self).is_ok());
        unsafe { std::mem::transmute(self) }
    }
}

pub type Zn<UInt, UIntLarge, const MODULUS_BITS: usize> = RingValue<ZnBase<UInt, UIntLarge, MODULUS_BITS>>;

///
/// An implementation of `Z/nZ` using Barett-reductions. Arithmetic might even be slightly slower
/// than [`feanor_math::rings::zn::zn_42::Zn`], but the generic arguments here allow using the smallest
/// possible type in each place, thus reducing the memory footprint.
/// 
pub struct ZnBase<UInt, UIntLarge, const MODULUS_BITS: usize>
    where UInt: PrimitiveUnsigned + TryFrom<UIntLarge>, UIntLarge: PrimitiveUnsigned + From<UInt>
{
    prime_modulus: UInt,
    modulus_half: UInt,
    inv_modulus: UInt,
    uint_large: PhantomData<UIntLarge>
}

impl<UInt, UIntLarge, const MODULUS_BITS: usize> Copy for ZnBase<UInt, UIntLarge, MODULUS_BITS>
    where UInt: PrimitiveUnsigned + TryFrom<UIntLarge>, UIntLarge: PrimitiveUnsigned + From<UInt>
{}


impl<UInt, UIntLarge, const MODULUS_BITS: usize> Clone for ZnBase<UInt, UIntLarge, MODULUS_BITS>
    where UInt: PrimitiveUnsigned + TryFrom<UIntLarge>, UIntLarge: PrimitiveUnsigned + From<UInt>
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<UInt, UIntLarge, const MODULUS_BITS: usize> ZnBase<UInt, UIntLarge, MODULUS_BITS>
    where UInt: PrimitiveUnsigned + TryFrom<UIntLarge>, UIntLarge: PrimitiveUnsigned + From<UInt>
{
    pub fn new(prime_modulus: UInt) -> RingValue<Self> {
        Self::check_assumptions(prime_modulus);
        RingValue::from(ZnBase {
            prime_modulus: prime_modulus,
            modulus_half: prime_modulus >> 1,
            inv_modulus: UInt::try_from((UIntLarge::from(1) << (2 * MODULUS_BITS)) / UIntLarge::from(prime_modulus)).ok().unwrap(),
            uint_large: PhantomData
        })
    }

    fn check_assumptions(prime_modulus: UInt) {
        // so that we can perform add in UInt
        assert!(UInt::bits() >= MODULUS_BITS + 1);
        assert!(UIntLarge::bits() >= 3 * MODULUS_BITS);
        assert!(prime_modulus < (UInt::from(1) << MODULUS_BITS), "Prime modulus {} has more that {} bits", prime_modulus, MODULUS_BITS);
        assert!(prime_modulus > (UInt::from(1) << (max(2 * MODULUS_BITS, UInt::bits()) - UInt::bits())), "Prime modulus {} has less than {} bits", prime_modulus, max(2 * MODULUS_BITS, UInt::bits()) - UInt::bits());
    }

    ///
    /// Assumes that the input is smaller than `prime_modulus^2`.
    /// Then the output is smaller than `2 * prime_modulus` and congruent to input. 
    /// 
    fn bounded_reduce(&self, x: UIntLarge) -> UInt {
        debug_assert!(x < UIntLarge::from(self.prime_modulus) * UIntLarge::from(self.prime_modulus));
        let quotient = UInt::try_from((x * UIntLarge::from(self.inv_modulus)) >> (2 * MODULUS_BITS)).ok().unwrap();
        let result = UInt::try_from(x - UIntLarge::from(quotient) * UIntLarge::from(self.prime_modulus)).ok().unwrap();
        debug_assert!(result < UInt::from(2) * self.prime_modulus);
        return self.simple_reduce(result);
    }

    fn simple_reduce(&self, mut x: UInt) -> UInt {
        if x >= self.prime_modulus {
            x = x - self.prime_modulus;
        }
        debug_assert!(x < self.prime_modulus);
        return x;
    }

    pub fn compress_el<UIntShort: PrimitiveUnsigned>(&self, el: <Self as RingBase>::Element) -> UIntShort {
        assert!(UIntShort::max_value() >= StaticRing::<i64>::RING.coerce(&StaticRing::<UInt::Signed>::RING, *self.modulus()) as usize - 1);
        UIntShort::try_from(el.into()).ok().unwrap()
    }

    pub fn uncompress_el<UIntShort: PrimitiveUnsigned>(&self, el: UIntShort) -> <Self as RingBase>::Element {
        assert!(UIntShort::max_value() >= StaticRing::<i64>::RING.coerce(&StaticRing::<UInt::Signed>::RING, *self.modulus()) as usize - 1);
        UInt::try_from(el.into()).ok().unwrap()
    }
}

impl<UInt, UIntLarge, const MODULUS_BITS: usize> PartialEq for ZnBase<UInt, UIntLarge, MODULUS_BITS>
    where UInt: PrimitiveUnsigned + TryFrom<UIntLarge>, UIntLarge: PrimitiveUnsigned + From<UInt>
{
    fn eq(&self, other: &Self) -> bool {
        self.prime_modulus == other.prime_modulus
    }
}

impl<UInt, UIntLarge, const MODULUS_BITS: usize> RingBase for ZnBase<UInt, UIntLarge, MODULUS_BITS>
    where UInt: PrimitiveUnsigned + TryFrom<UIntLarge>, UIntLarge: PrimitiveUnsigned + From<UInt>
{
    type Element = UInt;

    fn clone_el(&self, val: &Self::Element) -> Self::Element { *val }

    fn add_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        debug_assert!(*lhs < self.prime_modulus);
        debug_assert!(rhs < self.prime_modulus);
        *lhs = self.simple_reduce(*lhs + rhs);
        debug_assert!(*lhs < self.prime_modulus);
    }
    
    fn negate_inplace(&self, lhs: &mut Self::Element) {
        debug_assert!(*lhs < self.prime_modulus);
        *lhs = self.simple_reduce(self.prime_modulus - *lhs);
        debug_assert!(*lhs < self.prime_modulus);
    }

    fn mul_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        debug_assert!(*lhs < self.prime_modulus);
        debug_assert!(rhs < self.prime_modulus);
        *lhs = self.bounded_reduce(UIntLarge::from(*lhs) * UIntLarge::from(rhs));
        debug_assert!(*lhs < self.prime_modulus);
    }

    fn zero(&self) -> Self::Element { self.from_int(0) }

    fn one(&self) -> Self::Element { self.from_int(1) }

    fn neg_one(&self) -> Self::Element { self.from_int(-1) }

    fn from_int(&self, value: i32) -> Self::Element {
        RingRef::new(self).coerce(&StaticRing::<i32>::RING, value)
    }

    fn eq_el(&self, lhs: &Self::Element, rhs: &Self::Element) -> bool {
        *lhs == *rhs
    }

    fn is_commutative(&self) -> bool { true }
    fn is_noetherian(&self) -> bool { true }

    fn dbg<'a>(&self, value: &Self::Element, out: &mut std::fmt::Formatter<'a>) -> std::fmt::Result {
        write!(out, "{}", value)
    }
}

impl<UInt1, UIntLarge1, UInt2, UIntLarge2, const MODULUS_BITS1: usize, const MODULUS_BITS2: usize> CanHomFrom<ZnBase<UInt1, UIntLarge1, MODULUS_BITS1>> for ZnBase<UInt2, UIntLarge2, MODULUS_BITS2>
    where UInt1: PrimitiveUnsigned + TryFrom<UIntLarge1>, UIntLarge1: PrimitiveUnsigned + From<UInt1>,
        UInt2: PrimitiveUnsigned + TryFrom<UIntLarge2>, UIntLarge2: PrimitiveUnsigned + From<UInt2>,
        UInt2: From<UInt1>
{
    type Homomorphism = ();

    fn has_canonical_hom(&self, from: &ZnBase<UInt1, UIntLarge1, MODULUS_BITS1>) -> Option<Self::Homomorphism> {
        if self.prime_modulus == UInt2::from(from.prime_modulus) {
            Some(())
        } else {
            None
        }
    }

    fn map_in(&self, _from: &ZnBase<UInt1, UIntLarge1, MODULUS_BITS1>, el: <ZnBase<UInt1, UIntLarge1, MODULUS_BITS1> as RingBase>::Element, _hom: &Self::Homomorphism) -> Self::Element {
        UInt2::from(el)
    }
}

impl<UInt1, UIntLarge1, UInt2, UIntLarge2, const MODULUS_BITS1: usize, const MODULUS_BITS2: usize> CanonicalIso<ZnBase<UInt1, UIntLarge1, MODULUS_BITS1>> for ZnBase<UInt2, UIntLarge2, MODULUS_BITS2>
    where UInt1: PrimitiveUnsigned + TryFrom<UIntLarge1>, UIntLarge1: PrimitiveUnsigned + From<UInt1>,
        UInt2: PrimitiveUnsigned + TryFrom<UIntLarge2>, UIntLarge2: PrimitiveUnsigned + From<UInt2>,
        UInt2: From<UInt1>,
        UInt1: From<UInt2>
{
    type Isomorphism = ();

    fn has_canonical_iso(&self, from: &ZnBase<UInt1, UIntLarge1, MODULUS_BITS1>) -> Option<Self::Homomorphism> {
        if self.prime_modulus == UInt2::from(from.prime_modulus) {
            Some(())
        } else {
            None
        }
    }

    fn map_out(&self, _from: &ZnBase<UInt1, UIntLarge1, MODULUS_BITS1>, el: Self::Element, _iso: &Self::Isomorphism) -> <ZnBase<UInt1, UIntLarge1, MODULUS_BITS1> as RingBase>::Element {
        UInt1::from(el)
    }
}

impl<UInt, UIntLarge, const MODULUS_BITS: usize> CanHomFrom<zn_42::ZnBase> for ZnBase<UInt, UIntLarge, MODULUS_BITS>
    where UInt: PrimitiveUnsigned + TryFrom<UIntLarge>, UIntLarge: PrimitiveUnsigned + From<UInt>
{
    type Homomorphism = ();

    fn has_canonical_hom(&self, from: &zn_42::ZnBase) -> Option<Self::Homomorphism> {
        if self.prime_modulus == UInt::try_from(*from.modulus() as u128).ok()? {
            Some(())
        } else {
            None
        }
    }

    fn map_in(&self, from: &zn_42::ZnBase, el: <zn_42::ZnBase as RingBase>::Element, _hom: &Self::Homomorphism) -> Self::Element {
        UInt::try_from(from.smallest_positive_lift(el) as u128).ok().unwrap()
    }
} 

impl<UInt, UIntLarge, const MODULUS_BITS: usize> CanonicalIso<zn_42::ZnBase> for ZnBase<UInt, UIntLarge, MODULUS_BITS>
    where UInt: PrimitiveUnsigned + TryFrom<UIntLarge>, UIntLarge: PrimitiveUnsigned + From<UInt>
{
    type Isomorphism = <zn_42::ZnBase as CanHomFrom<StaticRingBase<i128>>>::Homomorphism;

    fn has_canonical_iso(&self, from: &zn_42::ZnBase) -> Option<Self::Isomorphism> {
        if self.prime_modulus == UInt::try_from(*from.modulus() as u128).ok()? {
            from.has_canonical_hom(StaticRing::<i128>::RING.get_ring())
        } else {
            None
        }
    }

    fn map_out(&self, from: &zn_42::ZnBase, el: Self::Element, hom: &Self::Isomorphism) -> <zn_42::ZnBase as RingBase>::Element {
        from.map_in(StaticRing::<i128>::RING.get_ring(), UInt::into(el) as i128, hom)
    }
}

pub trait MapFromIntegerRing: IntegerRing + CanonicalIso<StaticRingBase<i128>> {}

pub struct IntegerToZnHom<I: IntegerRing + CanonicalIso<StaticRingBase<i128>>> {
    highbit_bound: usize,
    highbit_mod: usize,
    int_ring: PhantomData<I>,
    hom: <I as CanHomFrom<StaticRingBase<i128>>>::Homomorphism, 
    iso: <I as CanonicalIso<StaticRingBase<i128>>>::Isomorphism
}

impl<I, UInt, UIntLarge, const MODULUS_BITS: usize> CanHomFrom<I> for ZnBase<UInt, UIntLarge, MODULUS_BITS>
    where UInt: PrimitiveUnsigned + TryFrom<UIntLarge>, UIntLarge: PrimitiveUnsigned + From<UInt>,
        I: MapFromIntegerRing
{
    type Homomorphism = IntegerToZnHom<I>;

    fn has_canonical_hom(&self, from: &I) -> Option<Self::Homomorphism> {
        Some(IntegerToZnHom {
            highbit_bound: StaticRing::<i128>::RING.abs_highest_set_bit(&(UInt::into(self.prime_modulus) as i128 * UInt::into(self.prime_modulus) as i128)).unwrap(),
            highbit_mod: StaticRing::<i128>::RING.abs_highest_set_bit(&(UInt::into(self.prime_modulus) as i128)).unwrap(),
            int_ring: PhantomData,
            hom: from.has_canonical_hom(StaticRing::<i128>::RING.get_ring())?, 
            iso: from.has_canonical_iso(StaticRing::<i128>::RING.get_ring())?
        })
    }

    fn map_in(&self, from: &I, el: <I as RingBase>::Element, hom: &Self::Homomorphism) -> Self::Element {
        // similar approach to `feanor_math::rings::zn::zn_42`
        let (neg, n) = if from.is_neg(&el) {
            (true, from.negate(el))
        } else {
            (false, el)
        };
        let as_u128 = |x: <I as RingBase>::Element| from.map_out(StaticRing::<i128>::RING.get_ring(), x, &hom.iso) as u128;
        let as_ElI = |x: UInt| from.map_in(StaticRing::<i128>::RING.get_ring(), UInt::into(x) as i128, &hom.hom);
        let highbit_el = from.abs_highest_set_bit(&n).unwrap_or(0);
        let reduced = if highbit_el < hom.highbit_mod {
            UInt::try_from(as_u128(n)).ok().unwrap()
        } else if highbit_el < hom.highbit_bound {
            self.bounded_reduce(UIntLarge::try_from(as_u128(n)).ok().unwrap())
        } else {
            UInt::try_from(as_u128(from.euclidean_rem(n, &as_ElI(self.prime_modulus)))).ok().unwrap()
        };
        debug_assert!(reduced < self.prime_modulus);
        if neg {
            self.negate(reduced)
        } else {
            reduced
        }
    }
} 

impl<UInt, UIntLarge, const MODULUS_BITS: usize> DivisibilityRing for ZnBase<UInt, UIntLarge, MODULUS_BITS>
    where UInt: PrimitiveUnsigned + TryFrom<UIntLarge>, UIntLarge: PrimitiveUnsigned + From<UInt>
{
    fn checked_left_div(&self, lhs: &Self::Element, rhs: &Self::Element) -> Option<Self::Element> {
        let ring = zn_42::Zn::new(UInt::into(self.prime_modulus) as u64);
        let self_ref =RingRef::new(self);
        let iso = self_ref.can_iso(&ring).unwrap();
        Some(RingRef::new(self).coerce(&ring, ring.checked_div(&iso.map(*lhs), &iso.map(*rhs))?))
    }
}

pub struct ZnBaseElementsIter<'a, UInt, UIntLarge, const MODULUS_BITS: usize> 
    where UInt: PrimitiveUnsigned + TryFrom<UIntLarge>, UIntLarge: PrimitiveUnsigned + From<UInt>
{
    ring: &'a ZnBase<UInt, UIntLarge, MODULUS_BITS>,
    current: UInt
}

impl<'a, UInt, UIntLarge, const MODULUS_BITS: usize> Copy for ZnBaseElementsIter<'a, UInt, UIntLarge, MODULUS_BITS>
    where UInt: PrimitiveUnsigned + TryFrom<UIntLarge>, UIntLarge: PrimitiveUnsigned + From<UInt>
{}

impl<'a, UInt, UIntLarge, const MODULUS_BITS: usize> Clone for ZnBaseElementsIter<'a, UInt, UIntLarge, MODULUS_BITS>
    where UInt: PrimitiveUnsigned + TryFrom<UIntLarge>, UIntLarge: PrimitiveUnsigned + From<UInt>
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, UInt, UIntLarge, const MODULUS_BITS: usize> Iterator for ZnBaseElementsIter<'a, UInt, UIntLarge, MODULUS_BITS>
    where UInt: PrimitiveUnsigned + TryFrom<UIntLarge>, UIntLarge: PrimitiveUnsigned + From<UInt>
{
    type Item = UInt;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current < self.ring.prime_modulus {
            let result = self.current;
            self.current = self.current + UInt::from(1);
            return Some(result);
        } else {
            return None;
        }
    }
}

impl<UInt, UIntLarge, const MODULUS_BITS: usize> FiniteRing for ZnBase<UInt, UIntLarge, MODULUS_BITS>
    where UInt: PrimitiveUnsigned + TryFrom<UIntLarge>, UIntLarge: PrimitiveUnsigned + From<UInt>
{
    type ElementsIter<'a> = ZnBaseElementsIter<'a, UInt, UIntLarge, MODULUS_BITS>
        where Self: 'a;

    fn elements<'a>(&'a self) -> Self::ElementsIter<'a> {
        ZnBaseElementsIter {
            ring: self,
            current: UInt::from(0)
        }
    }

    fn size<I: IntegerRingStore>(&self, ZZ: &I) -> El<I>
        where I::Type: IntegerRing
    {
        int_cast(*self.modulus(), ZZ, self.integer_ring())
    }

    fn random_element<G: FnMut() -> u64>(&self, rng: G) -> Self::Element {
        zn::generic_impls::random_element(self, rng)
    }
}

impl<UInt, UIntLarge, const MODULUS_BITS: usize> PrincipalIdealRing for ZnBase<UInt, UIntLarge, MODULUS_BITS>
    where UInt: PrimitiveUnsigned + TryFrom<UIntLarge>, UIntLarge: PrimitiveUnsigned + From<UInt>
{
    fn ideal_gen(&self, lhs: &Self::Element, rhs: &Self::Element) -> (Self::Element, Self::Element, Self::Element) {
        let (s, t, d) = self.integer_ring().ideal_gen(&self.smallest_lift(*lhs), &self.smallest_lift(*rhs));
        let modulo = RingRef::new(self).into_can_hom(self.integer_ring()).ok().unwrap();
        return (modulo.map(s), modulo.map(t), modulo.map(d));
    }
}

impl<UInt, UIntLarge, const MODULUS_BITS: usize> ZnRing for ZnBase<UInt, UIntLarge, MODULUS_BITS>
    where UInt: PrimitiveUnsigned + TryFrom<UIntLarge>, UIntLarge: PrimitiveUnsigned + From<UInt>
{
    type IntegerRingBase = StaticRingBase<UInt::Signed>;
    type Integers = StaticRing<UInt::Signed>;

    fn integer_ring(&self) -> &Self::Integers {
        &StaticRing::<UInt::Signed>::RING
    }

    fn modulus(&self) -> &El<Self::Integers> {
        (&self.prime_modulus).as_signed_ref()
    }

    fn smallest_positive_lift(&self, el: Self::Element) -> El<Self::Integers> {
        UInt::try_into(el).ok().unwrap()
    }

    fn smallest_lift(&self, el: Self::Element) -> El<Self::Integers> {
        if el <= self.modulus_half {
            UInt::try_into(el).ok().unwrap()
        } else {
            let mut result = UInt::try_into(el).ok().unwrap();
            result -= *self.modulus();
            return result;
        }
    }
}

impl MapFromIntegerRing for BigIntRingBase {}
impl<T: PrimitiveInt> MapFromIntegerRing for StaticRingBase<T> {}

///
/// For moduli of at most 8 bit, i.e. in `[2, 255]`.
/// 
pub type Zp8bit = Zn<u16, u32, 8>;

///
/// For moduli of at most 8 bit, i.e. in `[17, 1023]`.
/// 
pub type Zp10bit = Zn<u16, u32, 10>;

///
/// For moduli of at most 16 bit, i.e. in `[2, 65535]`
/// 
pub type Zp16bit = Zn<u32, u64, 16>;

///
/// For moduli between 8 and 20 bit, i.e. in `[257, 1048575]`
/// 
pub type Zp20bit = Zn<u32, u64, 20>;

///
/// For moduli between 20 and 42 bit, i.e. in `[1048577, 4398046511103]`
/// 
pub type Zp42bit = Zn<u64, u128, 42>;

#[cfg(test)]
use feanor_math::assert_el_eq;

#[test]
fn test_types_valid() {
    <Zp8bit as RingStore>::Type::new(2);
    <Zp8bit as RingStore>::Type::new(255);
    <Zp10bit as RingStore>::Type::new(17);
    <Zp10bit as RingStore>::Type::new(1023);
    <Zp16bit as RingStore>::Type::new(2);
    <Zp16bit as RingStore>::Type::new(65535);
    <Zp20bit as RingStore>::Type::new(257);
    <Zp20bit as RingStore>::Type::new(1048575);
    <Zp42bit as RingStore>::Type::new(1048577);
    <Zp42bit as RingStore>::Type::new(4398046511103);
}

#[test]
fn test_ring_axioms() {
    feanor_math::ring::generic_tests::test_ring_axioms(<Zp8bit as RingStore>::Type::new(2), [0, 1].into_iter());
    feanor_math::ring::generic_tests::test_ring_axioms(<Zp8bit as RingStore>::Type::new(255), [0, 1, 2, 4, 5, 7].into_iter());
    feanor_math::ring::generic_tests::test_ring_axioms(<Zp16bit as RingStore>::Type::new(2), [0, 1].into_iter());
    feanor_math::ring::generic_tests::test_ring_axioms(<Zp16bit as RingStore>::Type::new(65535), [0, 1, 2, 4, 5, 7].into_iter());
    feanor_math::ring::generic_tests::test_ring_axioms(<Zp20bit as RingStore>::Type::new(257), [0, 1, 2, 4, 5, 7].into_iter());
    feanor_math::ring::generic_tests::test_ring_axioms(<Zp20bit as RingStore>::Type::new(1048575), [0, 1, 2, 4, 5, 7].into_iter());
    feanor_math::ring::generic_tests::test_ring_axioms(<Zp42bit as RingStore>::Type::new(1048577), [0, 1, 2, 4, 5, 7].into_iter());
    feanor_math::ring::generic_tests::test_ring_axioms(<Zp42bit as RingStore>::Type::new(4398046511103), [0, 1, 2, 4, 5, 7].into_iter());
}

#[test]
fn test_zn_map_in_large_int() {
    let R = <Zp8bit as RingStore>::Type::new(2);
    zn::generic_tests::test_map_in_large_int(R);

    let ZZbig = BigIntRing::RING;
    let R = <Zp8bit as RingStore>::Type::new(2);
    assert_el_eq!(&R, &R.int_hom().map(1), &R.coerce(&ZZbig, ZZbig.sub(ZZbig.power_of_two(84), ZZbig.one())));
}

#[test]
fn test_zn_map_in_small_int() {
    let R = <Zp8bit as RingStore>::Type::new(2);
    assert_el_eq!(&R, &R.int_hom().map(1), &R.coerce(&StaticRing::<i8>::RING, 1));
}

#[test]
fn test_zn_ring_axioms() {
    zn::generic_tests::test_zn_axioms(<Zp8bit as RingStore>::Type::new(2));
    zn::generic_tests::test_zn_axioms(<Zp8bit as RingStore>::Type::new(255));
    zn::generic_tests::test_zn_axioms(<Zp16bit as RingStore>::Type::new(2));
    zn::generic_tests::test_zn_axioms(<Zp16bit as RingStore>::Type::new(257));
    zn::generic_tests::test_zn_axioms(<Zp20bit as RingStore>::Type::new(257));
    // skip larger rings for performance reasons
}