use feanor_math::rings::poly::{dense_poly::DensePolyRing, PolyRingStore, PolyRing};
use feanor_math::rings::zn::{ZnRingStore, ZnRing};
use feanor_math::rings::float_complex::{Complex64El, Complex64};
use feanor_math::algorithms;
use feanor_math::algorithms::fft::*;
use feanor_math::mempool::*;
use feanor_math::vector::*;
use feanor_math::homomorphism::*;
use feanor_math::integer::*;
use feanor_math::default_memory_provider;

use crate::globals::*;

type MemoryProviderZn = DefaultMemoryProvider;
type MemoryProviderComplex = DefaultMemoryProvider;

///
/// The ring `(Z/nZ)[X]/(X^N + 1)` for `N` power of two, using complex-valued
/// FFTs for arithmetic.
/// 
pub struct Pow2CyclotomicRingBase {
    base: Zn,
    fft_table: cooley_tuckey::FFTTableCooleyTuckey<RingValue<Complex64>>,
    twiddles: Vec<Complex64El>,
    inv_twiddles: Vec<Complex64El>
}

pub type Pow2CyclotomicRing = RingValue<Pow2CyclotomicRingBase>;

impl Pow2CyclotomicRingBase {

    pub fn new(base: Zn, log2_N: usize) -> RingValue<Self> {
        let CC = Complex64::RING;
        let fft_table = cooley_tuckey::FFTTableCooleyTuckey::for_complex(CC, log2_N);
        let twiddles = (0..(1 << log2_N)).map(|i| CC.root_of_unity(i, 2 << log2_N)).collect();
        let inv_twiddles = (0..(1 << log2_N)).map(|i| CC.root_of_unity((2 << log2_N) - i, 2 << log2_N)).collect();
        return RingValue::from(Self { base, fft_table, twiddles, inv_twiddles });
    }

    pub fn gen(&self) -> <Self as RingBase>::Element {
        self.memory_provider_zn().get_new_init(self.N(), |i| if i == 1 { self.base.one() } else { self.base.zero() })
    }

    pub fn modulus(&self) -> &i64 {
        self.base.modulus()
    }

    pub fn poly_repr<P>(&self, poly_ring: P, el: &<Self as RingBase>::Element) -> El<P>
        where P: PolyRingStore,
            P::Type: PolyRing,
            <P::Type as RingExtension>::BaseRing: RingStore<Type = zn_42::ZnBase>
    {
        // so that we can use elements without modification
        assert!(self.base.get_ring() == poly_ring.base_ring().get_ring());
        poly_ring.from_terms(
            (0..self.N())
                .filter(|i| !self.base.is_zero(&el[*i]))
                .map(|i| (el[i], i))
        )
    }

    pub fn smallest_lift<P>(&self, poly_ring: P, el: &<Self as RingBase>::Element) -> El<P>
        where P: PolyRingStore,
            P::Type: PolyRing,
            <P::Type as RingExtension>::BaseRing: IntegerRingStore,
            <<P::Type as RingExtension>::BaseRing as RingStore>::Type: IntegerRing + CanHomFrom<<<Zn as RingStore>::Type as ZnRing>::IntegerRingBase>
    {
        timed("ComplexPow2CyclotomicBase::smallest_lift", || {
            let hom = poly_ring.base_ring().can_hom(self.base_ring().integer_ring()).unwrap();
            poly_ring.from_terms(
                (0..self.N())
                    .filter(|i| !self.base.is_zero(&el[*i]))
                    .map(|i| (hom.map(self.base.smallest_lift(el[i])), i))
            )
        })
    }

    fn memory_provider_zn(&self) -> MemoryProviderZn {
        default_memory_provider!()
    }

    fn memory_provider_complex(&self) -> MemoryProviderComplex {
        default_memory_provider!()
    }

    fn N(&self) -> usize {
        self.fft_table.len()
    }
    
    fn fft_of(&self, el: &<MemoryProviderZn as MemoryProvider<El<Zn>>>::Object) -> <MemoryProviderComplex as MemoryProvider<Complex64El>>::Object {
        let CC = Complex64::RING;
        let mut result = self.memory_provider_complex().get_new_init(
            self.N(),
            |i| CC.int_hom().map(self.base.smallest_lift(el[i]) as i32)
        );
        for i in 0..self.N() {
            CC.mul_assign_ref(result.at_mut(i), self.twiddles.at(i));
        }
        self.fft_table.unordered_fft(&mut result[..], &self.memory_provider_complex(), &CC.identity());
        return result;
    }

    fn fft_back(&self, mut el: <MemoryProviderComplex as MemoryProvider<Complex64El>>::Object, out: &mut <MemoryProviderZn as MemoryProvider<El<Zn>>>::Object) {
        let CC = Complex64::RING;
        self.fft_table.unordered_inv_fft(&mut el[..], &self.memory_provider_complex(), &CC.identity());
        for i in 0..self.N() {
            CC.mul_assign_ref(el.at_mut(i), self.inv_twiddles.at(i));
        }
        for i in 0..self.N() {
            let (re, im) = CC.closest_gaussian_int(el[i]);
            debug_assert_eq!(im, 0);
            out[i] = self.base.int_hom().map(re as i32);
        }
    }

    fn mul_assign_fft(&self, lhs: &mut <MemoryProviderZn as MemoryProvider<El<Zn>>>::Object, rhs_fft: &<MemoryProviderComplex as MemoryProvider<Complex64El>>::Object) {
        let mut lhs_fft = self.fft_of(lhs);
        for i in 0..self.N() {
            Complex64::RING.mul_assign(&mut lhs_fft[i], rhs_fft[i]);
        }
        self.fft_back(lhs_fft, lhs);
    }
}

impl PartialEq for Pow2CyclotomicRingBase {

    fn eq(&self, other: &Self) -> bool {
        self.base.get_ring() == other.base.get_ring() && self.N() == other.N()
    }
}

impl RingBase for Pow2CyclotomicRingBase {
    
    type Element = <MemoryProviderZn as MemoryProvider<El<Zn>>>::Object;

    fn clone_el(&self, val: &Self::Element) -> Self::Element {
        self.memory_provider_zn().get_new_init(val.len(), |i| self.base.clone_el(&val[i]))
    }

    fn add_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        assert_eq!(self.N(), lhs.len());
        assert_eq!(self.N(), rhs.len());
        for i in 0..self.N() {
            self.base.add_assign_ref(&mut lhs[i], &rhs[i]);
        }
    }

    fn add_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        self.add_assign_ref(lhs, &rhs);
    }

    fn sub_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        self.negate_inplace(lhs);
        self.add_assign_ref(lhs, rhs);
        self.negate_inplace(lhs);
    }

    fn negate_inplace(&self, lhs: &mut Self::Element) {
        assert_eq!(self.N(), lhs.len());
        for i in 0..self.N() {
            self.base.negate_inplace(&mut lhs[i]);
        }
    }

    fn mul_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        self.mul_assign_ref(lhs, &rhs);
    }

    fn mul_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        assert_eq!(self.N(), lhs.len());
        assert_eq!(self.N(), rhs.len());
        self.mul_assign_fft(lhs, &self.fft_of(rhs));
    }
    
    fn from_int(&self, value: i32) -> Self::Element {
        self.from(self.base.int_hom().map(value))
    }

    fn eq_el(&self, lhs: &Self::Element, rhs: &Self::Element) -> bool {
        assert_eq!(self.N(), lhs.len());
        assert_eq!(self.N(), rhs.len());
        (0..self.N()).all(|i| self.base.eq_el(&lhs[i], &rhs[i]))
    }
    
    fn is_commutative(&self) -> bool { true }
    fn is_noetherian(&self) -> bool { true }

    fn dbg<'a>(&self, value: &Self::Element, out: &mut std::fmt::Formatter<'a>) -> std::fmt::Result {
        let poly_ring = DensePolyRing::new(&self.base, "ζ");
        poly_ring.get_ring().dbg(&self.poly_repr(&poly_ring, value), out)
    }

    fn square(&self, value: &mut Self::Element) {
        assert_eq!(self.N(), value.len());
        let mut value_fft = self.fft_of(value);
        for i in 0..self.N() {
            Complex64::RING.square(&mut value_fft[i]);
        }
        self.fft_back(value_fft, value);
    }

    fn pow_gen<R: IntegerRingStore>(&self, x: Self::Element, power: &El<R>, integers: R) -> Self::Element 
        where R::Type: IntegerRing
    {
        assert_eq!(self.N(), x.len());
        let x_fft = self.fft_of(&x);
        algorithms::sqr_mul::generic_abs_square_and_multiply(
            x_fft, 
            power, 
            integers, 
            |mut a| {
                self.square(&mut a);
                return a;
            }, 
            |a, mut b| {
                self.mul_assign_fft(&mut b, a);
                return b;
            }, 
            self.one()
        )
    }
}

impl RingExtension for Pow2CyclotomicRingBase {

    type BaseRing = Zn;

    fn base_ring<'a>(&'a self) -> &'a Self::BaseRing {
        &self.base
    }

    fn from(&self, x: El<Self::BaseRing>) -> Self::Element {
        self.memory_provider_zn().get_new_init(self.N(), |i| if i == 0 { x } else { self.base.zero() })
    }
}

impl<P: ?Sized> CanHomFrom<P> for Pow2CyclotomicRingBase
    where P: PolyRing,
        zn_42::ZnBase: CanHomFrom<<P::BaseRing as RingStore>::Type>
{
    type Homomorphism = <zn_42::ZnBase as CanHomFrom<<P::BaseRing as RingStore>::Type>>::Homomorphism;

    fn has_canonical_hom(&self, from: &P) -> Option<Self::Homomorphism> {
        self.base.get_ring().has_canonical_hom(from.base_ring().get_ring())
    }

    fn map_in(&self, from: &P, el: <P as RingBase>::Element, hom: &Self::Homomorphism) -> Self::Element {
        self.map_in_ref(from, &el, hom)
    }

    fn map_in_ref(&self, from: &P, el: &P::Element, hom: &Self::Homomorphism) -> Self::Element {
        let mut result = self.zero();
        for (c, i) in from.terms(el) {
            self.base.add_assign(&mut result[i % self.N()], self.base.get_ring().map_in_ref(from.base_ring().get_ring(), c, hom));
        }
        return result;
    }
}

impl CanHomFrom<Pow2CyclotomicRingBase> for Pow2CyclotomicRingBase {

    type Homomorphism = <zn_42::ZnBase as CanHomFrom<zn_42::ZnBase>>::Homomorphism;

    fn has_canonical_hom(&self, from: &Pow2CyclotomicRingBase) -> Option<Self::Homomorphism> {
        if self.N() == from.N() {
            self.base.get_ring().has_canonical_hom(from.base.get_ring())
        } else {
            None
        }
    }

    fn map_in(&self, from: &Pow2CyclotomicRingBase, el: <Pow2CyclotomicRingBase as RingBase>::Element, hom: &Self::Homomorphism) -> Self::Element {
        self.memory_provider_zn().get_new_init(self.N(), |i| self.base.get_ring().map_in(from.base.get_ring(), el[i], hom))
    }
}

impl CanonicalIso<Pow2CyclotomicRingBase> for Pow2CyclotomicRingBase {

    type Isomorphism = <zn_42::ZnBase as CanonicalIso<zn_42::ZnBase>>::Isomorphism;

    fn has_canonical_iso(&self, from: &Pow2CyclotomicRingBase) -> Option<Self::Homomorphism> {
        if self.N() == from.N() {
            self.base.get_ring().has_canonical_iso(from.base.get_ring())
        } else {
            None
        }
    }

    fn map_out(&self, from: &Pow2CyclotomicRingBase, el: Self::Element, iso: &Self::Isomorphism) -> <Pow2CyclotomicRingBase as RingBase>::Element {
        from.memory_provider_zn().get_new_init(from.N(), |i| self.base.get_ring().map_out(from.base.get_ring(), el[i], iso))
    }
}

#[cfg(test)]
use feanor_math::assert_el_eq;

#[test]
fn test_ring_axioms() {
    let ring = Pow2CyclotomicRingBase::new(Zn::new(7), 9);
    let x = ring.get_ring().gen();

    assert!(!ring.is_zero(&x));
    assert_el_eq!(&ring, &ring.negate(ring.one()), &ring.pow(ring.clone_el(&x), ring.get_ring().N()));

    let mut edge_case_elements = Vec::new();
    edge_case_elements.push(ring.zero());
    edge_case_elements.push(ring.one());
    edge_case_elements.push(ring.negate(ring.one()));
    edge_case_elements.push(ring.clone_el(&x));
    edge_case_elements.push(ring.pow(ring.clone_el(&x), ring.get_ring().N() - 1));

    feanor_math::ring::generic_tests::test_ring_axioms(ring, edge_case_elements.into_iter());
}