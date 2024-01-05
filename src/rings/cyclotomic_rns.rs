
use feanor_math::algorithms::fft::FFTTable;
use feanor_math::homomorphism::{CanHomFrom, CanonicalIso, Homomorphism};
use feanor_math::integer::*;
use feanor_math::rings::poly::{PolyRing, PolyRingStore};
use feanor_math::vector::VectorView;
use feanor_math::vector::vec_fn::VectorFn;
use feanor_math::algorithms;
use feanor_math::delegate::DelegateRing;

use crate::globals::*;
use super::*;
use super::base::*;

///
/// The ring `(Z/qZ)[X]/(X^N + 1)` for `N` a power of two.
/// 
/// This provides a nice interface, the actual implementation is in [`super::base`].
/// 
pub struct RNSPow2CyclotomicRingBase(RingValue<DoubleRNSTensorRingBase<NoopFFTTable>>);

pub type RNSPow2CyclotomicRing = RingValue<RNSPow2CyclotomicRingBase>;

pub struct Pow2CyclotomicEl(pub(super) El<DoubleRNSTensorRing<NoopFFTTable>>);

pub struct NoopFFTTable(Zn, El<Zn>);

impl PartialEq for NoopFFTTable {

    fn eq(&self, other: &Self) -> bool {
        self.0.get_ring() == other.0.get_ring() && self.0.eq_el(&self.1, &other.1)
    }
}

impl FFTTable for NoopFFTTable {

    type Ring = Zn;

    fn unordered_fft<V, S, M, H>(&self, _: V, _: &M, _: &H)
        where S: ?Sized + RingBase, 
            H: feanor_math::homomorphism::Homomorphism<<Self::Ring as RingStore>::Type, S>,
            V: feanor_math::vector::VectorViewMut<S::Element>,
            M: feanor_math::mempool::MemoryProvider<S::Element> {}

    fn unordered_inv_fft<V, S, M, H>(&self, _: V, _: &M, _: &H)
        where S: ?Sized + RingBase, 
            H: feanor_math::homomorphism::Homomorphism<<Self::Ring as RingStore>::Type, S>,
            V: feanor_math::vector::VectorViewMut<S::Element>,
            M: feanor_math::mempool::MemoryProvider<S::Element> {}

    fn len(&self) -> usize { 1 }

    fn ring(&self) -> &Self::Ring {
        &self.0
    }

    fn root_of_unity(&self) -> &El<Self::Ring> {
        &self.1
    }

    fn unordered_fft_permutation(&self, i: usize) -> usize {
        assert_eq!(i, 0);
        0
    }

    fn unordered_fft_permutation_inv(&self, i: usize) -> usize {
        assert_eq!(i, 0);
        0
    }
}

impl RNSPow2CyclotomicRingBase {

    pub fn new(base: RNSBase, log2_N: usize) -> RingValue<Self> {
        RNSPow2CyclotomicRing::from(RNSPow2CyclotomicRingBase(DoubleRNSTensorRingBase::create(
            base, 
            1, 
            log2_N, 
            |Fp, _| NoopFFTTable(Fp, Fp.one()), 
            1,
            |Fp, n| {
                assert!(n == 2 << log2_N);
                algorithms::unity_root::get_prim_root_of_unity_pow2(Fp, log2_N + 1).unwrap()
            }
        )))
    }

    pub(super) fn new_with_roots(base: RNSBase, log2_N: usize, mut roots_of_unity: Vec<El<Zn>>) -> RingValue<Self> {
        roots_of_unity.reverse();
        let result = RNSPow2CyclotomicRing::from(RNSPow2CyclotomicRingBase(DoubleRNSTensorRingBase::create(
            base, 
            1, 
            log2_N, 
            |Fp, _| NoopFFTTable(Fp, Fp.one()), 
            1,
            |_Fp, _n| roots_of_unity.pop().unwrap()
        )));
        assert!(roots_of_unity.len() == 0);
        return result;
    }

    ///
    /// See [`DoubleRNSTensorRing::reduce_rns_base()`].
    /// 
    pub fn reduce_rns_base(&self, count: usize) -> RingValue<Self> {
        RingValue::from(Self(self.0.get_ring().reduce_rns_base(count)))
    }

    ///
    /// See [`DoubleRNSTensorRing::modulus_switch()`].
    /// 
    pub fn modulus_switch(&self, target: &Self, el: <Self as RingBase>::Element, plaintext_modulus: i64) -> <Self as RingBase>::Element {
        Pow2CyclotomicEl(self.0.get_ring().modulus_switch(target.0.get_ring(), el.0, plaintext_modulus))
    }

    pub fn smallest_lift<P>(&self, poly_ring: P, Pow2CyclotomicEl(value): &<Self as RingBase>::Element) -> El<P>
        where P: PolyRingStore,
            P::Type: PolyRing,
            <P::Type as RingExtension>::BaseRing: IntegerRingStore,
            <<P::Type as RingExtension>::BaseRing as RingStore>::Type: IntegerRing + CanHomFrom<BigIntRingBase>
    {
        <DoubleRNSTensorRingBase<_>>::smallest_lift(self.0.get_ring(), poly_ring, value)
    }

    ///
    /// See [`DoubleRNSTensorRing::external_product_decompose()`].
    /// 
    pub fn external_product_decompose(&self, el: &<Self as RingBase>::Element, basis: El<BigIntRing>, len: usize) -> Vec<<Self as RingBase>::Element> {
        self.0.get_ring().external_product_decompose(&el.0, basis, len).into_iter().map(|x| Pow2CyclotomicEl(x)).collect()
    }

    pub fn rns_base(&self) -> &RNSBase {
        self.0.get_ring().rns_base()
    }

    pub fn N(&self) -> usize {
        self.0.get_ring().N()
    }

    pub fn rank(&self) -> usize {
        self.N()
    }

    pub fn gen(&self) -> <Self as RingBase>::Element {
        Pow2CyclotomicEl(self.0.get_ring().gen())
    }
    
    pub fn sample_uniform<G>(&self, rng: G) -> <Self as RingBase>::Element
        where G: FnMut() -> u64
    {
        Pow2CyclotomicEl(self.0.get_ring().sample_uniform(rng))
    }

    pub fn sample_coefficient_distribution<G>(&self, distr: G) -> <Self as RingBase>::Element
        where G: FnMut() -> i32
    {
        Pow2CyclotomicEl(self.0.get_ring().sample_coefficient_distribution(distr))
    }

    pub fn sample_binomial4<G>(&self, mut rng: G) -> <Self as RingBase>::Element
        where G: FnMut() -> u64
    {
        self.sample_coefficient_distribution(|| {
            let value = rng();
            return (value & 0xF).count_ones() as i32 - (value & 0xF0).count_ones() as i32;
        })
    }

    pub fn sample_ternary<G>(&self, mut rng: G) -> <Self as RingBase>::Element
        where G: FnMut() -> u64
    {
        self.sample_coefficient_distribution(|| (rng() % 3) as i32 - 1)
    }
    
    pub fn compress<'a, S, V>(&'a self, el: &'a <Self as RingBase>::Element, rings: &'a V) -> impl 'a + ExactSizeIterator<Item = El<S>>
        where V: 'a + VectorView<S>,
            S: 'a + RingStore,
            S::Type: CanHomFrom<zn_42::ZnBase>
    {
        self.0.get_ring().compress(&el.0, rings)
    }

    pub fn uncompress<S, V, W>(&self, el: V, rings: W) -> <Self as RingBase>::Element
        where V: VectorFn<El<S>>,
            W: VectorFn<S>,
            S: RingStore,
            S::Type: CanonicalIso<zn_42::ZnBase>
    {
        Pow2CyclotomicEl(self.0.get_ring().uncompress(el, rings))
    }

}

impl PartialEq for RNSPow2CyclotomicRingBase {
    
    fn eq(&self, other: &Self) -> bool {
        self.0.get_ring() == other.0.get_ring()
    }
}

impl RingExtension for RNSPow2CyclotomicRingBase {

    type BaseRing = RNSBase;

    fn base_ring<'a>(&'a self) -> &'a Self::BaseRing {
        self.0.base_ring()
    }

    fn from(&self, x: El<Self::BaseRing>) -> Self::Element {
        Pow2CyclotomicEl(self.0.inclusion().map(x))
    }

    fn mul_assign_base(&self, lhs: &mut Self::Element, rhs: &El<Self::BaseRing>) {
        self.0.inclusion().mul_assign_map_ref(&mut lhs.0, rhs)
    }
}

impl DelegateRing for RNSPow2CyclotomicRingBase {

    type Base = DoubleRNSTensorRingBase<NoopFFTTable>;
    type Element = Pow2CyclotomicEl;

    fn delegate(&self, Pow2CyclotomicEl(el): Self::Element) -> <Self::Base as RingBase>::Element { el }

    fn delegate_mut<'a>(&self, Pow2CyclotomicEl(el): &'a mut Self::Element) -> &'a mut <Self::Base as RingBase>::Element { el }

    fn delegate_ref<'a>(&self, Pow2CyclotomicEl(el): &'a Self::Element) -> &'a <Self::Base as RingBase>::Element { el }

    fn get_delegate(&self) -> &Self::Base { self.0.get_ring() }

    fn rev_delegate(&self, el: <Self::Base as RingBase>::Element) -> Self::Element { Pow2CyclotomicEl(el) }
}

impl<P: ?Sized> CanHomFrom<P> for RNSPow2CyclotomicRingBase
    where P: PolyRing,
        zn_42::ZnBase: CanHomFrom<<P::BaseRing as RingStore>::Type>
{
    type Homomorphism = <DoubleRNSTensorRingBase<NoopFFTTable> as CanHomFrom<P>>::Homomorphism;

    fn has_canonical_hom(&self, from: &P) -> Option<Self::Homomorphism> {
        self.0.get_ring().has_canonical_hom(from)
    }

    fn map_in(&self, from: &P, el: <P as RingBase>::Element, hom: &Self::Homomorphism) -> Self::Element {
        Pow2CyclotomicEl(self.0.get_ring().map_in_ref(from, &el, hom))
    }

    fn map_in_ref(&self, from: &P, el: &P::Element, hom: &Self::Homomorphism) -> Self::Element {
        Pow2CyclotomicEl(self.0.get_ring().map_in_ref(from, &el, hom))
    }
}

impl CanHomFrom<RNSPow2CyclotomicRingBase> for RNSPow2CyclotomicRingBase {

    type Homomorphism = <DoubleRNSTensorRingBase<NoopFFTTable> as CanHomFrom<DoubleRNSTensorRingBase<NoopFFTTable>>>::Homomorphism;

    fn has_canonical_hom(&self, from: &RNSPow2CyclotomicRingBase) -> Option<Self::Homomorphism> {
        self.0.get_ring().has_canonical_hom(from.0.get_ring())
    }

    fn map_in(&self, from: &RNSPow2CyclotomicRingBase, Pow2CyclotomicEl(el): <RNSPow2CyclotomicRingBase as RingBase>::Element, hom: &Self::Homomorphism) -> Self::Element {
        Pow2CyclotomicEl(self.0.get_ring().map_in(from.0.get_ring(), el, hom))
    }

    fn map_in_ref(&self, from: &RNSPow2CyclotomicRingBase, Pow2CyclotomicEl(el): &<RNSPow2CyclotomicRingBase as RingBase>::Element, hom: &Self::Homomorphism) -> Self::Element {
        Pow2CyclotomicEl(self.0.get_ring().map_in_ref(from.0.get_ring(), el, hom))
    }
}

impl CanonicalIso<RNSPow2CyclotomicRingBase> for RNSPow2CyclotomicRingBase {

    type Isomorphism = <DoubleRNSTensorRingBase<NoopFFTTable> as CanonicalIso<DoubleRNSTensorRingBase<NoopFFTTable>>>::Isomorphism;

    fn has_canonical_iso(&self, from: &RNSPow2CyclotomicRingBase) -> Option<Self::Isomorphism> {
        self.0.get_ring().has_canonical_iso(from.0.get_ring())
    }

    fn map_out(&self, from: &RNSPow2CyclotomicRingBase, Pow2CyclotomicEl(el): Self::Element, iso: &Self::Isomorphism) -> <RNSPow2CyclotomicRingBase as RingBase>::Element {
        Pow2CyclotomicEl(self.0.get_ring().map_out(from.0.get_ring(), el, iso))
    }
}

#[cfg(test)]
use feanor_math::assert_el_eq;
#[cfg(test)]
use feanor_math::default_memory_provider;

#[test]
fn test_ring_axioms() {
    let ring = RNSPow2CyclotomicRingBase::new(
        RNSBase::new([Zn::new(17), Zn::new(97)].into_iter().collect(), BigIntRing::RING, default_memory_provider!()), 
        3
    );
    let x = ring.get_ring().gen();

    assert!(!ring.is_zero(&x));
    assert_el_eq!(&ring, &ring.negate(ring.one()), &ring.pow(ring.clone_el(&x), 8));

    let mut edge_case_elements = Vec::new();
    for i in 0..8 {
        edge_case_elements.push(ring.pow(ring.clone_el(&x), i));
        edge_case_elements.push(ring.negate(ring.pow(ring.clone_el(&x), i)));
        edge_case_elements.push(ring.add(ring.pow(ring.clone_el(&x), i), ring.one()));
        edge_case_elements.push(ring.sub(ring.pow(ring.clone_el(&x), i), ring.one()));
    }

    feanor_math::ring::generic_tests::test_ring_axioms(ring, edge_case_elements.into_iter());
}
