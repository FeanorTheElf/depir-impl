use feanor_math::homomorphism::*;
use feanor_math::vector::VectorView;
use feanor_math::vector::vec_fn::VectorFn;
use feanor_math::algorithms;
use feanor_math::algorithms::fft::bluestein;
use feanor_math::delegate::DelegateRing;
use feanor_math::integer::IntegerRingStore;

use crate::globals::*;
use super::*;
use super::base::*;
use super::cyclotomic_rns::*;

///
/// The ring `(Z/qZ)[X]/(X^N + 1) ⊗ (Z/qZ)[X]/(X^m - 1)` with `N` a power of two and 
/// `m` odd, viewn as free module over the base ring `(Z/qZ)[X]/(X^N + 1)`.
/// 
/// This provides a nice interface, the actual implementation is in [`super::base`].
/// 
pub struct RNSCyclotomicTensorRingBase {
    ring: DoubleRNSTensorRing<bluestein::FFTTableBluestein<Zn>>, 
    base_ring: RNSPow2CyclotomicRing,
    generator: El<DoubleRNSTensorRing<bluestein::FFTTableBluestein<Zn>>>
}

pub type RNSCyclotomicTensorRing = RingValue<RNSCyclotomicTensorRingBase>;

impl RNSCyclotomicTensorRingBase {

    pub fn new(base: RNSBase, log2_N: usize, m: usize) -> RingValue<Self> {
        assert!(m % 2 == 1);
        assert!(m > 1);
        let bluestein_log2 = StaticRing::<i64>::RING.abs_log2_ceil(&(m as i64)).unwrap() + 1;
        let result = DoubleRNSTensorRingBase::create(
            base, 
            m, 
            log2_N, 
            |Fp, root_of_unity_base| bluestein::FFTTableBluestein::new(
                Fp, 
                Fp.pow(Fp.clone_el(&root_of_unity_base), 1 << (bluestein_log2 - 1)), 
                Fp.pow(root_of_unity_base, m), 
                m, 
                bluestein_log2
            ), 
            m << bluestein_log2,
            |Fp, n| {
                algorithms::unity_root::get_prim_root_of_unity(Fp, n).unwrap()
            }
        );
        let pow2_cyclotomic = result.get_ring().power_two_cyclotomic_factor();
        RNSCyclotomicTensorRing::from(RNSCyclotomicTensorRingBase {
            generator: result.pow(result.get_ring().gen(), 2 << log2_N),
            ring: result, 
            base_ring: pow2_cyclotomic, 
        })
    }

    pub fn m_th_root_of_unity(&self) -> &<Self as RingBase>::Element {
        &self.generator
    }

    pub fn N(&self) -> usize {
        self.ring.get_ring().N()
    }

    pub fn m(&self) -> usize {
        self.ring.get_ring().m()
    }

    pub fn rank(&self) -> usize {
        self.ring.get_ring().rank()
    }

    pub fn rns_base(&self) -> &RNSBase {
        self.ring.get_ring().rns_base()
    }

    ///
    /// Returns the vector corresponding to the given element, assuming this ring is
    /// a free module over `(Z/qZ)[X]/(X^N + 1)` w.r.t. the basis `1 ⊗ X^i`.
    /// 
    pub fn wrt_standard_basis(&self, el: <Self as RingBase>::Element) -> Vec<El<RNSPow2CyclotomicRing>> {
        self.ring.get_ring().tensor_decomp(el, self.base_ring().get_ring())
    }
    
    pub fn compress<'a, S, V>(&'a self, el: &'a <Self as RingBase>::Element, rings: &'a V) -> impl 'a + ExactSizeIterator<Item = El<S>>
        where V: 'a + VectorView<S>,
            S: 'a + RingStore,
            S::Type: CanHomFrom<zn_42::ZnBase>
    {
        self.ring.get_ring().compress(el, rings)
    }

    pub fn uncompress<S, V, W>(&self, el: V, rings: W) -> <Self as RingBase>::Element
        where V: VectorFn<El<S>>,
            W: VectorFn<S>,
            S: RingStore,
            S::Type: CanonicalIso<zn_42::ZnBase>
    {
        self.ring.get_ring().uncompress(el, rings)
    }
    
    pub fn reduce_rns_base(&self, count: usize) -> RingValue<Self> {
        let result = self.ring.get_ring().reduce_rns_base(count);
        RingValue::from(Self { 
            base_ring: result.get_ring().power_two_cyclotomic_factor(), 
            generator: result.pow(result.get_ring().gen(), 2 * result.get_ring().N()),
            ring: result
        })
    }
}

impl PartialEq for RNSCyclotomicTensorRingBase {
    
    fn eq(&self, other: &Self) -> bool {
        self.ring.get_ring() == other.ring.get_ring()
    }
}

impl DelegateRing for RNSCyclotomicTensorRingBase {

    type Base = DoubleRNSTensorRingBase<bluestein::FFTTableBluestein<Zn>>;
    type Element = DoubleRNSTensorRingEl;

    fn delegate(&self, el: Self::Element) -> <Self::Base as RingBase>::Element { el }

    fn delegate_mut<'a>(&self, el: &'a mut Self::Element) -> &'a mut <Self::Base as RingBase>::Element { el }

    fn delegate_ref<'a>(&self, el: &'a Self::Element) -> &'a <Self::Base as RingBase>::Element { el }

    fn get_delegate(&self) -> &Self::Base { self.ring.get_ring() }

    fn rev_delegate(&self, el: <Self::Base as RingBase>::Element) -> Self::Element { el }
}

impl CanHomFrom<RNSCyclotomicTensorRingBase> for RNSCyclotomicTensorRingBase {

    type Homomorphism = <DoubleRNSTensorRingBase<bluestein::FFTTableBluestein<Zn>> as CanHomFrom<DoubleRNSTensorRingBase<bluestein::FFTTableBluestein<Zn>>>>::Homomorphism;

    fn has_canonical_hom(&self, from: &RNSCyclotomicTensorRingBase) -> Option<Self::Homomorphism> {
        self.ring.get_ring().has_canonical_hom(from.ring.get_ring())
    }

    fn map_in(&self, from: &RNSCyclotomicTensorRingBase, el: <RNSCyclotomicTensorRingBase as RingBase>::Element, hom: &Self::Homomorphism) -> Self::Element {
        self.ring.get_ring().map_in(from.ring.get_ring(), el, hom)
    }

    fn map_in_ref(&self, from: &RNSCyclotomicTensorRingBase, el: &<RNSCyclotomicTensorRingBase as RingBase>::Element, hom: &Self::Homomorphism) -> Self::Element {
        self.ring.get_ring().map_in_ref(from.ring.get_ring(), el, hom)
    }
}

impl CanonicalIso<RNSCyclotomicTensorRingBase> for RNSCyclotomicTensorRingBase {

    type Isomorphism = <DoubleRNSTensorRingBase<bluestein::FFTTableBluestein<Zn>> as CanonicalIso<DoubleRNSTensorRingBase<bluestein::FFTTableBluestein<Zn>>>>::Isomorphism;

    fn has_canonical_iso(&self, from: &RNSCyclotomicTensorRingBase) -> Option<Self::Isomorphism> {
        self.ring.get_ring().has_canonical_iso(from.ring.get_ring())
    }

    fn map_out(&self, from: &RNSCyclotomicTensorRingBase, el: Self::Element, iso: &Self::Isomorphism) -> <RNSCyclotomicTensorRingBase as RingBase>::Element {
        self.ring.get_ring().map_out(from.ring.get_ring(), el, iso)
    }
}

impl RingExtension for RNSCyclotomicTensorRingBase {

    type BaseRing = RNSPow2CyclotomicRing;
    
    fn base_ring<'a>(&'a self) -> &'a Self::BaseRing {
        &self.base_ring
    }

    fn from(&self, x: El<Self::BaseRing>) -> Self::Element {
        self.ring.get_ring().tensor_x_one(x, self.base_ring().get_ring())
    }
}

#[cfg(test)]
use feanor_math::assert_el_eq;
#[cfg(test)]
use feanor_math::default_memory_provider;

#[test]
fn test_ring_axioms() {
    let ring = RNSCyclotomicTensorRingBase::new(RNSBase::new(vec![Zn::new(113), Zn::new(337)], BigIntRing::RING, default_memory_provider!()), 3, 7);
    let x = ring.get_ring().ring.get_ring().gen();

    assert!(!ring.is_zero(&x));
    assert_el_eq!(&ring, &ring.negate(ring.one()), &ring.pow(ring.clone_el(&x), 8 * 7));

    let mut edge_case_elements = Vec::new();
    for i in 0..8 {
        edge_case_elements.push(ring.pow(ring.clone_el(&x), i));
        edge_case_elements.push(ring.negate(ring.pow(ring.clone_el(&x), i)));
        edge_case_elements.push(ring.add(ring.pow(ring.clone_el(&x), i), ring.one()));
        edge_case_elements.push(ring.sub(ring.pow(ring.clone_el(&x), i), ring.one()));
    }

    feanor_math::ring::generic_tests::test_ring_axioms(ring, edge_case_elements.into_iter());
}
