use std::rc::Rc;

use feanor_math::homomorphism::*;
use feanor_math::integer::*;
use feanor_math::pid::EuclideanRingStore;
use feanor_math::{rings::{zn::*, finite::*, zn::zn_42::ZnBase, poly::{*, dense_poly::DensePolyRing}}, algorithms::{self, fft::*}, vector::{*, subvector::{Subvector, SelfSubvectorView}, vec_fn::VectorFn}, mempool::caching::CachingMemoryProvider, assert_el_eq, delegate::DelegateRing, integer::{IntegerRingStore, IntegerRing, int_cast}};
use feanor_math::default_memory_provider;

use super::cyclotomic_rns::*;
use super::*;
use crate::rnsconv::{bgv_rescale::CongruencePreservingRescaling, RNSOperation};

///
/// The ring `(Z/qZ)[X]/(X^N + 1) ⊗ (Z/qZ)[X]/(X^m - 1)` with `N` a power of two and `m` odd.
/// 
/// This contains the actual, generic implementation. For using it, see 
/// the sibling modules [`super::cyclotomic`] and especially [`super::tensor`].
/// 
pub struct DoubleRNSTensorRingBase<F>
    where F: FFTTable<Ring = Zn> + PartialEq
{
    base: RNSBase,
    m: usize,
    log2_N: usize,
    fft_tables: Vec<Rc<factor_fft::FFTTableGenCooleyTuckey<Zn, F, cooley_tuckey::FFTTableCooleyTuckey<Zn>>>>,
    inv_twiddles: Vec<El<Zn>>,
    twiddles: Vec<El<Zn>>
}

pub type DoubleRNSTensorRing<F> = RingValue<DoubleRNSTensorRingBase<F>>;

impl<F> DoubleRNSTensorRingBase<F>
    where F: FFTTable<Ring = Zn> + PartialEq
{
    fn create_fft_table<C>(modulus: &Zn, m: usize, log2_N: usize, mut fft_table_creator: C, root_of_unity: El<Zn>, helper_root_of_unity: El<Zn>) -> factor_fft::FFTTableGenCooleyTuckey<Zn, F, cooley_tuckey::FFTTableCooleyTuckey<Zn>>
        where C: FnMut(Zn, El<Zn>) -> F
    {
        timed("DoubleRNSTensorRing::create_fft_table", || {
            let result = factor_fft::FFTTableGenCooleyTuckey::new(
                modulus.pow(modulus.clone_el(&root_of_unity), 2),
                fft_table_creator(modulus.clone(), helper_root_of_unity),
                cooley_tuckey::FFTTableCooleyTuckey::new(modulus.clone(), modulus.pow(root_of_unity, 2 * m), log2_N)
            );
            assert_el_eq!(modulus, &modulus.pow(root_of_unity, 2), result.root_of_unity());
            return result;
        })
    }

    fn create_twiddles(modulus: &Zn, rank: usize, root_of_unity: El<Zn>, inv_twiddles: &mut Vec<El<Zn>>, twiddles: &mut Vec<El<Zn>>) {
        timed("DoubleRNSTensorRing::create_twiddles", || {
            let mut current_root_of_unity = modulus.one();
            let inv_root_of_unity = modulus.pow(root_of_unity, 2 * rank - 1);
            let mut current_inv_root_of_unity = modulus.one();
            for _ in 0..rank {
                // since the FFT evaluates at the inverse root of unity, also use the inverse here
                inv_twiddles.push(current_inv_root_of_unity);
                twiddles.push(current_root_of_unity);
                assert_el_eq!(modulus, &modulus.one(), &modulus.mul_ref(twiddles.last().unwrap(), &inv_twiddles.last().unwrap()));
                modulus.mul_assign_ref(&mut current_root_of_unity, &root_of_unity);
                modulus.mul_assign_ref(&mut current_inv_root_of_unity, &inv_root_of_unity);
            }
        })
    }

    pub fn create<C, D>(base: RNSBase, m: usize, log2_N: usize, mut fft_table_creator: C, helper_order: usize, mut root_of_unity_generator: D) -> RingValue<Self> 
        where D: FnMut(&Zn, usize) -> El<Zn>,
            C: FnMut(Zn, El<Zn>) -> F
    {
        timed("DoubleRNSTensorRingBase::create", || {
            assert!(m % 2 != 0);
            assert!(log2_N > 0);
            let mut fft_tables = Vec::new();
            let mut twiddles = Vec::new();
            let mut inv_twiddles = Vec::new();
            let rank = m << log2_N;
            for i in 0..base.get_ring().len() {
                let modulus = base.get_ring().at(i);
                let order_gcd = algorithms::eea::signed_gcd(2 * rank as i64, helper_order as i64, &StaticRing::<i64>::RING) as usize;
                let base_root_of_unity = root_of_unity_generator(modulus, 2 * rank * helper_order / order_gcd);
                let root_of_unity = modulus.pow(modulus.clone_el(&base_root_of_unity), helper_order / order_gcd);
                let helper_root_of_unity = modulus.pow(base_root_of_unity, 2 * rank / order_gcd);

                let fft_table = Self::create_fft_table(&modulus, m, log2_N, &mut fft_table_creator, root_of_unity, helper_root_of_unity);
                fft_tables.push(Rc::new(fft_table));

                Self::create_twiddles(&modulus, rank, root_of_unity, &mut twiddles, &mut inv_twiddles);
            }
            return RingValue::from(Self { base, m, fft_tables, inv_twiddles: twiddles, twiddles: inv_twiddles, log2_N });
        })
    }

    pub fn gen(&self) -> <Self as RingBase>::Element {
        timed("DoubleRNSTensorRingBase::gen", || {
            let mut result = self.zero();
            for (i, Fp) in self.rns_base_iter() {
                for j in self.rank_iter() {
                    let power_of_zeta = self.power_of_zeta_at_index(i, j);
                    if power_of_zeta < self.rank() {
                        *self.at_mut(&mut result, i, j) = Fp.clone_el(self.inv_twiddles.at(i * self.rank() + power_of_zeta));
                    } else {
                        *self.at_mut(&mut result, i, j) = Fp.clone_el(self.twiddles.at(i * self.rank() + 2 * self.rank() - power_of_zeta));
                    }
                }
            }
            return result;
        })
    }

    pub fn poly_repr<P>(&self, poly_ring: P, el: &<Self as RingBase>::Element) -> El<P>
        where P: PolyRingStore,
            P::Type: PolyRing,
            <P::Type as RingExtension>::BaseRing: RingStore<Type = <RNSBase as RingStore>::Type>
    {
        timed("DoubleRNSTensorRingBase::poly_repr", || {
            // so that we can use elements without modification
            assert!(self.base.get_ring() == poly_ring.base_ring().get_ring());
            let mut el_inv_fft = self.clone_el(el);
            self.inv_fft(&mut el_inv_fft.data);
            poly_ring.from_terms(
                self.rank_iter()
                    .filter(|j| !self.rns_base_iter().all(|(i, Fp)| Fp.is_zero(self.at(&el_inv_fft, i, *j))))
                    .map(|j| (self.base.get_ring().from_congruence((&el_inv_fft.data[j..]).stride(self.rank_iter().len()).iter().cloned()), j))
            )
        })
    }

    pub fn smallest_lift<P>(&self, poly_ring: P, el: &<Self as RingBase>::Element) -> El<P>
        where P: PolyRingStore,
            P::Type: PolyRing,
            <P::Type as RingExtension>::BaseRing: IntegerRingStore,
            <<P::Type as RingExtension>::BaseRing as RingStore>::Type: IntegerRing + CanHomFrom<<<RNSBase as RingStore>::Type as ZnRing>::IntegerRingBase>
    {
        timed("DoubleRNSTensorRingBase::smallest_lift", || {
            let hom = poly_ring.base_ring().can_hom(self.base_ring().integer_ring()).unwrap();
            let mut el_inv_fft = self.clone_el(el);
            self.inv_fft(&mut el_inv_fft.data);
            poly_ring.from_terms(
                self.rank_iter()
                    .filter(|j| !self.rns_base_iter().all(|(i, Fp)| Fp.is_zero(self.at(&el_inv_fft, i, *j))))
                    .map(|j| (hom.map(self.base.smallest_lift(self.base.get_ring().from_congruence((&el_inv_fft.data[j..]).stride(self.rank_iter().len()).iter().cloned()))), j))
            )
        })
    }

    ///
    /// Returns the ring `(Z/qZ)[X]/(X^N + 1)`, where this ring is `(Z/qZ)[X]/(X^N + 1) ⊗ (Z/qZ)[X]/(X^m - 1)`.
    /// 
    pub fn power_two_cyclotomic_factor(&self) -> RNSPow2CyclotomicRing {
        // we have to ensure that the roots of unity and thus the FFT order are compatible
        let mut roots_of_unity = Vec::new();
        for (i, Fp) in self.rns_base_iter() {
            roots_of_unity.push(Fp.clone_el(self.twiddles.at(i * self.rank() + self.m())));
        }
        return RNSPow2CyclotomicRingBase::new_with_roots(self.base.clone(), self.log2_N, roots_of_unity);
    } 

    ///
    /// Computes the map `(Z/qZ)[X]/(X^N + 1) -> (Z/qZ)[X]/(X^N + 1) ⊗ (Z/qZ)[X]/(X^m - 1), x -> x ⊗ 1`.
    /// 
    /// Note that it is only defined up to isomorphism. If the rings are fixed, this isomorphism is always
    /// the same, i.e. multiple calls of this function are compatible (internal note: the isomorphism depends
    /// on which root of unity the FFTTables use).
    /// 
    /// We only assume that the `i`-th RNS factor of this ring has a canonical embedding from the `i`-th RNS
    /// factor of the given ring.
    /// 
    pub fn tensor_x_one(&self, mut lhs: El<RNSPow2CyclotomicRing>, lhs_ring: &RNSPow2CyclotomicRingBase) -> <Self as RingBase>::Element {
        timed("DoubleRNSTensorRingBase::tensor_x_one", || {
            let lhs_base_ring = lhs_ring.get_delegate();
            assert_eq!(lhs_base_ring.log2_N, self.log2_N);
            assert_eq!(lhs_base_ring.rns_base_iter().len(), self.rns_base_iter().len());
            let mut result = self.zero();

            for (i, _) in lhs_base_ring.rns_base_iter() {
                let hom = self.base.get_ring().at(i).can_hom(lhs_base_ring.base.get_ring().at(i)).unwrap();
                // otherwise the FFT coefficients will have the wrong order
                assert_el_eq!(self.base.get_ring().at(i), &hom.map_ref(lhs_base_ring.inv_twiddles.at(1)), self.inv_twiddles.at(self.m()));
                let mut data_part = Subvector::new(&mut lhs.0.data).subvector((i * lhs_base_ring.rank())..((i + 1) * lhs_base_ring.rank()));

                // bring the coefficients of data into the order corresponding to (z^1, z^3, z^5, ...)
                permute::permute_inv(
                    &mut data_part, 
                    |j| lhs_base_ring.fft_tables.at(i).unordered_fft_permutation(j), 
                    &default_memory_provider!()
                );

                // transfer the coefficients of data into the result
                for j in self.rank_iter() {
                    let zeta_power = self.power_of_zeta_at_index(i, j) % (1 << (self.log2_N + 1));
                    assert!((zeta_power - 1) % 2 == 0);
                    *self.at_mut(&mut result, i, j) = hom.map_ref(data_part.at((zeta_power - 1) / 2));
                }
            }
            return result;
        })
    }

    ///
    /// Finds the preimage under the linear map
    /// ```text
    /// ((Z/qZ)[X]/(X^N + 1))^m -> (Z/qZ)[X]/(X^N + 1) ⊗ (Z/qZ)[X]/(X^m - 1), (a_i) -> sum_i a_i ⊗ X^i
    /// ```
    /// 
    pub fn tensor_decomp(&self, mut val: <Self as RingBase>::Element, out_ring: &RNSPow2CyclotomicRingBase) -> Vec<El<RNSPow2CyclotomicRing>> {
        timed("DoubleRNSTensorRingBase::tensor_decomp", || {
            let out_base_ring = out_ring.get_delegate();
            assert_eq!(out_base_ring.log2_N, self.log2_N);
            assert_eq!(out_base_ring.rns_base_iter().len(), self.rns_base_iter().len());

            // technically, we only have to undo the order-m fft, but since we do not have a dedicated order-m fft table,
            // it is simplest to undo all the fft, and then redo the order-2N ffts. This also avoids all the problems due
            // to non-unique ordering of FFT coefficients
            self.inv_fft(&mut val.data);
            let mut result = Iterator::map(0..self.m, |_| out_ring.zero()).collect::<Vec<_>>();
            for k in 0..self.m() {
                for l in 0..self.N() {
                    let power = ((k << (1 + self.log2_N)) + l * self.m) % (self.m << (1 + self.log2_N));
                    if power > self.rank() {
                        for (i, Fp) in self.rns_base_iter() {
                            result[k].0.data[i * out_base_ring.rank() + l] = Fp.negate(Fp.clone_el(&val.data[i * self.rank() + power - self.rank()]));
                        }
                    } else {
                        for (i, Fp) in self.rns_base_iter() {
                            result[k].0.data[i * out_base_ring.rank() + l] = Fp.clone_el(&val.data[i * self.rank() + power]);
                        }
                    }
                }
            }
            for k in 0..self.m() {
                out_base_ring.fft(&mut result[k].0.data);
            }
            return result;
        })
    }

    pub fn compress<'a, S, V>(&'a self, el: &'a <Self as RingBase>::Element, rings: &'a V) -> impl 'a + ExactSizeIterator<Item = El<S>>
        where V: 'a + VectorView<S>,
            S: 'a + RingStore,
            S::Type: CanHomFrom<ZnBase>
    {
        assert_eq!(rings.len(), self.base.get_ring().len());
        let homs = self.rns_base_iter()
            .map(|(i, from)| rings.at(i).can_hom(&*from).ok_or(()))
            .collect::<Result<Vec<_>, ()>>().ok().unwrap();
        return el.data.iter().enumerate().map(move |(i, x)| homs[i / self.rank()].map_ref(x));
    }

    pub fn uncompress<S, V, W>(&self, el: V, rings: W) -> <Self as RingBase>::Element
        where V: VectorFn<El<S>>,
            W: VectorFn<S>,
            S: RingStore,
            S::Type: CanonicalIso<ZnBase>
    {
        assert_eq!(rings.len(), self.base.get_ring().len());
        assert_eq!(self.element_len(), el.len());
        let mut result = Vec::with_capacity(self.element_len());
        for (i, to) in self.rns_base_iter() {
            let from = rings.at(i);
            let hom = from.can_iso(to).unwrap();
            for j in self.rank_iter() {
                result.push(hom.map(el.at(i * self.rank() + j)));
            }
        }
        return DoubleRNSTensorRingEl { data: result };
    }

    ///
    /// Drops the first `count` moduli of the RNS base, to gain a new ring representing the same extension
    /// w.r.t. a different modulus. This will usually be used together with [`Self::modulus_switch()`].
    /// 
    pub fn reduce_rns_base(&self, count: usize) -> RingValue<Self> {
        let use_range = (self.rns_base().get_ring().len() - count)..self.rns_base().get_ring().len();
        let rank = self.rank();
        RingValue::from(Self {
            base: zn_rns::Zn::new(use_range.clone().map(|i| self.rns_base().get_ring().at(i).clone()).collect(), BigIntRing::RING, default_memory_provider!()),
            m: self.m,
            log2_N: self.log2_N,
            fft_tables: use_range.clone().map(|i| self.fft_tables.at(i).clone()).collect(),
            twiddles: use_range.clone().flat_map(|i| (0..rank).map(move |j| self.twiddles[i * rank + j])).collect(),
            inv_twiddles: use_range.clone().flat_map(|i| (0..rank).map(move |j| self.inv_twiddles[i * rank + j])).collect()
        })
    }

    ///
    /// Computes `y` in the target ring such that `target_ring.modulus() * el` is both close to and congruent to `self.modulus() * y`
    /// modulo `plaintext_modulus` (more concretely: their respective shortest lifts).
    /// 
    /// This basically implements the modulus-switching technique from the BGV-cryptosystem, i.e. modulus-switching all components of a
    /// ciphertext gives a new ciphertext that encrypts a related message, but in a ciphertext ring with different modulus. Usually, this
    /// is used to control the absolute magnitude of the noise, in our case, we just use it to reduce the size of messages that have to be
    /// transmitted.
    /// 
    /// In particular, modulus-switching a ciphertext to a lower modulus, and then back again, yields a new ciphertext encrypting the same
    /// message (under the same key), with its noise increased by an additive amount. 
    /// 
    pub fn modulus_switch(&self, target: &Self, mut el: <Self as RingBase>::Element, plaintext_modulus: i64) -> <Self as RingBase>::Element {
        assert_eq!(self.rank(), target.rank());

        timed("DoubleRNSTensorRingBase::modulus_switch", || {

            let self_rns_base_len = self.rns_base().get_ring().len();
            let target_rns_base_len = target.rns_base().get_ring().len();

            if self_rns_base_len > target_rns_base_len {

                for i in 0..target_rns_base_len {
                    assert!(self.rns_base().get_ring().at(i + self_rns_base_len - target_rns_base_len).get_ring() == target.rns_base().get_ring().at(i).get_ring());
                }

                let rescaling = CongruencePreservingRescaling::scale_down(
                    self.rns_base().get_ring().iter().cloned().collect(), 
                    self_rns_base_len - target_rns_base_len, 
                    Zn::new(plaintext_modulus as u64), 
                    Zn::new(65539), 
                    CachingMemoryProvider::new(2), 
                    CachingMemoryProvider::new(2)
                );
                self.inv_fft(&mut el.data);
                let mut result = target.zero();
                for j in 0..self.rank() {
                    let rescaled_coefficient = rescaling.apply((&el.data[j..]).stride(self.rank()).iter().zip(self.rns_base().get_ring().iter()).map(|(x, R)| R.clone_el(x)));
                    for i in 0..target_rns_base_len {
                        result.data[i * target.rank() + j] = target.rns_base().get_ring().at(i).clone_el(rescaled_coefficient.at(i));
                    }
                }
                target.fft(&mut result.data);
                return result;

            } else {

                for i in 0..self_rns_base_len {
                    assert!(self.rns_base().get_ring().at(i).get_ring() == target.rns_base().get_ring().at(i + target_rns_base_len - self_rns_base_len).get_ring());
                }

                let scale_up = ZZbig.prod((0..(target_rns_base_len - self_rns_base_len))
                    .map(|i| int_cast(target.rns_base().get_ring().at(i).integer_ring().clone_el(target.rns_base().get_ring().at(i).modulus()), ZZbig, target.rns_base().get_ring().at(i).integer_ring()))
                );
                let factors = (0..self_rns_base_len)
                    .map(|i| self.rns_base().get_ring().at(i).coerce(&ZZbig, ZZbig.clone_el(&scale_up)))
                    .collect::<Vec<_>>();

                let mut result = target.zero();
                for j in 0..self.rank() {
                    for i in 0..self_rns_base_len {
                        result.data[(i + target_rns_base_len - self_rns_base_len) * target.rank() + j] = self.rns_base().get_ring().at(i).mul_ref(&el.data[i * self.rank() + j], &factors[i]);
                    }
                }
                return result;
            }
        })
    }

    ///
    /// Computes a vector of "short" elements (i.e. elements whose shortest lift is bounded by `basis`)
    /// such that `sum result[i] * basis^i == el`.
    /// 
    /// This is a standard technique used for relinearization and key-switching. We remark that it is usually
    /// faster to do a RNS-based decomposition instead of using the powers of `basis`, but in our cases, the
    /// performance loss is negligible compared to the actual protocol.
    /// 
    pub fn external_product_decompose(&self, el: &<Self as RingBase>::Element, basis: El<BigIntRing>, len: usize) -> Vec<<Self as RingBase>::Element> {
        timed("DoubleRNSTensorRingBase::external_product_decompose", || {
            let poly_ring = DensePolyRing::new(ZZbig, "X");
            let mut integers = self.smallest_lift(&poly_ring, el);
            let mut result = (0..len).map(|_| self.zero()).collect::<Vec<_>>();
            for i in 0..len {
                for j in 0..integers.len() {
                    let (quo, rem) = ZZbig.euclidean_div_rem(std::mem::replace(&mut integers[j], ZZbig.zero()), &basis);
                    integers[j] = quo;
                    for (k, Fp) in self.rns_base_iter() {
                        result[i].data[j + k * self.rank()] = Fp.can_hom(&ZZbig).unwrap().map_ref(&rem);
                    }
                }
                self.fft(&mut result[i].data);
            }
            for j in 0..integers.len() {
                assert_el_eq!(&ZZbig, &ZZbig.zero(), &integers[j]);
            }
            let mut check = self.zero();
            for i in (0..len).rev() {
                self.mul_assign(&mut check, self.from(self.base_ring().coerce(&ZZbig, ZZbig.clone_el(&basis))));
                self.add_assign_ref(&mut check, &result[i]);
            }
            assert_el_eq!(&RingRef::new(self), el, &check);
            return result;
        })
    }

    pub fn mth_root_of_unity(&self) -> <Self as RingBase>::Element {
        RingRef::new(self).pow(self.gen(), 2 << self.log2_N)
    }

    fn rns_base_iter<'a>(&'a self) -> impl ExactSizeIterator<Item = (usize, &'a Zn)> {
        Iterator::map(0..self.base.get_ring().len(), |i| (i, self.base.get_ring().at(i)))
    }

    pub fn rns_base(&self) -> &RNSBase {
        &self.base
    }

    fn rank_iter(&self) -> impl ExactSizeIterator<Item = usize> {
        0..self.rank()
    }

    pub fn rank(&self) -> usize {
        self.m << self.log2_N
    }

    pub fn N(&self) -> usize {
        1 << self.log2_N
    }

    pub fn m(&self) -> usize {
        self.m
    }

    fn at<'a>(&self, el: &'a DoubleRNSTensorRingEl, i: usize, j: usize) -> &'a El<Zn> {
        &el.data[i * self.rank() + j]
    }

    fn at_mut<'a>(&self, el: &'a mut DoubleRNSTensorRingEl, i: usize, j: usize) -> &'a mut El<Zn> {
        &mut el.data[i * self.rank() + j]
    }

    fn element_len(&self) -> usize {
        self.rns_base_iter().len() * self.rank()
    }

    pub fn sample_uniform<G>(&self, mut rng: G) -> <Self as RingBase>::Element
        where G: FnMut() -> u64
    {
        let mut data = self.zero();
        for (i, Fp) in self.rns_base_iter() {
            for j in self.rank_iter() {
                *self.at_mut(&mut data, i, j) = Fp.random_element(&mut rng);
            }
        }
        return data;
    }

    pub fn sample_coefficient_distribution<G>(&self, mut distr: G) -> <Self as RingBase>::Element
        where G: FnMut() -> i32
    {
        timed("DoubleRNSTensorRingBase::sample_coefficient_distribution", || {
            let mut data = self.zero();
            for j in self.rank_iter() {
                let value = distr();
                for (i, Fp) in self.rns_base_iter() {
                    *self.at_mut(&mut data, i, j) = Fp.int_hom().map(value);
                }
            }
            self.fft(&mut data.data);
            return data;
        })
    }

    ///
    /// Returns `k` such that the `(i * self.rank() + j)`-th index of an element contains
    /// the evaluation of the polynomial to represent at `z^k`, where `z` is a `2Nm`-th root of unity
    /// such that `z^-2` underlies `self.fft_tables.at(i)` (the inverse is necessary as the FFT
    /// traditionally is the evaluation at). 
    /// Note that since we consider the `2N`-th cyclotomic ring, this means that `k` must be odd.
    /// 
    fn power_of_zeta_at_index(&self, i: usize, j: usize) -> usize {
        (self.fft_tables.at(i).unordered_fft_permutation(j) * 2 + 1) % (self.m << (self.log2_N + 1))
    }

    ///
    /// Performs the weighted FFT, i.e. transforms the coefficients of a polynomial `f` into the list of
    /// evaluations `f(z^k)` for all `k in (Z/2NZ)^* x Z/mZ`, i.e. the odd `k` in `Z/2NmZ`.
    /// 
    /// In particular, this allows component-wise multiplication to perform multiplication in the specific
    /// ring.
    /// 
    #[inline(never)]
    fn fft(&self, data: &mut [El<Zn>]) {
        timed("DoubleRNSTensorRingBase::fft", || {
            for (i, Fp) in self.rns_base_iter() {
                for j in self.rank_iter() {
                    Fp.mul_assign_ref(&mut data[i * self.rank() + j], &self.inv_twiddles[i * self.rank() + j]);
                }
                self.fft_tables.at(i).unordered_fft(&mut data[(i * self.rank())..((i + 1) * self.rank())], &default_memory_provider!(), &Fp.identity());
            }
        })
    }

    ///
    /// Inverse to [`fft()`].
    /// 
    fn inv_fft(&self, data: &mut [El<Zn>]) {
        timed("DoubleRNSTensorRingBase::inv_fft", || {
            for (i, Fp) in self.rns_base_iter() {
                self.fft_tables.at(i).unordered_inv_fft(&mut data[(i * self.rank())..((i + 1) * self.rank())], &default_memory_provider!(), &Fp.identity());
                for j in self.rank_iter() {
                    assert_el_eq!(Fp, &Fp.one(), &Fp.mul_ref(&self.twiddles[i * self.rank() + j], &self.inv_twiddles[i * self.rank() + j]));
                    Fp.mul_assign_ref(&mut data[i * self.rank() + j], &self.twiddles[i * self.rank() + j]);
                }
            }
        })
    }
}

impl<F> PartialEq for DoubleRNSTensorRingBase<F>
    where F: FFTTable<Ring = Zn> + PartialEq
{
    fn eq(&self, other: &Self) -> bool {
        self.base.get_ring() == other.base.get_ring() && self.log2_N == other.log2_N && self.m == other.m && self.fft_tables.iter().zip(other.fft_tables.iter()).all(|(a, b)| a == b)
    }
}

pub struct DoubleRNSTensorRingEl {
    data: Vec<El<Zn>>
}

impl<F> RingBase for DoubleRNSTensorRingBase<F>
    where F: FFTTable<Ring = Zn> + PartialEq
{
    type Element = DoubleRNSTensorRingEl;

    fn clone_el(&self, val: &Self::Element) -> Self::Element {
        DoubleRNSTensorRingEl { data: val.data.clone() }
    }

    fn add_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        assert_eq!(self.element_len(), lhs.data.len());
        assert_eq!(self.element_len(), rhs.data.len());
        for (i, Fp) in self.rns_base_iter() {
            for j in self.rank_iter() {
                Fp.add_assign_ref(self.at_mut(lhs, i, j), self.at(rhs, i, j));
            }
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
        assert_eq!(self.element_len(), lhs.data.len());
        for (i, Fp) in self.rns_base_iter() {
            for j in self.rank_iter() {
                Fp.negate_inplace(self.at_mut(lhs, i, j));
            }
        }
    }

    fn mul_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        self.mul_assign_ref(lhs, &rhs);
    }

    fn mul_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        assert_eq!(self.element_len(), lhs.data.len());
        assert_eq!(self.element_len(), rhs.data.len());
        for (i, Fp) in self.rns_base_iter() {
            for j in self.rank_iter() {
                Fp.mul_assign_ref(self.at_mut(lhs, i, j), self.at(rhs, i, j));
            }
        }
    }
    
    fn from_int(&self, value: i32) -> Self::Element {
        self.from(self.base.int_hom().map(value))
    }

    fn zero(&self) -> Self::Element {
        let mut data = Vec::new();
        for (_, Fp) in self.rns_base_iter() {
            for _ in self.rank_iter() {
                data.push(Fp.zero());
            }
        }
        return DoubleRNSTensorRingEl { data };
    }

    fn eq_el(&self, lhs: &Self::Element, rhs: &Self::Element) -> bool {
        assert_eq!(self.element_len(), lhs.data.len());
        assert_eq!(self.element_len(), rhs.data.len());
        for (i, Fp) in self.rns_base_iter() {
            for j in self.rank_iter() {
                if !Fp.eq_el(self.at(lhs, i, j), self.at(rhs, i, j)) {
                    return false;
                }
            }
        }
        return true;
    }
    
    fn is_commutative(&self) -> bool { true }
    fn is_noetherian(&self) -> bool { true }

    fn dbg<'a>(&self, value: &Self::Element, out: &mut std::fmt::Formatter<'a>) -> std::fmt::Result {
        let poly_ring = DensePolyRing::new(&self.base, "ζ");
        poly_ring.get_ring().dbg(&self.poly_repr(&poly_ring, value), out)
    }

    fn square(&self, lhs: &mut Self::Element) {
        assert_eq!(self.element_len(), lhs.data.len());
        for (i, Fp) in self.rns_base_iter() {
            for j in self.rank_iter() {
                Fp.square(self.at_mut(lhs, i, j));
            }
        }
    }

    fn pow_gen<R: IntegerRingStore>(&self, mut x: Self::Element, power: &El<R>, integers: R) -> Self::Element 
        where R::Type: IntegerRing
    {
        timed("DoubleRNSTensorRingBase::pow_gen", || {
            assert_eq!(self.element_len(), x.data.len());
            for (i, Fp) in self.rns_base_iter() {
                for j in self.rank_iter() {
                    *self.at_mut(&mut x, i, j) = Fp.pow_gen(Fp.clone_el(self.at(&x, i, j)), power, &integers);
                }
            }
            return x;
        })
    }
}

impl<F> RingExtension for DoubleRNSTensorRingBase<F>
    where F: FFTTable<Ring = Zn> + PartialEq
{
    type BaseRing = RNSBase;

    fn base_ring<'a>(&'a self) -> &'a Self::BaseRing {
        &self.base
    }

    fn from(&self, x: El<Self::BaseRing>) -> Self::Element {
        let mut result = self.zero();
        for (i, Fp) in self.rns_base_iter() {
            // the FFT of 1 is (1, 1, 1, ...)
            for j in self.rank_iter() {
                *self.at_mut(&mut result, i, j) = Fp.clone_el(self.base_ring().get_ring().get_congruence(&x).at(i));
            }
        }
        return result;
    }
}

impl<F, P: ?Sized> CanHomFrom<P> for DoubleRNSTensorRingBase<F>
    where F: FFTTable<Ring = Zn> + PartialEq,
        P: PolyRing,
        zn_42::ZnBase: CanHomFrom<<P::BaseRing as RingStore>::Type>
{
    type Homomorphism = Vec<<zn_42::ZnBase as CanHomFrom<<P::BaseRing as RingStore>::Type>>::Homomorphism>;

    fn has_canonical_hom(&self, from: &P) -> Option<Self::Homomorphism> {
        self.rns_base_iter().map(|(_, Fp)| Fp.get_ring().has_canonical_hom(from.base_ring().get_ring()).ok_or(())).collect::<Result<Vec<_>, ()>>().ok()
    }

    fn map_in(&self, from: &P, el: <P as RingBase>::Element, hom: &Self::Homomorphism) -> Self::Element {
        self.map_in_ref(from, &el, hom)
    }

    fn map_in_ref(&self, from: &P, el: &P::Element, hom: &Self::Homomorphism) -> Self::Element {
        let mut result = self.zero();
        for (c, j) in from.terms(el) {
            assert!(j < self.rank_iter().len());
            for (i, Fp) in self.rns_base_iter() {
                *self.at_mut(&mut result, i, j) = Fp.get_ring().map_in_ref(from.base_ring().get_ring(), c, hom.at(i));
            }
        }
        self.fft(&mut result.data);
        return result;
    }
}

impl<F> CanHomFrom<DoubleRNSTensorRingBase<F>> for DoubleRNSTensorRingBase<F>
    where F: FFTTable<Ring = Zn> + PartialEq
{
    type Homomorphism = Vec<<zn_42::ZnBase as CanHomFrom<zn_42::ZnBase>>::Homomorphism>;

    fn has_canonical_hom(&self, from: &DoubleRNSTensorRingBase<F>) -> Option<Self::Homomorphism> {
        if self.log2_N == from.log2_N && self.m == from.m && self.rns_base_iter().len() == from.rns_base_iter().len() {
            let mut result = Vec::new();
            for i in 0..self.rns_base_iter().len() {
                // If the ring FFT tables use a different order, this might or might not be a problem.
                // At the moment, it is completely ok to forbid it
                if self.fft_tables.at(i) != from.fft_tables.at(i) {
                    return None;
                }
                result.push(self.base.get_ring().at(i).get_ring().has_canonical_hom(from.base.get_ring().at(i).get_ring())?);
            }
            return Some(result);
        } else {
            None
        }
    }

    fn map_in(&self, from: &DoubleRNSTensorRingBase<F>, el: <DoubleRNSTensorRingBase<F> as RingBase>::Element, hom: &Self::Homomorphism) -> Self::Element {
        let mut result = Vec::new();
        let mut el_data = el.data.into_iter();
        for ((i, to_Fp), (_, from_Fp)) in self.rns_base_iter().zip(from.rns_base_iter()) {
            for j in self.rank_iter() {
                // we already checked that the fft tables have same order, assert here anyway
                debug_assert_eq!(from.fft_tables.at(i).unordered_fft_permutation(j), self.fft_tables.at(i).unordered_fft_permutation(j));
                result.push(to_Fp.get_ring().map_in(from_Fp.get_ring(), el_data.next().unwrap(), hom.at(i)));
            }
        }
        return DoubleRNSTensorRingEl { data: result };
    }
}

impl<F> CanonicalIso<DoubleRNSTensorRingBase<F>> for DoubleRNSTensorRingBase<F>
    where F: FFTTable<Ring = Zn> + PartialEq
{
    type Isomorphism = Vec<<zn_42::ZnBase as CanHomFrom<zn_42::ZnBase>>::Homomorphism>;

    fn has_canonical_iso(&self, from: &DoubleRNSTensorRingBase<F>) -> Option<Self::Homomorphism> {
        from.has_canonical_hom(self)
    }

    fn map_out(&self, from: &DoubleRNSTensorRingBase<F>, el: <DoubleRNSTensorRingBase<F> as RingBase>::Element, iso: &Self::Isomorphism) -> Self::Element {
        from.map_in(self, el, iso)
    }
}

#[cfg(test)]
fn test_ring_7_16() -> DoubleRNSTensorRing<bluestein::FFTTableBluestein<Zn>> {
    DoubleRNSTensorRingBase::create(
        RNSBase::new(vec![Zn::new(113), Zn::new(337)], BigIntRing::RING, default_memory_provider!()),
        7,
        3,
        |Fp, root_of_unity| bluestein::FFTTableBluestein::new(
            Fp.clone(),
            Fp.pow(Fp.clone_el(&root_of_unity), 8),
            Fp.pow(Fp.clone_el(&root_of_unity), 7),
            7,
            4
        ),
        7 * 16,
        |Fp, n| algorithms::unity_root::get_prim_root_of_unity(Fp, n).unwrap()
    )
}

#[cfg(test)]
fn test_ring_7_16_different_modulus() -> DoubleRNSTensorRing<bluestein::FFTTableBluestein<Zn>> {
    // for some reasons, this modulus gives crazy behavior
    DoubleRNSTensorRingBase::create(
        RNSBase::new(vec![Zn::new(7841)], BigIntRing::RING, default_memory_provider!()),
        7,
        3,
        |Fp, root_of_unity| bluestein::FFTTableBluestein::new(
            Fp.clone(),
            Fp.pow(Fp.clone_el(&root_of_unity), 8),
            Fp.pow(Fp.clone_el(&root_of_unity), 7),
            7,
            4
        ),
        7 * 16,
        |Fp, n| algorithms::unity_root::get_prim_root_of_unity(Fp, n).unwrap()
    )
}

#[cfg(test)]
fn edge_case_elements(ring: &DoubleRNSTensorRing<bluestein::FFTTableBluestein<Zn>>) -> impl Iterator<Item = El<DoubleRNSTensorRing<bluestein::FFTTableBluestein<Zn>>>> {
    let mut result = Vec::new();
    for i in (0..16).chain(32..48).step_by(4) {
        result.push(ring.pow(ring.get_ring().gen(), i));
        result.push(ring.negate(ring.pow(ring.get_ring().gen(), i)));
        result.push(ring.add(ring.pow(ring.get_ring().gen(), i), ring.one()));
        result.push(ring.sub(ring.pow(ring.get_ring().gen(), i), ring.one()));
    }
    return result.into_iter();
}

#[test]
fn test_gen() {
    let ring = test_ring_7_16();
    let ring = ring.get_ring();
    let x = ring.gen();

    let mut check = ring.zero();
    for (i, Fp) in ring.rns_base_iter() {
        *ring.at_mut(&mut check, i, 1) = Fp.one();
    }
    ring.fft(&mut check.data);
    for (i, Fp) in ring.rns_base_iter() {
        for j in ring.rank_iter() {
            assert_el_eq!(Fp, ring.at(&check, i, j), ring.at(&x, i, j));
        }
    }
}

#[test]
fn test_reduce_rns_base() {
    let ring = test_ring_7_16();
    let ring2 = ring.get_ring().reduce_rns_base(1);
    let x = ring.get_ring().gen();
    let x2 = ring2.get_ring().gen();

    for (i, Fp) in ring2.get_ring().rns_base_iter() {
        for j in ring2.get_ring().rank_iter() {
            assert!(Fp.get_ring() == ring.get_ring().rns_base().get_ring().at(1).get_ring());
            assert_el_eq!(Fp, x.data.at((i + 1) * ring.get_ring().rank() + j), x2.data.at(i * ring2.get_ring().rank() + j));
        }
    }
}

#[test]
fn test_modulus_switch() {
    let ring = test_ring_7_16();
    let ring2 = ring.get_ring().reduce_rns_base(1);
    let ring = ring.get_ring();
    let ring2 = ring2.get_ring();

    let actual = ring2.modulus_switch(ring, ring.modulus_switch(ring2, ring.gen(), 17), 17);
    assert!((-30..=30).any(|i| ring.eq_el(&ring.mul_int(ring.gen(), 1 + 17 * i), &actual)));

    let actual = ring2.modulus_switch(ring, ring.modulus_switch(ring2, ring.from_int(1 + 113 * 100), 17), 17);
    assert!((-30..=30).any(|i| ring.eq_el(&ring.from_int(1 + 113 * 100 + 17 * i), &actual)));
}

#[test]
fn test_ring_axioms_trivial_m() {
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

#[test]
fn test_ring_axioms_nontrivial_m() {
    let ring = test_ring_7_16();
    let x = ring.get_ring().gen();

    assert!(!ring.is_zero(&x));
    assert_el_eq!(&ring, &ring.negate(ring.one()), &ring.pow(ring.clone_el(&x), 8 * 7));
    assert_el_eq!(&ring, &ring.negate(ring.pow(ring.clone_el(&x), 4)), &ring.mul(ring.pow(ring.clone_el(&x), 28), ring.pow(ring.clone_el(&x), 32)));

    feanor_math::ring::generic_tests::test_ring_axioms(&ring, edge_case_elements(&ring));

    let ring = test_ring_7_16_different_modulus();
    let x = ring.get_ring().gen();

    assert!(!ring.is_zero(&x));
    assert_el_eq!(&ring, &ring.negate(ring.one()), &ring.pow(ring.clone_el(&x), 8 * 7));
    assert_el_eq!(&ring, &ring.negate(ring.pow(ring.clone_el(&x), 4)), &ring.mul(ring.pow(ring.clone_el (&x), 28), ring.pow(ring.clone_el(&x), 32)));

    feanor_math::ring::generic_tests::test_ring_axioms(&ring, edge_case_elements(&ring));
}

#[test]
fn test_tensor_one() {
    let ring = test_ring_7_16();
    let x = ring.get_ring().gen();

    let base_ring = ring.get_ring().power_two_cyclotomic_factor();
    assert_el_eq!(&ring, &ring.one(), &ring.get_ring().tensor_x_one(base_ring.one(), base_ring.get_ring()));
    assert_el_eq!(&ring, &ring.pow(x, 7), &ring.get_ring().tensor_x_one(base_ring.get_ring().gen(), base_ring.get_ring()));
    assert_el_eq!(&ring, &ring.pow(ring.get_ring().gen(), 28), &ring.get_ring().tensor_x_one(base_ring.pow(base_ring.get_ring().gen(), 4), base_ring.get_ring()));

    let ring = test_ring_7_16_different_modulus();
    let x = ring.get_ring().gen();

    let base_ring = ring.get_ring().power_two_cyclotomic_factor();
    assert_el_eq!(&ring, &ring.one(), &ring.get_ring().tensor_x_one(base_ring.one(), base_ring.get_ring()));
    assert_el_eq!(&ring, &ring.pow(x, 7), &ring.get_ring().tensor_x_one(base_ring.get_ring().gen(), base_ring.get_ring()));
    assert_el_eq!(&ring, &ring.pow(ring.get_ring().gen(), 28), &ring.get_ring().tensor_x_one(base_ring.pow(base_ring.get_ring().gen(), 4), base_ring.get_ring()));
}

#[test]
fn test_tensor_decomp() {
    let ring = test_ring_7_16();
    let base_ring = ring.get_ring().power_two_cyclotomic_factor();
    
    let x = ring.get_ring().gen();
    let decomposition = ring.get_ring().tensor_decomp(ring.pow(x, 16), base_ring.get_ring());

    assert_el_eq!(&base_ring, &base_ring.zero(), decomposition.at(0));
    assert_el_eq!(&base_ring, &base_ring.one(), decomposition.at(1));
    assert_el_eq!(&base_ring, &base_ring.zero(), decomposition.at(2));
    assert_el_eq!(&base_ring, &base_ring.zero(), decomposition.at(3));
    assert_el_eq!(&base_ring, &base_ring.zero(), decomposition.at(4));
    assert_el_eq!(&base_ring, &base_ring.zero(), decomposition.at(5));
    assert_el_eq!(&base_ring, &base_ring.zero(), decomposition.at(6));

    let x = ring.get_ring().gen();
    let decomposition = ring.get_ring().tensor_decomp(ring.pow(x, 1), base_ring.get_ring());

    assert_el_eq!(&base_ring, &base_ring.zero(), decomposition.at(0));
    assert_el_eq!(&base_ring, &base_ring.zero(), decomposition.at(1));
    assert_el_eq!(&base_ring, &base_ring.zero(), decomposition.at(2));
    assert_el_eq!(&base_ring, &base_ring.zero(), decomposition.at(3));
    assert_el_eq!(&base_ring, &base_ring.pow(base_ring.get_ring().gen(), 7), decomposition.at(4));
    assert_el_eq!(&base_ring, &base_ring.zero(), decomposition.at(5));
    assert_el_eq!(&base_ring, &base_ring.zero(), decomposition.at(6));

    let x = ring.get_ring().gen();
    for a in edge_case_elements(&ring) {
        let decomposition = ring.get_ring().tensor_decomp(ring.clone_el(&a), base_ring.get_ring());
        let actual = ring.sum(
           Iterator::map(0..ring.get_ring().m, |i| ring.mul(
                    ring.get_ring().tensor_x_one(base_ring.clone_el(decomposition.at(i)), base_ring.get_ring()), 
                    ring.pow(ring.clone_el(&x), 2 * ring.get_ring().N() * i)
                ))
            );
        assert_el_eq!(&ring, &a, &actual);
    }
}

#[test]
fn test_canonical_hom_self() {
    let ring = test_ring_7_16();

    assert!(ring.get_ring().has_canonical_hom(ring.get_ring()).is_some());
    // further checks are done by generic_test_ring_axioms() above

    // if RNS base is different
    let other_ring = DoubleRNSTensorRingBase::create(
        RNSBase::new(vec![Zn::new(337), Zn::new(113)], BigIntRing::RING, default_memory_provider!()),
        7,
        3,
        |Fp, root_of_unity| bluestein::FFTTableBluestein::new(
            Fp.clone(),
            Fp.pow(Fp.clone_el(&root_of_unity), 8),
            Fp.pow(Fp.clone_el(&root_of_unity), 7),
            7,
            4
        ),
        7 * 16,
        |Fp, n| algorithms::unity_root::get_prim_root_of_unity(Fp, n).unwrap()
    );
    assert!(ring.get_ring().has_canonical_hom(other_ring.get_ring()).is_none());
    assert!(other_ring.get_ring().has_canonical_hom(ring.get_ring()).is_none());

    // if log2_N is different
    let other_ring = DoubleRNSTensorRingBase::create(
        RNSBase::new(vec![Zn::new(113), Zn::new(337)], BigIntRing::RING, default_memory_provider!()),
        7,
        2,
        |Fp, root_of_unity| bluestein::FFTTableBluestein::new(
            Fp.clone(),
            Fp.pow(Fp.clone_el(&root_of_unity), 8),
            Fp.pow(Fp.clone_el(&root_of_unity), 7),
            7,
            4
        ),
        7 * 16,
        |Fp, n| algorithms::unity_root::get_prim_root_of_unity(Fp, n).unwrap()
    );
    assert!(ring.get_ring().has_canonical_hom(other_ring.get_ring()).is_none());
    assert!(other_ring.get_ring().has_canonical_hom(ring.get_ring()).is_none());

    // if order of FFT is different
    let unit_in_power_ring = 3; // unit in Z/7Z
    let other_ring = DoubleRNSTensorRingBase::create(
        RNSBase::new(vec![Zn::new(113), Zn::new(337)], BigIntRing::RING, default_memory_provider!()),
        7,
        3,
        |Fp, root_of_unity| bluestein::FFTTableBluestein::new(
            Fp.clone(),
            Fp.pow(Fp.clone_el(&root_of_unity), 8),
            Fp.pow(Fp.clone_el(&root_of_unity), 7),
            7,
            4
        ),
        7 * 16,
        |Fp, n| Fp.pow(algorithms::unity_root::get_prim_root_of_unity(Fp, n).unwrap(), unit_in_power_ring)
    );
    assert!(ring.get_ring().has_canonical_hom(other_ring.get_ring()).is_none());
    assert!(other_ring.get_ring().has_canonical_hom(ring.get_ring()).is_none());

    // if only bluestein helper FFT has different order
    let unit_in_power_ring = 3; // unit in Z/8Z
    let other_ring = DoubleRNSTensorRingBase::create(
        RNSBase::new(vec![Zn::new(113), Zn::new(337)], BigIntRing::RING, default_memory_provider!()),
        7,
        3,
        |Fp, root_of_unity| bluestein::FFTTableBluestein::new(
            Fp.clone(),
            Fp.pow(Fp.clone_el(&root_of_unity), 8),
            Fp.pow(Fp.clone_el(&root_of_unity), 7 * unit_in_power_ring),
            7,
            4
        ),
        7 * 16,
        |Fp, n| algorithms::unity_root::get_prim_root_of_unity(Fp, n).unwrap()
    );
    assert!(ring.get_ring().has_canonical_hom(other_ring.get_ring()).is_some());
    assert!(other_ring.get_ring().has_canonical_hom(ring.get_ring()).is_some());
    feanor_math::ring::generic_tests::test_hom_axioms(&other_ring, &ring, edge_case_elements(&other_ring));
    feanor_math::ring::generic_tests::test_hom_axioms(&ring, &other_ring, edge_case_elements(&ring));
    assert_el_eq!(&ring, &ring.get_ring().gen(), &ring.coerce(&other_ring, other_ring.get_ring().gen()));
}