
//
// Temporarily copied over from bfv-rust, until that becomes a
// proper library
// 

use feanor_math::homomorphism::*;
use feanor_math::integer::*;
use feanor_math::primitive_int::*;
use feanor_math::mempool::*;
use feanor_math::vector::{VectorView, VectorViewMut};
use feanor_math::ordered::OrderedRingStore;
use feanor_math::rings::zn::{ZnRingStore, ZnRing};
use feanor_math::divisibility::DivisibilityRingStore;
use feanor_math::ring::*;

use super::RNSOperation;

///
/// Stores values for an almost exact conversion between RNS bases.
/// A complete conversion refers to the function
/// ```text
/// Z/QZ -> Z/Q'Z, x -> [lift(x)]
/// ```
/// In our case, the output of the function is allowed to have an error of `{ -Q, 0, Q }`,
/// unless the shortest lift of the input is bounded by `Q/4`, in which case the result
/// is always correct.
/// 
/// # Implementation
/// 
/// The implementation is based on eprint.iacr.org/2016/510.pdf.
/// 
/// The basic idea is to perform a fast base conversion to `{ to_summands, m }`
/// with a helper prime m. Furthermore, we can do this in a way to get the result
/// in montgomery form, i.e. get `x_q m` instead of `x_q`. Hence, we end up with y
/// w.r.t. `{ to_summands, m }` such that
/// ```text
/// y in [shortest_lift(x)]_Q' m + Q { -k/2, -k/2 + 1, ..., k/2 }
/// ```
/// From this, we can now perform a montgomery reduction to find `[shortest_lift(x)]_Q'`
/// 
pub struct AlmostExactBaseConversion<V_from, V_to, R, R_intermediate, M_Zn, M_Int>
    where V_from: VectorView<R>, V_to: VectorView<R>,
        R: ZnRingStore, R_intermediate: ZnRingStore<Type = R::Type>,
        R::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        M_Zn: MemoryProvider<El<R>>,
        M_Int: MemoryProvider<El<<R::Type as ZnRing>::Integers>>
{
    from_summands: V_from,
    to_summands: V_to,
    // F_m for the helper factor m
    Zm: R_intermediate,
    /// the values `m q/Q mod q` for each RNS factor q dividing Q
    qm_over_Q: Vec<El<R>>,
    /// the values `Q/q mod q'` for each RNS factor q dividing Q and q' dividing Q'
    Q_over_q: (Vec<El<R>>, Vec<El<R_intermediate>>),
    // the value `Q^-1 mod m`
    Q_inv_Zm: El<R_intermediate>,
    // the value `Q mod q'` for q' dividing Q'
    Q_Zq: Vec<El<R>>,
    // the value `m^-1 mod q'` for q' dividing Q'
    m_inv_Zq: Vec<El<R>>,
    memory_provider_int: M_Int,
    memory_provider_zn: M_Zn
}

const ZZbig: BigIntRing = BigIntRing::RING;

fn inner_product<V_lhs, V_rhs, R_to, I>(lhs: V_lhs, rhs: V_rhs, ring: R_to, ZZ: &I) -> El<R_to>
    where 
        I: IntegerRingStore,
        I::Type: IntegerRing,
        R_to: ZnRingStore,
        R_to::Type: ZnRing + CanHomFrom<I::Type>,
        V_lhs: VectorView<El<I>>,
        V_rhs: VectorView<El<R_to>>
{
    debug_assert_eq!(lhs.len(), rhs.len());
    let hom = ring.can_hom(&ZZ).unwrap();
    <_ as RingStore>::sum(&ring,
        lhs.iter()
            .map(|x| hom.map_ref(x))
            .zip(rhs.iter())
            .map(|(x, y)| ring.mul_ref_snd(x, y))
    )
}

impl<V_from, V_to, R, R_intermediate, M_Zn, M_Int> AlmostExactBaseConversion<V_from, V_to, R, R_intermediate, M_Zn, M_Int> 
    where V_from: VectorView<R>, V_to: VectorView<R>,
        R: ZnRingStore, R_intermediate: ZnRingStore<Type = R::Type>,
        R::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        M_Zn: MemoryProvider<El<R>>,
        M_Int: MemoryProvider<El<<R::Type as ZnRing>::Integers>>
{
    pub fn new(from_summands: V_from, to_summands: V_to, Zm: R_intermediate, memory_provider_int: M_Int, memory_provider_zn: M_Zn) -> Self {
        assert!(from_summands.len() > 0);
        assert!(to_summands.len() > 0);
        // we need `2 * ` such that the almost-exact conversion is an exact conversion in the case of small input
        assert!(Zm.integer_ring().is_gt(Zm.modulus(), &int_cast(2 * from_summands.len() as i64, Zm.integer_ring(), &StaticRing::<i64>::RING)));
        
        let ZZ = Zm.integer_ring();
        // we want to use integers without casts, so check ring equality here
        for Zk in from_summands.iter() {
            assert!(Zk.integer_ring().get_ring() == ZZ.get_ring());
        }
        for Zk in to_summands.iter() {
            assert!(Zk.integer_ring().get_ring() == ZZ.get_ring());
        }

        let m = int_cast(Zm.integer_ring().clone_el(Zm.modulus()), &ZZ, Zm.integer_ring());
        let from_moduli = from_summands.iter().map(|R| ZZ.clone_el(R.modulus()));
        let Q = ZZbig.prod(from_moduli.clone().map(|n| int_cast(n, &ZZbig, ZZ)));
        let q_over_Q = from_summands.iter()
            .zip(from_moduli.clone())
            .map(|(R, n)| R.invert(&R.coerce(&ZZbig, ZZbig.checked_div(&Q, &int_cast(n, &ZZbig, ZZ)).unwrap())).unwrap())
            .collect::<Vec<_>>();
        let qm_over_Q = from_summands.iter()
            .zip(q_over_Q.iter())
            .map(|(R, x)| R.mul_ref_fst(x, R.coerce(ZZ, ZZ.clone_el(&m))))
            .collect::<Vec<_>>();
        let Q_over_q = (
            to_summands.iter()
                .flat_map(|to| from_moduli.clone()
                    .map(|n| ZZbig.checked_div(&Q, &int_cast(n, &ZZbig, ZZ)).unwrap())
                    .map(|x| to.coerce(&ZZbig, x))
                )
                .collect::<Vec<_>>(),
            from_moduli.clone()
                .map(|n| ZZbig.checked_div(&Q, &int_cast(n, &ZZbig, ZZ)).unwrap())
                .map(|x| Zm.coerce(&ZZbig, x))
                .collect::<Vec<_>>()
        );
        let Q_inv_Zm = Zm.invert(&Zm.coerce(&ZZbig, ZZbig.clone_el(&Q))).unwrap();
        let Q_Zq = to_summands.iter().map(|R| R.coerce(&ZZbig, ZZbig.clone_el(&Q))).collect();
        let m_inv_Zq = to_summands.iter().map(|R| R.invert(&R.coerce(R.integer_ring(), R.integer_ring().coerce(&ZZ, ZZ.clone_el(&m)))).unwrap()).collect();

        AlmostExactBaseConversion { from_summands, to_summands, Zm, qm_over_Q, Q_over_q, Q_inv_Zm, Q_Zq, m_inv_Zq, memory_provider_int, memory_provider_zn }
    }

    ///
    /// Performs a variant of the fast base conversion to the intermediate 
    /// RNS base `{ to_summands, m }`. The only difference is that the result
    /// will be in Montgomery form, i.e. is
    /// ```text
    /// shortest_lift(x m mod Q) + Q { -k/2, -k/2 + 1, ..., k/2 }
    /// ```
    /// modulo `{ to_summands, m }`.
    /// 
    #[inline(never)]
    fn fast_convert_assign_montgomery<V, W>(&self, mut input: V, mut target: W, target_Zm: &mut El<R_intermediate>)
        where V: ExactSizeIterator<Item = El<R>>, W: VectorViewMut<El<R>>
    {
        debug_assert_eq!(input.len(), self.from_summands.len());
        debug_assert_eq!(self.qm_over_Q.len(), self.from_summands.len());
        debug_assert!(target.len() == self.to_summands.len());
        debug_assert!(self.Q_over_q.0.len() == self.to_summands.len() * self.from_summands.len());
        debug_assert!(self.Q_over_q.1.len() == self.from_summands.len());

        // this is the same for all rings, as checked in [`Self::new()`]
        let ZZ = self.Zm.integer_ring();

        let el_qm_over_Q = self.memory_provider_int.get_new_init(self.from_summands.len(), |i| 
            self.from_summands.at(i).smallest_lift(self.from_summands.at(i).mul_ref_snd(input.next().unwrap(), self.qm_over_Q.at(i)))
        );

        for i in 0..self.to_summands.len() {
            let to = self.to_summands.at(i);
            *target.at_mut(i) = inner_product(&*el_qm_over_Q, &self.Q_over_q.0[(i * self.from_summands.len())..((i + 1) * self.from_summands.len())], to, ZZ);
        }
        *target_Zm = inner_product(&*el_qm_over_Q, &self.Q_over_q.1, &self.Zm, ZZ);
    }

    ///
    /// Performs the reduction mod q of the output of the intermediate fast base conversion
    ///
    #[inline(never)]
    fn reduce_mod_q_inplace<W>(&self, mut target: W, helper: El<R_intermediate>)
        where W: VectorViewMut<El<R>>
    {
        debug_assert!(target.len() == self.to_summands.len());

        // Lemma 4 in
        // "A Full RNS Variant of FV like Somewhat Homomorphic Encryption Schemes",
        // Jean-Claude Bajard, Julien Eynard, Anwar Hasan, and Vincent Zucca,
        // eprint.iacr.org/2016/510.pdf
        //
        // The input is `x = [cm]_q + uq`, where c is the smallest representative mod q
        // and `|u| <= #Q/2`;
        // Write `[cm]_q = cm + wq`, so `|w| < m/2`. Now we have
        //```text
        // [cm]_q + uq = cm + (u + w)q
        //```
        // and hence find `u + w = x q^-1 mod m`. This allows us to determine `u + w` up to `+/- 1`,
        // as `|u + w| <= m/2 + #Q/2 <= m`, assuming that `m > #Q`.
        //
        // Furthermore, if `c <= q/4` then `[cm]_q = cm + wq` for `|w| <= m/4`. Thus `|u + w| <= m/2`, assuming
        // that `m >= 2#Q`.

        let ZZ = self.Zm.integer_ring();
        let r = self.Zm.smallest_lift(self.Zm.mul_ref_snd(helper, &self.Q_inv_Zm));
        for i in 0..self.to_summands.len() {
            let diff = self.to_summands.at(i).mul_ref_fst(&self.Q_Zq[i], self.to_summands.at(i).coerce(ZZ, ZZ.clone_el(&r)));
            self.to_summands.at(i).sub_assign(target.at_mut(i), diff);
            self.to_summands.at(i).mul_assign_ref(target.at_mut(i), &self.m_inv_Zq[i]);
        }
    }
}


impl<V_from, V_to, R, R_intermediate, M_Zn, M_Int> RNSOperation for AlmostExactBaseConversion<V_from, V_to, R, R_intermediate, M_Zn, M_Int> 
    where V_from: VectorView<R>, V_to: VectorView<R>,
        R: ZnRingStore, R_intermediate: ZnRingStore<Type = R::Type>,
        R::Type: ZnRing + CanHomFrom<BigIntRingBase>,
        M_Zn: MemoryProvider<El<R>>,
        M_Int: MemoryProvider<<<<R as RingStore>::Type as ZnRing>::IntegerRingBase as RingBase>::Element>
{
    type Ring = R;
    type RingType = R::Type;
    type InRings<'a> = &'a V_from
        where Self: 'a;
    type OutRings<'a> = &'a V_to
        where Self: 'a;
    type Result<'a> = M_Zn::Object
        where Self: 'a;

    fn input_rings<'a>(&'a self) -> Self::InRings<'a> {
        &self.from_summands
    }

    fn output_rings<'a>(&'a self) -> Self::OutRings<'a> {
        &self.to_summands
    }

    ///
    /// Performs the (almost) exact RNS base conversion
    /// ```text
    ///     Z/QZ -> Z/Q'Z, x -> smallest_lift(x) + kQ mod Q''
    /// ```
    /// where `k in { -1, 0, 1 }`.
    /// 
    /// Furthermore, if the shortest lift of the input is bounded by `Q/4`,
    /// then the result is guaranteed to be exact.
    /// 
    fn apply<'a, V>(&'a self, el: V) -> Self::Result<'a>
        where V: ExactSizeIterator<Item = El<R>>
    {
        let mut result = self.memory_provider_zn.get_new_init(self.output_rings().len(), |i| self.output_rings().at(i).zero());
        let mut helper = self.Zm.zero();
        self.fast_convert_assign_montgomery(el, &mut result, &mut helper);
        self.reduce_mod_q_inplace(&mut result, helper);
        return result;
    }

}

#[cfg(test)]
use feanor_math::rings::zn::zn_42::*;
#[cfg(test)]
use feanor_math::assert_el_eq;
#[cfg(test)]
use feanor_math::default_memory_provider;

#[test]
fn test_rns_base_conversion() {
    let from = vec![Zn::new(17), Zn::new(97)];
    let to = vec![Zn::new(17), Zn::new(97), Zn::new(113), Zn::new(257)];

    let table = AlmostExactBaseConversion::new(from.clone(), to.clone(), Zn::new(65537), default_memory_provider!(), default_memory_provider!());

    for k in -412..=412 {
        let input = from.iter().map(|R| R.int_hom().map(k)).collect::<Vec<_>>();
        let output = to.iter().map(|R| R.int_hom().map(k)).collect::<Vec<_>>();

        let actual = table.apply(input.into_iter());

        for j in 0..to.len() {
            assert_el_eq!(to.at(j), output.at(j), actual.at(j));
        }
    }
}

#[test]
fn test_rns_base_conversion_small_helper() {
    let from = vec![Zn::new(97), Zn::new(3)];
    let to = vec![Zn::new(17)];
    let table = AlmostExactBaseConversion::new(from.clone(), to.clone(), Zn::new(5), default_memory_provider!(), default_memory_provider!());
    
    for k in 0..291 {
        let target = table.apply([from[0].int_hom().map(k), from[1].int_hom().map(k)].into_iter());
        assert!(
            to[0].eq_el(&to[0].int_hom().map(k), target.at(0)) ||
            to[0].eq_el(&to[0].int_hom().map(k + 97 * 3), target.at(0)) ||
            to[0].eq_el(&to[0].int_hom().map(k - 97 * 3), target.at(0))
        );
    }
}

#[test]
fn test_rns_base_conversion_not_coprime() {
    let from = vec![Zn::new(17), Zn::new(97), Zn::new(113)];
    let to = vec![Zn::new(17), Zn::new(97), Zn::new(113), Zn::new(257)];
    let table = AlmostExactBaseConversion::new(from.clone(), to.clone(), Zn::new(65537), default_memory_provider!(), default_memory_provider!());

    for k in &[0, 1, 2, 17, 97, 113, 17 * 113, 18, 98, 114] {
        let x = from.iter().map(|R| R.int_hom().map(*k)).collect::<Vec<_>>();
        let y = to.iter().map(|R| R.int_hom().map(*k)).collect::<Vec<_>>();
        let result = table.apply((&x[..]).iter().cloned());
        
        for i in 0..y.len() {
            assert!(to[i].eq_el(&y[i], result.at(i)));
        }
    }
}

#[test]
fn test_rns_base_conversion_coprime() {
    let from = vec![Zn::new(17), Zn::new(97), Zn::new(113)];
    let to = vec![Zn::new(19), Zn::new(23), Zn::new(257)];
    let table = AlmostExactBaseConversion::new(from.iter().cloned().collect::<Vec<_>>(), to.iter().cloned().collect::<Vec<_>>(), Zn::new(65537), default_memory_provider!(), default_memory_provider!());

    for k in &[0, 1, 2, 17, 97, 113, 17 * 113, 18, 98, 114] {
        let x = from.iter().map(|R| R.int_hom().map(*k)).collect::<Vec<_>>();
        let y = to.iter().map(|R| R.int_hom().map(*k)).collect::<Vec<_>>();
        let result = table.apply((&x[..]).iter().cloned());
        
        for i in 0..y.len() {
            assert!(to[i].eq_el(&y[i], result.at(i)));
        }
    }
}
