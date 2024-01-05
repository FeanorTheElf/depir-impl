#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
#![feature(iter_advance_by)]
#![feature(cell_update)]
#![feature(core_intrinsics)]
#![feature(generic_const_exprs)]

use feanor_math::rings::zn::ZnRingStore;
use feanor_math::rings::poly::dense_poly::DensePolyRing;
use feanor_math::rings::field::AsField;
use feanor_math::vector::*;
use feanor_math::vector::vec_fn::{VectorFn, IntoVectorFn};
use feanor_math::homomorphism::*;

use rand::{SeedableRng, RngCore, CryptoRng};
use rings::cyclotomic_cc::Pow2CyclotomicRing;
use rings::cyclotomic_rns::RNSPow2CyclotomicRing;
use rings::tensor::RNSCyclotomicTensorRing;

use std::{fs::File, io::{BufWriter, Write, BufReader, Read}, borrow::Borrow, sync::Mutex};

use globals::*;
use params::*;
use evaluation::Evaluator;
use interpolate::InterpolationMatrix;

extern crate base64;
extern crate feanor_math;
extern crate rand;
extern crate crossbeam;
extern crate windows_sys;
extern crate append_only_vec;

pub mod globals;

pub mod params;

///
/// For computing the "total degree interpolation polynomial", i.e. the
/// multivariate polynomial of total degree D that interpolates binomial(D + m, m)
/// points.
/// 
pub mod interpolate;

pub mod iters;

///
/// Implementations of the rings over which we define the ASHE scheme, in particular
/// `Zq[X]/(Phi_2N(X))` and `Zq[X, Y]/(Phi_2N(X), Y^k - 1)`
/// 
pub mod rings;

///
/// For the datastructure-based evaluation of a polynomial `f(X1, ..., Xm)` on points
/// in `Zp`
/// 
pub mod evaluation;

///
/// For BGV-style modulus switching in the ASHE scheme.
/// 
pub mod rnsconv;

type PlaintextRing = Pow2CyclotomicRing;

///
/// The ciphertext ring of the ASHE scheme, namely `Zq[X, Y]/(Phi_2N(X), Y^k - 1)`
/// 
type CiphertextRing = RNSCyclotomicTensorRing;

///
/// The cyclotomic ring the ASHE scheme is based on, namely `Zq[X]/(Phi_2N(X))`
/// 
type MainRing = RNSPow2CyclotomicRing;
 
type CiphertextSeed = u64;
type CompressedEl = u32;

///
/// The first component is the concatenation of the double-RNS representation
/// components of `c10, ..., cm0`. The second component are the `m` seeds used
/// to generate `c11, ..., cm1`. Here `(ci0, ci1)` is the ciphertext encrypting
/// the value for the `i`-th variable (note that `m = variable_number`). 
/// 
type Query = (Vec<CompressedEl>, Vec<CiphertextSeed>);

type Reply = Vec<CompressedEl>;

static DEBUG_SK: Mutex<Option<El<MainRing>>> = Mutex::new(None);

#[allow(unused)]
fn debug_decrypt_println<I>(ciphertext_ring: &MainRing, ct: I, plaintext_mod: i64) 
    where I: DoubleEndedIterator<Item = El<MainRing>>
{
    let noisy_result = evaluate_poly_naive(
        &ciphertext_ring, 
        ct.into_iter(), 
        DEBUG_SK.lock().unwrap().as_ref().unwrap()
    );

    let poly_ring = DensePolyRing::new(ZZbig, "X");
    let result = ciphertext_ring.get_ring().smallest_lift(&poly_ring, &noisy_result);
    let plaintext_poly_ring = DensePolyRing::new(Zn::new(plaintext_mod as u64), "X");
    plaintext_poly_ring.println(&plaintext_poly_ring.coerce(&poly_ring, result));
}

pub struct Server<'a> {
    params: DEPIRParams,
    point_grid: InterpolationMatrix<Zn, Vec<El<Zn>>>,
    db_polynomial: Option<Vec<Int>>,
    scalar_ring: AsField<Zn>,
    ciphertext_ring: CiphertextRing,
    preevaluations: Option<Evaluator<'a>>,
    reply_ciphertext_ring: MainRing
}

fn unpack_ring_element(ciphertext_ring: &MainRing, element_seed: CiphertextSeed) -> El<MainRing> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(element_seed);
    return ciphertext_ring.get_ring().sample_uniform(|| rng.next_u64());
}

impl<'a> Server<'a> {

    #[inline(never)]
    pub fn create(params: DEPIRParams, top_level_primes: &[TopLevelModulus]) -> Self {
        params.validate(top_level_primes);
        Server {
            params: params,
            point_grid: params.create_point_grid(),
            scalar_ring: params.create_scalar_plaintext_ring().as_field().ok().unwrap(),
            ciphertext_ring: params.create_ciphertext_ring(top_level_primes),
            db_polynomial: None,
            preevaluations: None,
            reply_ciphertext_ring: params.create_reply_ciphertext_ring(top_level_primes)
        }
    }

    pub fn scalar_ring(&self) -> &AsField<Zn> {
        &self.scalar_ring
    }

    pub fn set_db_int<V>(&mut self, db: V)
        where V: VectorView<Int>
    {
        self.db_polynomial = Some(self.get_db_polynomial(db.into_fn().map(|x| self.scalar_ring().base_ring().int_hom().map(x as i32)), self.scalar_ring().base_ring()));
    }

    pub fn set_db<V>(&mut self, db: V, db_ring: &Zn) 
        where V: VectorFn<El<Zn>>
    {
        self.db_polynomial = Some(self.get_db_polynomial(db, db_ring));
    }

    #[inline(never)]
    fn get_db_polynomial<V>(&self, db: V, db_ring: &Zn) -> Vec<Int>
        where V: VectorFn<El<Zn>>
    {
        timed("Server::set_db", || {
            let hom = self.scalar_ring().can_hom(db_ring).unwrap();
            let d = self.params.d();

            assert!(db.len() <= self.point_grid.size(d));
            let mut result = Vec::new();
            result.extend(Iterator::map(0..db.len(), |i| hom.map(db.at(i))));
            result.extend(Iterator::map(0..(self.point_grid.size(d) - db.len()), |_| self.scalar_ring().zero()));

            self.point_grid.solve_inplace(&mut result[..], d, &self.scalar_ring().inclusion());

            return result.into_iter().map(|x| self.scalar_ring().smallest_lift(x)).inspect(|x| assert!(x.abs() <= *self.scalar_ring.modulus() / 2)).collect();
        })
    }

    #[inline(never)]
    pub fn load_db(&mut self, identifier: &str) {
        timed("Server::load_db", || {
            let file = File::open(format!("{}poly_{}", PATH, identifier)).unwrap();
            let mut poly = Vec::new();
            let mut reader = BufReader::new(file);
            let mut value = [0; std::mem::size_of::<Int>()];
            for _ in 0..self.params.N() {
                reader.read_exact(&mut value[..]).unwrap();
                poly.push(readwrite::read_Int(&mut value[..]));
            }
            assert!(poly.iter().copied().all(|x| x.abs() <= *self.scalar_ring.modulus() / 2));
            assert!(reader.bytes().next().is_none());
            self.db_polynomial = Some(poly);
        })
    }

    ///
    /// Takes the `m` ciphertexts stored in query (in packed form), unpacks them, maps
    /// them into the total ciphertext ring `Z[X,Y]/(X^n + 1, Y^k - 1)`, and groups the
    /// FFT coefficients by prime ring.
    /// 
    /// In particular, the output are `top_level_primes.len()` many vectors, each containing
    /// the FFT coefficients of all inputs w.r.t. the prime `top_level_primes[i]`.
    /// 
    #[inline(never)]
    pub fn unpack_query(&self, (c0_data, c1_seeds): Query, top_level_primes: &[TopLevelModulus]) -> Vec<Vec<El<Zn>>> {
        timed("Server::unpack_query", || {
            let N = self.ciphertext_ring.base_ring().get_ring().N();
            let m = self.params.m();
            let rank = self.ciphertext_ring.get_ring().rank();
            assert_eq!(c1_seeds.len(), m);
            assert_eq!(c0_data.len(), N * self.params.ciphertext_moduli_count() * m);
    
            let mut result = Vec::new();
            result.resize_with(top_level_primes.len(), || {
                let mut res = Vec::new();
                res.resize(rank * m, dummy());
                return res;
            });
    
            let segment_len = N * self.params.ciphertext_moduli_count();
            for k in 0..m {
                let c0 = self.ciphertext_ring.base_ring().get_ring().uncompress((&c0_data[(k * segment_len)..((k + 1) * segment_len)]).into_fn().map(|x| x as u64), top_level_primes.into_fn());
                let c1 = unpack_ring_element(self.ciphertext_ring.base_ring(), c1_seeds[k]);
                
                let ciphertext = self.ciphertext_ring.add(self.ciphertext_ring.inclusion().map(c0), self.ciphertext_ring.mul_ref_snd(self.ciphertext_ring.inclusion().map(c1), self.ciphertext_ring.get_ring().m_th_root_of_unity()));
                let data = self.ciphertext_ring.get_ring().compress(&ciphertext, self.ciphertext_ring.get_ring().rns_base().get_ring());
                for (i, x) in data.enumerate() {
                    result[i / rank][(i % rank) * m + k] = x;
                }
            }
            return result;
        })
    }

    #[inline(never)]
    pub fn pack_reply<V>(&self, reply: V, top_level_primes: &[TopLevelModulus]) -> Reply
        where V: ExactSizeIterator<Item = El<MainRing>>
    {
        let relin_moduli = &top_level_primes[(top_level_primes.len() - self.params.reply_ciphertext_moduli_count())..];
        let mut result = Vec::with_capacity(reply.len() * self.ciphertext_ring.get_ring().N() * self.params.reply_ciphertext_moduli_count());
        for x in reply {
            let mod_switched = self.ciphertext_ring.base_ring().get_ring().modulus_switch(self.reply_ciphertext_ring.get_ring(), x, *self.scalar_ring.modulus());
            let components = self.reply_ciphertext_ring.get_ring().compress(&mod_switched,&relin_moduli);
            assert_eq!(components.len(), self.params.reply_ciphertext_moduli_count() * self.reply_ciphertext_ring.get_ring().rank());
            result.extend(components.into_iter().map(|x| <_ as TryInto<CompressedEl>>::try_into(x).unwrap()));
        }
        return result;
    }

    #[inline(never)]
    pub fn evaluate_query(&self, el: Query, top_level_primes: &[TopLevelModulus]) -> Reply {
        let result_el = timed("Server::evaluate_query", || {
            
            let preevaluation_db = self.preevaluations.as_ref().unwrap();
            let rank = self.ciphertext_ring.get_ring().rank();
            let mut result = Vec::with_capacity(rank * self.params.ciphertext_moduli_count());
            let input = self.unpack_query(el, top_level_primes);

            crossbeam::scope::<_, ()>(|s| {
                let mut handles: Vec<crossbeam::thread::ScopedJoinHandle<'_, Vec<_>>> = Vec::new();
                for i in 0..self.params.ciphertext_moduli_count() {
                    let input_slice = &input[i];
                    handles.push(s.spawn(move |_| {
                        let mut result = Vec::new();
                        result.resize(rank, top_level_primes[0].zero());
                        preevaluation_db.evaluate_many(i, input_slice, &mut result[..]);
                        result
                    }));
                }
                for handle in handles.into_iter() {
                    result.extend(handle.join().unwrap().into_iter());
                }
            }).unwrap();

            self.ciphertext_ring.get_ring().uncompress(result.into_fn(), top_level_primes.into_fn())
        });
        timed("Server::relin_pack_reply", || {
            self.pack_reply(
                self.ciphertext_ring.get_ring().wrt_standard_basis(self.ciphertext_ring.clone_el(&result_el)).into_iter().take(self.params.d() + 1), 
                top_level_primes
            )
        })
    }

    #[inline(never)]
    pub fn preprocess_db(&mut self, primes: &'a AllModuli) {
        assert_eq!(primes.level_2_primes.len(), self.params.ciphertext_moduli_count());

        timed("Server::preprocess_db", || {
            self.preevaluations = Some(Evaluator::new(
                primes,
                self.params.poly_params(), 
                &self.db_polynomial.as_ref().unwrap()[..]
            ))
        })
    }

    #[inline(never)]
    pub fn save_db(&self) {
        timed("Server::save_poly", || {
            let file = File::create(format!("{}poly_{}", PATH, poly_hash(&self.db_polynomial.as_ref().unwrap()[..], ZZ))).unwrap();
            let mut writer = BufWriter::new(file);
            for c in &self.db_polynomial.as_ref().unwrap()[..] {
                writer.write(readwrite::write_Int(*c).borrow()).unwrap();
            }
        })
    }
}

// used for debugging/testing purposes
fn evaluate_poly_naive<R: RingStore, I>(ring: &R, poly: I, point: &El<R>) -> El<R>
    where I: DoubleEndedIterator<Item = El<R>>
{
    let mut current = ring.zero();
    let mut current_pow = ring.one();
    for c in poly {
        ring.add_assign(&mut current, ring.mul_ref_snd(c, &current_pow));
        ring.mul_assign_ref(&mut current_pow, point);
    }
    return current;
}

pub struct Client {
    params: DEPIRParams,
    point_grid: InterpolationMatrix<Zn, Vec<El<Zn>>>,
    plaintext: PlaintextRing,
    ciphertext_ring: CiphertextRing,
    reply_ciphertext_ring: MainRing,
    sk: Option<El<MainRing>>
}

impl Client {

    #[inline(never)]
    pub fn create(params: DEPIRParams, top_level_primes: &[TopLevelModulus]) -> Self {
        timed("Client::create", || {
            params.validate(top_level_primes);

            return Client {
                params: params,
                ciphertext_ring: params.create_ciphertext_ring(top_level_primes),
                plaintext: params.create_plaintext_ring(),
                point_grid: params.create_point_grid(),
                reply_ciphertext_ring: params.create_reply_ciphertext_ring(top_level_primes),
                sk: None
            }
        })
    }

    #[inline(never)]
    pub fn keygen<G: RngCore + CryptoRng>(&mut self, rng: &mut G) {
        timed("Client::keygen", || {
            self.sk = Some(self.ciphertext_ring.base_ring().get_ring().sample_ternary(|| rng.next_u64()));
            *DEBUG_SK.lock().unwrap() = Some(self.ciphertext_ring.base_ring().clone_el(self.sk.as_ref().unwrap()))
        })
    }

    #[inline(never)]
    pub fn enc<G: RngCore + CryptoRng>(&self, data: &El<PlaintextRing>, rng: &mut G) -> (El<MainRing>, u64) {
        timed("Client::enc", || {
            let R = self.ciphertext_ring.base_ring();
            let a_seed = rng.next_u64();
            let a = unpack_ring_element(&R, a_seed);
            let mut b = R.mul_ref(&a, &self.sk.as_ref().expect("Need to call keygen() before enc()"));
            R.negate_inplace(&mut b);
            let mut e = R.get_ring().sample_binomial4(|| rng.next_u64());
            R.int_hom().mul_assign_map(&mut e, *self.plaintext.get_ring().modulus() as i32);
            R.add_assign(&mut b, e);

            let plaintext_poly_ring = DensePolyRing::new(ZZ, "X");
            let m = R.coerce(&plaintext_poly_ring, self.plaintext.get_ring().smallest_lift(&plaintext_poly_ring, data));

            // this is now a BGV encryption of m
            let c0 = R.add(m, b);
            return (c0, a_seed);
        })
    }

    #[inline(never)]
    pub fn dec(&self, ct: Vec<El<MainRing>>) -> El<PlaintextRing> {
        timed("Client::dec", || {

            // we do not want to choose the ciphertext modulus `q` and the reply ciphertext modulus `q'` to be `q = q' mod t`,
            // hence we cannot decrypt in `self.reply_ciphertext_ring`
            let noisy_result = evaluate_poly_naive(
                self.ciphertext_ring.base_ring(), 
                ct.into_iter().map(|x| self.reply_ciphertext_ring.get_ring().modulus_switch(self.ciphertext_ring.base_ring().get_ring(), x, *self.plaintext.base_ring().modulus())), 
                self.sk.as_ref().unwrap()
            );

            let plaintext_poly_ring = DensePolyRing::new(ZZbig, "X");
            let result = self.ciphertext_ring.base_ring().get_ring().smallest_lift(&plaintext_poly_ring, &noisy_result);
            return self.plaintext.coerce(&plaintext_poly_ring, result);
        })
    }
    
    #[inline(never)]
    pub fn query_single<G: RngCore + CryptoRng>(&self, index: usize, rng: &mut G, top_level_primes: &[TopLevelModulus]) -> Query {
        timed("Client::query_single", || {
            let point = self.point_grid.point_at_index(self.params.d(), index);
            let m = self.params.m();
            let mut result_data: Vec<u32> = Vec::new();
            let mut result_seeds = Vec::new();
            for k in 0..m {
                let (c0, c1_seed) = self.enc(&self.plaintext.inclusion().map(point[k]), rng);
                result_seeds.push(c1_seed);
                result_data.extend(
                    self.ciphertext_ring.base_ring().get_ring().compress(&c0, &top_level_primes)
                        .map(|x| <_ as TryInto<CompressedEl>>::try_into(x).unwrap())
                );
            }
            return (result_data, result_seeds);
        })
    }

    #[inline(never)]
    pub fn unpack_reply(&self, reply: Reply, top_level_primes: &[TopLevelModulus]) -> Vec<El<MainRing>> {
        let reply_segment_len = self.params.reply_ciphertext_moduli_count() * self.reply_ciphertext_ring.get_ring().rank();
        let reply_moduli = &top_level_primes[(top_level_primes.len() - self.params.reply_ciphertext_moduli_count())..];
        let reply_len = reply.len() / reply_segment_len;
        assert_eq!(reply.len(), reply_segment_len * reply_len);
        let mut result = Vec::with_capacity(2);
        for i in 0..reply_len {
            result.push(self.reply_ciphertext_ring.get_ring().uncompress(
                (&reply[(i * reply_segment_len)..((i + 1) * reply_segment_len)]).into_fn().map(|x| x as u64), 
                reply_moduli.into_fn()
            ));
        }
        return result;
    }

    #[inline(never)]
    pub fn reply_single(&self, response: Reply, top_level_primes: &[TopLevelModulus]) -> El<PlaintextRing> {
        timed("Client::response_single", || {
            return self.dec(self.unpack_reply(response, top_level_primes));
        })
    }
}

#[allow(unused)]
fn example_run() {
    let params = SSD_PERF_PARAMS;
    let primes = params.create_primes();
    let moduli = AllModuli::new(primes);
    params.print(&moduli);

    let mut server = Server::create(params, &moduli.level_2_primes);
    server.set_db_int([0, 1, 2, 3, 4, 5, 6, 7]);
    server.save_db();
    server.preprocess_db(&moduli);
    
    let mut client = Client::create(params, &moduli.level_2_primes);
    client.keygen(&mut rand::thread_rng());
    
    let query = client.query_single(2, &mut rand::thread_rng(), &moduli.level_2_primes);
    let reply = server.evaluate_query(query, &moduli.level_2_primes);
    client.plaintext.println(&client.reply_single(reply, &moduli.level_2_primes));
}

fn main() {
    bench_run();
}

#[allow(unused)]
fn bench_run() {
    let params = BENCH_PARAMS;
    let primes = params.create_primes();
    let moduli = AllModuli::new(primes);
    params.print(&moduli);

    let mut server = Server::create(params, &moduli.level_2_primes);
    server.set_db_int([0, 1, 2, 3, 4, 5, 6, 7, 8]);
    server.save_db();
    server.preprocess_db(&moduli);
    
    let mut client = Client::create(params, &moduli.level_2_primes);
    client.keygen(&mut rand::thread_rng());
    
    let query = client.query_single(2, &mut rand::thread_rng(), &moduli.level_2_primes);
    let reply = server.evaluate_query(query, &moduli.level_2_primes);
    println!("Reply size: {} B", reply.len() * std::mem::size_of::<CompressedEl>());
    client.plaintext.println(&client.reply_single(reply, &moduli.level_2_primes));
}

#[cfg(test)]
use feanor_math::assert_el_eq;

#[test]
fn test_scheme() {
    let params = TEST_PARAMS;
    let primes = params.create_primes();
    let moduli = AllModuli::new(primes);

    params.print(&moduli);

    let mut server = Server::create(params, &moduli.level_2_primes);
    server.set_db_int([0, 1, 2, 3, 4, 5, 6, 7, 8]);
    server.save_db();
    server.preprocess_db(&moduli);
    
    let mut client = Client::create(params, &moduli.level_2_primes);
    client.keygen(&mut rand::thread_rng());
    
    let query = client.query_single(3, &mut rand::thread_rng(), &moduli.level_2_primes);
    let reply = server.evaluate_query(query, &moduli.level_2_primes);
    let result = client.reply_single(reply, &moduli.level_2_primes);
    assert_el_eq!(&client.plaintext, &client.plaintext.int_hom().map(3), &result);
}