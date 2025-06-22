// Copyright 2024-2025 Irreducible Inc.

use std::{cmp, marker::PhantomData};

use binius_field::{BinaryField, PackedField, TowerField};
use binius_math::BinarySubspace;
use binius_utils::bail;

use super::{
	additive_ntt::{AdditiveNTT, NTTShape},
	error::Error,
	twiddle::TwiddleAccess,
};
use crate::twiddle::{
	self, OnTheFlyTwiddleAccess, PrecomputedTwiddleAccess, expand_subspace_evals,
};

/// Implementation of `AdditiveNTT` that performs the computation single-threaded.
#[derive(Debug)]
pub struct SingleThreadedNTT<F: BinaryField, TA: TwiddleAccess<F> = OnTheFlyTwiddleAccess<F>> {
	// TODO: Figure out how to make this private, it should not be `pub(super)`.
	pub(super) s_evals: Vec<TA>,
	_marker: PhantomData<F>,
}

impl<F: BinaryField> SingleThreadedNTT<F> {
	/// Default constructor constructs an NTT over the canonical subspace for the field using
	/// on-the-fly computed twiddle factors.
	pub fn new(log_domain_size: usize) -> Result<Self, Error> {
		let subspace = BinarySubspace::with_dim(log_domain_size)?;
		Self::with_subspace(&subspace)
	}

	/// Constructs an NTT over an isomorphic subspace for the given domain field using on-the-fly
	/// computed twiddle factors.
	pub fn with_domain_field<FDomain>(log_domain_size: usize) -> Result<Self, Error>
	where
		FDomain: BinaryField,
		F: From<FDomain>,
	{
		let subspace = BinarySubspace::<FDomain>::with_dim(log_domain_size)?.isomorphic();
		Self::with_subspace(&subspace)
	}

	pub fn with_subspace(subspace: &BinarySubspace<F>) -> Result<Self, Error> {
		let twiddle_access = OnTheFlyTwiddleAccess::generate(subspace)?;
		Ok(Self::with_twiddle_access(twiddle_access))
	}

	pub fn precompute_twiddles(&self) -> SingleThreadedNTT<F, PrecomputedTwiddleAccess<F>> {
		SingleThreadedNTT::with_twiddle_access(expand_subspace_evals(&self.s_evals))
	}
}

impl<F: TowerField> SingleThreadedNTT<F> {
	/// A specialization of [`with_domain_field`](Self::with_domain_field) to the canonical tower
	/// field.
	pub fn with_canonical_field(log_domain_size: usize) -> Result<Self, Error> {
		Self::with_domain_field::<F::Canonical>(log_domain_size)
	}
}

impl<F: BinaryField, TA: TwiddleAccess<F>> SingleThreadedNTT<F, TA> {
	const fn with_twiddle_access(twiddle_access: Vec<TA>) -> Self {
		Self {
			s_evals: twiddle_access,
			_marker: PhantomData,
		}
	}
}

impl<F: BinaryField, TA: TwiddleAccess<F>> SingleThreadedNTT<F, TA> {
	pub fn twiddles(&self) -> &[TA] {
		&self.s_evals
	}
}

impl<F, TA> AdditiveNTT<F> for SingleThreadedNTT<F, TA>
where
	F: BinaryField,
	TA: TwiddleAccess<F>,
{
	fn log_domain_size(&self) -> usize {
		self.s_evals.len()
	}

	fn subspace(&self, i: usize) -> BinarySubspace<F> {
		let (subspace, shift) = self.s_evals[self.log_domain_size() - i].affine_subspace();
		debug_assert_eq!(shift, F::ZERO, "s_evals subspaces must be linear by construction");
		subspace
	}

	fn get_subspace_eval(&self, i: usize, j: usize) -> F {
		self.s_evals[self.log_domain_size() - i].get(j)
	}

	fn forward_transform<P: PackedField<Scalar = F>>(
		&self,
		data: &mut [P],
		shape: NTTShape,
		coset: usize,
		coset_bits: usize,
		skip_rounds: usize,
	) -> Result<(), Error> {
		forward_transform(
			self.log_domain_size(),
			&self.s_evals,
			data,
			shape,
			coset,
			coset_bits,
			skip_rounds,
		)
	}

	fn inverse_transform<P: PackedField<Scalar = F>>(
		&self,
		data: &mut [P],
		shape: NTTShape,
		coset: usize,
		coset_bits: usize,
		skip_rounds: usize,
	) -> Result<(), Error> {
		inverse_transform(
			self.log_domain_size(),
			&self.s_evals,
			data,
			shape,
			coset,
			coset_bits,
			skip_rounds,
		)
	}
}

// ----------------------------------------------------------------------------------------------------------
// Similar to https://github.com/starkware-libs/stwo/blob/dev/crates/prover/src/core/backend/simd/fft/rfft.rs
//
pub fn forward_transform<F: BinaryField, P: PackedField<Scalar = F>>(
	log_domain_size: usize,
	s_evals: &[impl TwiddleAccess<F>],
	data: &mut [P],
	shape: NTTShape,
	coset: usize,
	coset_bits: usize,
	skip_rounds: usize,
) -> Result<(), Error> {
	check_batch_transform_inputs_and_params(
		log_domain_size,
		data,
		shape,
		coset,
		coset_bits,
		skip_rounds,
	)?;

	match data.len() {
		0 => return Ok(()),
		1 => {
			return match P::LOG_WIDTH {
				0 => Ok(()),
				_ => {
					let mut buffer = [data[0], P::zero()];
					forward_transform(
						log_domain_size,
						s_evals,
						&mut buffer,
						shape,
						coset,
						coset_bits,
						skip_rounds,
					)?;
					data[0] = buffer[0];
					Ok(())
				}
			};
		}
		_ => {}
	};

	let NTTShape {
		log_x,
		log_y,
		log_z,
	} = shape;
	let log_w = P::LOG_WIDTH;
	let cutoff = log_w.saturating_sub(log_x);
	let s_evals = &s_evals[log_domain_size - (log_y + coset_bits)..];

	// Process layers above cutoff with cache-blocking optimization
	if log_y > skip_rounds {
		forward_transform_above_cutoff(data, s_evals, shape, coset, cutoff, skip_rounds);
	}

	// Process layers below cutoff with packed operations
	if cutoff > 0 {
		forward_transform_below_cutoff(data, s_evals, shape, coset, cutoff, skip_rounds);
	}

	Ok(())
}

/// Optimized using cache-efficient blocking
#[inline(always)]
fn forward_transform_above_cutoff<F: BinaryField, P: PackedField<Scalar = F>>(
	data: &mut [P],
	s_evals: &[impl TwiddleAccess<F>],
	shape: NTTShape,
	coset: usize,
	cutoff: usize,
	skip_rounds: usize,
) {
	let NTTShape {
		log_x,
		log_y,
		log_z,
	} = shape;
	let log_w = P::LOG_WIDTH;

	// Cache block size - tune this based on your cache size
	const CACHE_BLOCK_LOG_SIZE: usize = 6; // 64 elements per block

	for i in (cutoff..(log_y - skip_rounds)).rev() {
		let s_evals_i = &s_evals[i];
		let coset_offset = coset << (log_y - 1 - i);
		let stride = 1 << (log_x + i - log_w);
		let num_blocks = 1 << (log_y - 1 - i);
		let block_size = 1 << (i + log_x - log_w);

		if log_z + (log_y - 1 - i) + (i + log_x - log_w) >= CACHE_BLOCK_LOG_SIZE {
			let outer_blocks = 1 << log_z;
			let blocks_per_cache_block = 1 << CACHE_BLOCK_LOG_SIZE.min(log_y - 1 - i);

			for outer_block in 0..outer_blocks {
				let base_j = outer_block << (log_x + log_y - log_w);

				for block_chunk_start in (0..num_blocks).step_by(blocks_per_cache_block) {
					let block_chunk_end =
						(block_chunk_start + blocks_per_cache_block).min(num_blocks);

					// Prefetch twiddle factors for this chunk
					let chunk_twiddles: Vec<_> = (block_chunk_start..block_chunk_end)
						.map(|k| s_evals_i.get(coset_offset | k))
						.collect();

					// Process each block in the chunk
					for (block_idx, &twiddle) in chunk_twiddles.iter().enumerate() {
						let k = block_chunk_start + block_idx;
						let block_base = base_j | (k << (log_x + i + 1 - log_w));

						// Vectorized butterfly operations within the block
						butterfly_block_simd(data, block_base, stride, block_size, twiddle);
					}
				}
			}
		} else {
			// Fall back to simpler approach for smaller transforms
			forward_transform_above_cutoff_simple(
				data,
				s_evals_i,
				log_x,
				log_y,
				log_z,
				log_w,
				i,
				coset_offset,
			);
		}
	}
}

use std::arch::x86_64::*;
#[target_feature(enable = "avx2")]
unsafe fn butterfly_block_simd_u32(
	data: *mut u32,
	block_base: usize,
	stride: usize,
	block_size: usize,
	twiddle: u32,
) {
	// Broadcast twiddle into all 8 lanes of a 256-bit register
	let tw_vec = _mm256_set1_epi32(twiddle as i32);

	unsafe {
		let mut i = 0;
		while i + 8 <= block_size {
			let ptr0 = data.add(block_base + i) as *const __m256i;
			let ptr1 = data.add(block_base + i + stride) as *const __m256i;

			let u_vec = _mm256_loadu_si256(ptr0);
			let v_vec = _mm256_loadu_si256(ptr1);

			let v_mul = _mm256_mullo_epi32(v_vec, tw_vec);

			let t0 = _mm256_add_epi32(u_vec, v_mul);

			let t1 = _mm256_add_epi32(v_vec, t0);

			_mm256_storeu_si256(data.add(block_base + i) as *mut __m256i, t0);
			_mm256_storeu_si256(data.add(block_base + i + stride) as *mut __m256i, t1);

			i += 8;
		}

		for j in i..block_size {
			let p0 = data.add(block_base + j);
			let p1 = data.add(block_base + j + stride);
			let u = *p0;
			let v = *p1;

			let t0 = u.wrapping_add(v.wrapping_mul(twiddle));
			*p0 = t0;
			*p1 = v.wrapping_add(t0);
		}
	}
}

/// Safe wrapper for your `&mut [P]` slice—requires `P` to be `repr(transparent)` over `u32`.
///! This does not work, the twiddle is hard-coded could not convert it to u32
pub fn butterfly_block_simd<F: BinaryField, P: PackedField<Scalar = F>>(
	data: &mut [P],
	block_base: usize,
	stride: usize,
	block_size: usize,
	twiddle: F,
) where
	P: Copy, // ensures no padding, repr(transparent)
{
	// cast `&mut [P]` → raw `*mut u32`
	let ptr = data.as_mut_ptr() as *mut u32;

	// We tried to hotfix this by casting coversion through string. In the end we could not pass the
	// test let x = twiddle.to_string();
	// let bytes = x.as_bytes();
	// let twiddleu32 = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);

	debug_assert!(is_x86_feature_detected!("avx2"), "AVX2 required");
	unsafe { butterfly_block_simd_u32(ptr, block_base, stride, block_size, 1) }
}

/// Simple version for smaller transforms
#[inline(always)]
fn forward_transform_above_cutoff_simple<F: BinaryField, P: PackedField<Scalar = F>>(
	data: &mut [P],
	s_evals_i: &impl TwiddleAccess<F>,
	log_x: usize,
	log_y: usize,
	log_z: usize,
	log_w: usize,
	i: usize,
	coset_offset: usize,
) {
	for j in 0..1 << log_z {
		for k in 0..1 << (log_y - 1 - i) {
			let twiddle = P::broadcast(s_evals_i.get(coset_offset | k));
			for l in 0..1 << (i + log_x - log_w) {
				let idx0 = j << (log_x + log_y - log_w) | k << (log_x + i + 1 - log_w) | l;
				let idx1 = idx0 | 1 << (log_x + i - log_w);
				data[idx0] += data[idx1] * twiddle;
				data[idx1] += data[idx0];
			}
		}
	}
}

/// Optimized transform for layers below cutoff with improved packed operations
#[inline(always)]
fn forward_transform_below_cutoff<F: BinaryField, P: PackedField<Scalar = F>>(
	data: &mut [P],
	s_evals: &[impl TwiddleAccess<F>],
	shape: NTTShape,
	coset: usize,
	cutoff: usize,
	skip_rounds: usize,
) {
	let NTTShape {
		log_x,
		log_y,
		log_z,
	} = shape;
	let log_w = P::LOG_WIDTH;

	for i in (0..cmp::min(cutoff, log_y - skip_rounds)).rev() {
		let s_evals_i = &s_evals[i];
		let coset_offset = coset << (log_y - 1 - i);

		// Pre-calculate packed additive twiddle once per layer
		let block_twiddle = calculate_packed_additive_twiddle::<P>(s_evals_i, shape, i);

		let log_block_len = i + log_x;
		let log_packed_count = (log_y - 1).saturating_sub(cutoff);
		let outer_count = 1 << (log_x + log_y + log_z).saturating_sub(log_w + log_packed_count + 1);
		let packed_count = 1 << log_packed_count;

		const OUTER_BLOCK_SIZE: usize = 64;

		for j_block in (0..outer_count).step_by(OUTER_BLOCK_SIZE) {
			let j_end = (j_block + OUTER_BLOCK_SIZE).min(outer_count);

			let twiddles: Vec<_> = (0..packed_count)
				.map(|k| {
					P::broadcast(s_evals_i.get(coset_offset | k << (cutoff - i))) + block_twiddle
				})
				.collect();

			for j in j_block..j_end {
				for k in 0..packed_count {
					let twiddle = twiddles[k];
					let index = k << 1 | j << (log_packed_count + 1);

					let (mut u, mut v) = data[index].interleave(data[index | 1], log_block_len);
					u += v * twiddle;
					v += u;
					(data[index], data[index | 1]) = u.interleave(v, log_block_len);
				}
			}
		}
	}
}

pub fn inverse_transform<F: BinaryField, P: PackedField<Scalar = F>>(
	log_domain_size: usize,
	s_evals: &[impl TwiddleAccess<F>],
	data: &mut [P],
	shape: NTTShape,
	coset: usize,
	coset_bits: usize,
	skip_rounds: usize,
) -> Result<(), Error> {
	check_batch_transform_inputs_and_params(
		log_domain_size,
		data,
		shape,
		coset,
		coset_bits,
		skip_rounds,
	)?;

	match data.len() {
		0 => return Ok(()),
		1 => {
			return match P::LOG_WIDTH {
				0 => Ok(()),
				_ => {
					// Special case when there is only one packed element: since we cannot
					// interleave with another packed element, the code below will panic when there
					// is only one.
					//
					// Handle the case of one packed element by batch transforming the original
					// data with dummy data and extracting the transformed result.
					let mut buffer = [data[0], P::zero()];
					inverse_transform(
						log_domain_size,
						s_evals,
						&mut buffer,
						shape,
						coset,
						coset_bits,
						skip_rounds,
					)?;
					data[0] = buffer[0];
					Ok(())
				}
			};
		}
		_ => {}
	};

	let NTTShape {
		log_x,
		log_y,
		log_z,
	} = shape;

	let log_w = P::LOG_WIDTH;

	// Cutoff is the stage of the NTT where each the butterfly units are contained within
	// packed base field elements.
	let cutoff = log_w.saturating_sub(log_x);

	// Choose the twiddle factors so that NTTs on differently sized domains, with the same
	// coset_bits, share the final layer twiddles.
	let s_evals = &s_evals[log_domain_size - (log_y + coset_bits)..];

	#[allow(clippy::needless_range_loop)]
	for i in 0..cutoff.min(log_y - skip_rounds) {
		let s_evals_i = &s_evals[i];
		let coset_offset = coset << (log_y - 1 - i);

		// A block is a block of butterfly units that all have the same twiddle factor. Since we
		// are below the cutoff round, the block length is less than the packing width, and
		// therefore each packed multiplication is with a non-uniform twiddle. Since the subspace
		// polynomials are linear, we can calculate an additive factor that can be added to the
		// packed twiddles for all packed butterfly units.
		let block_twiddle = calculate_packed_additive_twiddle::<P>(s_evals_i, shape, i);

		let log_block_len = i + log_x;
		let log_packed_count = (log_y - 1).saturating_sub(cutoff);
		for j in 0..1 << (log_x + log_y + log_z).saturating_sub(log_w + log_packed_count + 1) {
			for k in 0..1 << log_packed_count {
				let twiddle =
					P::broadcast(s_evals_i.get(coset_offset | k << (cutoff - i))) + block_twiddle;
				let index = k << 1 | j << (log_packed_count + 1);
				let (mut u, mut v) = data[index].interleave(data[index | 1], log_block_len);
				v += u;
				u += v * twiddle;
				(data[index], data[index | 1]) = u.interleave(v, log_block_len);
			}
		}
	}

	// i indexes the layer of the NTT network, also the binary subspace.
	#[allow(clippy::needless_range_loop)]
	for i in cutoff..(log_y - skip_rounds) {
		let s_evals_i = &s_evals[i];
		let coset_offset = coset << (log_y - 1 - i);

		// j indexes the outer Z tensor axis.
		for j in 0..1 << log_z {
			// k indexes the block within the layer. Each block performs butterfly operations with
			// the same twiddle factor.
			for k in 0..1 << (log_y - 1 - i) {
				let twiddle = s_evals_i.get(coset_offset | k);
				for l in 0..1 << (i + log_x - log_w) {
					let idx0 = j << (log_x + log_y - log_w) | k << (log_x + i + 1 - log_w) | l;
					let idx1 = idx0 | 1 << (log_x + i - log_w);
					data[idx1] += data[idx0];
					data[idx0] += data[idx1] * twiddle;
				}
			}
		}
	}

	Ok(())
}

pub fn check_batch_transform_inputs_and_params<PB: PackedField>(
	log_domain_size: usize,
	data: &[PB],
	shape: NTTShape,
	coset: usize,
	coset_bits: usize,
	skip_rounds: usize,
) -> Result<(), Error> {
	let NTTShape {
		log_x,
		log_y,
		log_z,
	} = shape;

	if !data.len().is_power_of_two() {
		bail!(Error::PowerOfTwoLengthRequired);
	}
	if skip_rounds > log_y {
		bail!(Error::SkipRoundsTooLarge);
	}

	let full_sized_y = (data.len() * PB::WIDTH) >> (log_x + log_z);

	// Verify that our log_y exactly matches the data length, except when we are NTT-ing one packed
	// field
	if (1 << log_y != full_sized_y && data.len() > 2) || (1 << log_y > full_sized_y) {
		bail!(Error::BatchTooLarge);
	}

	if coset >= (1 << coset_bits) {
		bail!(Error::CosetIndexOutOfBounds { coset, coset_bits });
	}

	// The domain size should be at least large enough to represent the given coset.
	let log_required_domain_size = log_y + coset_bits;
	if log_required_domain_size > log_domain_size {
		bail!(Error::DomainTooSmall {
			log_required_domain_size
		});
	}

	Ok(())
}

#[inline]
fn calculate_packed_additive_twiddle<P>(
	s_evals: &impl TwiddleAccess<P::Scalar>,
	shape: NTTShape,
	ntt_round: usize,
) -> P
where
	P: PackedField<Scalar: BinaryField>,
{
	let NTTShape {
		log_x,
		log_y,
		log_z,
	} = shape;
	debug_assert!(log_y > 0);

	let log_block_len = ntt_round + log_x;
	debug_assert!(log_block_len < P::LOG_WIDTH);

	let packed_log_len = (log_x + log_y + log_z).min(P::LOG_WIDTH);
	let log_blocks_count = packed_log_len.saturating_sub(log_block_len + 1);

	let packed_log_z = packed_log_len.saturating_sub(log_x + log_y);
	let packed_log_y = packed_log_len - packed_log_z - log_x;

	let twiddle_stride = P::LOG_WIDTH
		.saturating_sub(log_x)
		.min(log_blocks_count - packed_log_z);

	let mut twiddle = P::default();
	for i in 0..1 << (log_blocks_count - twiddle_stride) {
		for j in 0..1 << twiddle_stride {
			let (subblock_twiddle_0, subblock_twiddle_1) = if packed_log_y == log_y {
				let same_twiddle = s_evals.get(j);
				(same_twiddle, same_twiddle)
			} else {
				s_evals.get_pair(twiddle_stride, j)
			};
			let idx0 = j << (log_block_len + 1) | i << (log_block_len + twiddle_stride + 1);
			let idx1 = idx0 | 1 << log_block_len;

			for k in 0..1 << log_block_len {
				twiddle.set(idx0 | k, subblock_twiddle_0);
				twiddle.set(idx1 | k, subblock_twiddle_1);
			}
		}
	}
	twiddle
}

#[cfg(test)]
mod tests {
	use std::iter::repeat_with;

	use assert_matches::assert_matches;
	use binius_field::{
		BinaryField8b, BinaryField16b, Field, PackedBinaryField8x16b, PackedFieldIndexable,
	};
	use binius_math::Error as MathError;
	use rand::{SeedableRng, rngs::StdRng};

	use super::*;

	#[test]
	fn test_additive_ntt_fails_with_field_too_small() {
		assert_matches!(
			SingleThreadedNTT::<BinaryField8b>::new(10),
			Err(Error::MathError(MathError::DomainSizeTooLarge))
		);
	}

	#[test]
	fn test_subspace_size_agrees_with_domain_size() {
		let ntt = SingleThreadedNTT::<BinaryField16b>::new(10).expect("msg");
		assert_eq!(ntt.subspace(10).dim(), 10);
		assert_eq!(ntt.subspace(1).dim(), 1);
	}

	/// The additive NTT has a useful property that the NTT output of an internally zero-padded
	/// input has the structure of repeating codeword symbols.
	///
	/// More precisely, let $m \in K^{2^\ell}$ be a message and $c = \text{NTT}_{\ell}(m)$ be its
	/// NTT evaluation on the domain $S^{(\ell)}$. Define $m' \in K^{2^{\ell+\nu}}$ to be a
	/// sequence with $m'_{i * 2^\nu} = m_i$ and $m'_j = 0$ when $j \ne 0 \mod 2^\nu$. The
	/// property is that $c' = \text{NTT}_{\ell+\nu}(m')$ will have the structure
	/// $c'_j = c_{\lfloor j / 2^\nu \rfloor}$.
	///
	/// So for $\nu = 2$, then $m' = (m_0, 0, 0, 0, m_1, 0, 0, 0, \ldots, m_{\ell-1}, 0, 0, 0)$ and
	/// $c' = (c_0, c_0, c_0, c_0, c_1, c_1, c_1, c_1, \ldots, c_{\ell-1}, c_{\ell-1}, c_{\ell-1},
	/// c_{\ell-1})$.
	#[test]
	fn test_repetition_property() {
		let log_len = 8;
		let ntt = SingleThreadedNTT::<BinaryField16b>::new(log_len + 2).unwrap();

		let mut rng = StdRng::seed_from_u64(0);
		let msg = repeat_with(|| <BinaryField16b as Field>::random(&mut rng))
			.take(1 << log_len)
			.collect::<Vec<_>>();

		let mut msg_padded = vec![BinaryField16b::ZERO; 1 << (log_len + 2)];
		for i in 0..1 << log_len {
			msg_padded[i << 2] = msg[i];
		}

		let mut out = msg;
		ntt.forward_transform(
			&mut out,
			NTTShape {
				log_y: log_len,
				..Default::default()
			},
			0,
			0,
			0,
		)
		.unwrap();
		let mut out_rep = msg_padded;
		ntt.forward_transform(
			&mut out_rep,
			NTTShape {
				log_y: log_len + 2,
				..Default::default()
			},
			0,
			0,
			0,
		)
		.unwrap();
		for i in 0..1 << (log_len + 2) {
			assert_eq!(out_rep[i], out[i >> 2]);
		}
	}

	#[test]
	fn one_packed_field_forward() {
		let s = SingleThreadedNTT::<BinaryField16b>::new(10).expect("msg");
		let mut packed = [PackedBinaryField8x16b::random(StdRng::from_os_rng())];

		let mut packed_copy = packed;

		let unpacked = PackedBinaryField8x16b::unpack_scalars_mut(&mut packed_copy);

		let shape = NTTShape {
			log_x: 0,
			log_y: 3,
			log_z: 0,
		};
		let _ = s.forward_transform(&mut packed, shape, 3, 2, 0);
		let _ = s.forward_transform(unpacked, shape, 3, 2, 0);

		for (i, unpacked_item) in unpacked.iter().enumerate().take(8) {
			assert_eq!(packed[0].get(i), *unpacked_item);
		}
	}

	#[test]
	fn one_packed_field_inverse() {
		let s = SingleThreadedNTT::<BinaryField16b>::new(10).expect("msg");
		let mut packed = [PackedBinaryField8x16b::random(StdRng::from_os_rng())];

		let mut packed_copy = packed;

		let unpacked = PackedBinaryField8x16b::unpack_scalars_mut(&mut packed_copy);

		let shape = NTTShape {
			log_x: 0,
			log_y: 3,
			log_z: 0,
		};
		let _ = s.inverse_transform(&mut packed, shape, 3, 2, 0);
		let _ = s.inverse_transform(unpacked, shape, 3, 2, 0);

		for (i, unpacked_item) in unpacked.iter().enumerate().take(8) {
			assert_eq!(packed[0].get(i), *unpacked_item);
		}
	}

	#[test]
	fn smaller_embedded_batch_forward() {
		let s = SingleThreadedNTT::<BinaryField16b>::new(10).expect("msg");
		let mut packed = [PackedBinaryField8x16b::random(StdRng::from_os_rng())];

		let mut packed_copy = packed;

		let unpacked = &mut PackedBinaryField8x16b::unpack_scalars_mut(&mut packed_copy)[0..4];

		let shape = NTTShape {
			log_x: 0,
			log_y: 2,
			log_z: 0,
		};
		let _ = forward_transform(s.log_domain_size(), &s.s_evals, &mut packed, shape, 3, 2, 0);
		let _ = s.forward_transform(unpacked, shape, 3, 2, 0);

		for (i, unpacked_item) in unpacked.iter().enumerate().take(4) {
			assert_eq!(packed[0].get(i), *unpacked_item);
		}
	}

	#[test]
	fn smaller_embedded_batch_inverse() {
		let s = SingleThreadedNTT::<BinaryField16b>::new(10).expect("msg");
		let mut packed = [PackedBinaryField8x16b::random(StdRng::from_os_rng())];

		let mut packed_copy = packed;

		let unpacked = &mut PackedBinaryField8x16b::unpack_scalars_mut(&mut packed_copy)[0..4];

		let shape = NTTShape {
			log_x: 0,
			log_y: 2,
			log_z: 0,
		};
		let _ = inverse_transform(s.log_domain_size(), &s.s_evals, &mut packed, shape, 3, 2, 0);
		let _ = s.inverse_transform(unpacked, shape, 3, 2, 0);

		for (i, unpacked_item) in unpacked.iter().enumerate().take(4) {
			assert_eq!(packed[0].get(i), *unpacked_item);
		}
	}

	// TODO: Write test that compares polynomial evaluation via additive NTT with naive Lagrange
	// polynomial interpolation. A randomized test should suffice for larger NTT sizes.
}
