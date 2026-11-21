/*
 * Copyright (C) 2018-2019 Ehsan Dalvand <dalvand.ehsan@gmail.com>, Alireza Jahandideh <ar.jahandideh@gmail.com>
 *
 * This program is free software: you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation: either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 * Argon2 on the GPU, 32 threads per lane holding a 1 KB block in registers
 * (4 qwords each): the Blake2b helpers that build the first blocks and the
 * tag, and argon2_fill, which computes every other block. After the
 * Dynamic-GPU-Miner kernels as reworked in ccminer-tpfuemp (runtime geometry,
 * per-lane shared cache); the data-independent addressing follows argon2-gpu
 * by Ondrej Mosnacek (MIT). Only built into one translation unit.
 */
#pragma once
#include <stdint.h>
#include "cuda_compat.h"

#define THREADS_PER_LANE	32
#define QWORDS_PER_THREAD	4		/* 128 qwords per block / 32 threads */

#define ARGON2_BLOCK_SIZE		1024
#define ARGON2_QWORDS_IN_BLOCK		128
#define ARGON2_ADDRESSES_IN_BLOCK	128
#define ARGON2_SYNC_POINTS		4
#define ARGON2_VERSION_10		0x10
#define ARGON2_VERSION_13		0x13
#define ARGON2_D			0
#define ARGON2_I			1
#define ARGON2_ID			2
#define BLAKE_BLOCKBYTES		128
/* argon2_fill keeps columns 2..7 of each lane only in shared memory */
#define ARGON2_SHARED_BLOCKS_PER_LANE	6

struct partialState { uint64_t a, b; };
struct block_g { uint64_t data[ARGON2_QWORDS_IN_BLOCK]; };
struct block_th { uint64_t a, b, c, d; };
struct uint64x8 { uint64_t s0, s1, s2, s3, s4, s5, s6, s7; };

/* message word order of the 12 Blake2b rounds, 16 nibble-packed bytes per row */
static __constant__ const uint64_t a2_sigma[12][2] = {
	{ 506097522914230528ULL, 1084818905618843912ULL },
	{ 436021270388410894ULL, 217587900856929281ULL },
	{ 940973067642603531ULL, 290764780619369994ULL },
	{ 1011915791265892615ULL, 580682894302053890ULL },
	{ 1083683067090239497ULL, 937601969488068878ULL },
	{ 218436676723543042ULL, 648815278989708548ULL },
	{ 721716194318550284ULL, 794887571959580416ULL },
	{ 649363922558061325ULL, 721145521830297605ULL },
	{ 576464098234863366ULL, 363107122416517644ULL },
	{ 360576072368521738ULL, 3672381957147407ULL },
	{ 506097522914230528ULL, 1084818905618843912ULL },
	{ 436021270388410894ULL, 217587900856929281ULL },
};

/* Blake2b IV with the parameter block of a 64-byte digest (0x...948) and of a
 * 32-byte one (0x...928) folded into word 0 */
static __constant__ const uint64_t a2_blake2b_init64[8] = {
	0x6A09E667F2BDC948ULL, 0xBB67AE8584CAA73BULL, 0x3C6EF372FE94F82BULL, 0xA54FF53A5F1D36F1ULL,
	0x510E527FADE682D1ULL, 0x9B05688C2B3E6C1FULL, 0x1F83D9ABFB41BD6BULL, 0x5BE0CD19137E2179ULL
};
static __constant__ const uint64_t a2_blake2b_init32[8] = {
	0x6A09E667F2BDC928ULL, 0xBB67AE8584CAA73BULL, 0x3C6EF372FE94F82BULL, 0xA54FF53A5F1D36F1ULL,
	0x510E527FADE682D1ULL, 0x9B05688C2B3E6C1FULL, 0x1F83D9ABFB41BD6BULL, 0x5BE0CD19137E2179ULL
};
static __constant__ const uint64_t a2_blake2b_iv[8] = {
	0x6A09E667F3BCC908ULL, 0xBB67AE8584CAA73BULL, 0x3C6EF372FE94F82BULL, 0xA54FF53A5F1D36F1ULL,
	0x510E527FADE682D1ULL, 0x9B05688C2B3E6C1FULL, 0x1F83D9ABFB41BD6BULL, 0x5BE0CD19137E2179ULL
};

static __device__ __forceinline__ uint64_t a2_rotr64(const uint64_t x, const uint32_t n)
{
	return (x >> n) | (x << (64 - n));
}

static __device__ __forceinline__ uint64_t a2_u64(uint32_t hi, uint32_t lo)
{
	return ((uint64_t)hi << 32) | (uint64_t)lo;
}

/* 64-bit shuffle from two 32-bit ones (CUDA 8 has no 64-bit __shfl) */
static __device__ __forceinline__ uint64_t a2_shfl64(uint64_t v, int src, int width = 32)
{
	const uint32_t lo = __shfl((uint32_t)v, src, width);
	const uint32_t hi = __shfl((uint32_t)(v >> 32), src, width);
	return a2_u64(hi, lo);
}

/* ---- Blake2b, one thread per hash ----------------------------------------- */

#define A2_CAT(x, y) A2_CAT_(x, y)
#define A2_CAT_(x, y) x ## y
#define A2_G(a, b, c, d, x, col) { \
	ref1 = a2_sigma[r][col] >> 16 * x; \
	ref2 = a2_sigma[r][col] >> (16 * x + 8); \
	A2_CAT(v, a) += A2_CAT(v, b) + m[ref1]; \
	A2_CAT(v, d) = a2_rotr64(A2_CAT(v, d) ^ A2_CAT(v, a), 32); \
	A2_CAT(v, c) += A2_CAT(v, d); \
	A2_CAT(v, b) = a2_rotr64(A2_CAT(v, b) ^ A2_CAT(v, c), 24); \
	A2_CAT(v, a) += A2_CAT(v, b) + m[ref2]; \
	A2_CAT(v, d) = a2_rotr64(A2_CAT(v, d) ^ A2_CAT(v, a), 16); \
	A2_CAT(v, c) += A2_CAT(v, d); \
	A2_CAT(v, b) = a2_rotr64(A2_CAT(v, b) ^ A2_CAT(v, c), 63); \
}

/* one compression; step is the 1-based block number, lastChunkSize the bytes
 * of the final block */
static __device__ void a2_blake2b_compress_1w(uint64x8 *state, const uint64_t *m,
	const uint32_t step, const bool lastChunk = false, const uint32_t lastChunkSize = 0)
{
	uint64_t v0 = state->s0, v1 = state->s1, v2 = state->s2, v3 = state->s3;
	uint64_t v4 = state->s4, v5 = state->s5, v6 = state->s6, v7 = state->s7;
	uint64_t v8 = a2_blake2b_iv[0], v9 = a2_blake2b_iv[1], v10 = a2_blake2b_iv[2], v11 = a2_blake2b_iv[3];
	uint64_t v12, v13 = a2_blake2b_iv[5], v14, v15 = a2_blake2b_iv[7];

	if (lastChunk) {
		v12 = a2_blake2b_iv[4] ^ ((uint64_t)(step - 1) * BLAKE_BLOCKBYTES + lastChunkSize);
		v14 = a2_blake2b_iv[6] ^ (uint64_t)-1;
	} else {
		v12 = a2_blake2b_iv[4] ^ ((uint64_t)step * BLAKE_BLOCKBYTES);
		v14 = a2_blake2b_iv[6];
	}

	#pragma unroll 12
	for (int r = 0; r < 12; r++) {
		uint8_t ref1, ref2;
		A2_G(0, 4, 8, 12, 0, 0);
		A2_G(1, 5, 9, 13, 1, 0);
		A2_G(2, 6, 10, 14, 2, 0);
		A2_G(3, 7, 11, 15, 3, 0);
		A2_G(0, 5, 10, 15, 0, 1);
		A2_G(1, 6, 11, 12, 1, 1);
		A2_G(2, 7, 8, 13, 2, 1);
		A2_G(3, 4, 9, 14, 3, 1);
	}

	state->s0 ^= v0 ^ v8;
	state->s1 ^= v1 ^ v9;
	state->s2 ^= v2 ^ v10;
	state->s3 ^= v3 ^ v11;
	state->s4 ^= v4 ^ v12;
	state->s5 ^= v5 ^ v13;
	state->s6 ^= v6 ^ v14;
	state->s7 ^= v7 ^ v15;
}

/* ---- Blake2b, four threads per hash (two state words each) ---------------- */

static __device__ __forceinline__ void a2_g4(uint64_t *a, uint64_t *b, uint64_t *c, uint64_t *d,
	const uint64_t *m1, const uint64_t *m2)
{
	*a = *a + *b + *m1;
	*d = a2_rotr64(*d ^ *a, 32);
	*c = *c + *d;
	*b = a2_rotr64(*b ^ *c, 24);
	*a = *a + *b + *m2;
	*d = a2_rotr64(*d ^ *a, 16);
	*c = *c + *d;
	*b = a2_rotr64(*b ^ *c, 63);
}

/* every thread of the block must call it (it contains a barrier) */
static __device__ void a2_blake2b_compress_4w(partialState *state, uint64_t *m,
	uint32_t step, uint32_t idx, bool lastChunk = false, uint32_t lastChunkSize = 0)
{
	const uint64_t counter = (idx == 0 ? step : 0);
	uint64_t a = state->a, b = state->b, c = a2_blake2b_iv[idx], d;

	if (lastChunk) {
		if (idx == 0)
			d = a2_blake2b_iv[4] ^ ((uint64_t)(step - 1) * BLAKE_BLOCKBYTES + lastChunkSize);
		else if (idx == 2)
			d = a2_blake2b_iv[6] ^ (uint64_t)-1;
		else
			d = a2_blake2b_iv[idx + 4];
	} else {
		d = a2_blake2b_iv[idx + 4] ^ counter * BLAKE_BLOCKBYTES;
	}

	__syncthreads();

	for (uint32_t r = 0; r < 12; ++r) {
		uint8_t ref1 = a2_sigma[r][0] >> 8 * 2 * idx;
		uint8_t ref2 = a2_sigma[r][0] >> 8 * (2 * idx + 1);
		a2_g4(&a, &b, &c, &d, &m[ref1], &m[ref2]);
		b = a2_shfl64(b, idx + 1, 4);
		c = a2_shfl64(c, idx + 2, 4);
		d = a2_shfl64(d, idx + 3, 4);

		ref1 = a2_sigma[r][1] >> 8 * 2 * idx;
		ref2 = a2_sigma[r][1] >> 8 * (2 * idx + 1);
		a2_g4(&a, &b, &c, &d, &m[ref1], &m[ref2]);
		b = a2_shfl64(b, idx - 1, 4);
		c = a2_shfl64(c, idx - 2, 4);
		d = a2_shfl64(d, idx - 3, 4);
	}

	state->a = state->a ^ a ^ c;
	state->b = state->b ^ b ^ d;
}

/* Blocks 0 and 1 of each lane: Blake2b-long(1024) of H0 || column || row.
 * buffer: H0 in words 1..16, the rest zero; threadIdx.x picks row and column,
 * the job is blockIdx.x * blockDim.y + threadIdx.y. */
static __device__ void a2_fill_first_block(block_g *memory, uint32_t *buffer,
	uint32_t lanes, uint32_t total_blocks)
{
	const uint32_t row = threadIdx.x / lanes;
	const uint32_t column = threadIdx.x % lanes;
	block_g *cell = memory + (size_t)(blockIdx.x * blockDim.y + threadIdx.y) * total_blocks
		+ row * lanes + column;
	uint64_t *b64 = (uint64_t *)buffer;
	uint64x8 state;

	state.s0 = a2_blake2b_init64[0]; state.s1 = a2_blake2b_init64[1];
	state.s2 = a2_blake2b_init64[2]; state.s3 = a2_blake2b_init64[3];
	state.s4 = a2_blake2b_init64[4]; state.s5 = a2_blake2b_init64[5];
	state.s6 = a2_blake2b_init64[6]; state.s7 = a2_blake2b_init64[7];

	buffer[0] = 1024;
	buffer[17] = row;
	buffer[18] = column;
	a2_blake2b_compress_1w(&state, b64, 1, true, 76);

	cell->data[0] = state.s0;
	cell->data[1] = state.s1;
	cell->data[2] = state.s2;
	cell->data[3] = state.s3;

	for (int i = 0; i < 8; i++)
		b64[i + 8] = 0;

	for (uint32_t i = 1; i < 31; i++) {
		b64[0] = state.s0; b64[1] = state.s1; b64[2] = state.s2; b64[3] = state.s3;
		b64[4] = state.s4; b64[5] = state.s5; b64[6] = state.s6; b64[7] = state.s7;

		state.s0 = a2_blake2b_init64[0]; state.s1 = a2_blake2b_init64[1];
		state.s2 = a2_blake2b_init64[2]; state.s3 = a2_blake2b_init64[3];
		state.s4 = a2_blake2b_init64[4]; state.s5 = a2_blake2b_init64[5];
		state.s6 = a2_blake2b_init64[6]; state.s7 = a2_blake2b_init64[7];
		a2_blake2b_compress_1w(&state, b64, 1, true, 64);

		cell->data[(i << 2) + 0] = state.s0;
		cell->data[(i << 2) + 1] = state.s1;
		cell->data[(i << 2) + 2] = state.s2;
		cell->data[(i << 2) + 3] = state.s3;
	}

	cell->data[124] = state.s4;
	cell->data[125] = state.s5;
	cell->data[126] = state.s6;
	cell->data[127] = state.s7;
}

/* ---- the fill: one block per lane per step, 32 threads per lane ----------- */

static __device__ __forceinline__ uint64_t a2_ld(const uint64_t *ptr)
{
	uint32_t a, b;
	asm("ld.global.ca.v2.u32 {%0, %1}, [%2];" : "=r"(a), "=r"(b) : "l"(ptr));
	return a2_u64(b, a);
}

static __device__ __forceinline__ void a2_st(uint64_t *ptr, const uint64_t value)
{
	asm("st.global.wb.v2.u32 [%0], {%1, %2};" :: "l"(ptr), "r"((uint32_t)value), "r"((uint32_t)(value >> 32)));
}

static __device__ __forceinline__ uint64_t a2_cmpeq_mask(uint32_t test, uint32_t ref)
{
	const uint32_t x = -(uint32_t)(test == ref);
	return a2_u64(x, x);
}

static __device__ __forceinline__ uint64_t a2_th_get(const block_th *b, uint32_t idx)
{
	uint64_t res = 0;
	res ^= a2_cmpeq_mask(idx, 0) & b->a;
	res ^= a2_cmpeq_mask(idx, 1) & b->b;
	res ^= a2_cmpeq_mask(idx, 2) & b->c;
	res ^= a2_cmpeq_mask(idx, 3) & b->d;
	return res;
}

static __device__ __forceinline__ void a2_th_set(block_th *b, uint32_t idx, uint64_t v)
{
	b->a ^= a2_cmpeq_mask(idx, 0) & (v ^ b->a);
	b->b ^= a2_cmpeq_mask(idx, 1) & (v ^ b->b);
	b->c ^= a2_cmpeq_mask(idx, 2) & (v ^ b->c);
	b->d ^= a2_cmpeq_mask(idx, 3) & (v ^ b->d);
}

static __device__ __forceinline__ void a2_xor(block_th *dst, const block_th *src)
{
	dst->a ^= src->a; dst->b ^= src->b; dst->c ^= src->c; dst->d ^= src->d;
}

static __device__ __forceinline__ void a2_load(block_th *dst, const block_g *src, uint32_t t)
{
	dst->a = a2_ld(&src->data[0 * THREADS_PER_LANE + t]);
	dst->b = a2_ld(&src->data[1 * THREADS_PER_LANE + t]);
	dst->c = a2_ld(&src->data[2 * THREADS_PER_LANE + t]);
	dst->d = a2_ld(&src->data[3 * THREADS_PER_LANE + t]);
}

static __device__ __forceinline__ void a2_load_xor(block_th *dst, const block_g *src, uint32_t t)
{
	dst->a ^= a2_ld(&src->data[0 * THREADS_PER_LANE + t]);
	dst->b ^= a2_ld(&src->data[1 * THREADS_PER_LANE + t]);
	dst->c ^= a2_ld(&src->data[2 * THREADS_PER_LANE + t]);
	dst->d ^= a2_ld(&src->data[3 * THREADS_PER_LANE + t]);
}

static __device__ __forceinline__ void a2_store(block_g *dst, const block_th *src, uint32_t t)
{
	a2_st(&dst->data[0 * THREADS_PER_LANE + t], src->a);
	a2_st(&dst->data[1 * THREADS_PER_LANE + t], src->b);
	a2_st(&dst->data[2 * THREADS_PER_LANE + t], src->c);
	a2_st(&dst->data[3 * THREADS_PER_LANE + t], src->d);
}

static __device__ __forceinline__ void a2_sh_load(block_th *dst, const block_g *src, uint32_t t)
{
	dst->a = src->data[0 * THREADS_PER_LANE + t];
	dst->b = src->data[1 * THREADS_PER_LANE + t];
	dst->c = src->data[2 * THREADS_PER_LANE + t];
	dst->d = src->data[3 * THREADS_PER_LANE + t];
}

static __device__ __forceinline__ void a2_sh_load_xor(block_th *dst, const block_g *src, uint32_t t)
{
	dst->a ^= src->data[0 * THREADS_PER_LANE + t];
	dst->b ^= src->data[1 * THREADS_PER_LANE + t];
	dst->c ^= src->data[2 * THREADS_PER_LANE + t];
	dst->d ^= src->data[3 * THREADS_PER_LANE + t];
}

static __device__ __forceinline__ void a2_sh_store(block_g *dst, const block_th *src, uint32_t t)
{
	dst->data[0 * THREADS_PER_LANE + t] = src->a;
	dst->data[1 * THREADS_PER_LANE + t] = src->b;
	dst->data[2 * THREADS_PER_LANE + t] = src->c;
	dst->data[3 * THREADS_PER_LANE + t] = src->d;
}

/* BlaMka: x + y + 2 * lo32(x) * lo32(y) */
static __device__ __forceinline__ uint64_t a2_f(uint64_t x, uint64_t y)
{
	const uint32_t xlo = (uint32_t)x, ylo = (uint32_t)y;
	return x + y + 2 * a2_u64(__umulhi(xlo, ylo), xlo * ylo);
}

static __device__ __forceinline__ void a2_g(block_th *block)
{
	uint64_t a = block->a, b = block->b, c = block->c, d = block->d;
	a = a2_f(a, b);
	d = a2_rotr64(d ^ a, 32);
	c = a2_f(c, d);
	b = a2_rotr64(b ^ c, 24);
	a = a2_f(a, b);
	d = a2_rotr64(d ^ a, 16);
	c = a2_f(c, d);
	b = a2_rotr64(b ^ c, 63);
	block->a = a; block->b = b; block->c = c; block->d = d;
}

/* source thread of qword i for the column and row passes of the permutation */
struct a2_shift1   { static __device__ __forceinline__ uint32_t apply(uint32_t t, uint32_t i) { return (t & 0x1c) | ((t + i) & 0x3); } };
struct a2_unshift1 { static __device__ __forceinline__ uint32_t apply(uint32_t t, uint32_t i) { i = (QWORDS_PER_THREAD - i) % QWORDS_PER_THREAD; return (t & 0x1c) | ((t + i) & 0x3); } };
struct a2_shift2   { static __device__ __forceinline__ uint32_t apply(uint32_t t, uint32_t i) { uint32_t lo = ((t & 0x1) | ((t & 0x10) >> 3)) + i; lo &= 0x3; return ((lo & 0x2) << 3) | (t & 0xe) | (lo & 0x1); } };
struct a2_unshift2 { static __device__ __forceinline__ uint32_t apply(uint32_t t, uint32_t i) { i = (QWORDS_PER_THREAD - i) % QWORDS_PER_THREAD; uint32_t lo = ((t & 0x1) | ((t & 0x10) >> 3)) + i; lo &= 0x3; return ((lo & 0x2) << 3) | (t & 0xe) | (lo & 0x1); } };

template <class S>
static __device__ __forceinline__ void a2_apply_shuffle(block_th *block, uint32_t t)
{
	for (uint32_t i = 0; i < QWORDS_PER_THREAD; i++) {
		uint64_t v = a2_th_get(block, i);
		v = a2_shfl64(v, S::apply(t, i));
		a2_th_set(block, i, v);
	}
}

static __device__ __forceinline__ void a2_transpose(block_th *block, uint32_t t)
{
	const uint32_t group = (t & 0x0C) >> 2;
	for (uint32_t i = 1; i < QWORDS_PER_THREAD; i++) {
		const uint32_t idx = group ^ i;
		uint64_t v = a2_th_get(block, idx);
		v = a2_shfl64(v, (i << 2) ^ t);
		a2_th_set(block, idx, v);
	}
}

/* the Argon2 permutation P over the block, rows then columns */
static __device__ void a2_shuffle_block(block_th *block, uint32_t t)
{
	a2_transpose(block, t);
	a2_g(block);
	a2_apply_shuffle<a2_shift1>(block, t);
	a2_g(block);
	a2_apply_shuffle<a2_unshift1>(block, t);
	a2_transpose(block, t);
	a2_g(block);
	a2_apply_shuffle<a2_shift2>(block, t);
	a2_g(block);
	a2_apply_shuffle<a2_unshift2>(block, t);
}

static __device__ void a2_next_addresses(block_th *addr, block_th *tmp, uint32_t thread_input, uint32_t t)
{
	addr->a = a2_u64(0, thread_input);
	addr->b = 0;
	addr->c = 0;
	addr->d = 0;
	a2_shuffle_block(addr, t);
	addr->a ^= a2_u64(0, thread_input);
	*tmp = *addr;
	a2_shuffle_block(addr, t);
	a2_xor(addr, tmp);
}

static __device__ void a2_ref_pos(uint32_t lanes, uint32_t segment_blocks,
	uint32_t pass, uint32_t lane, uint32_t slice, uint32_t offset,
	uint32_t *ref_lane, uint32_t *ref_index)
{
	const uint32_t lane_blocks = ARGON2_SYNC_POINTS * segment_blocks;

	*ref_lane = *ref_lane & (lanes - 1);   /* lanes is a power of two */

	uint32_t base;
	if (pass != 0) {
		base = lane_blocks - segment_blocks;
	} else {
		if (slice == 0)
			*ref_lane = lane;
		base = slice * segment_blocks;
	}

	uint32_t ref_area_size = base + offset - 1;
	if (*ref_lane != lane)
		ref_area_size = min(ref_area_size, base);

	*ref_index = __umulhi(*ref_index, *ref_index);
	*ref_index = ref_area_size - 1 - __umulhi(ref_area_size, *ref_index);

	if (pass != 0 && slice != ARGON2_SYNC_POINTS - 1) {
		*ref_index += (slice + 1) * segment_blocks;
		if (*ref_index >= lane_blocks)
			*ref_index -= lane_blocks;
	}
}

/* One block: G(prev XOR ref), XORed into the old block from the second pass on
 * in v1.3. Columns 2..7 live only in the shared cache c (their global slots
 * are never read); the last column must go to global, its natural slot would
 * alias the column-2 cache entry another lane may still read in this slice. */
static __device__ void a2_core(block_g *memory, block_g *mem_curr, block_th *prev, block_th *tmp,
	block_g *c, uint32_t lanes, uint32_t t, uint32_t pass, uint32_t version,
	uint32_t ref_index, uint32_t ref_lane, uint32_t curr_index, uint32_t lane)
{
	block_g *mem_ref = memory + ref_index * lanes + ref_lane;
	const bool ref_cached = ref_index < 8 && ref_index > 1;
	const bool curr_cached = curr_index < 8 && curr_index > 1;

	if (version != ARGON2_VERSION_10 && pass != 0) {
		if (curr_cached)
			a2_sh_load(tmp, &c[lane * ARGON2_SHARED_BLOCKS_PER_LANE + curr_index - 2], t);
		else
			a2_load(tmp, mem_curr, t);
		if (ref_cached)
			a2_sh_load_xor(prev, &c[ref_lane * ARGON2_SHARED_BLOCKS_PER_LANE + ref_index - 2], t);
		else
			a2_load_xor(prev, mem_ref, t);
		a2_xor(tmp, prev);
	} else {
		if (ref_cached)
			a2_sh_load_xor(prev, &c[ref_lane * ARGON2_SHARED_BLOCKS_PER_LANE + ref_index - 2], t);
		else
			a2_load_xor(prev, mem_ref, t);
		*tmp = *prev;
	}

	a2_shuffle_block(prev, t);
	a2_xor(prev, tmp);

	if (curr_cached)
		a2_sh_store(&c[lane * ARGON2_SHARED_BLOCKS_PER_LANE + curr_index - 2], prev, t);
	else
		a2_store(mem_curr, prev, t);
}

static __device__ void a2_step(block_g *memory, block_g *mem_curr, block_th *prev, block_th *tmp,
	block_th *addr, block_g *c, uint32_t lanes, uint32_t segment_blocks, uint32_t t,
	uint32_t *thread_input, uint32_t lane, uint32_t pass, uint32_t slice, uint32_t offset,
	uint32_t version, uint32_t type)
{
	uint32_t ref_index, ref_lane;

	if (type == ARGON2_I || (type == ARGON2_ID && pass == 0 && slice < ARGON2_SYNC_POINTS / 2)) {
		/* data-independent: J1/J2 from a generated address block */
		const uint32_t addr_index = offset % ARGON2_ADDRESSES_IN_BLOCK;
		if (addr_index == 0) {
			if (t == 6)
				++*thread_input;
			a2_next_addresses(addr, tmp, *thread_input, t);
		}
		uint64_t v = a2_th_get(addr, addr_index / THREADS_PER_LANE);
		v = a2_shfl64(v, addr_index % THREADS_PER_LANE);
		ref_index = (uint32_t)v;
		ref_lane = (uint32_t)(v >> 32);
	} else {
		const uint64_t v = a2_shfl64(prev->a, 0);
		ref_index = (uint32_t)v;
		ref_lane = (uint32_t)(v >> 32);
	}

	a2_ref_pos(lanes, segment_blocks, pass, lane, slice, offset, &ref_lane, &ref_index);
	a2_core(memory, mem_curr, prev, tmp, c, lanes, t, pass, version, ref_index, ref_lane,
		slice * segment_blocks + offset, lane);
}

/* Blocks 2.. of every lane for every pass. Launch: grid (1, 1, jobs), block
 * (32, lanes, 1), dynamic shared lanes * 6 blocks. Each job owns lanes * 4 *
 * segment_blocks contiguous blocks; the final XOR of the lanes' last column
 * is written into the job's block 0. */
__global__ void argon2_fill(block_g *memory, uint32_t passes, uint32_t lanes,
	uint32_t segment_blocks, uint32_t version, uint32_t type)
{
	const uint32_t job_id = blockIdx.z * blockDim.z + threadIdx.z;
	const uint32_t lane = threadIdx.y;
	uint32_t t = threadIdx.x;
	const uint32_t lane_blocks = ARGON2_SYNC_POINTS * segment_blocks;

	memory += (size_t)job_id * lanes * lane_blocks;

	block_th prev, addr, tmp;
	extern __shared__ block_g a2_cache[];

	/* address generator input, one word per thread: 0 pass, 1 lane, 2 slice,
	 * 3 memory blocks, 4 passes, 5 type, 6 address block counter */
	uint32_t thread_input = 0;
	if (type == ARGON2_I || type == ARGON2_ID) {
		switch (t) {
		case 1: thread_input = lane; break;
		case 3: thread_input = lanes * lane_blocks; break;
		case 4: thread_input = passes; break;
		case 5: thread_input = type; break;
		default: thread_input = 0; break;
		}
		/* offsets 0 and 1 are filled already: seed the first address block */
		if (segment_blocks > 2) {
			if (t == 6)
				++thread_input;
			a2_next_addresses(&addr, &tmp, thread_input, t);
		}
	}

	block_g *mem_lane = memory + lane;
	block_g *mem_prev = mem_lane + 1 * lanes;
	block_g *mem_curr = mem_lane + 2 * lanes;

	a2_load(&prev, mem_prev, t);

	uint32_t skip = 2;
	for (uint32_t pass = 0; pass < passes; ++pass) {
		for (uint32_t slice = 0; slice < ARGON2_SYNC_POINTS; ++slice) {
			for (uint32_t offset = 0; offset < segment_blocks; ++offset) {
				if (skip > 0) {
					--skip;
					continue;
				}
				a2_step(memory, mem_curr, &prev, &tmp, &addr, a2_cache, lanes, segment_blocks,
					t, &thread_input, lane, pass, slice, offset, version, type);
				mem_curr += lanes;
			}

			__syncthreads();

			if (type == ARGON2_I || type == ARGON2_ID) {
				if (t == 2)
					++thread_input;		/* next slice */
				if (t == 6)
					thread_input = 0;	/* restart the address block counter */
			}
		}

		/* Argon2id uses addresses only in pass 0 */
		if (type == ARGON2_I) {
			if (t == 0)
				++thread_input;
			if (t == 2)
				thread_input = 0;
		}
		mem_curr = mem_lane;
	}

	__syncthreads();

	/* final block: XOR of the lanes' last-column blocks into block 0 */
	t = threadIdx.x + blockDim.x * threadIdx.y;
	const uint32_t *last_col = (const uint32_t *)(memory + (size_t)(lane_blocks - 1) * lanes);
	for (uint32_t j = t; j < 256; j += blockDim.x * blockDim.y) {
		uint32_t buf = 0;
		for (uint32_t i = 0; i < lanes; i++)
			buf ^= last_col[j + i * 256];
		((uint32_t *)memory)[j] = buf;
	}
}
