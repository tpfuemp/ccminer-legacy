// SPDX-License-Identifier: GPL-3.0-or-later
/*
 * SHA-256 building blocks for host and device (from the tpruvot sha256t
 * kernels); algo kernels call these instead of re-implementing them.
 */

#ifndef CUDA_SHA256_DEVICE_CUH
#define CUDA_SHA256_DEVICE_CUH

#include <stdint.h>
#include <cuda_helper.h> // ROTR32 (single SHF on device, shift/or on host)

/* shared by the host and the device copies */
#define SHA256_K_INIT { \
	0x428A2F98U, 0x71374491U, 0xB5C0FBCFU, 0xE9B5DBA5U, 0x3956C25BU, 0x59F111F1U, 0x923F82A4U, 0xAB1C5ED5U, \
	0xD807AA98U, 0x12835B01U, 0x243185BEU, 0x550C7DC3U, 0x72BE5D74U, 0x80DEB1FEU, 0x9BDC06A7U, 0xC19BF174U, \
	0xE49B69C1U, 0xEFBE4786U, 0x0FC19DC6U, 0x240CA1CCU, 0x2DE92C6FU, 0x4A7484AAU, 0x5CB0A9DCU, 0x76F988DAU, \
	0x983E5152U, 0xA831C66DU, 0xB00327C8U, 0xBF597FC7U, 0xC6E00BF3U, 0xD5A79147U, 0x06CA6351U, 0x14292967U, \
	0x27B70A85U, 0x2E1B2138U, 0x4D2C6DFCU, 0x53380D13U, 0x650A7354U, 0x766A0ABBU, 0x81C2C92EU, 0x92722C85U, \
	0xA2BFE8A1U, 0xA81A664BU, 0xC24B8B70U, 0xC76C51A3U, 0xD192E819U, 0xD6990624U, 0xF40E3585U, 0x106AA070U, \
	0x19A4C116U, 0x1E376C08U, 0x2748774CU, 0x34B0BCB5U, 0x391C0CB3U, 0x4ED8AA4AU, 0x5B9CCA4FU, 0x682E6FF3U, \
	0x748F82EEU, 0x78A5636FU, 0x84C87814U, 0x8CC70208U, 0x90BEFFFAU, 0xA4506CEBU, 0xBEF9A3F7U, 0xC67178F2U }

#define SHA256_H_INIT { \
	0x6A09E667U, 0xBB67AE85U, 0x3C6EF372U, 0xA54FF53AU, \
	0x510E527FU, 0x9B05688CU, 0x1F83D9ABU, 0x5BE0CD19U }

static const uint32_t h_sha256_K[64] = SHA256_K_INIT;
static const uint32_t h_sha256_H[8]  = SHA256_H_INIT;

#ifdef __CUDACC__
/* statically initialized, one copy per translation unit */
static __constant__ uint32_t c_sha256_K[64] = SHA256_K_INIT;
static __constant__ uint32_t c_sha256_H[8]  = SHA256_H_INIT;
#endif

/* --------------------------------------------------------------------------
 * Round primitives (Ch/Maj map to one LOP3 on sm_50+).
 * ------------------------------------------------------------------------ */

__host__ __device__ __forceinline__
uint32_t sha256_ch(const uint32_t e, const uint32_t f, const uint32_t g)
{
	return ((f ^ g) & e) ^ g; // xandx form
}

__host__ __device__ __forceinline__
uint32_t sha256_maj(const uint32_t a, const uint32_t b, const uint32_t c)
{
	return (b & c) | ((b | c) & a); // andor form
}

__host__ __device__ __forceinline__
uint32_t sha256_bsg0(const uint32_t x) { return ROTR32(x, 2) ^ ROTR32(x, 13) ^ ROTR32(x, 22); }

__host__ __device__ __forceinline__
uint32_t sha256_bsg1(const uint32_t x) { return ROTR32(x, 6) ^ ROTR32(x, 11) ^ ROTR32(x, 25); }

__host__ __device__ __forceinline__
uint32_t sha256_ssg0(const uint32_t x) { return ROTR32(x, 7) ^ ROTR32(x, 18) ^ (x >> 3); }

__host__ __device__ __forceinline__
uint32_t sha256_ssg1(const uint32_t x) { return ROTR32(x, 17) ^ ROTR32(x, 19) ^ (x >> 10); }

/* One compression round, message word supplied by the caller (rounds 0..15,
 * or any round whose schedule word is precomputed). */
__host__ __device__ __forceinline__
void sha256_round(const uint32_t a, const uint32_t b, const uint32_t c, uint32_t &d,
	const uint32_t e, const uint32_t f, const uint32_t g, uint32_t &h,
	const uint32_t in, const uint32_t k)
{
	const uint32_t t1 = h + sha256_bsg1(e) + sha256_ch(e, f, g) + k + in;
	const uint32_t t2 = sha256_bsg0(a) + sha256_maj(a, b, c);
	d += t1;
	h = t1 + t2;
}

/* One compression round for rounds 16..63 over a 16-word rolling schedule:
 * extends in[pc] in place, then compresses. */
__host__ __device__ __forceinline__
void sha256_round_sched(const uint32_t a, const uint32_t b, const uint32_t c, uint32_t &d,
	const uint32_t e, const uint32_t f, const uint32_t g, uint32_t &h,
	uint32_t *in, const uint32_t pc, const uint32_t k)
{
	const uint32_t inx1 = in[(pc - 2) & 0xF];
	const uint32_t inx2 = in[(pc - 7) & 0xF];
	const uint32_t inx3 = in[(pc - 15) & 0xF];

	in[pc] += sha256_ssg1(inx1) + inx2 + sha256_ssg0(inx3);

	sha256_round(a, b, c, d, e, f, g, h, in[pc], k);
}

/* --------------------------------------------------------------------------
 * Transforms. `in` (16 words, host order) is overwritten, `state` is updated
 * in place. `k`: h_sha256_K on the host, c_sha256_K on the device.
 * ------------------------------------------------------------------------ */

__host__ __device__ static inline
void sha256_transform_full(uint32_t *in, uint32_t *state, const uint32_t *k)
{
	uint32_t a = state[0];
	uint32_t b = state[1];
	uint32_t c = state[2];
	uint32_t d = state[3];
	uint32_t e = state[4];
	uint32_t f = state[5];
	uint32_t g = state[6];
	uint32_t h = state[7];

	sha256_round(a, b, c, d, e, f, g, h, in[ 0], k[ 0]);
	sha256_round(h, a, b, c, d, e, f, g, in[ 1], k[ 1]);
	sha256_round(g, h, a, b, c, d, e, f, in[ 2], k[ 2]);
	sha256_round(f, g, h, a, b, c, d, e, in[ 3], k[ 3]);
	sha256_round(e, f, g, h, a, b, c, d, in[ 4], k[ 4]);
	sha256_round(d, e, f, g, h, a, b, c, in[ 5], k[ 5]);
	sha256_round(c, d, e, f, g, h, a, b, in[ 6], k[ 6]);
	sha256_round(b, c, d, e, f, g, h, a, in[ 7], k[ 7]);
	sha256_round(a, b, c, d, e, f, g, h, in[ 8], k[ 8]);
	sha256_round(h, a, b, c, d, e, f, g, in[ 9], k[ 9]);
	sha256_round(g, h, a, b, c, d, e, f, in[10], k[10]);
	sha256_round(f, g, h, a, b, c, d, e, in[11], k[11]);
	sha256_round(e, f, g, h, a, b, c, d, in[12], k[12]);
	sha256_round(d, e, f, g, h, a, b, c, in[13], k[13]);
	sha256_round(c, d, e, f, g, h, a, b, in[14], k[14]);
	sha256_round(b, c, d, e, f, g, h, a, in[15], k[15]);

	for (int i = 0; i < 3; i++)
	{
		sha256_round_sched(a, b, c, d, e, f, g, h, in,  0, k[16 + 16 * i]);
		sha256_round_sched(h, a, b, c, d, e, f, g, in,  1, k[17 + 16 * i]);
		sha256_round_sched(g, h, a, b, c, d, e, f, in,  2, k[18 + 16 * i]);
		sha256_round_sched(f, g, h, a, b, c, d, e, in,  3, k[19 + 16 * i]);
		sha256_round_sched(e, f, g, h, a, b, c, d, in,  4, k[20 + 16 * i]);
		sha256_round_sched(d, e, f, g, h, a, b, c, in,  5, k[21 + 16 * i]);
		sha256_round_sched(c, d, e, f, g, h, a, b, in,  6, k[22 + 16 * i]);
		sha256_round_sched(b, c, d, e, f, g, h, a, in,  7, k[23 + 16 * i]);
		sha256_round_sched(a, b, c, d, e, f, g, h, in,  8, k[24 + 16 * i]);
		sha256_round_sched(h, a, b, c, d, e, f, g, in,  9, k[25 + 16 * i]);
		sha256_round_sched(g, h, a, b, c, d, e, f, in, 10, k[26 + 16 * i]);
		sha256_round_sched(f, g, h, a, b, c, d, e, in, 11, k[27 + 16 * i]);
		sha256_round_sched(e, f, g, h, a, b, c, d, in, 12, k[28 + 16 * i]);
		sha256_round_sched(d, e, f, g, h, a, b, c, in, 13, k[29 + 16 * i]);
		sha256_round_sched(c, d, e, f, g, h, a, b, in, 14, k[30 + 16 * i]);
		sha256_round_sched(b, c, d, e, f, g, h, a, in, 15, k[31 + 16 * i]);
	}

	state[0] += a;
	state[1] += b;
	state[2] += c;
	state[3] += d;
	state[4] += e;
	state[5] += f;
	state[6] += g;
	state[7] += h;
}

/* Truncated final transform: only state[6] and state[7] are valid (rounds
 * 62/63 skipped). For target checks only, never as a digest. */
__host__ __device__ static inline
void sha256_final_to_target(uint32_t *in, uint32_t *state, const uint32_t *k)
{
	uint32_t a = state[0];
	uint32_t b = state[1];
	uint32_t c = state[2];
	uint32_t d = state[3];
	uint32_t e = state[4];
	uint32_t f = state[5];
	uint32_t g = state[6];
	uint32_t h = state[7];

	sha256_round(a, b, c, d, e, f, g, h, in[ 0], k[ 0]);
	sha256_round(h, a, b, c, d, e, f, g, in[ 1], k[ 1]);
	sha256_round(g, h, a, b, c, d, e, f, in[ 2], k[ 2]);
	sha256_round(f, g, h, a, b, c, d, e, in[ 3], k[ 3]);
	sha256_round(e, f, g, h, a, b, c, d, in[ 4], k[ 4]);
	sha256_round(d, e, f, g, h, a, b, c, in[ 5], k[ 5]);
	sha256_round(c, d, e, f, g, h, a, b, in[ 6], k[ 6]);
	sha256_round(b, c, d, e, f, g, h, a, in[ 7], k[ 7]);
	sha256_round(a, b, c, d, e, f, g, h, in[ 8], k[ 8]);
	sha256_round(h, a, b, c, d, e, f, g, in[ 9], k[ 9]);
	sha256_round(g, h, a, b, c, d, e, f, in[10], k[10]);
	sha256_round(f, g, h, a, b, c, d, e, in[11], k[11]);
	sha256_round(e, f, g, h, a, b, c, d, in[12], k[12]);
	sha256_round(d, e, f, g, h, a, b, c, in[13], k[13]);
	sha256_round(c, d, e, f, g, h, a, b, in[14], k[14]);
	sha256_round(b, c, d, e, f, g, h, a, in[15], k[15]);

	for (int i = 0; i < 2; i++)
	{
		sha256_round_sched(a, b, c, d, e, f, g, h, in,  0, k[16 + 16 * i]);
		sha256_round_sched(h, a, b, c, d, e, f, g, in,  1, k[17 + 16 * i]);
		sha256_round_sched(g, h, a, b, c, d, e, f, in,  2, k[18 + 16 * i]);
		sha256_round_sched(f, g, h, a, b, c, d, e, in,  3, k[19 + 16 * i]);
		sha256_round_sched(e, f, g, h, a, b, c, d, in,  4, k[20 + 16 * i]);
		sha256_round_sched(d, e, f, g, h, a, b, c, in,  5, k[21 + 16 * i]);
		sha256_round_sched(c, d, e, f, g, h, a, b, in,  6, k[22 + 16 * i]);
		sha256_round_sched(b, c, d, e, f, g, h, a, in,  7, k[23 + 16 * i]);
		sha256_round_sched(a, b, c, d, e, f, g, h, in,  8, k[24 + 16 * i]);
		sha256_round_sched(h, a, b, c, d, e, f, g, in,  9, k[25 + 16 * i]);
		sha256_round_sched(g, h, a, b, c, d, e, f, in, 10, k[26 + 16 * i]);
		sha256_round_sched(f, g, h, a, b, c, d, e, in, 11, k[27 + 16 * i]);
		sha256_round_sched(e, f, g, h, a, b, c, d, in, 12, k[28 + 16 * i]);
		sha256_round_sched(d, e, f, g, h, a, b, c, in, 13, k[29 + 16 * i]);
		sha256_round_sched(c, d, e, f, g, h, a, b, in, 14, k[30 + 16 * i]);
		sha256_round_sched(b, c, d, e, f, g, h, a, in, 15, k[31 + 16 * i]);
	}

	sha256_round_sched(a, b, c, d, e, f, g, h, in,  0, k[16 + 16 * 2]);
	sha256_round_sched(h, a, b, c, d, e, f, g, in,  1, k[17 + 16 * 2]);
	sha256_round_sched(g, h, a, b, c, d, e, f, in,  2, k[18 + 16 * 2]);
	sha256_round_sched(f, g, h, a, b, c, d, e, in,  3, k[19 + 16 * 2]);
	sha256_round_sched(e, f, g, h, a, b, c, d, in,  4, k[20 + 16 * 2]);
	sha256_round_sched(d, e, f, g, h, a, b, c, in,  5, k[21 + 16 * 2]);
	sha256_round_sched(c, d, e, f, g, h, a, b, in,  6, k[22 + 16 * 2]);
	sha256_round_sched(b, c, d, e, f, g, h, a, in,  7, k[23 + 16 * 2]);
	sha256_round_sched(a, b, c, d, e, f, g, h, in,  8, k[24 + 16 * 2]);
	sha256_round_sched(h, a, b, c, d, e, f, g, in,  9, k[25 + 16 * 2]);
	sha256_round_sched(g, h, a, b, c, d, e, f, in, 10, k[26 + 16 * 2]);
	sha256_round_sched(f, g, h, a, b, c, d, e, in, 11, k[27 + 16 * 2]);
	sha256_round_sched(e, f, g, h, a, b, c, d, in, 12, k[28 + 16 * 2]);
	sha256_round_sched(d, e, f, g, h, a, b, c, in, 13, k[29 + 16 * 2]);

	state[6] += g;
	state[7] += h;
}

/* Block 2 of an 80-byte header resumed at round 4, from
 * sha256_prehash_split_host(nsplit 3) and sha256_preextend_w3_host().
 * in[4..15] hold the padding, `ms` is the input state, `out` the digest. */
__device__ static inline
void sha256_transform_80_from_pre4(uint32_t *in, const uint32_t *pre, const uint32_t *wx,
	const uint32_t nonce, const uint32_t *ms, uint32_t *out, const uint32_t *k)
{
	/* physical registers hold the logical round-4 state shifted by 4 (the
	 * call pattern below continues the standard rotation at round 4) */
	uint32_t e = pre[0] + nonce;
	uint32_t f = pre[1];
	uint32_t g = pre[2];
	uint32_t h = pre[3];
	uint32_t a = pre[4] + nonce;
	uint32_t b = pre[5];
	uint32_t c = pre[6];
	uint32_t d = pre[7];

	sha256_round(e, f, g, h, a, b, c, d, in[ 4], k[ 4]);
	sha256_round(d, e, f, g, h, a, b, c, in[ 5], k[ 5]);
	sha256_round(c, d, e, f, g, h, a, b, in[ 6], k[ 6]);
	sha256_round(b, c, d, e, f, g, h, a, in[ 7], k[ 7]);
	sha256_round(a, b, c, d, e, f, g, h, in[ 8], k[ 8]);
	sha256_round(h, a, b, c, d, e, f, g, in[ 9], k[ 9]);
	sha256_round(g, h, a, b, c, d, e, f, in[10], k[10]);
	sha256_round(f, g, h, a, b, c, d, e, in[11], k[11]);
	sha256_round(e, f, g, h, a, b, c, d, in[12], k[12]);
	sha256_round(d, e, f, g, h, a, b, c, in[13], k[13]);
	sha256_round(c, d, e, f, g, h, a, b, in[14], k[14]);
	sha256_round(b, c, d, e, f, g, h, a, in[15], k[15]);

	/* rounds 16..19: schedule words precomputed up to their nonce term */
	in[0] = wx[0];
	sha256_round(a, b, c, d, e, f, g, h, in[0], k[16]);
	in[1] = wx[1];
	sha256_round(h, a, b, c, d, e, f, g, in[1], k[17]);
	in[2] = wx[2] + sha256_ssg0(nonce);
	sha256_round(g, h, a, b, c, d, e, f, in[2], k[18]);
	in[3] = wx[3] + nonce;
	sha256_round(f, g, h, a, b, c, d, e, in[3], k[19]);

	sha256_round_sched(e, f, g, h, a, b, c, d, in,  4, k[20]);
	sha256_round_sched(d, e, f, g, h, a, b, c, in,  5, k[21]);
	sha256_round_sched(c, d, e, f, g, h, a, b, in,  6, k[22]);
	sha256_round_sched(b, c, d, e, f, g, h, a, in,  7, k[23]);
	sha256_round_sched(a, b, c, d, e, f, g, h, in,  8, k[24]);
	sha256_round_sched(h, a, b, c, d, e, f, g, in,  9, k[25]);
	sha256_round_sched(g, h, a, b, c, d, e, f, in, 10, k[26]);
	sha256_round_sched(f, g, h, a, b, c, d, e, in, 11, k[27]);
	sha256_round_sched(e, f, g, h, a, b, c, d, in, 12, k[28]);
	sha256_round_sched(d, e, f, g, h, a, b, c, in, 13, k[29]);
	sha256_round_sched(c, d, e, f, g, h, a, b, in, 14, k[30]);
	sha256_round_sched(b, c, d, e, f, g, h, a, in, 15, k[31]);

	for (int i = 1; i < 3; i++)
	{
		sha256_round_sched(a, b, c, d, e, f, g, h, in,  0, k[16 + 16 * i]);
		sha256_round_sched(h, a, b, c, d, e, f, g, in,  1, k[17 + 16 * i]);
		sha256_round_sched(g, h, a, b, c, d, e, f, in,  2, k[18 + 16 * i]);
		sha256_round_sched(f, g, h, a, b, c, d, e, in,  3, k[19 + 16 * i]);
		sha256_round_sched(e, f, g, h, a, b, c, d, in,  4, k[20 + 16 * i]);
		sha256_round_sched(d, e, f, g, h, a, b, c, in,  5, k[21 + 16 * i]);
		sha256_round_sched(c, d, e, f, g, h, a, b, in,  6, k[22 + 16 * i]);
		sha256_round_sched(b, c, d, e, f, g, h, a, in,  7, k[23 + 16 * i]);
		sha256_round_sched(a, b, c, d, e, f, g, h, in,  8, k[24 + 16 * i]);
		sha256_round_sched(h, a, b, c, d, e, f, g, in,  9, k[25 + 16 * i]);
		sha256_round_sched(g, h, a, b, c, d, e, f, in, 10, k[26 + 16 * i]);
		sha256_round_sched(f, g, h, a, b, c, d, e, in, 11, k[27 + 16 * i]);
		sha256_round_sched(e, f, g, h, a, b, c, d, in, 12, k[28 + 16 * i]);
		sha256_round_sched(d, e, f, g, h, a, b, c, in, 13, k[29 + 16 * i]);
		sha256_round_sched(c, d, e, f, g, h, a, b, in, 14, k[30 + 16 * i]);
		sha256_round_sched(b, c, d, e, f, g, h, a, in, 15, k[31 + 16 * i]);
	}

	out[0] = ms[0] + a;
	out[1] = ms[1] + b;
	out[2] = ms[2] + c;
	out[3] = ms[3] + d;
	out[4] = ms[4] + e;
	out[5] = ms[5] + f;
	out[6] = ms[6] + g;
	out[7] = ms[7] + h;
}

/* --------------------------------------------------------------------------
 * Host helpers.
 * ------------------------------------------------------------------------ */

/* Midstate of the first 64-byte block of an 80-byte header (words already in
 * host order). */
static inline void sha256_midstate_host(const uint32_t *data16, uint32_t midstate[8])
{
	uint32_t in[16];
	for (int i = 0; i < 16; i++) in[i] = data16[i];
	for (int i = 0; i < 8; i++)  midstate[i] = h_sha256_H[i];
	sha256_transform_full(in, midstate, h_sha256_K);
}

/* State after `nsplit` constant rounds, with round `nsplit` (word = nonce)
 * split into constant parts: pre[0] = a - nonce, pre[4] = e - nonce. The
 * device adds the nonce and resumes at round nsplit+1. */
static inline void sha256_prehash_split_host(const uint32_t midstate[8], const uint32_t *w,
	const int nsplit, uint32_t pre[8])
{
	uint32_t a = midstate[0], b = midstate[1], c = midstate[2], d = midstate[3];
	uint32_t e = midstate[4], f = midstate[5], g = midstate[6], h = midstate[7];
	for (int i = 0; i < nsplit; i++) {
		const uint32_t t1 = h + sha256_bsg1(e) + sha256_ch(e, f, g) + h_sha256_K[i] + w[i];
		const uint32_t t2 = sha256_bsg0(a) + sha256_maj(a, b, c);
		h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
	}
	const uint32_t T1c = h + sha256_bsg1(e) + sha256_ch(e, f, g) + h_sha256_K[nsplit];
	const uint32_t T2c = sha256_bsg0(a) + sha256_maj(a, b, c);
	pre[0] = T1c + T2c;
	pre[1] = a;
	pre[2] = b;
	pre[3] = c;
	pre[4] = d + T1c;
	pre[5] = e;
	pre[6] = f;
	pre[7] = g;
}

/* Schedule words 16..19 without their nonce terms (word 3 = nonce): wx[0],
 * wx[1] complete; the device adds ssg0(nonce) to wx[2] and nonce to wx[3]. */
static inline void sha256_preextend_w3_host(const uint32_t *w, uint32_t wx[4])
{
	wx[0] = w[0] + sha256_ssg0(w[1]) + w[9]  + sha256_ssg1(w[14]);
	wx[1] = w[1] + sha256_ssg0(w[2]) + w[10] + sha256_ssg1(w[15]);
	wx[2] = w[2] + w[11] + sha256_ssg1(wx[0]);
	wx[3] = sha256_ssg0(w[4]) + w[12] + sha256_ssg1(wx[1]);
}

#endif /* CUDA_SHA256_DEVICE_CUH */
