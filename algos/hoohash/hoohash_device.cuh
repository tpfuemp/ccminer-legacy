/*
 * HoohashV110 device code, after ccminer-tpfuemp: a translation of ref/hoohash.c
 * (PePe-core) with libdevice transcendentals. The file that includes it must be
 * built with --fmad=false: the reference rounds every product and sum.
 *
 * The 64x64 matmul is sequential (the running sum of a row and the switch value
 * sw carry over every step), so one thread hashes one nonce; the lanes of a warp
 * share the costly transform steps (hoo_matmul).
 */
#pragma once
#include <stdint.h>
#include <math.h>
#include "cuda/blake3_device.cuh"
#include "cuda/cuda_compat.h"

#define HOO_PI  3.14159265358979323846
#define HOO_EPS 1e-9
#define HOO_CTM 0.000001   /* COMPLEX_TRANSFORM_MULTIPLIER */

struct hoo_xoshiro { uint64_t s0, s1, s2, s3; };

__device__ __forceinline__ uint64_t hoo_rotl64(uint64_t x, int k)
{
	return (x << k) | (x >> (64 - k));
}

__device__ __forceinline__ uint64_t hoo_read_u64le(const uint8_t *d)
{
	uint64_t r = 0;
	for (int i = 0; i < 8; i++) r |= ((uint64_t)d[i]) << (i * 8);
	return r;
}

__device__ __forceinline__ uint32_t hoo_read_u32be(const uint8_t *d)
{
	return ((uint32_t)d[0] << 24) | ((uint32_t)d[1] << 16) | ((uint32_t)d[2] << 8) | (uint32_t)d[3];
}

__device__ __forceinline__ uint64_t hoo_xoshiro_gen(hoo_xoshiro *x)
{
	const uint64_t res = hoo_rotl64(x->s0 + x->s3, 23) + x->s0;
	const uint64_t t = x->s1 << 17;
	x->s2 ^= x->s0;
	x->s3 ^= x->s1;
	x->s1 ^= x->s2;
	x->s0 ^= x->s3;
	x->s2 ^= t;
	x->s3 = hoo_rotl64(x->s3, 45);
	return res;
}

/* exp(sin(x) + cos(x)); sincos shares one argument reduction and gives the same
 * values as separate sin and cos */
__device__ __forceinline__ double hoo_Medium(double x)
{
	double s, c;
	sincos(x, &s, &c);
	return exp(s + c);
}

__device__ __forceinline__ double hoo_Intermediate(double x)
{
	if (fabs(x - HOO_PI / 2) < HOO_EPS || fabs(x - 3 * HOO_PI / 2) < HOO_EPS)
		return 0.0;
	const double s = sin(x);
	return s * s;
}

__device__ __forceinline__ double hoo_High(double x)
{
	return 1.0 / sqrt(fabs(x) + 1.0);
}

/* f1 = frac(u), f2 = frac(2u) for u = x * CTM / 8: the reference's second floor of
 * x * CTM / 4 gives the same f2 exactly (2u and 2 * f1 are exact) */
__device__ double hoo_ComplexNonLinear(double x)
{
	const double u  = (x * HOO_CTM) / 8.0;
	const double f1 = u - floor(u);
	const double t  = f1 + f1;
	const double f2 = (t >= 1.0) ? (t - 1.0) : t;
	if (f1 < 0.33) {
		if      (f2 < 0.25) return hoo_Medium(x + (1 + f2));
		else if (f2 < 0.5)  return hoo_Medium(x - (1 + f2));
		else if (f2 < 0.75) return hoo_Medium(x * (1 + f2));
		else                return hoo_Medium(x / (1 + f2));
	} else if (f1 < 0.66) {
		if      (f2 < 0.25) return hoo_Intermediate(x + (1 + f2));
		else if (f2 < 0.5)  return hoo_Intermediate(x - (1 + f2));
		else if (f2 < 0.75) return hoo_Intermediate(x * (1 + f2));
		else                return hoo_Intermediate(x / (1 + f2));
	} else {
		if      (f2 < 0.25) return hoo_High(x + (1 + f2));
		else if (f2 < 0.5)  return hoo_High(x - (1 + f2));
		else if (f2 < 0.75) return hoo_High(x * (1 + f2));
		else                return hoo_High(x / (1 + f2));
	}
}

/* as the reference: on NaN or Inf the loop only shrinks input until it returns 0 */
__device__ double hoo_SafeComplexTransform(double input)
{
	double rounds = 1;
	const double v = hoo_ComplexNonLinear(input);
	while (isnan(v) || isinf(v)) {
		input = input * 0.1;
		if (input <= 0.0000000000001) return 0;
		rounds++;
	}
	return v * rounds;
}

/* sw <= 0.02 with sw = p / 1024 - floor(p / 1024), for a finite p >= 0: p / 1024 and
 * the fraction are exact, so this is (p mod 1024) <= 1024 * 0.02, compared exactly on
 * the bit patterns (after ccminer-tpfuemp) */
__device__ __forceinline__ bool hoo_sw_low(double p)
{
	const uint64_t b  = (uint64_t)__double_as_longlong(p);
	const uint64_t cb = (uint64_t)__double_as_longlong(0.02 * 1024.0);
	const int e = (int)(b >> 52) - 1023;
	if (e < 10) return b <= cb;       /* p < 1024: the remainder is p itself */
	if (e >= 62) return true;         /* p is a multiple of 1024 */
	const uint64_t m   = (b & 0xFFFFFFFFFFFFFull) | (1ull << 52);
	const uint64_t rem = m & ((1ull << (62 - e)) - 1);
	const uint64_t cm  = (cb & 0xFFFFFFFFFFFFFull) | (1ull << 52);
	return rem <= (cm >> (e - ((int)(cb >> 52) - 1023)));
}

/* mat from the seed, and matdiv = mat * 0.0001 (the reference's mat * divider) */
__device__ void hoo_generateMatrix(const uint8_t *seed, double mat[64][64], double matdiv[64][64])
{
	hoo_xoshiro st;
	st.s0 = hoo_read_u64le(seed + 0);
	st.s1 = hoo_read_u64le(seed + 8);
	st.s2 = hoo_read_u64le(seed + 16);
	st.s3 = hoo_read_u64le(seed + 24);
	const double normalize = 1000000.0;
	for (int i = 0; i < 64; i++)
		for (int j = 0; j < 64; j++) {
			const uint32_t lo = (uint32_t)(hoo_xoshiro_gen(&st) & 0xFFFFFFFFu);
			mat[i][j] = (double)lo / (double)0xFFFFFFFFu * normalize;
			matdiv[i][j] = mat[i][j] * 0.0001;
		}
}

/* The transform steps of a warp run together once this many lanes wait for one, or
 * when no lane can take a cheap step (set per arch by the includer) */
#ifndef HOO_WAIT_K
#define HOO_WAIT_K 24
#endif

/* hashBytes = BLAKE3 of the full header, nonceMod = the header's byte 76 (the nonce's
 * low byte as the node reads it). tab[v * 4096 + i * 64 + j] = matdiv[i][j] * v.
 *
 * Each lane walks its own (i, j) in the reference order, so its sums are the
 * reference's bit for bit. A lane whose next step needs the transform (sw <= 0.02)
 * waits while others take cheap steps; the waiting lanes then evaluate the transform
 * together instead of one after another. The whole warp must call this (full-mask
 * votes); a lane with live == false only votes and writes no output. */
__device__ void hoo_matmul(const double mat[64][64], const double *tab,
	const uint8_t *hashBytes, uint8_t *output, const double nonceMod, const bool live)
{
	uint32_t H[8];
	for (int k = 0; k < 8; k++) H[k] = hoo_read_u32be(hashBytes + k * 4);
	const double hashXor = (double)(H[0] ^ H[1] ^ H[2] ^ H[3] ^ H[4] ^ H[5] ^ H[6] ^ H[7]);
	const double multiplier = 1234;

	/* the 64 nibbles (vector[] of the reference), high nibble of each byte first */
	const uint64_t nw[4] = {
		((uint64_t)H[0] << 32) | H[1], ((uint64_t)H[2] << 32) | H[3],
		((uint64_t)H[4] << 32) | H[5], ((uint64_t)H[6] << 32) | H[7] };

	uint8_t result[32];
	int i = live ? 0 : 64, j = 0;
	double prod = 0.0;
	bool low = true;      /* sw <= 0.02; sw starts at 0 and carries over the rows */
	uint64_t pend = 0;

	for (;;) {
		const bool done = i >= 64;
		const uint64_t w = j < 16 ? nw[0] : j < 32 ? nw[1] : j < 48 ? nw[2] : nw[3];
		const uint32_t v = done ? 0 : (uint32_t)(w >> (60 - 4 * (j & 15))) & 0xF;
		/* v == 0 adds +0.0 on both branches (the transform is finite and >= 0):
		 * it is a cheap step, sw is still recomputed */
		const bool needT = !done && v != 0 && low;
		const uint32_t mT = WARP_BALLOT(needT);
		const uint32_t mC = WARP_BALLOT(!done && !needT);
		if (!(mT | mC))
			break;
		bool step = false;
		if (mT && (__popc(mT) >= HOO_WAIT_K || !mC)) {
			if (needT) {
				const double input = (mat[i][j] * hashXor * (double)v + nonceMod);
				prod += hoo_SafeComplexTransform(input) * (double)v * multiplier;
				step = true;
			}
		} else if (!done && !needT) {
			if (v) prod += tab[v * 4096 + i * 64 + j];
			step = true;
		}
		if (step) {
			low = hoo_sw_low(prod);
			if (++j == 64) {
				/* the row sums are >= 0 and small, so the conversion matches the CPU's */
				const uint64_t pv = (uint64_t)prod;
				if (i & 1) result[i >> 1] = hashBytes[i >> 1] ^ (uint8_t)((pend + pv) & 0xFF);
				else pend = pv;
				i++; j = 0; prod = 0.0;
			}
		}
	}
	if (live)
		blake3_256(result, 32, output);
}
