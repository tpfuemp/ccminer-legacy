/*
 * HoohashV110 device code, after ccminer-tpfuemp: a translation of ref/hoohash.c
 * (PePe-core) with libdevice transcendentals. The file that includes it must be
 * built with --fmad=false: the reference rounds every product and sum.
 *
 * The 64x64 matmul is sequential (the running sum of a row and the switch value
 * sw carry over every step), so one thread hashes one nonce.
 */
#pragma once
#include <stdint.h>
#include <math.h>
#include "cuda/blake3_device.cuh"

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

__device__ __forceinline__ double hoo_TransformFactor(double x)
{
	const double granularity = 1024.0;
	return x / granularity - floor(x / granularity);
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

/* hashBytes = BLAKE3 of the full header, nonce = LE u32 at byte 76 */
__device__ void hoo_matmul(const double mat[64][64], const double matdiv[64][64],
	const uint8_t *hashBytes, uint8_t *output, uint64_t nonce)
{
	uint8_t vector[64], result[32];
	double product[64];
	uint32_t H[8];

	for (int i = 0; i < 64; i++) product[i] = 0.0;
	for (int i = 0; i < 8; i++) H[i] = hoo_read_u32be(hashBytes + i * 4);
	const double hashXor = (double)(H[0] ^ H[1] ^ H[2] ^ H[3] ^ H[4] ^ H[5] ^ H[6] ^ H[7]);
	const double nonceMod = (double)(nonce & 0xFF);
	const double multiplier = 1234;
	double sw = 0.0;

	for (int i = 0; i < 32; i++) {
		vector[2 * i]     = hashBytes[i] >> 4;
		vector[2 * i + 1] = hashBytes[i] & 0x0F;
	}

	for (int i = 0; i < 64; i++) {
		for (int j = 0; j < 64; j++) {
			/* vector[j] == 0 adds +0.0 on both branches (the transform is finite
			 * and >= 0): skipping it is exact, sw is still recomputed */
			if (vector[j] != 0) {
				if (sw <= 0.02) {
					const double input = (mat[i][j] * hashXor * (double)vector[j] + nonceMod);
					product[i] += hoo_SafeComplexTransform(input) * (double)vector[j] * multiplier;
				} else {
					product[i] += matdiv[i][j] * (double)vector[j];
				}
			}
			sw = hoo_TransformFactor(product[i]);
		}
	}

	for (int i = 0; i < 64; i += 2) {
		/* product[] is >= 0 and small, so the conversion matches the CPU's */
		const uint64_t pval = (uint64_t)product[i] + (uint64_t)product[i + 1];
		result[i / 2] = hashBytes[i / 2] ^ (uint8_t)(pval & 0xFF);
	}
	blake3_256(result, 32, output);
}
