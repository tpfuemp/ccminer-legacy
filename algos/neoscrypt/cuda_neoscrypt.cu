/**
 * NeoScrypt: FastKDF-BLAKE2s of the header, ChaCha20 and Salsa20 SMix (N 128, r 2,
 * 32 KiB per hash), FastKDF-BLAKE2s again. sm_50+ runs one thread per hash, older GPUs
 * four lanes per hash with warp shuffles.
 *
 * djm34 (ccminer-sp-neoscrypt), Nanashi Meiyo-Meijin (1.7.6-r10), tpruvot, KlausT
 * (ccminerKlausTyescrypt)
 */
#include <stdio.h>
#include <memory.h>

#include "miner.h"
#include "cuda_helper.h"
#include "neoscrypt.h"

#define rotateR ROTR32
#define rotateL ROTL32

/* low word of hi:lo >> s; s is a multiple of 8 in 0..32 here (sm_30: byte select) */
#if __CUDA_ARCH__ >= 320
#define shr_c(lo, hi, s) __funnelshift_rc(lo, hi, s)
#else
#define shr_c(lo, hi, s) __byte_perm(lo, hi, 0x3210 + 0x1111 * ((s) >> 3))
#endif

#ifndef __LDG_PTR
#if defined(_WIN64) || defined(__LP64__)
#define __LDG_PTR "l"
#else
#define __LDG_PTR "r"
#endif
#endif

/* threads per block and blocks per SM (BPM1 SMix, BPM2 FastKDF), mirrored in neoscrypt_cpu_init() */
#if __CUDA_ARCH__ >= 520
#define TPB 128
#define BPM1 4
#define BPM2 3
#elif __CUDA_ARCH__ >= 500
#define TPB 128
#define BPM1 4
#define BPM2 2
#else
#define TPB 192
#define BPM1 3
#define BPM2 1
#endif

/* SMix blocks per hash */
#define SHIFT 128U

typedef struct __align__(32) uint8 {
	uint32_t s0, s1, s2, s3, s4, s5, s6, s7;
} uint8;

typedef struct __align__(64) uint16 {
	uint8 lo, hi;
} uint16;

static __host__ __device__ __forceinline__ uint8 operator^(const uint8 &a, const uint8 &b)
{
	uint8 r = { a.s0 ^ b.s0, a.s1 ^ b.s1, a.s2 ^ b.s2, a.s3 ^ b.s3,
		a.s4 ^ b.s4, a.s5 ^ b.s5, a.s6 ^ b.s6, a.s7 ^ b.s7 };
	return r;
}
static __host__ __device__ __forceinline__ void operator^=(uint8 &a, const uint8 &b) { a = a ^ b; }

static __device__ __forceinline__ uint4 operator^(const uint4 &a, const uint4 &b)
{
	return make_uint4(a.x ^ b.x, a.y ^ b.y, a.z ^ b.z, a.w ^ b.w);
}
static __device__ __forceinline__ uint4 operator+(const uint4 &a, const uint4 &b)
{
	return make_uint4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
static __device__ __forceinline__ void operator^=(uint4 &a, const uint4 &b) { a = a ^ b; }
static __device__ __forceinline__ void operator+=(uint4 &a, const uint4 &b) { a = a + b; }

__device__ uint4 *W;
__device__ uint4 *Tr;
__device__ uint4 *Tr2;
__device__ uint4 *Input;

__constant__ uint32_t c_data[64];
__constant__ uint32_t key_init[16];
__constant__ uint32_t input_init[16];

static const __constant__ uint8 BLAKE2S_IV_Vec = {
	0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
	0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19
};

static const uint8 BLAKE2S_IV_Vechost = {
	0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
	0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19
};

static const uint32_t BLAKE2S_SIGMA_host[10][16] = {
	{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
	{ 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
	{ 11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 },
	{ 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 },
	{ 9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13 },
	{ 2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9 },
	{ 12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11 },
	{ 13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10 },
	{ 6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5 },
	{ 10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0 },
};

__constant__ uint32_t BLAKE2S_SIGMA[10][16] = {
	{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
	{ 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
	{ 11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 },
	{ 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 },
	{ 9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13 },
	{ 2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9 },
	{ 12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11 },
	{ 13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10 },
	{ 6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5 },
	{ 10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0 },
};

#define BLOCK_SIZE         64U
#define BLAKE2S_BLOCK_SIZE 64U
#define BLAKE2S_OUT_SIZE   32U

/* width-4 shuffles of the 4-lane SMix */
__device__ __forceinline__ uint32_t WarpShuffle(uint32_t a, uint32_t b, uint32_t c)
{
	return (uint32_t) SHFL((int) a, (int) b, (int) c);
}

__device__ __forceinline__ void WarpShuffle3(uint32_t &d0, uint32_t &d1, uint32_t &d2,
	uint32_t a0, uint32_t a1, uint32_t a2, uint32_t b0, uint32_t b1, uint32_t b2, uint32_t c)
{
	d0 = WarpShuffle(a0, b0, c);
	d1 = WarpShuffle(a1, b1, c);
	d2 = WarpShuffle(a2, b2, c);
}

static __device__ __forceinline__ uint4 __ldL1(const uint4 *ptr)
{
	uint4 ret;
	asm("ld.global.ca.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w) : __LDG_PTR(ptr));
	return ret;

}

static __device__ __forceinline__ void __stL1(const uint4 *ptr, const uint4 value)
{
	asm("st.global.wb.v4.u32 [%0], {%1, %2, %3, %4};" ::__LDG_PTR(ptr), "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w));
}

#define BLAKE_G(idx0, idx1, a, b, c, d, key) { \
	idx = BLAKE2S_SIGMA[idx0][idx1]; a += key[idx]; \
	a += b; d = __byte_perm(d^a, 0, 0x1032); \
	c += d; b = rotateR(b^c, 12); \
	idx = BLAKE2S_SIGMA[idx0][(idx1)+1]; a += key[idx]; \
	a += b; d = __byte_perm(d^a, 0, 0x0321); \
	c += d; b = rotateR(b^c, 7); \
}

#define BLAKE(a, b, c, d, key1,key2) { \
	a += key1; \
	a += b; d = __byte_perm(d^a, 0, 0x1032); \
	c += d; b = rotateR(b^c, 12); \
	a += key2; \
	a += b; d = __byte_perm(d^a, 0, 0x0321); \
	c += d; b = rotateR(b^c, 7); \
}

#define BLAKE_G_PRE(idx0,idx1, a, b, c, d, key) { \
	a += key[idx0]; \
	a += b; d = __byte_perm(d^a, 0, 0x1032); \
	c += d; b = rotateR(b^c, 12); \
	a += key[idx1]; \
	a += b; d = __byte_perm(d^a, 0, 0x0321); \
	c += d; b = rotateR(b^c, 7); \
}

#define BLAKE_G_PRE0(idx0,idx1, a, b, c, d, key) { \
	a += b; d = __byte_perm(d^a, 0, 0x1032); \
	c += d; b = rotateR(b^c, 12); \
	a += b; d = __byte_perm(d^a, 0, 0x0321); \
	c += d; b = rotateR(b^c, 7); \
}

#define BLAKE_G_PRE1(idx0,idx1, a, b, c, d, key) { \
	a += key[idx0]; \
	a += b; d = __byte_perm(d^a, 0, 0x1032); \
	c += d; b = rotateR(b^c, 12); \
	a += b; d = __byte_perm(d^a, 0, 0x0321); \
	c += d; b = rotateR(b^c, 7); \
}

#define BLAKE_G_PRE2(idx0,idx1, a, b, c, d, key) { \
	a += b; d = __byte_perm(d^a, 0, 0x1032); \
	c += d; b = rotateR(b^c, 12); \
	a += key[idx1]; \
	a += b; d = __byte_perm(d^a, 0, 0x0321); \
	c += d; b = rotateR(b^c, 7); \
}

static __forceinline__ __device__
void Blake2S(uint32_t *out, const uint32_t* __restrict__  inout, const  uint32_t * __restrict__ TheKey)
{
	uint16 V;
	uint8 tmpblock;

	V.hi = BLAKE2S_IV_Vec;
	V.lo = BLAKE2S_IV_Vec;
	V.lo.s0 ^= 0x01012020;

	// Copy input block for later
	tmpblock = V.lo;

	V.hi.s4 ^= BLAKE2S_BLOCK_SIZE;

	//	{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
	BLAKE_G_PRE(0, 1, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, TheKey);
	BLAKE_G_PRE(2, 3, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, TheKey);
	BLAKE_G_PRE(4, 5, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, TheKey);
	BLAKE_G_PRE(6, 7, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, TheKey);
	BLAKE_G_PRE0(8, 9, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, TheKey);
	BLAKE_G_PRE0(10, 11, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, TheKey);
	BLAKE_G_PRE0(12, 13, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, TheKey);
	BLAKE_G_PRE0(14, 15, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, TheKey);
	// { 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
	BLAKE_G_PRE0(14, 10, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, TheKey);
	BLAKE_G_PRE1(4, 8, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, TheKey);
	BLAKE_G_PRE0(9, 15, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, TheKey);
	BLAKE_G_PRE2(13, 6, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, TheKey);
	BLAKE_G_PRE1(1, 12, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, TheKey);
	BLAKE_G_PRE(0, 2, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, TheKey);
	BLAKE_G_PRE2(11, 7, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, TheKey);
	BLAKE_G_PRE(5, 3, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, TheKey);
	// { 11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 },
	BLAKE_G_PRE0(11, 8, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, TheKey);
	BLAKE_G_PRE2(12, 0, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, TheKey);
	BLAKE_G_PRE(5, 2, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, TheKey);
	BLAKE_G_PRE0(15, 13, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, TheKey);
	BLAKE_G_PRE0(10, 14, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, TheKey);
	BLAKE_G_PRE(3, 6, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, TheKey);
	BLAKE_G_PRE(7, 1, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, TheKey);
	BLAKE_G_PRE2(9, 4, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, TheKey);
	// { 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 },
	BLAKE_G_PRE1(7, 9, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, TheKey);
	BLAKE_G_PRE(3, 1, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, TheKey);
	BLAKE_G_PRE0(13, 12, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, TheKey);
	BLAKE_G_PRE0(11, 14, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, TheKey);
	BLAKE_G_PRE(2, 6, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, TheKey);
	BLAKE_G_PRE1(5, 10, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, TheKey);
	BLAKE_G_PRE(4, 0, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, TheKey);
	BLAKE_G_PRE0(15, 8, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, TheKey);
	// { 9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13 },
	BLAKE_G_PRE2(9, 0, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, TheKey);
	BLAKE_G_PRE(5, 7, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, TheKey);
	BLAKE_G_PRE(2, 4, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, TheKey);
	BLAKE_G_PRE0(10, 15, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, TheKey);
	BLAKE_G_PRE2(14, 1, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, TheKey);
	BLAKE_G_PRE0(11, 12, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, TheKey);
	BLAKE_G_PRE1(6, 8, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, TheKey);
	BLAKE_G_PRE1(3, 13, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, TheKey);
	// { 2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9 },
	BLAKE_G_PRE1(2, 12, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, TheKey);
	BLAKE_G_PRE1(6, 10, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, TheKey);
	BLAKE_G_PRE1(0, 11, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, TheKey);
	BLAKE_G_PRE2(8, 3, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, TheKey);
	BLAKE_G_PRE1(4, 13, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, TheKey);
	BLAKE_G_PRE(7, 5, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, TheKey);
	BLAKE_G_PRE0(15, 14, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, TheKey);
	BLAKE_G_PRE1(1, 9, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, TheKey);
	// { 12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11 },
	BLAKE_G_PRE2(12, 5, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, TheKey);
	BLAKE_G_PRE1(1, 15, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, TheKey);
	BLAKE_G_PRE0(14, 13, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, TheKey);
	BLAKE_G_PRE1(4, 10, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, TheKey);
	BLAKE_G_PRE(0, 7, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, TheKey);
	BLAKE_G_PRE(6, 3, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, TheKey);
	BLAKE_G_PRE2(9, 2, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, TheKey);
	BLAKE_G_PRE0(8, 11, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, TheKey);
	// { 13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10 },
	BLAKE_G_PRE0(13, 11, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, TheKey);
	BLAKE_G_PRE1(7, 14, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, TheKey);
	BLAKE_G_PRE2(12, 1, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, TheKey);
	BLAKE_G_PRE1(3, 9, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, TheKey);
	BLAKE_G_PRE(5, 0, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, TheKey);
	BLAKE_G_PRE2(15, 4, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, TheKey);
	BLAKE_G_PRE2(8, 6, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, TheKey);
	BLAKE_G_PRE(2, 10, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, TheKey);
	// { 6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5 },
	BLAKE_G_PRE1(6, 15, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, TheKey);
	BLAKE_G_PRE0(14, 9, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, TheKey);
	BLAKE_G_PRE2(11, 3, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, TheKey);
	BLAKE_G_PRE1(0, 8, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, TheKey);
	BLAKE_G_PRE2(12, 2, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, TheKey);
	BLAKE_G_PRE2(13, 7, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, TheKey);
	BLAKE_G_PRE(1, 4, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, TheKey);
	BLAKE_G_PRE2(10, 5, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, TheKey);
	// { 10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0 },
	BLAKE_G_PRE2(10, 2, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, TheKey);
	BLAKE_G_PRE2(8, 4, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, TheKey);
	BLAKE_G_PRE(7, 6, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, TheKey);
	BLAKE_G_PRE(1, 5, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, TheKey);
	BLAKE_G_PRE0(15, 11, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, TheKey);
	BLAKE_G_PRE0(9, 14, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, TheKey);
	BLAKE_G_PRE1(3, 12, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, TheKey);
	BLAKE_G_PRE2(13, 0, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, TheKey);

	V.lo ^= V.hi;
	V.lo ^= tmpblock;

	V.hi = BLAKE2S_IV_Vec;
	tmpblock = V.lo;

	V.hi.s4 ^= 128;
	V.hi.s6 = ~V.hi.s6;

	// { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
	BLAKE_G_PRE(0, 1, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, inout);
	BLAKE_G_PRE(2, 3, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, inout);
	BLAKE_G_PRE(4, 5, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, inout);
	BLAKE_G_PRE(6, 7, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, inout);
	BLAKE_G_PRE(8, 9, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, inout);
	BLAKE_G_PRE(10, 11, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, inout);
	BLAKE_G_PRE(12, 13, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, inout);
	BLAKE_G_PRE(14, 15, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, inout);
	// { 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
	BLAKE_G_PRE(14, 10, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, inout);
	BLAKE_G_PRE(4, 8, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, inout);
	BLAKE_G_PRE(9, 15, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, inout);
	BLAKE_G_PRE(13, 6, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, inout);
	BLAKE_G_PRE(1, 12, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, inout);
	BLAKE_G_PRE(0, 2, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, inout);
	BLAKE_G_PRE(11, 7, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, inout);
	BLAKE_G_PRE(5, 3, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, inout);
	// { 11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 },
	BLAKE_G_PRE(11, 8, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, inout);
	BLAKE_G_PRE(12, 0, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, inout);
	BLAKE_G_PRE(5, 2, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, inout);
	BLAKE_G_PRE(15, 13, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, inout);
	BLAKE_G_PRE(10, 14, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, inout);
	BLAKE_G_PRE(3, 6, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, inout);
	BLAKE_G_PRE(7, 1, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, inout);
	BLAKE_G_PRE(9, 4, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, inout);
	// { 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 },
	BLAKE_G_PRE(7, 9, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, inout);
	BLAKE_G_PRE(3, 1, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, inout);
	BLAKE_G_PRE(13, 12, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, inout);
	BLAKE_G_PRE(11, 14, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, inout);
	BLAKE_G_PRE(2, 6, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, inout);
	BLAKE_G_PRE(5, 10, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, inout);
	BLAKE_G_PRE(4, 0, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, inout);
	BLAKE_G_PRE(15, 8, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, inout);

	BLAKE(V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, inout[9], inout[0]);
	BLAKE(V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, inout[5], inout[7]);
	BLAKE(V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, inout[2], inout[4]);
	BLAKE(V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, inout[10], inout[15]);
	BLAKE(V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, inout[14], inout[1]);
	BLAKE(V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, inout[11], inout[12]);
	BLAKE(V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, inout[6], inout[8]);
	BLAKE(V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, inout[3], inout[13]);

	BLAKE(V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, inout[2], inout[12]);
	BLAKE(V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, inout[6], inout[10]);
	BLAKE(V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, inout[0], inout[11]);
	BLAKE(V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, inout[8], inout[3]);
	BLAKE(V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, inout[4], inout[13]);
	BLAKE(V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, inout[7], inout[5]);
	BLAKE(V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, inout[15], inout[14]);
	BLAKE(V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, inout[1], inout[9]);

	BLAKE(V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, inout[12], inout[5]);
	BLAKE(V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, inout[1], inout[15]);
	BLAKE(V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, inout[14], inout[13]);
	BLAKE(V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, inout[4], inout[10]);
	BLAKE(V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, inout[0], inout[7]);
	BLAKE(V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, inout[6], inout[3]);
	BLAKE(V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, inout[9], inout[2]);
	BLAKE(V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, inout[8], inout[11]);
	// 13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10,
	BLAKE(V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, inout[13], inout[11]);
	BLAKE(V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, inout[7], inout[14]);
	BLAKE(V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, inout[12], inout[1]);
	BLAKE(V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, inout[3], inout[9]);
	BLAKE(V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, inout[5], inout[0]);
	BLAKE(V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, inout[15], inout[4]);
	BLAKE(V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, inout[8], inout[6]);
	BLAKE(V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, inout[2], inout[10]);
	// 6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5,
	BLAKE(V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, inout[6], inout[15]);
	BLAKE(V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, inout[14], inout[9]);
	BLAKE(V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, inout[11], inout[3]);
	BLAKE(V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, inout[0], inout[8]);
	BLAKE(V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, inout[12], inout[2]);
	BLAKE(V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, inout[13], inout[7]);
	BLAKE(V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, inout[1], inout[4]);
	BLAKE(V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, inout[10], inout[5]);
	// 10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0,
	BLAKE(V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, inout[10], inout[2]);
	BLAKE(V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, inout[8], inout[4]);
	BLAKE(V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, inout[7], inout[6]);
	BLAKE(V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, inout[1], inout[5]);
	BLAKE(V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, inout[15], inout[11]);
	BLAKE(V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, inout[9], inout[14]);
	BLAKE(V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, inout[3], inout[12]);
	BLAKE(V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, inout[13], inout[0]);

	V.lo ^= V.hi;
	V.lo ^= tmpblock;

	((uint8*)out)[0] = V.lo;
}

#define SALSA(a,b,c,d) { \
	uint32_t t; \
	t = rotateL(a+d,  7U); b ^= t; \
	t = rotateL(b+a,  9U); c ^= t; \
	t = rotateL(c+b, 13U); d ^= t; \
	t = rotateL(d+c, 18U); a ^= t; \
}

#define CHACHA_STEP(a,b,c,d) { \
	a += b; d = __byte_perm(d^a, 0, 0x1032); \
	c += d; b = rotateL(b^c, 12); \
	a += b; d = __byte_perm(d^a, 0, 0x2103); \
	c += d; b = rotateL(b^c, 7); \
}

#define SALSA_CORE(state) { \
	SALSA(state[0], state[4], state[8], state[12]); \
	SALSA(state[5], state[9], state[13], state[1]); \
	SALSA(state[10], state[14], state[2], state[6]); \
	SALSA(state[15], state[3], state[7], state[11]); \
	SALSA(state[0], state[1], state[2], state[3]); \
	SALSA(state[5], state[6], state[7], state[4]); \
	SALSA(state[10], state[11], state[8], state[9]); \
	SALSA(state[15], state[12], state[13], state[14]); \
}

#define CHACHA_CORE(state)	{ \
	CHACHA_STEP(state[0], state[4], state[8], state[12]); \
	CHACHA_STEP(state[1], state[5], state[9], state[13]); \
	CHACHA_STEP(state[2], state[6], state[10], state[14]); \
	CHACHA_STEP(state[3], state[7], state[11], state[15]); \
	CHACHA_STEP(state[0], state[5], state[10], state[15]); \
	CHACHA_STEP(state[1], state[6], state[11], state[12]); \
	CHACHA_STEP(state[2], state[7], state[8], state[13]); \
	CHACHA_STEP(state[3], state[4], state[9], state[14]); \
}

#define SALSA_CORE_PARALLEL(state) { \
	SALSA(state.x, state.y, state.z, state.w); \
	WarpShuffle3(state.y, state.z, state.w, state.y, state.z, state.w, threadIdx.x + 3, threadIdx.x + 2, threadIdx.x + 1, 4); \
	SALSA(state.x, state.w, state.z, state.y); \
	WarpShuffle3(state.y, state.z, state.w, state.y, state.z, state.w, threadIdx.x + 1, threadIdx.x + 2, threadIdx.x + 3, 4); \
}

#define CHACHA_CORE_PARALLEL(state)	{ \
	CHACHA_STEP(state.x, state.y, state.z, state.w); \
	WarpShuffle3(state.y, state.z, state.w, state.y, state.z, state.w, threadIdx.x + 1, threadIdx.x + 2, threadIdx.x + 3, 4); \
	CHACHA_STEP(state.x, state.y, state.z, state.w); \
	WarpShuffle3(state.y, state.z, state.w, state.y, state.z, state.w, threadIdx.x + 3, threadIdx.x + 2, threadIdx.x + 1, 4); \
}

__forceinline__ __device__
void salsa_small_rnd(uint4 X[4])
{
	uint32_t state[16];
	((uint4*)state)[0] = X[0];
	((uint4*)state)[1] = X[1];
	((uint4*)state)[2] = X[2];
	((uint4*)state)[3] = X[3];

#pragma nounroll
	for (int i = 0; i < 10; i++) {
		SALSA_CORE(state);
	}

	X[0] += ((uint4*)state)[0];
	X[1] += ((uint4*)state)[1];
	X[2] += ((uint4*)state)[2];
	X[3] += ((uint4*)state)[3];
}

__device__ __forceinline__
void chacha_small_rnd(uint4 X[4])
{
	uint32_t state[16];
	((uint4*)state)[0] = X[0];
	((uint4*)state)[1] = X[1];
	((uint4*)state)[2] = X[2];
	((uint4*)state)[3] = X[3];

#pragma nounroll
	for (int i = 0; i < 10; i++) {
		CHACHA_CORE(state);
	}

	X[0] += ((uint4*)state)[0];
	X[1] += ((uint4*)state)[1];
	X[2] += ((uint4*)state)[2];
	X[3] += ((uint4*)state)[3];
}

__forceinline__ __device__
uint4 salsa_small_parallel_rnd(uint4 X)
{
	uint4 state = X;

#pragma nounroll
	for (int i = 0; i < 10; i++) {
		SALSA_CORE_PARALLEL(state);
	}

	return (X + state);
}

__device__ __forceinline__
uint4 chacha_small_parallel_rnd(uint4 X)
{
	uint4 state = X;

#pragma nounroll
	for (int i = 0; i < 10; i++) {
		CHACHA_CORE_PARALLEL(state);
	}

	return (X + state);
}

__device__ __forceinline__
void neoscrypt_chacha(uint4 XV[16])
{
	uint4 temp[4];

	XV[0] ^= XV[12];
	XV[1] ^= XV[13];
	XV[2] ^= XV[14];
	XV[3] ^= XV[15];
	chacha_small_rnd(&XV[0]);
	temp[0] = XV[4] ^ XV[0];
	temp[1] = XV[5] ^ XV[1];
	temp[2] = XV[6] ^ XV[2];
	temp[3] = XV[7] ^ XV[3];
	chacha_small_rnd(temp);
	XV[4] = XV[8] ^ temp[0];
	XV[5] = XV[9] ^ temp[1];
	XV[6] = XV[10] ^ temp[2];
	XV[7] = XV[11] ^ temp[3];
	chacha_small_rnd(&XV[4]);
	XV[12] ^= XV[4];
	XV[13] ^= XV[5];
	XV[14] ^= XV[6];
	XV[15] ^= XV[7];
	chacha_small_rnd(&XV[12]);
	XV[8] = temp[0];
	XV[9] = temp[1];
	XV[10] = temp[2];
	XV[11] = temp[3];
}

__device__ __forceinline__
void neoscrypt_salsa(uint4 XV[16])
{
	uint4 temp[4];

	XV[0] ^= XV[12];
	XV[1] ^= XV[13];
	XV[2] ^= XV[14];
	XV[3] ^= XV[15];
	salsa_small_rnd(&XV[0]);
	temp[0] = XV[4] ^ XV[0];
	temp[1] = XV[5] ^ XV[1];
	temp[2] = XV[6] ^ XV[2];
	temp[3] = XV[7] ^ XV[3];
	salsa_small_rnd(temp);
	XV[4] = XV[8] ^ temp[0];
	XV[5] = XV[9] ^ temp[1];
	XV[6] = XV[10] ^ temp[2];
	XV[7] = XV[11] ^ temp[3];
	salsa_small_rnd(&XV[4]);
	XV[12] ^= XV[4];
	XV[13] ^= XV[5];
	XV[14] ^= XV[6];
	XV[15] ^= XV[7];
	salsa_small_rnd(&XV[12]);
	XV[8] = temp[0];
	XV[9] = temp[1];
	XV[10] = temp[2];
	XV[11] = temp[3];
}

__device__ __forceinline__
void neoscrypt_chacha_parallel(uint4 XV[4])
{
	uint4 temp;

	XV[0] = chacha_small_parallel_rnd(XV[0] ^ XV[3]);
	temp = chacha_small_parallel_rnd(XV[1] ^ XV[0]);
	XV[1] = chacha_small_parallel_rnd(XV[2] ^ temp);
	XV[3] = chacha_small_parallel_rnd(XV[3] ^ XV[1]);
	XV[2] = temp;
}

__device__ __forceinline__
void neoscrypt_salsa_parallel(uint4 XV[4])
{
	uint4 temp;

	XV[0] = salsa_small_parallel_rnd(XV[0] ^ XV[3]);
	temp =salsa_small_parallel_rnd(XV[1] ^ XV[0]);
	XV[1] = salsa_small_parallel_rnd(XV[2] ^ temp);
	XV[3] = salsa_small_parallel_rnd(XV[3] ^ XV[1]);
	XV[2] = temp;
}

static __forceinline__ __device__
void fastkdf256(const uint32_t thread, const uint32_t nonce)
{
	extern __shared__ uint32_t B[];

	uint32_t bufidx, qbuf, rbuf, bitbuf;
	uint32_t shift, a, b, noncepos;
	uint32_t input[16];
	uint32_t key[16];
	uint32_t temp[9];

	for (int i = 0; i < 64; i++)
		B[i * TPB + threadIdx.x] = c_data[i];

	B[19 * TPB + threadIdx.x] = nonce;
	B[39 * TPB + threadIdx.x] = nonce;
	B[59 * TPB + threadIdx.x] = nonce;

#pragma unroll
	for (int i = 0; i < 16; i++) input[i] = input_init[i];
#pragma unroll
	for (int i = 0; i < 8; i++) key[i] = key_init[i];
#pragma unroll
	for (int i = 8; i < 16; i++) key[i] = 0;

#pragma unroll 1
	for (int i = 0; i < 31; i++)
	{
		bufidx = 0;
#pragma unroll
		for (int x = 0; x < 8; ++x)
			bufidx += (input[x] & 0x00ff00ff) + ((input[x] & 0xff00ff00) >> 8);
		bufidx += bufidx >> 16;
		bufidx &= 0x000000ff;
		qbuf = bufidx >> 2;
		rbuf = bufidx & 3;
		bitbuf = rbuf << 3;

		shift = 32U - bitbuf;
		temp[0] = B[((qbuf + 0) & 0x3f) * TPB + threadIdx.x] ^ (input[0] << bitbuf);
		temp[1] = B[((qbuf + 1) & 0x3f) * TPB + threadIdx.x] ^ shr_c(input[0], input[1], shift);
		temp[2] = B[((qbuf + 2) & 0x3f) * TPB + threadIdx.x] ^ shr_c(input[1], input[2], shift);
		temp[3] = B[((qbuf + 3) & 0x3f) * TPB + threadIdx.x] ^ shr_c(input[2], input[3], shift);
		temp[4] = B[((qbuf + 4) & 0x3f) * TPB + threadIdx.x] ^ shr_c(input[3], input[4], shift);
		temp[5] = B[((qbuf + 5) & 0x3f) * TPB + threadIdx.x] ^ shr_c(input[4], input[5], shift);
		temp[6] = B[((qbuf + 6) & 0x3f) * TPB + threadIdx.x] ^ shr_c(input[5], input[6], shift);
		temp[7] = B[((qbuf + 7) & 0x3f) * TPB + threadIdx.x] ^ shr_c(input[6], input[7], shift);
		temp[8] = B[((qbuf + 8) & 0x3f) * TPB + threadIdx.x] ^ shr_c(input[7], 0, shift);

		B[((qbuf + 0) & 0x3f) * TPB + threadIdx.x] = temp[0];
		B[((qbuf + 1) & 0x3f) * TPB + threadIdx.x] = temp[1];
		B[((qbuf + 2) & 0x3f) * TPB + threadIdx.x] = temp[2];
		B[((qbuf + 3) & 0x3f) * TPB + threadIdx.x] = temp[3];
		B[((qbuf + 4) & 0x3f) * TPB + threadIdx.x] = temp[4];
		B[((qbuf + 5) & 0x3f) * TPB + threadIdx.x] = temp[5];
		B[((qbuf + 6) & 0x3f) * TPB + threadIdx.x] = temp[6];
		B[((qbuf + 7) & 0x3f) * TPB + threadIdx.x] = temp[7];
		B[((qbuf + 8) & 0x3f) * TPB + threadIdx.x] = temp[8];

		noncepos = qbuf < 20U ? 19U : 39U;
		noncepos = qbuf < 40U ? noncepos : 59U;

		a = (qbuf + 0) & 0x3f;
		a = a == noncepos ? nonce : c_data[a];

#pragma unroll
		for (int k = 0; k < 16; k += 2)
		{
			b = (qbuf + k + 1) & 0x3f;
			b = b == noncepos ? nonce : c_data[b];
			input[k + 0] = shr_c(a, b, bitbuf);

			a = (qbuf + k + 2) & 0x3f;
			a = a == noncepos ? nonce : c_data[a];
			input[k + 1] = shr_c(b, a, bitbuf);
		}

		key[0] = shr_c(temp[0], temp[1], bitbuf);
		key[1] = shr_c(temp[1], temp[2], bitbuf);
		key[2] = shr_c(temp[2], temp[3], bitbuf);
		key[3] = shr_c(temp[3], temp[4], bitbuf);
		key[4] = shr_c(temp[4], temp[5], bitbuf);
		key[5] = shr_c(temp[5], temp[6], bitbuf);
		key[6] = shr_c(temp[6], temp[7], bitbuf);
		key[7] = shr_c(temp[7], temp[8], bitbuf);

		Blake2S(input, input, key);
	}

	bufidx = 0;
#pragma unroll
	for (int x = 0; x < 8; ++x)
		bufidx += (input[x] & 0x00ff00ff) + ((input[x] & 0xff00ff00) >> 8);
	bufidx += bufidx >> 16;
	bufidx &= 0x000000ff;
	qbuf = bufidx >> 2;
	rbuf = bufidx & 3;
	bitbuf = rbuf << 3;

	a = B[((qbuf + 0) & 0x3f) * TPB + threadIdx.x];
#pragma unroll
	for (int i = 0; i < 8; i += 4)
	{
		b = B[((qbuf + i + 1) & 0x3f) * TPB + threadIdx.x];
		temp[0] = shr_c(a, b, bitbuf) ^ input[i + 0] ^ c_data[i + 0];
		a = B[((qbuf + i + 2) & 0x3f) * TPB + threadIdx.x];
		temp[1] = shr_c(b, a, bitbuf) ^ input[i + 1] ^ c_data[i + 1];
		b = B[((qbuf + i + 3) & 0x3f) * TPB + threadIdx.x];
		temp[2] = shr_c(a, b, bitbuf) ^ input[i + 2] ^ c_data[i + 2];
		a = B[((qbuf + i + 4) & 0x3f) * TPB + threadIdx.x];
		temp[3] = shr_c(b, a, bitbuf) ^ input[i + 3] ^ c_data[i + 3];
		Input[16U * thread + (i >> 2)] = make_uint4(temp[0], temp[1], temp[2], temp[3]);
	}
#pragma unroll
	for (int i = 8; i < 16; i += 4)
	{
		b = B[((qbuf + i + 1) & 0x3f) * TPB + threadIdx.x];
		temp[0] = shr_c(a, b, bitbuf) ^ c_data[i + 0];
		a = B[((qbuf + i + 2) & 0x3f) * TPB + threadIdx.x];
		temp[1] = shr_c(b, a, bitbuf) ^ c_data[i + 1];
		b = B[((qbuf + i + 3) & 0x3f) * TPB + threadIdx.x];
		temp[2] = shr_c(a, b, bitbuf) ^ c_data[i + 2];
		a = B[((qbuf + i + 4) & 0x3f) * TPB + threadIdx.x];
		temp[3] = shr_c(b, a, bitbuf) ^ c_data[i + 3];
		Input[16U * thread + (i >> 2)] = make_uint4(temp[0], temp[1], temp[2], temp[3]);
	}
	b = B[((qbuf + 17) & 0x3f) * TPB + threadIdx.x];
	temp[0] = shr_c(a, b, bitbuf) ^ c_data[16];
	a = B[((qbuf + 18) & 0x3f) * TPB + threadIdx.x];
	temp[1] = shr_c(b, a, bitbuf) ^ c_data[17];
	b = B[((qbuf + 19) & 0x3f) * TPB + threadIdx.x];
	temp[2] = shr_c(a, b, bitbuf) ^ c_data[18];
	a = B[((qbuf + 20) & 0x3f) * TPB + threadIdx.x];
	temp[3] = shr_c(b, a, bitbuf) ^ nonce;
	Input[16U * thread + 4] = make_uint4(temp[0], temp[1], temp[2], temp[3]);
#pragma unroll
	for (int i = 20; i < 36; i += 4)
	{
		b = B[((qbuf + i + 1) & 0x3f) * TPB + threadIdx.x];
		temp[0] = shr_c(a, b, bitbuf) ^ c_data[i + 0];
		a = B[((qbuf + i + 2) & 0x3f) * TPB + threadIdx.x];
		temp[1] = shr_c(b, a, bitbuf) ^ c_data[i + 1];
		b = B[((qbuf + i + 3) & 0x3f) * TPB + threadIdx.x];
		temp[2] = shr_c(a, b, bitbuf) ^ c_data[i + 2];
		a = B[((qbuf + i + 4) & 0x3f) * TPB + threadIdx.x];
		temp[3] = shr_c(b, a, bitbuf) ^ c_data[i + 3];
		Input[16U * thread + (i >> 2)] = make_uint4(temp[0], temp[1], temp[2], temp[3]);
	}
	b = B[((qbuf + 37) & 0x3f) * TPB + threadIdx.x];
	temp[0] = shr_c(a, b, bitbuf) ^ c_data[36];
	a = B[((qbuf + 38) & 0x3f) * TPB + threadIdx.x];
	temp[1] = shr_c(b, a, bitbuf) ^ c_data[37];
	b = B[((qbuf + 39) & 0x3f) * TPB + threadIdx.x];
	temp[2] = shr_c(a, b, bitbuf) ^ c_data[38];
	a = B[((qbuf + 40) & 0x3f) * TPB + threadIdx.x];
	temp[3] = shr_c(b, a, bitbuf) ^ nonce;
	Input[16U * thread + 9] = make_uint4(temp[0], temp[1], temp[2], temp[3]);
#pragma unroll
	for (int i = 40; i < 56; i += 4)
	{
		b = B[((qbuf + i + 1) & 0x3f) * TPB + threadIdx.x];
		temp[0] = shr_c(a, b, bitbuf) ^ c_data[i + 0];
		a = B[((qbuf + i + 2) & 0x3f) * TPB + threadIdx.x];
		temp[1] = shr_c(b, a, bitbuf) ^ c_data[i + 1];
		b = B[((qbuf + i + 3) & 0x3f) * TPB + threadIdx.x];
		temp[2] = shr_c(a, b, bitbuf) ^ c_data[i + 2];
		a = B[((qbuf + i + 4) & 0x3f) * TPB + threadIdx.x];
		temp[3] = shr_c(b, a, bitbuf) ^ c_data[i + 3];
		Input[16U * thread + (i >> 2)] = make_uint4(temp[0], temp[1], temp[2], temp[3]);
	}
	b = B[((qbuf + 57) & 0x3f) * TPB + threadIdx.x];
	temp[0] = shr_c(a, b, bitbuf) ^ c_data[56];
	a = B[((qbuf + 58) & 0x3f) * TPB + threadIdx.x];
	temp[1] = shr_c(b, a, bitbuf) ^ c_data[57];
	b = B[((qbuf + 59) & 0x3f) * TPB + threadIdx.x];
	temp[2] = shr_c(a, b, bitbuf) ^ c_data[58];
	a = B[((qbuf + 60) & 0x3f) * TPB + threadIdx.x];
	temp[3] = shr_c(b, a, bitbuf) ^ nonce;
	Input[16U * thread + 14] = make_uint4(temp[0], temp[1], temp[2], temp[3]);
	b = B[((qbuf + 61) & 0x3f) * TPB + threadIdx.x];
	temp[0] = shr_c(a, b, bitbuf) ^ c_data[60];
	a = B[((qbuf + 62) & 0x3f) * TPB + threadIdx.x];
	temp[1] = shr_c(b, a, bitbuf) ^ c_data[61];
	b = B[((qbuf + 63) & 0x3f) * TPB + threadIdx.x];
	temp[2] = shr_c(a, b, bitbuf) ^ c_data[62];
	a = B[((qbuf + 0) & 0x3f) * TPB + threadIdx.x];
	temp[3] = shr_c(b, a, bitbuf) ^ c_data[63];
	Input[16U * thread + 15] = make_uint4(temp[0], temp[1], temp[2], temp[3]);
}

/* hash words 6 and 7 */
static __forceinline__ __device__
uint2 fastkdf32(uint32_t thread, const uint32_t nonce)
{
	extern __shared__ uint32_t B[];

	uint32_t bufidx, qbuf, rbuf, bitbuf;
	uint32_t shift, a, b, noncepos;
	uint32_t input[16];
	uint32_t key[16];
	uint32_t temp[9];

#pragma unroll
	for (int i = 0; i < 16; i++) {
		((uint4*)key)[0] = __ldg(Tr2 + 16U * thread + i) ^ __ldg(Tr + 16U * thread + i);
		B[(i * 4 + 0) * TPB + threadIdx.x] = key[0];
		B[(i * 4 + 1) * TPB + threadIdx.x] = key[1];
		B[(i * 4 + 2) * TPB + threadIdx.x] = key[2];
		B[(i * 4 + 3) * TPB + threadIdx.x] = key[3];
	}

#pragma unroll
	for (int i = 0; i < 16; i++) input[i] = c_data[i];
#pragma unroll
	for (int i = 0; i < 8; i++) key[i] = B[i * TPB + threadIdx.x];
#pragma unroll
	for (int i = 8; i < 16; i++) key[i] = 0;

	for (int i = 0; i < 31; i++)
	{
		Blake2S(input, input, key);

		bufidx = 0;
#pragma unroll
		for (int x = 0; x < 8; ++x)
			bufidx += (input[x] & 0x00ff00ff) + ((input[x] & 0xff00ff00) >> 8);
		bufidx += bufidx >> 16;
		bufidx &= 0x000000ff;
		qbuf = bufidx >> 2;
		rbuf = bufidx & 3;
		bitbuf = rbuf << 3;

		shift = 32U - bitbuf;
		temp[0] = B[((qbuf + 0) & 0x3f) * TPB + threadIdx.x] ^ (input[0] << bitbuf);
		temp[1] = B[((qbuf + 1) & 0x3f) * TPB + threadIdx.x] ^ shr_c(input[0], input[1], shift);
		temp[2] = B[((qbuf + 2) & 0x3f) * TPB + threadIdx.x] ^ shr_c(input[1], input[2], shift);
		temp[3] = B[((qbuf + 3) & 0x3f) * TPB + threadIdx.x] ^ shr_c(input[2], input[3], shift);
		temp[4] = B[((qbuf + 4) & 0x3f) * TPB + threadIdx.x] ^ shr_c(input[3], input[4], shift);
		temp[5] = B[((qbuf + 5) & 0x3f) * TPB + threadIdx.x] ^ shr_c(input[4], input[5], shift);
		temp[6] = B[((qbuf + 6) & 0x3f) * TPB + threadIdx.x] ^ shr_c(input[5], input[6], shift);
		temp[7] = B[((qbuf + 7) & 0x3f) * TPB + threadIdx.x] ^ shr_c(input[6], input[7], shift);
		temp[8] = B[((qbuf + 8) & 0x3f) * TPB + threadIdx.x] ^ shr_c(input[7], 0, shift);

		B[((qbuf + 0) & 0x3f) * TPB + threadIdx.x] = temp[0];
		B[((qbuf + 1) & 0x3f) * TPB + threadIdx.x] = temp[1];
		B[((qbuf + 2) & 0x3f) * TPB + threadIdx.x] = temp[2];
		B[((qbuf + 3) & 0x3f) * TPB + threadIdx.x] = temp[3];
		B[((qbuf + 4) & 0x3f) * TPB + threadIdx.x] = temp[4];
		B[((qbuf + 5) & 0x3f) * TPB + threadIdx.x] = temp[5];
		B[((qbuf + 6) & 0x3f) * TPB + threadIdx.x] = temp[6];
		B[((qbuf + 7) & 0x3f) * TPB + threadIdx.x] = temp[7];
		B[((qbuf + 8) & 0x3f) * TPB + threadIdx.x] = temp[8];

		noncepos = qbuf < 20U ? 19U : 39U;
		noncepos = qbuf < 40U ? noncepos : 59U;

		a = (qbuf + 0) & 0x3f;
		a = a == noncepos ? nonce : c_data[a];

#pragma unroll
		for (int k = 0; k < 16; k += 2)
		{
			b = (qbuf + k + 1) & 0x3f;
			b = b == noncepos ? nonce : c_data[b];
			input[k + 0] = shr_c(a, b, bitbuf);

			a = (qbuf + k + 2) & 0x3f;
			a = a == noncepos ? nonce : c_data[a];
			input[k + 1] = shr_c(b, a, bitbuf);
		}

		key[0] = shr_c(temp[0], temp[1], bitbuf);
		key[1] = shr_c(temp[1], temp[2], bitbuf);
		key[2] = shr_c(temp[2], temp[3], bitbuf);
		key[3] = shr_c(temp[3], temp[4], bitbuf);
		key[4] = shr_c(temp[4], temp[5], bitbuf);
		key[5] = shr_c(temp[5], temp[6], bitbuf);
		key[6] = shr_c(temp[6], temp[7], bitbuf);
		key[7] = shr_c(temp[7], temp[8], bitbuf);
	}

	Blake2S(input, input, key);

	bufidx = 0;
#pragma unroll
	for (int x = 0; x < 8; ++x)
		bufidx += (input[x] & 0x00ff00ff) + ((input[x] & 0xff00ff00) >> 8);
	bufidx += bufidx >> 16;
	bufidx &= 0x000000ff;
	qbuf = bufidx >> 2;
	rbuf = bufidx & 3;
	bitbuf = rbuf << 3;

	temp[6] = B[((qbuf + 6) & 0x3f) * TPB + threadIdx.x];
	temp[7] = B[((qbuf + 7) & 0x3f) * TPB + threadIdx.x];
	temp[8] = B[((qbuf + 8) & 0x3f) * TPB + threadIdx.x];

	return make_uint2(shr_c(temp[6], temp[7], bitbuf) ^ input[6] ^ c_data[6],
		shr_c(temp[7], temp[8], bitbuf) ^ input[7] ^ c_data[7]);
}

#define BLAKE_Ghost(idx0, idx1, a, b, c, d, key) { \
	idx = BLAKE2S_SIGMA_host[idx0][idx1]; a += key[idx]; \
	a += b; d = ROTR32(d^a,16); \
	c += d; b = ROTR32(b^c, 12); \
	idx = BLAKE2S_SIGMA_host[idx0][(idx1)+1]; a += key[idx]; \
	a += b; d = ROTR32(d^a,8); \
	c += d; b = ROTR32(b^c, 7); \
}

static void Blake2Shost(uint32_t * inout, const uint32_t * inkey)
{
	uint16 V;
	uint32_t idx;
	uint8 tmpblock;

	V.hi = BLAKE2S_IV_Vechost;
	V.lo = BLAKE2S_IV_Vechost;
	V.lo.s0 ^= 0x01012020;

	// Copy input block for later
	tmpblock = V.lo;

	V.hi.s4 ^= BLAKE2S_BLOCK_SIZE;

	for (int x = 0; x < 10; ++x)
	{
		BLAKE_Ghost(x, 0x00, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, inkey);
		BLAKE_Ghost(x, 0x02, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, inkey);
		BLAKE_Ghost(x, 0x04, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, inkey);
		BLAKE_Ghost(x, 0x06, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, inkey);
		BLAKE_Ghost(x, 0x08, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, inkey);
		BLAKE_Ghost(x, 0x0A, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, inkey);
		BLAKE_Ghost(x, 0x0C, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, inkey);
		BLAKE_Ghost(x, 0x0E, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, inkey);
	}

	V.lo ^= V.hi;
	V.lo ^= tmpblock;

	V.hi = BLAKE2S_IV_Vechost;
	tmpblock = V.lo;

	V.hi.s4 ^= 128;
	V.hi.s6 = ~V.hi.s6;

	for (int x = 0; x < 10; ++x)
	{
		BLAKE_Ghost(x, 0x00, V.lo.s0, V.lo.s4, V.hi.s0, V.hi.s4, inout);
		BLAKE_Ghost(x, 0x02, V.lo.s1, V.lo.s5, V.hi.s1, V.hi.s5, inout);
		BLAKE_Ghost(x, 0x04, V.lo.s2, V.lo.s6, V.hi.s2, V.hi.s6, inout);
		BLAKE_Ghost(x, 0x06, V.lo.s3, V.lo.s7, V.hi.s3, V.hi.s7, inout);
		BLAKE_Ghost(x, 0x08, V.lo.s0, V.lo.s5, V.hi.s2, V.hi.s7, inout);
		BLAKE_Ghost(x, 0x0A, V.lo.s1, V.lo.s6, V.hi.s3, V.hi.s4, inout);
		BLAKE_Ghost(x, 0x0C, V.lo.s2, V.lo.s7, V.hi.s0, V.hi.s5, inout);
		BLAKE_Ghost(x, 0x0E, V.lo.s3, V.lo.s4, V.hi.s1, V.hi.s6, inout);
	}

	V.lo ^= V.hi ^ tmpblock;

	((uint8*)inout)[0] = V.lo;
}

__global__
__launch_bounds__(TPB, BPM2)
void neoscrypt_gpu_hash_start(const uint32_t threads, const uint32_t startNonce)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	const uint32_t nonce = startNonce + thread;
	if (thread < threads)
		fastkdf256(thread, cuda_swab32(nonce));
}

__global__
__launch_bounds__(TPB, BPM1)
void neoscrypt_gpu_hash_chacha1(const uint32_t threads, const uint32_t start)
{
#if __CUDA_ARCH__ >= 500
	const uint32_t thread = blockIdx.x * TPB + threadIdx.x;

	if (thread < threads) {
		uint4 X[16];
#pragma unroll
		for (int i = 0; i < 16; i++) {
			X[i] = __ldg(Input + 16U * (thread + start) + i);
		}

#pragma nounroll
		for (uint32_t i = 0; i < SHIFT; i++)
		{
			__stL1(W + (SHIFT * thread + i) * 16U + 0, X[0]);
			__stL1(W + (SHIFT * thread + i) * 16U + 1, X[1]);
			__stL1(W + (SHIFT * thread + i) * 16U + 2, X[2]);
			__stL1(W + (SHIFT * thread + i) * 16U + 3, X[3]);
			__stL1(W + (SHIFT * thread + i) * 16U + 4, X[4]);
			__stL1(W + (SHIFT * thread + i) * 16U + 5, X[5]);
			__stL1(W + (SHIFT * thread + i) * 16U + 6, X[6]);
			__stL1(W + (SHIFT * thread + i) * 16U + 7, X[7]);
			__stL1(W + (SHIFT * thread + i) * 16U + 8, X[8]);
			__stL1(W + (SHIFT * thread + i) * 16U + 9, X[9]);
			__stL1(W + (SHIFT * thread + i) * 16U + 10, X[10]);
			__stL1(W + (SHIFT * thread + i) * 16U + 11, X[11]);
			__stL1(W + (SHIFT * thread + i) * 16U + 12, X[12]);
			__stL1(W + (SHIFT * thread + i) * 16U + 13, X[13]);
			__stL1(W + (SHIFT * thread + i) * 16U + 14, X[14]);
			__stL1(W + (SHIFT * thread + i) * 16U + 15, X[15]);

			neoscrypt_chacha(X);
		}

#pragma nounroll
		for (uint32_t t = 0; t < SHIFT; t++)
		{
			uint32_t index = X[12].x & 0x7F;

			X[0] ^= __ldL1(W + (SHIFT * thread + index) * 16U + 0);
			X[1] ^= __ldL1(W + (SHIFT * thread + index) * 16U + 1);
			X[2] ^= __ldL1(W + (SHIFT * thread + index) * 16U + 2);
			X[3] ^= __ldL1(W + (SHIFT * thread + index) * 16U + 3);
			X[4] ^= __ldL1(W + (SHIFT * thread + index) * 16U + 4);
			X[5] ^= __ldL1(W + (SHIFT * thread + index) * 16U + 5);
			X[6] ^= __ldL1(W + (SHIFT * thread + index) * 16U + 6);
			X[7] ^= __ldL1(W + (SHIFT * thread + index) * 16U + 7);
			X[8] ^= __ldL1(W + (SHIFT * thread + index) * 16U + 8);
			X[9] ^= __ldL1(W + (SHIFT * thread + index) * 16U + 9);
			X[10] ^= __ldL1(W + (SHIFT * thread + index) * 16U + 10);
			X[11] ^= __ldL1(W + (SHIFT * thread + index) * 16U + 11);
			X[12] ^= __ldL1(W + (SHIFT * thread + index) * 16U + 12);
			X[13] ^= __ldL1(W + (SHIFT * thread + index) * 16U + 13);
			X[14] ^= __ldL1(W + (SHIFT * thread + index) * 16U + 14);
			X[15] ^= __ldL1(W + (SHIFT * thread + index) * 16U + 15);

			neoscrypt_chacha(X);
		}

#pragma unroll
		for (uint32_t i = 0; i < 16; i++)
		{
			Tr[16U * (thread + start) + i] = X[i];
		}
	}
#else
	const uint32_t thread = blockIdx.x * (TPB >> 2) + threadIdx.y;

	if (thread < threads) {
		uint4 X[4];
#pragma unroll
		for (uint32_t i = 0; i < 4; i++)
		{
			X[i].x = __ldg((uint32_t*)Input + ((thread + start) * 4U + i) * 16U + 0 * 4 + threadIdx.x);
			X[i].y = __ldg((uint32_t*)Input + ((thread + start) * 4U + i) * 16U + 1 * 4 + threadIdx.x);
			X[i].z = __ldg((uint32_t*)Input + ((thread + start) * 4U + i) * 16U + 2 * 4 + threadIdx.x);
			X[i].w = __ldg((uint32_t*)Input + ((thread + start) * 4U + i) * 16U + 3 * 4 + threadIdx.x);
		}

#pragma nounroll
		for (uint32_t i = 0; i < SHIFT; i++)
		{
			__stL1(W + (SHIFT * thread + i) * 16U + 4 * 0 + threadIdx.x, X[0]);
			__stL1(W + (SHIFT * thread + i) * 16U + 4 * 1 + threadIdx.x, X[1]);
			__stL1(W + (SHIFT * thread + i) * 16U + 4 * 2 + threadIdx.x, X[2]);
			__stL1(W + (SHIFT * thread + i) * 16U + 4 * 3 + threadIdx.x, X[3]);

			neoscrypt_chacha_parallel(X);
		}

#pragma nounroll
		for (uint32_t t = 0; t < SHIFT; t++)
		{
			uint32_t index = WarpShuffle(X[3].x, 0, 4) & 0x7F;

			X[0] ^= __ldL1(W + (SHIFT * thread + index) * 16U + 4 * 0 + threadIdx.x);
			X[1] ^= __ldL1(W + (SHIFT * thread + index) * 16U + 4 * 1 + threadIdx.x);
			X[2] ^= __ldL1(W + (SHIFT * thread + index) * 16U + 4 * 2 + threadIdx.x);
			X[3] ^= __ldL1(W + (SHIFT * thread + index) * 16U + 4 * 3 + threadIdx.x);

			neoscrypt_chacha_parallel(X);
		}

#pragma unroll
		for (int i = 0; i < 4; i++)
		{
			((uint32_t*)Tr)[((thread + start) * 4U + i) * 16U + 0 * 4 + threadIdx.x] = X[i].x;
			((uint32_t*)Tr)[((thread + start) * 4U + i) * 16U + 1 * 4 + threadIdx.x] = X[i].y;
			((uint32_t*)Tr)[((thread + start) * 4U + i) * 16U + 2 * 4 + threadIdx.x] = X[i].z;
			((uint32_t*)Tr)[((thread + start) * 4U + i) * 16U + 3 * 4 + threadIdx.x] = X[i].w;
		}
	}
#endif
}

__global__
__launch_bounds__(TPB, BPM1)
void neoscrypt_gpu_hash_salsa1(const uint32_t threads, const uint32_t start)
{
#if __CUDA_ARCH__ >= 500
	const uint32_t thread = blockIdx.x * TPB + threadIdx.x;

	if (thread < threads) {
		uint4 X[16];
#pragma unroll
		for (uint32_t i = 0; i < 16; i++)
		{
			X[i] = __ldg(Input + 16U * (thread + start) + i);
		}

#pragma nounroll
		for (uint32_t i = 0; i < SHIFT; i++)
		{
			__stL1(W + (SHIFT * thread + i) * 16U + 0, X[0]);
			__stL1(W + (SHIFT * thread + i) * 16U + 1, X[1]);
			__stL1(W + (SHIFT * thread + i) * 16U + 2, X[2]);
			__stL1(W + (SHIFT * thread + i) * 16U + 3, X[3]);
			__stL1(W + (SHIFT * thread + i) * 16U + 4, X[4]);
			__stL1(W + (SHIFT * thread + i) * 16U + 5, X[5]);
			__stL1(W + (SHIFT * thread + i) * 16U + 6, X[6]);
			__stL1(W + (SHIFT * thread + i) * 16U + 7, X[7]);
			__stL1(W + (SHIFT * thread + i) * 16U + 8, X[8]);
			__stL1(W + (SHIFT * thread + i) * 16U + 9, X[9]);
			__stL1(W + (SHIFT * thread + i) * 16U + 10, X[10]);
			__stL1(W + (SHIFT * thread + i) * 16U + 11, X[11]);
			__stL1(W + (SHIFT * thread + i) * 16U + 12, X[12]);
			__stL1(W + (SHIFT * thread + i) * 16U + 13, X[13]);
			__stL1(W + (SHIFT * thread + i) * 16U + 14, X[14]);
			__stL1(W + (SHIFT * thread + i) * 16U + 15, X[15]);

			neoscrypt_salsa(X);
		}

#pragma nounroll
		for (uint32_t t = 0; t < SHIFT; t++)
		{
			uint32_t index = X[12].x & 0x7F;

			X[0] ^= __ldL1(W + (SHIFT * thread + index) * 16U + 0);
			X[1] ^= __ldL1(W + (SHIFT * thread + index) * 16U + 1);
			X[2] ^= __ldL1(W + (SHIFT * thread + index) * 16U + 2);
			X[3] ^= __ldL1(W + (SHIFT * thread + index) * 16U + 3);
			X[4] ^= __ldL1(W + (SHIFT * thread + index) * 16U + 4);
			X[5] ^= __ldL1(W + (SHIFT * thread + index) * 16U + 5);
			X[6] ^= __ldL1(W + (SHIFT * thread + index) * 16U + 6);
			X[7] ^= __ldL1(W + (SHIFT * thread + index) * 16U + 7);
			X[8] ^= __ldL1(W + (SHIFT * thread + index) * 16U + 8);
			X[9] ^= __ldL1(W + (SHIFT * thread + index) * 16U + 9);
			X[10] ^= __ldL1(W + (SHIFT * thread + index) * 16U + 10);
			X[11] ^= __ldL1(W + (SHIFT * thread + index) * 16U + 11);
			X[12] ^= __ldL1(W + (SHIFT * thread + index) * 16U + 12);
			X[13] ^= __ldL1(W + (SHIFT * thread + index) * 16U + 13);
			X[14] ^= __ldL1(W + (SHIFT * thread + index) * 16U + 14);
			X[15] ^= __ldL1(W + (SHIFT * thread + index) * 16U + 15);

			neoscrypt_salsa(X);
		}

#pragma unroll
		for (uint32_t i = 0; i < 16; i++)
		{
			Tr2[16U * (thread + start) + i] = X[i];
		}
	}
#else
	const uint32_t thread = blockIdx.x * (TPB >> 2) + threadIdx.y;

	if (thread < threads) {
		uint4 X[4];
#pragma unroll
		for (uint32_t i = 0; i < 4; i++)
		{
			X[i].x = __ldg((uint32_t*)Input + ((thread + start) * 4U + i) * 16U + ((0 + threadIdx.x) & 3) * 4 + threadIdx.x);
			X[i].y = __ldg((uint32_t*)Input + ((thread + start) * 4U + i) * 16U + ((1 + threadIdx.x) & 3) * 4 + threadIdx.x);
			X[i].z = __ldg((uint32_t*)Input + ((thread + start) * 4U + i) * 16U + ((2 + threadIdx.x) & 3) * 4 + threadIdx.x);
			X[i].w = __ldg((uint32_t*)Input + ((thread + start) * 4U + i) * 16U + ((3 + threadIdx.x) & 3) * 4 + threadIdx.x);
		}

#pragma nounroll
		for (uint32_t i = 0; i < SHIFT; i++)
		{
			__stL1(W + (SHIFT * thread + i) * 16U + 4 * 0 + threadIdx.x, X[0]);
			__stL1(W + (SHIFT * thread + i) * 16U + 4 * 1 + threadIdx.x, X[1]);
			__stL1(W + (SHIFT * thread + i) * 16U + 4 * 2 + threadIdx.x, X[2]);
			__stL1(W + (SHIFT * thread + i) * 16U + 4 * 3 + threadIdx.x, X[3]);

			neoscrypt_salsa_parallel(X);
		}

#pragma nounroll
		for (uint32_t t = 0; t < SHIFT; t++)
		{
			uint32_t index = WarpShuffle(X[3].x, 0, 4) & 0x7F;

			X[0] ^= __ldL1(W + (SHIFT * thread + index) * 16U + 4 * 0 + threadIdx.x);
			X[1] ^= __ldL1(W + (SHIFT * thread + index) * 16U + 4 * 1 + threadIdx.x);
			X[2] ^= __ldL1(W + (SHIFT * thread + index) * 16U + 4 * 2 + threadIdx.x);
			X[3] ^= __ldL1(W + (SHIFT * thread + index) * 16U + 4 * 3 + threadIdx.x);

			neoscrypt_salsa_parallel(X);
		}

#pragma unroll
		for (int i = 0; i < 4; i++)
		{
			((uint32_t*)Tr2)[((thread + start) * 4U + i) * 16U + ((0 + threadIdx.x) & 3) * 4 + threadIdx.x] = X[i].x;
			((uint32_t*)Tr2)[((thread + start) * 4U + i) * 16U + ((1 + threadIdx.x) & 3) * 4 + threadIdx.x] = X[i].y;
			((uint32_t*)Tr2)[((thread + start) * 4U + i) * 16U + ((2 + threadIdx.x) & 3) * 4 + threadIdx.x] = X[i].z;
			((uint32_t*)Tr2)[((thread + start) * 4U + i) * 16U + ((3 + threadIdx.x) & 3) * 4 + threadIdx.x] = X[i].w;
		}
	}
#endif
}

__global__
__launch_bounds__(TPB, BPM2)
void neoscrypt_gpu_hash_ending(const uint32_t threads, const uint32_t startNonce,
	const uint64_t target, uint32_t *resNonce)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads) {
		const uint32_t nonce = startNonce + thread;
		const uint2 h = fastkdf32(thread, cuda_swab32(nonce));
		// top 64 bits of the 256-bit hash against target words 6..7
		if ((((uint64_t) h.y << 32) | h.x) <= target) {
			const uint32_t pos = atomicInc(&resNonce[0], UINT32_MAX) + 1;
			if (pos < NEOSCRYPT_MAX_RESULTS)
				resNonce[pos] = nonce;
		}
	}
}

static uint4 *d_W[MAX_GPUS], *d_Tr[MAX_GPUS], *d_Tr2[MAX_GPUS], *d_Input[MAX_GPUS];
static uint32_t s_tpb[MAX_GPUS], s_smem2[MAX_GPUS];
static bool s_lanes4[MAX_GPUS];

/* the launch shape follows the compiled arch (cuda_arch), not the device */
__host__
bool neoscrypt_cpu_init(int thr_id, uint32_t threads)
{
	const int dev_id = device_map[thr_id];
	const int arch = cuda_get_arch(thr_id);
	s_lanes4[thr_id] = arch < 500;
	s_tpb[thr_id] = arch >= 500 ? 128 : 192;

	// sm_50+ SMix: dynamic shared memory only caps the blocks per SM, so that the
	// scratch of the running blocks fits the L2 cache
	s_smem2[thr_id] = 0;
	if (!s_lanes4[thr_id]) {
		cudaDeviceProp p;
		if (cudaGetDeviceProperties(&p, dev_id) != cudaSuccess)
			return false;
		const double cores = (double) p.multiProcessorCount * (p.major == 6 && p.minor == 0 ? 64 : 128);
		uint32_t bpm = (uint32_t) ((double) p.l2CacheSize / (cores * 256 * 1.2));
		if (bpm < 1) bpm = 1;
		if (bpm < 4) // BPM1 of the sm_50+ shapes
			s_smem2[thr_id] = (uint32_t) min((size_t) p.sharedMemPerMultiprocessor / bpm,
				(size_t) p.sharedMemPerBlock);
	}

	const size_t n = threads;
	if (cudaMalloc(&d_W[thr_id], n * SHIFT * 256) != cudaSuccess ||
	    cudaMalloc(&d_Tr[thr_id], n * 256) != cudaSuccess ||
	    cudaMalloc(&d_Tr2[thr_id], n * 256) != cudaSuccess ||
	    cudaMalloc(&d_Input[thr_id], n * 256) != cudaSuccess)
		return false;
	cudaMemcpyToSymbol(W, &d_W[thr_id], sizeof(uint4*), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Tr, &d_Tr[thr_id], sizeof(uint4*), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Tr2, &d_Tr2[thr_id], sizeof(uint4*), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(Input, &d_Input[thr_id], sizeof(uint4*), 0, cudaMemcpyHostToDevice);

	const cudaFuncCache kdf = s_lanes4[thr_id] ? cudaFuncCachePreferEqual : cudaFuncCachePreferL1;
	cudaFuncSetCacheConfig(neoscrypt_gpu_hash_start, kdf);
	cudaFuncSetCacheConfig(neoscrypt_gpu_hash_ending, kdf);
	cudaFuncSetCacheConfig(neoscrypt_gpu_hash_chacha1, cudaFuncCachePreferL1);
	cudaFuncSetCacheConfig(neoscrypt_gpu_hash_salsa1, cudaFuncCachePreferL1);
	return cudaGetLastError() == cudaSuccess;
}

__host__
void neoscrypt_cpu_free(int thr_id)
{
	cudaFree(d_W[thr_id]);
	cudaFree(d_Tr[thr_id]);
	cudaFree(d_Tr2[thr_id]);
	cudaFree(d_Input[thr_id]);
	d_W[thr_id] = d_Tr[thr_id] = d_Tr2[thr_id] = d_Input[thr_id] = NULL;
}

/* one launch series over [startNonce, startNonce + threads), returns the candidate count */
__host__
uint32_t neoscrypt_scan(int thr_id, uint32_t threads, uint32_t startNonce, uint64_t target,
	uint32_t *d_res, uint32_t *h_res)
{
	const uint32_t tpb = s_tpb[thr_id];
	const uint32_t smem1 = tpb * 256;
	dim3 grid1((threads + tpb - 1) / tpb);
	dim3 block1(tpb);

	cudaMemset(d_res, 0, sizeof(uint32_t));
	neoscrypt_gpu_hash_start <<<grid1, block1, smem1>>> (threads, startNonce);
	if (s_lanes4[thr_id]) {
		dim3 grid3((threads * 4 + tpb - 1) / tpb);
		dim3 block3(4, tpb >> 2);
		neoscrypt_gpu_hash_chacha1 <<<grid3, block3>>> (threads, 0);
		neoscrypt_gpu_hash_salsa1 <<<grid3, block3>>> (threads, 0);
	} else {
		neoscrypt_gpu_hash_chacha1 <<<grid1, block1, s_smem2[thr_id]>>> (threads, 0);
		neoscrypt_gpu_hash_salsa1 <<<grid1, block1, s_smem2[thr_id]>>> (threads, 0);
	}
	neoscrypt_gpu_hash_ending <<<grid1, block1, smem1>>> (threads, startNonce, target, d_res);
	cudaMemcpy(h_res, d_res, NEOSCRYPT_MAX_RESULTS * sizeof(uint32_t), cudaMemcpyDeviceToHost);
	return h_res[0];
}

/* 80-byte header as 20 words, nonce word ignored */
__host__
void neoscrypt_setBlock(const uint32_t *pdata)
{
	uint32_t PaddedMessage[64];
	uint32_t input[16], key[16] = { 0 };

	for (int i = 0; i < 19; i++) {
		PaddedMessage[i] = pdata[i];
		PaddedMessage[i + 20] = pdata[i];
		PaddedMessage[i + 40] = pdata[i];
	}
	for (int i = 0; i < 4; i++)
		PaddedMessage[i + 60] = pdata[i];
	PaddedMessage[19] = 0;
	PaddedMessage[39] = 0;
	PaddedMessage[59] = 0;

	memcpy(input, pdata, 64);
	memcpy(key, pdata, 32);
	Blake2Shost(input, key);

	cudaMemcpyToSymbol(input_init, input, 64, 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(key_init, key, 64, 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(c_data, PaddedMessage, 64 * sizeof(uint32_t), 0, cudaMemcpyHostToDevice);
}
