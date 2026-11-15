/**
 * Copyright 2020-2021 CryptoGraphics (VerthashMiner), GPL-2.0-or-later.
 *
 * Verthash kernels, after VerthashMiner (src/vhCuda/verthash.cu):
 * SHA3-256 of the header per nonce, then 4 threads per nonce finish the 8 SHA3-512
 * states (first 72-byte block absorbed on the host) into a 512-byte subset in shared
 * memory and run the 4096 data file reads, each thread on 8 of the 32 bytes.
 */
#include <stdint.h>
#include <string.h>

#include "cuda_helper.h"
#include "cuda_compat.h"
#include "sph/sha3.h"
#include "verthash.h"

#define VH_TPB 64            /* IO kernel threads per block */
#define VH_GROUPS (VH_TPB / 4)

static __constant__ uint32_t c_header[19]; /* header bytes 0..75 */
static __constant__ uint32_t c_mdiv;

static __constant__ uint2 c_rc[24] = {
	{ 0x00000001, 0x00000000 }, { 0x00008082, 0x00000000 }, { 0x0000808a, 0x80000000 }, { 0x80008000, 0x80000000 },
	{ 0x0000808b, 0x00000000 }, { 0x80000001, 0x00000000 }, { 0x80008081, 0x80000000 }, { 0x00008009, 0x80000000 },
	{ 0x0000008a, 0x00000000 }, { 0x00000088, 0x00000000 }, { 0x80008009, 0x00000000 }, { 0x8000000a, 0x00000000 },
	{ 0x8000808b, 0x00000000 }, { 0x0000008b, 0x80000000 }, { 0x00008089, 0x80000000 }, { 0x00008003, 0x80000000 },
	{ 0x00008002, 0x80000000 }, { 0x00000080, 0x80000000 }, { 0x0000800a, 0x00000000 }, { 0x8000000a, 0x80000000 },
	{ 0x80008081, 0x80000000 }, { 0x00008080, 0x80000000 }, { 0x80000001, 0x00000000 }, { 0x80008008, 0x80000000 }
};

__device__ __forceinline__ static void keccakf(uint2 st[25])
{
	uint2 t[5], u[5], v, w;

	#pragma unroll 1
	for (int r = 0; r < 24; r++) {
		#pragma unroll
		for (int i = 0; i < 5; i++)
			t[i] = st[i] ^ st[i + 5] ^ st[i + 10] ^ st[i + 15] ^ st[i + 20];
		#pragma unroll
		for (int i = 0; i < 5; i++)
			u[i] = ROL2(t[(i + 1) % 5], 1) ^ t[(i + 4) % 5];
		#pragma unroll
		for (int i = 0; i < 25; i++)
			st[i] ^= u[i % 5];

		v = st[1];
		st[1]  = ROL2(st[6], 44);
		st[6]  = ROL2(st[9], 20);
		st[9]  = ROL2(st[22], 61);
		st[22] = ROL2(st[14], 39);
		st[14] = ROL2(st[20], 18);
		st[20] = ROL2(st[2], 62);
		st[2]  = ROL2(st[12], 43);
		st[12] = ROL2(st[13], 25);
		st[13] = ROL2(st[19], 8);
		st[19] = ROL2(st[23], 56);
		st[23] = ROL2(st[15], 41);
		st[15] = ROL2(st[4], 27);
		st[4]  = ROL2(st[24], 14);
		st[24] = ROL2(st[21], 2);
		st[21] = ROL2(st[8], 55);
		st[8]  = ROL2(st[16], 45);
		st[16] = ROL2(st[5], 36);
		st[5]  = ROL2(st[3], 28);
		st[3]  = ROL2(st[18], 21);
		st[18] = ROL2(st[17], 15);
		st[17] = ROL2(st[11], 10);
		st[11] = ROL2(st[7], 6);
		st[7]  = ROL2(st[10], 3);
		st[10] = ROL2(v, 1);

		#pragma unroll
		for (int j = 0; j < 25; j += 5) {
			v = st[j]; w = st[j + 1];
			st[j]     ^= ~w & st[j + 2];
			st[j + 1] ^= ~st[j + 2] & st[j + 3];
			st[j + 2] ^= ~st[j + 3] & st[j + 4];
			st[j + 3] ^= ~st[j + 4] & v;
			st[j + 4] ^= ~v & w;
		}
		st[0] ^= c_rc[r];
	}
}

__device__ __forceinline__ static uint32_t fnv1a(const uint32_t a, const uint32_t b)
{
	return (a ^ b) * 0x1000193U;
}

/* SHA3-256 of the 80-byte header with the nonce little-endian in bytes 76..79 */
__global__ __launch_bounds__(256)
void verthash_sha3_256(uint2 *iohashes, const uint32_t startNonce)
{
	const uint32_t thread = blockDim.x * blockIdx.x + threadIdx.x;
	uint2 st[25];

	#pragma unroll
	for (int i = 0; i < 9; i++)
		st[i] = make_uint2(c_header[2 * i], c_header[2 * i + 1]);
	st[9] = make_uint2(c_header[18], startNonce + thread);
	st[10] = make_uint2(0x06, 0);
	#pragma unroll
	for (int i = 11; i < 25; i++)
		st[i] = make_uint2(0, 0);
	st[16].y = 0x80000000; /* last byte of the 136-byte rate */

	keccakf(st);

	#pragma unroll
	for (int i = 0; i < 4; i++)
		iohashes[4 * thread + i] = st[i];
}

/* 4 threads per nonce, lane l holds hash words 2l and 2l + 1. DUAL: odd indexes read the
 * copy shifted by 16 bytes, so every 32-byte read is aligned. */
template <bool DUAL>
__global__ __launch_bounds__(VH_TPB)
void verthash_io(uint2 *iohashes, const uint2 *__restrict__ kstates, const uint2 *__restrict__ memory,
	const uint2 *__restrict__ memory_odd, const uint32_t threads, const uint32_t startNonce,
	uint32_t *resNonces, const uint32_t target7)
{
	const uint32_t gid = blockDim.x * blockIdx.x + threadIdx.x;
	const uint32_t group = (gid & (VH_TPB - 1)) >> 2;
	const uint32_t nid = gid >> 2;
	const uint32_t lane = gid & 3;
	const uint32_t nonce = startNonce + nid;

	/* subset word w of a group at [w * VH_GROUPS + group]: the loop reads one word index
	 * across the block, transposed it hits 16 banks instead of one */
	__shared__ uint32_t subset[128 * VH_GROUPS];

	#pragma unroll 1
	for (int s = 0; s < 2; s++) {
		const uint2 *k = kstates + (2 * lane + s) * 25;
		uint2 st[25];
		#pragma unroll
		for (int i = 0; i < 25; i++)
			st[i] = k[i];
		st[0] ^= make_uint2(c_header[18], nonce); /* second block: header bytes 72..79 */
		st[1].x ^= 0x06;
		st[8].y ^= 0x80000000; /* last byte of the 72-byte rate */

		keccakf(st);

		#pragma unroll
		for (int i = 0; i < 8; i++) {
			const uint32_t w = lane * 32 + s * 16 + 2 * i;
			subset[w * VH_GROUPS + group] = st[i].x;
			subset[(w + 1) * VH_GROUPS + group] = st[i].y;
		}
	}
	__syncthreads();

	uint2 h = iohashes[gid];
	uint32_t acc = 0x811c9dc5U;

	#pragma unroll 1
	for (uint32_t i = 0; i < 4096; i++) {
		const uint32_t r = i >> 7;
		const uint32_t x = subset[(i & 127) * VH_GROUPS + group];
		const uint32_t seek = r ? (x << r) | (x >> (32 - r)) : x;
		const uint32_t idx = fnv1a(seek, acc) % c_mdiv;

		uint2 d;
		if (DUAL)
			d = ((idx & 1) ? memory_odd : memory)[((idx & ~1U) << 1) + lane];
		else
			d = memory[(idx << 1) + lane];

		h.x = fnv1a(h.x, d.x);
		h.y = fnv1a(h.y, d.y);
		#pragma unroll
		for (int l = 0; l < 4; l++) {
			acc = fnv1a(acc, SHFL((int) d.x, l, 4));
			acc = fnv1a(acc, SHFL((int) d.y, l, 4));
		}
	}

	iohashes[gid] = h;
	if (lane == 3 && nid < threads && h.y <= target7) {
		const uint32_t pos = atomicInc(&resNonces[0], UINT32_MAX) + 1;
		if (pos < VH_MAX_RESULTS)
			resNonces[pos] = nonce;
	}
}

void verthash_cuda_set_mdiv(uint32_t mdiv)
{
	cudaMemcpyToSymbol(c_mdiv, &mdiv, sizeof(mdiv), 0, cudaMemcpyHostToDevice);
}

/* header words (endiandata) and the 8 SHA3-512 states after the first 72-byte block */
void verthash_cuda_setBlock(const uint32_t *endiandata, uint64_t *d_kstates)
{
	uint64_t st[8][25];
	uint8_t in[72];
	memcpy(in, endiandata, 72);
	for (int i = 0; i < 8; i++) {
		in[0]++;
		memset(st[i], 0, sizeof(st[i]));
		for (int j = 0; j < 9; j++) {
			uint64_t w;
			memcpy(&w, in + 8 * j, 8);
			st[i][j] = w;
		}
		sha3_keccakf(st[i]);
	}
	cudaMemcpyToSymbol(c_header, endiandata, sizeof(c_header), 0, cudaMemcpyHostToDevice);
	cudaMemcpy(d_kstates, st, sizeof(st), cudaMemcpyHostToDevice);
}

void verthash_cuda_scan(uint32_t threads, uint32_t startNonce, uint32_t target7,
	const uint64_t *d_kstates, const uint64_t *d_memory, const uint64_t *d_memory_odd,
	uint64_t *d_iohashes, uint32_t *d_res)
{
	const uint32_t padded = (threads + VH_PAD - 1) / VH_PAD * VH_PAD;
	const uint32_t blocks = padded * 4 / VH_TPB;

	verthash_sha3_256 <<<padded / 256, 256>>> ((uint2 *) d_iohashes, startNonce);
	if (d_memory_odd)
		verthash_io<true> <<<blocks, VH_TPB>>> ((uint2 *) d_iohashes, (const uint2 *) d_kstates,
			(const uint2 *) d_memory, (const uint2 *) d_memory_odd, threads, startNonce, d_res, target7);
	else
		verthash_io<false> <<<blocks, VH_TPB>>> ((uint2 *) d_iohashes, (const uint2 *) d_kstates,
			(const uint2 *) d_memory, NULL, threads, startNonce, d_res, target7);
}
