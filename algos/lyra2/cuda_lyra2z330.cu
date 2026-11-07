/**
 * Lyra2 over the 80-byte header (password = salt = header) with a global matrix of
 * nRows x nCols blocks per hash, for lyra2z330. Kepler: 4 lanes per hash, sm_50+: 16 lanes
 * per hash; a batch runs in 4 or 16 sub-launches that share one matrix buffer.
 * Nanashi Meiyo-Meijin, KlausT (ccminerKlausTyescrypt)
 */
#include <stdio.h>
#include <stdint.h>
#include <memory.h>

#include "miner.h"
#include "cuda_helper.h"
#include "cuda_lyra2_vectors.h"
#include "cuda_lyra2z330.h"

#define TPB 32

__constant__ uint2 c_data[10];
__constant__ uint64_t c_target;

__device__ uint2 *DMatrix;
__device__ uint2 *DState;

__device__ __forceinline__ uint2 WarpShuffle(uint2 a, uint32_t b, uint32_t c)
{
	return make_uint2((uint32_t) SHFL((int) a.x, (int) b, (int) c), (uint32_t) SHFL((int) a.y, (int) b, (int) c));
}

__device__ __forceinline__ void WarpShuffle2(uint2 &d0, uint2 &d1, uint2 a0, uint2 a1, uint32_t b0, uint32_t b1, uint32_t c)
{
	d0 = WarpShuffle(a0, b0, c);
	d1 = WarpShuffle(a1, b1, c);
}

__device__ __forceinline__ void WarpShuffle3(uint2 &d0, uint2 &d1, uint2 &d2, uint2 a0, uint2 a1, uint2 a2, uint32_t b0, uint32_t b1, uint32_t b2, uint32_t c)
{
	d0 = WarpShuffle(a0, b0, c);
	d1 = WarpShuffle(a1, b1, c);
	d2 = WarpShuffle(a2, b2, c);
}

__device__ __forceinline__ void WarpShuffle4(uint2 &d0, uint2 &d1, uint2 &d2, uint2 &d3, uint2 a0, uint2 a1, uint2 a2, uint2 a3, uint32_t b0, uint32_t b1, uint32_t b2, uint32_t b3, uint32_t c)
{
	d0 = WarpShuffle(a0, b0, c);
	d1 = WarpShuffle(a1, b1, c);
	d2 = WarpShuffle(a2, b2, c);
	d3 = WarpShuffle(a3, b3, c);
}

static __device__ __forceinline__ uint2 __ldL1(const uint2 *ptr)
{
	uint2 ret;
	asm("ld.global.ca.v2.u32 {%0, %1}, [%2];" : "=r"(ret.x), "=r"(ret.y) : __LDG_PTR(ptr));
	return ret;

}

static __device__ __forceinline__ void __stL1(const uint2 *ptr, const uint2 value)
{
	asm("st.global.wb.v2.u32 [%0], {%1, %2};" ::__LDG_PTR(ptr), "r"(value.x), "r"(value.y));
}

static __device__ __forceinline__ void prefetch(const uint2 *ptr)
{
	asm("prefetch.global.L1 [%0+0];" ::__LDG_PTR(ptr));
	asm("prefetch.global.L1 [%0+4];" ::__LDG_PTR(ptr));
}

__device__ __forceinline__
void G(uint2 &a, uint2 &b, uint2 &c, uint2 &d)
{
	uint32_t tmp;
	a += b; d ^= a; tmp = d.x; d.x = d.y; d.y = tmp;
	c += d; b ^= c; b = ROR2(b, 24);
	a += b; d ^= a; d = ROR2(d, 16);
	c += d; b ^= c; b = ROR2(b, 63);
}

__device__ __forceinline__
void round_lyra(uint2x4 s[4], const uint32_t count)
{
	G(s[0].x, s[1].x, s[2].x, s[3].x);
	G(s[0].y, s[1].y, s[2].y, s[3].y);
	G(s[0].z, s[1].z, s[2].z, s[3].z);
	G(s[0].w, s[1].w, s[2].w, s[3].w);

	G(s[0].x, s[1].y, s[2].z, s[3].w);
	G(s[0].y, s[1].z, s[2].w, s[3].x);
	G(s[0].z, s[1].w, s[2].x, s[3].y);
	G(s[0].w, s[1].x, s[2].y, s[3].z);
}

__device__ __forceinline__ void round_lyra(uint2 s[4])
{
	G(s[0], s[1], s[2], s[3]);
	WarpShuffle3(s[1], s[2], s[3], s[1], s[2], s[3], threadIdx.x + 1, threadIdx.x + 2, threadIdx.x + 3, 4);
	G(s[0], s[1], s[2], s[3]);
	WarpShuffle3(s[1], s[2], s[3], s[1], s[2], s[3], threadIdx.x + 3, threadIdx.x + 2, threadIdx.x + 1, 4);
}

__device__ __forceinline__ void round_lyra(uint2 &s)
{
	uint2 s0, s1, s2, s3;
	WarpShuffle4(s0, s1, s2, s3, s, s, s, s, threadIdx.x + 0, threadIdx.x + 4, threadIdx.x + 8, threadIdx.x + 12, 16);
	G(s0, s1, s2, s3);
	WarpShuffle3(s1, s2, s3, s1, s2, s3, threadIdx.x + 1, threadIdx.x + 2, threadIdx.x + 3, 4);
	G(s0, s1, s2, s3);
	WarpShuffle3(s1, s2, s3, s1, s2, s3, threadIdx.x + 3, threadIdx.x + 2, threadIdx.x + 1, 4);
	if (threadIdx.y == 0) s = s0;
	else if (threadIdx.y == 1) s = s1;
	else if (threadIdx.y == 2) s = s2;
	else if (threadIdx.y == 3) s = s3;
}

__device__ __forceinline__ void blake2bLyra(uint2x4 v[4]) {
	round_lyra(v, 0);
	round_lyra(v, 1);
	round_lyra(v, 2);
	round_lyra(v, 3);
	round_lyra(v, 4);
	round_lyra(v, 5);
	round_lyra(v, 6);
	round_lyra(v, 7);
	round_lyra(v, 8);
	round_lyra(v, 9);
	round_lyra(v, 10);
	round_lyra(v, 11);
}

__device__ __forceinline__ void reducedBlake2bLyra(uint2 v[4]) {
	round_lyra(v);
}

__device__ __forceinline__ void reducedBlake2bLyra(uint2 &v) {
	round_lyra(v);
}

__global__
__launch_bounds__(TPB, 1)
void lyra2_gpu_hash_80_1(uint32_t threads, const uint32_t startNonce, const uint32_t timeCost, const uint32_t nRows, const uint32_t nCols)
{
	const uint32_t thread = blockDim.x * blockIdx.x + threadIdx.x;

	const uint2x4 blake2b_IV[2] = {
		0xf3bcc908UL, 0x6a09e667UL, 0x84caa73bUL, 0xbb67ae85UL,
		0xfe94f82bUL, 0x3c6ef372UL, 0x5f1d36f1UL, 0xa54ff53aUL,
		0xade682d1UL, 0x510e527fUL, 0x2b3e6c1fUL, 0x9b05688cUL,
		0xfb41bd6bUL, 0x1f83d9abUL, 0x137e2179UL, 0x5be0cd19UL
	};

	const uint2x4 Mask[3] = {
		0x00000020UL, 0x00000000UL, 0x00000050UL, 0x00000000UL,
		0x00000050UL, 0x00000000UL, timeCost    , 0x00000000UL,
		nRows       , 0x00000000UL, nCols       , 0x00000000UL,
		0x00000080UL, 0x00000000UL, 0x00000000UL, 0x00000000UL,
		0x00000000UL, 0x00000000UL, 0x00000000UL, 0x00000000UL,
		0x00000000UL, 0x00000000UL, 0x00000000UL, 0x01000000UL
	};

	uint2x4 state[4];

	if (thread < threads)
	{
		state[0].x = c_data[0];
		state[0].y = c_data[1];
		state[0].z = c_data[2];
		state[0].w = c_data[3];
		state[1].x = c_data[4];
		state[1].y = c_data[5];
		state[1].z = c_data[6];
		state[1].w = c_data[7];
		state[2] = blake2b_IV[0];
		state[3] = blake2b_IV[1];

		blake2bLyra(state);

		state[0].x ^= c_data[8];
		state[0].y.x ^= c_data[9].x;
		state[0].y.y ^= cuda_swab32(startNonce + thread);
		state[0].z ^= c_data[0];
		state[0].w ^= c_data[1];
		state[1].x ^= c_data[2];
		state[1].y ^= c_data[3];
		state[1].z ^= c_data[4];
		state[1].w ^= c_data[5];

		blake2bLyra(state);

		state[0].x ^= c_data[6];
		state[0].y ^= c_data[7];
		state[0].z ^= c_data[8];
		state[0].w.x ^= c_data[9].x;
		state[0].w.y ^= cuda_swab32(startNonce + thread);
		state[1] ^= Mask[0];

		blake2bLyra(state);

		state[0] ^= Mask[1];
		state[1] ^= Mask[2];

		blake2bLyra(state);

		((uint2x4*)DState)[thread * 4 + 0] = state[0];
		((uint2x4*)DState)[thread * 4 + 1] = state[1];
		((uint2x4*)DState)[thread * 4 + 2] = state[2];
		((uint2x4*)DState)[thread * 4 + 3] = state[3];
	}
}

#define Mdev(r,c) DMatrix[(((blockIdx.x * nRows + (r)) * nCols + (c)) * 2 + threadIdx.z) * 12 + threadIdx.y * 4 + threadIdx.x]
#define Ddev(r,c,a) DMatrix[((((blockIdx.x * nRows + (r)) * nCols + (c)) * 3 + (a)) * 8 + threadIdx.y) * 4 + threadIdx.x]

__global__
__launch_bounds__(TPB, 1)
void lyra2_gpu_hash_32_2_16way(uint32_t threads, uint32_t offset, const uint32_t timeCost, const uint32_t nRows, const uint32_t nCols)
{
	const uint32_t thread = blockIdx.x * blockDim.z + threadIdx.z;
	uint32_t subthread = threadIdx.y * 4 + threadIdx.x;

	if (subthread == 0) subthread = 11;
	else subthread -= 1;

	if ((thread + offset) < threads)
	{
		uint32_t row, prev, rowa, step, window, gap;
		uint2 bufIn, bufInOut, bufOut, bufstate;

		uint2 state = __ldg(&DState[(thread + offset) * 16 + threadIdx.y * 4 + threadIdx.x]);

		for (uint32_t i = 0; i < nCols; i++) {
			if (threadIdx.y < 3) __stL1(&Mdev(0, nCols - i - 1), state);

			reducedBlake2bLyra(state);
		}

		for (uint32_t i = 0; i < nCols; i++) {
			if (threadIdx.y < 3) {
				bufIn = __ldL1(&Mdev(0, i));
				if (i < nCols - 1) {
					prefetch(&Mdev(0, i + 1));
				}
				state ^= bufIn;
			}

			reducedBlake2bLyra(state);

			if (threadIdx.y < 3) __stL1(&Mdev(1, nCols - i - 1), bufIn ^ state);
		}

		for (row = 2, prev = 1, rowa = 0, step = 1, window = 2, gap = 1; row < nRows; row++) {

			for (uint32_t i = 0; i < nCols; i++) {
				if (threadIdx.y < 3) {
					bufIn = __ldL1(&Mdev(prev, i));
					bufInOut = __ldL1(&Mdev(rowa, i));
					if (i < nCols - 1) {
						prefetch(&Mdev(prev, i + 1));
						prefetch(&Mdev(rowa, i + 1));
					}
					state ^= bufIn + bufInOut;
				}

				reducedBlake2bLyra(state);

				bufstate = WarpShuffle(state, subthread, 16);

				if (threadIdx.y < 3) {
					bufOut = bufIn ^ state;
					__stL1(&Mdev(row, nCols - i - 1), bufOut);
					__stL1(&Mdev(rowa, i), bufInOut ^ bufstate);
				}
			}

			rowa = (rowa + step) & (window - 1);
			prev = row;

			if (rowa == 0) {
				step = window + gap;
				window <<= 1;
				gap = -gap;
			}
		}
		DState[(thread + offset) * 16 + threadIdx.y * 4 + threadIdx.x] = state;
	}
}

__global__
__launch_bounds__(TPB, 1)
void lyra2_gpu_hash_32_2_4way(uint32_t threads, uint32_t offset, const uint32_t timeCost, const uint32_t nRows, const uint32_t nCols)
{
	const uint32_t thread = blockIdx.x * blockDim.y + threadIdx.y;

	if ((thread + offset) < threads)
	{
		uint32_t row, prev, rowa, step, window, gap;
		uint2 bufIn[3], bufInOut[3], bufOut[3], bufstate[3];

		uint2 state[4];
		state[0] = __ldg(&DState[(thread + offset) * 16 + 0 * 4 + threadIdx.x]);
		state[1] = __ldg(&DState[(thread + offset) * 16 + 1 * 4 + threadIdx.x]);
		state[2] = __ldg(&DState[(thread + offset) * 16 + 2 * 4 + threadIdx.x]);
		state[3] = __ldg(&DState[(thread + offset) * 16 + 3 * 4 + threadIdx.x]);

		for (uint32_t i = 0; i < nCols; i++) {
			__stL1(&Ddev(0, nCols - i - 1, 0), state[0]);
			__stL1(&Ddev(0, nCols - i - 1, 1), state[1]);
			__stL1(&Ddev(0, nCols - i - 1, 2), state[2]);

			reducedBlake2bLyra(state);
		}

		for (uint32_t i = 0; i < nCols; i++) {
			bufIn[0] = __ldL1(&Ddev(0, i, 0));
			bufIn[1] = __ldL1(&Ddev(0, i, 1));
			bufIn[2] = __ldL1(&Ddev(0, i, 2));
			if (i < nCols - 1) {
				prefetch(&Ddev(0, i + 1, 0));
				prefetch(&Ddev(0, i + 1, 1));
				prefetch(&Ddev(0, i + 1, 2));
			}
			state[0] ^= bufIn[0];
			state[1] ^= bufIn[1];
			state[2] ^= bufIn[2];

			reducedBlake2bLyra(state);

			__stL1(&Ddev(1, nCols - i - 1, 0), bufIn[0] ^ state[0]);
			__stL1(&Ddev(1, nCols - i - 1, 1), bufIn[1] ^ state[1]);
			__stL1(&Ddev(1, nCols - i - 1, 2), bufIn[2] ^ state[2]);
		}

		for (row = 2, prev = 1, rowa = 0, step = 1, window = 2, gap = 1; row < nRows; row++) {

			for (uint32_t i = 0; i < nCols; i++) {
				bufIn[0] = __ldL1(&Ddev(prev, i, 0));
				bufIn[1] = __ldL1(&Ddev(prev, i, 1));
				bufIn[2] = __ldL1(&Ddev(prev, i, 2));
				bufInOut[0] = __ldL1(&Ddev(rowa, i, 0));
				bufInOut[1] = __ldL1(&Ddev(rowa, i, 1));
				bufInOut[2] = __ldL1(&Ddev(rowa, i, 2));
				if (i < nCols - 1) {
					prefetch(&Ddev(prev, i + 1, 0));
					prefetch(&Ddev(prev, i + 1, 1));
					prefetch(&Ddev(prev, i + 1, 2));
					prefetch(&Ddev(rowa, i + 1, 0));
					prefetch(&Ddev(rowa, i + 1, 1));
					prefetch(&Ddev(rowa, i + 1, 2));
				}
				state[0] ^= bufIn[0] + bufInOut[0];
				state[1] ^= bufIn[1] + bufInOut[1];
				state[2] ^= bufIn[2] + bufInOut[2];

				reducedBlake2bLyra(state);

				WarpShuffle3(bufstate[0], bufstate[1], bufstate[2], state[0], state[1], state[2], threadIdx.x + 3, threadIdx.x + 3, threadIdx.x + 3, 4);

				bufOut[0] = bufIn[0] ^ state[0];
				bufOut[1] = bufIn[1] ^ state[1];
				bufOut[2] = bufIn[2] ^ state[2];
				__stL1(&Ddev(row, nCols - i - 1, 0), bufOut[0]);
				__stL1(&Ddev(row, nCols - i - 1, 1), bufOut[1]);
				__stL1(&Ddev(row, nCols - i - 1, 2), bufOut[2]);

				if (threadIdx.x == 0) {
					bufInOut[0] ^= bufstate[2];
					bufInOut[1] ^= bufstate[0];
					bufInOut[2] ^= bufstate[1];
				}
				else {
					bufInOut[0] ^= bufstate[0];
					bufInOut[1] ^= bufstate[1];
					bufInOut[2] ^= bufstate[2];
				}

				__stL1(&Ddev(rowa, i, 0), bufInOut[0]);
				__stL1(&Ddev(rowa, i, 1), bufInOut[1]);
				__stL1(&Ddev(rowa, i, 2), bufInOut[2]);
			}

			rowa = (rowa + step) & (window - 1);
			prev = row;

			if (rowa == 0) {
				step = window + gap;
				window <<= 1;
				gap = -gap;
			}
		}
		DState[(thread + offset) * 16 + 0 * 4 + threadIdx.x] = state[0];
		DState[(thread + offset) * 16 + 1 * 4 + threadIdx.x] = state[1];
		DState[(thread + offset) * 16 + 2 * 4 + threadIdx.x] = state[2];
		DState[(thread + offset) * 16 + 3 * 4 + threadIdx.x] = state[3];
	}
}

__global__
__launch_bounds__(TPB, 1)
void lyra2_gpu_hash_32_3_16way(uint32_t threads, uint32_t offset, const uint32_t timeCost, const uint32_t nRows, const uint32_t nCols, uint32_t tau, uint32_t prev)
{
	const uint32_t thread = blockIdx.x * blockDim.z + threadIdx.z;

	if ((thread + offset) < threads)
	{
		uint2 state = __ldg(&DState[(thread + offset) * 16 + threadIdx.y * 4 + threadIdx.x]);
		uint32_t subthread = threadIdx.y * 4 + threadIdx.x;

		if (subthread == 0) subthread = 11;
		else subthread -= 1;

		uint32_t row = 0;
		uint32_t rowa;
		uint2 bufIn, bufInOut, bufOut, bufstate;
		uint64_t step = (tau & 1) == 0 ? 0xFFFFFFFFFFFFFFFF : (uint64_t)((nRows >> 1) - 1);

		do {
			rowa = (uint32_t)(devectorize(WarpShuffle(state, 0, 16)) % (uint64_t)nRows);

			for (uint32_t i = 0; i < nCols; i++) {
				if (threadIdx.y < 3) {
					bufIn = __ldL1(&Mdev(prev, i));
					bufInOut = __ldL1(&Mdev(rowa, i));
					bufOut = __ldL1(&Mdev(row, i));
					if (i < nCols - 1) {
						prefetch(&Mdev(prev, i + 1));
						prefetch(&Mdev(rowa, i + 1));
						prefetch(&Mdev(row, i + 1));
					}
					state ^= bufIn + bufInOut;
				}

				reducedBlake2bLyra(state);

				bufstate = WarpShuffle(state, subthread, 16);

				if (threadIdx.y < 3) {
					bufOut ^= state;
					if (row == rowa) bufInOut = bufOut;
					else __stL1(&Mdev(row, i), bufOut);
					__stL1(&Mdev(rowa, i), bufInOut ^ bufstate);
				}
			}

			prev = row;

			row = (uint32_t)(((uint64_t)row + step) % (uint64_t)nRows);
		} while (row != 0);

		if (tau == timeCost && threadIdx.y < 3) state ^= __ldL1(&Mdev(rowa, 0));

		DState[(thread + offset) * 16 + threadIdx.y * 4 + threadIdx.x] = state;
	}
}

__global__
__launch_bounds__(TPB, 1)
void lyra2_gpu_hash_32_3_4way(uint32_t threads, uint32_t offset, const uint32_t timeCost, const uint32_t nRows, const uint32_t nCols, uint32_t tau, uint32_t prev)
{
	const uint32_t thread = blockIdx.x * blockDim.y + threadIdx.y;

	if ((thread + offset) < threads)
	{
		uint2 state[4];
		state[0] = __ldg(&DState[(thread + offset) * 16 + 0 * 4 + threadIdx.x]);
		state[1] = __ldg(&DState[(thread + offset) * 16 + 1 * 4 + threadIdx.x]);
		state[2] = __ldg(&DState[(thread + offset) * 16 + 2 * 4 + threadIdx.x]);
		state[3] = __ldg(&DState[(thread + offset) * 16 + 3 * 4 + threadIdx.x]);

		uint32_t row = 0;
		uint32_t rowa;
		uint2 bufIn[3], bufInOut[3], bufOut[3], bufstate[3];
		uint64_t step = (tau & 1) == 0 ? 0xFFFFFFFFFFFFFFFF : (uint64_t)((nRows >> 1) - 1);

		do {
			rowa = (uint32_t)(devectorize(WarpShuffle(state[0], 0, 4)) % (uint64_t)nRows);

			for (uint32_t i = 0; i < nCols; i++) {
				bufIn[0] = __ldL1(&Ddev(prev, i, 0));
				bufIn[1] = __ldL1(&Ddev(prev, i, 1));
				bufIn[2] = __ldL1(&Ddev(prev, i, 2));
				bufInOut[0] = __ldL1(&Ddev(rowa, i, 0));
				bufInOut[1] = __ldL1(&Ddev(rowa, i, 1));
				bufInOut[2] = __ldL1(&Ddev(rowa, i, 2));
				bufOut[0] = __ldL1(&Ddev(row, i, 0));
				bufOut[1] = __ldL1(&Ddev(row, i, 1));
				bufOut[2] = __ldL1(&Ddev(row, i, 2));
				if (i < nCols - 1) {
					prefetch(&Ddev(prev, i + 1, 0));
					prefetch(&Ddev(prev, i + 1, 1));
					prefetch(&Ddev(prev, i + 1, 2));
					prefetch(&Ddev(rowa, i + 1, 0));
					prefetch(&Ddev(rowa, i + 1, 1));
					prefetch(&Ddev(rowa, i + 1, 2));
					prefetch(&Ddev(row, i + 1, 0));
					prefetch(&Ddev(row, i + 1, 1));
					prefetch(&Ddev(row, i + 1, 2));
				}
				state[0] ^= bufIn[0] + bufInOut[0];
				state[1] ^= bufIn[1] + bufInOut[1];
				state[2] ^= bufIn[2] + bufInOut[2];

				reducedBlake2bLyra(state);

				WarpShuffle3(bufstate[0], bufstate[1], bufstate[2], state[0], state[1], state[2], threadIdx.x + 3, threadIdx.x + 3, threadIdx.x + 3, 4);

				bufOut[0] ^= state[0];
				bufOut[1] ^= state[1];
				bufOut[2] ^= state[2];
				if (row == rowa) {
					bufInOut[0] = bufOut[0];
					bufInOut[1] = bufOut[1];
					bufInOut[2] = bufOut[2];
				}
				else {
					__stL1(&Ddev(row, i, 0), bufOut[0]);
					__stL1(&Ddev(row, i, 1), bufOut[1]);
					__stL1(&Ddev(row, i, 2), bufOut[2]);
				}

				if (threadIdx.x == 0) {
					bufInOut[0] ^= bufstate[2];
					bufInOut[1] ^= bufstate[0];
					bufInOut[2] ^= bufstate[1];
				}
				else {
					bufInOut[0] ^= bufstate[0];
					bufInOut[1] ^= bufstate[1];
					bufInOut[2] ^= bufstate[2];
				}

				__stL1(&Ddev(rowa, i, 0), bufInOut[0]);
				__stL1(&Ddev(rowa, i, 1), bufInOut[1]);
				__stL1(&Ddev(rowa, i, 2), bufInOut[2]);
			}

			prev = row;

			row = (uint32_t)(((uint64_t)row + step) % (uint64_t)nRows);
		} while (row != 0);

		if (tau == timeCost) {
			state[0] ^= __ldL1(&Ddev(rowa, 0, 0));
			state[1] ^= __ldL1(&Ddev(rowa, 0, 1));
			state[2] ^= __ldL1(&Ddev(rowa, 0, 2));
		}

		DState[(thread + offset) * 16 + 0 * 4 + threadIdx.x] = state[0];
		DState[(thread + offset) * 16 + 1 * 4 + threadIdx.x] = state[1];
		DState[(thread + offset) * 16 + 2 * 4 + threadIdx.x] = state[2];
		DState[(thread + offset) * 16 + 3 * 4 + threadIdx.x] = state[3];
	}
}

__global__
__launch_bounds__(TPB, 1)
void lyra2_gpu_hash_32_4(uint32_t threads, uint32_t count, uint32_t startNounce, uint32_t *resNonce)
{
	const uint32_t thread = blockDim.x * blockIdx.x + threadIdx.x;

	uint2x4 state[4];

	if (thread < threads)
	{
		state[0] = __ldg4(&((uint2x4*)DState)[thread * 4 + 0]);
		state[1] = __ldg4(&((uint2x4*)DState)[thread * 4 + 1]);
		state[2] = __ldg4(&((uint2x4*)DState)[thread * 4 + 2]);
		state[3] = __ldg4(&((uint2x4*)DState)[thread * 4 + 3]);

		blake2bLyra(state);

		// hash words 6..7 against the target top; padding lanes past count never report
		if (thread < count && devectorize(state[0].w) <= c_target) {
			const uint32_t pos = atomicInc(&resNonce[0], UINT32_MAX) + 1;
			if (pos < LYRA2Z330_MAX_RESULTS)
				resNonce[pos] = startNounce + thread;
		}
	}
}

static uint64_t *d_matrix[MAX_GPUS];
static uint64_t *d_state[MAX_GPUS];
static bool s_16way[MAX_GPUS];

/* lanes per hash by the compiled arch: 16 on sm_50+, 4 below; the matrix buffer holds
 * one sub-launch (threads / lanes hashes) */
__host__
bool lyra2z330_cpu_init(int thr_id, uint32_t threads)
{
	const int dev_id = device_map[thr_id];
	s_16way[thr_id] = cuda_arch[dev_id] >= 500;
	const size_t hashes = threads / (s_16way[thr_id] ? 16 : 4);

	if (cudaMalloc(&d_matrix[thr_id], LYRA2Z330_MATRIX_BYTES * hashes) != cudaSuccess ||
	    cudaMalloc(&d_state[thr_id], 16 * sizeof(uint64_t) * threads) != cudaSuccess)
		return false;
	cudaMemcpyToSymbol(DMatrix, &d_matrix[thr_id], sizeof(uint64_t*), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(DState, &d_state[thr_id], sizeof(uint64_t*), 0, cudaMemcpyHostToDevice);
	return cudaGetLastError() == cudaSuccess;
}

__host__
void lyra2z330_cpu_free(int thr_id)
{
	cudaFree(d_matrix[thr_id]);
	cudaFree(d_state[thr_id]);
	d_matrix[thr_id] = d_state[thr_id] = NULL;
}

/* 80-byte header as 20 words, byte-swapped as hashed; nonce word ignored */
__host__
void lyra2z330_setBlock(const uint32_t *endiandata, uint64_t target)
{
	cudaMemcpyToSymbol(c_data, endiandata, 80, 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(c_target, &target, sizeof(target), 0, cudaMemcpyHostToDevice);
}

__host__
void lyra2z330_cpu_hash(int thr_id, uint32_t threads, uint32_t count, uint32_t startNounce, uint32_t *d_resNonce)
{
	const uint32_t timeCost = LYRA2Z330_TIMECOST, nRows = LYRA2Z330_ROWS, nCols = LYRA2Z330_COLS;
	dim3 grid((threads + TPB - 1) / TPB);
	dim3 block1(TPB);
	dim3 block2(4, 4, TPB / 16);
	dim3 block3(4, TPB / 4);

	cudaMemset(d_resNonce, 0, sizeof(uint32_t));
	lyra2_gpu_hash_80_1 <<< grid, block1 >>> (threads, startNounce, timeCost, nRows, nCols);
	// last row of an odd tau: nRows / 2 + 1, of an even one: 1; a sync after each matrix
	// kernel keeps the driver from batching them past the display watchdog
	if (s_16way[thr_id]) {
		for (uint32_t i = 0; i < 16; i++) {
			uint32_t prev = nRows - 1;
			lyra2_gpu_hash_32_2_16way <<< grid, block2 >>> (threads, grid.x * 2 * i, timeCost, nRows, nCols);
			cudaDeviceSynchronize();
			for (uint32_t tau = 1; tau <= timeCost; tau++) {
				lyra2_gpu_hash_32_3_16way <<< grid, block2 >>> (threads, grid.x * 2 * i, timeCost, nRows, nCols, tau, prev);
				cudaDeviceSynchronize();
				prev = (tau & 1) == 0 ? 1 : (nRows >> 1) + 1;
			}
		}
	} else {
		for (uint32_t i = 0; i < 4; i++) {
			uint32_t prev = nRows - 1;
			lyra2_gpu_hash_32_2_4way <<< grid, block3 >>> (threads, grid.x * 8 * i, timeCost, nRows, nCols);
			cudaDeviceSynchronize();
			for (uint32_t tau = 1; tau <= timeCost; tau++) {
				lyra2_gpu_hash_32_3_4way <<< grid, block3 >>> (threads, grid.x * 8 * i, timeCost, nRows, nCols, tau, prev);
				cudaDeviceSynchronize();
				prev = (tau & 1) == 0 ? 1 : (nRows >> 1) + 1;
			}
		}
	}
	lyra2_gpu_hash_32_4 <<< grid, block1 >>> (threads, count, startNounce, d_resNonce);
}
