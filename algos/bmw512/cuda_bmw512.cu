// SPDX-License-Identifier: GPL-3.0-or-later
/*
 * BMW-512 over the 80-byte header, both compressions and the target check in
 * one kernel. Compression code from tpruvot ccminer 2.3.1 (quark/cuda_bmw512.cu),
 * uint2 variant on all archs.
 */
#include <stdio.h>
#include <memory.h>

#include "miner.h"
#include "cuda_helper.h"
#include "bmw512.h"

__constant__ static uint64_t c_PaddedMessage80[16]; // 80-byte header + padding

#define SHR(x, n) SHR2(x, n)
#define SHL(x, n) SHL2(x, n)
#define ROL(x, n) ROL2(x, n)

#define CONST_EXP2(i) \
	q[i+0] + ROL(q[i+1], 5)  + q[i+2] + ROL(q[i+3], 11) + \
	q[i+4] + ROL(q[i+5], 27) + q[i+6] + SWAPUINT2(q[i+7]) + \
	q[i+8] + ROL(q[i+9], 37) + q[i+10] + ROL(q[i+11], 43) + \
	q[i+12] + ROL(q[i+13], 53) + (SHR(q[i+14],1) ^ q[i+14]) + (SHR(q[i+15],2) ^ q[i+15])

__device__ static
void Compression512(uint2 *msg, uint2 *hash)
{
	uint2 q[32];
	uint2 tmp;

	tmp = (msg[ 5] ^ hash[ 5]) - (msg[ 7] ^ hash[ 7]) + (msg[10] ^ hash[10]) + (msg[13] ^ hash[13]) + (msg[14] ^ hash[14]);
	q[0] = (SHR(tmp, 1) ^ SHL(tmp, 3) ^ ROL(tmp,  4) ^ ROL(tmp, 37)) + hash[1];
	tmp = (msg[ 6] ^ hash[ 6]) - (msg[ 8] ^ hash[ 8]) + (msg[11] ^ hash[11]) + (msg[14] ^ hash[14]) - (msg[15] ^ hash[15]);
	q[1] = (SHR(tmp, 1) ^ SHL(tmp, 2) ^ ROL(tmp, 13) ^ ROL(tmp, 43)) + hash[2];
	tmp = (msg[ 0] ^ hash[ 0]) + (msg[ 7] ^ hash[ 7]) + (msg[ 9] ^ hash[ 9]) - (msg[12] ^ hash[12]) + (msg[15] ^ hash[15]);
	q[2] = (SHR(tmp, 2) ^ SHL(tmp, 1) ^ ROL(tmp, 19) ^ ROL(tmp, 53)) + hash[3];
	tmp = (msg[ 0] ^ hash[ 0]) - (msg[ 1] ^ hash[ 1]) + (msg[ 8] ^ hash[ 8]) - (msg[10] ^ hash[10]) + (msg[13] ^ hash[13]);
	q[3] = (SHR(tmp, 2) ^ SHL(tmp, 2) ^ ROL(tmp, 28) ^ ROL(tmp, 59)) + hash[4];
	tmp = (msg[ 1] ^ hash[ 1]) + (msg[ 2] ^ hash[ 2]) + (msg[ 9] ^ hash[ 9]) - (msg[11] ^ hash[11]) - (msg[14] ^ hash[14]);
	q[4] = (SHR(tmp, 1) ^ tmp) + hash[5];
	tmp = (msg[ 3] ^ hash[ 3]) - (msg[ 2] ^ hash[ 2]) + (msg[10] ^ hash[10]) - (msg[12] ^ hash[12]) + (msg[15] ^ hash[15]);
	q[5] = (SHR(tmp, 1) ^ SHL(tmp, 3) ^ ROL(tmp,  4) ^ ROL(tmp, 37)) + hash[6];
	tmp = (msg[ 4] ^ hash[ 4]) - (msg[ 0] ^ hash[ 0]) - (msg[ 3] ^ hash[ 3]) - (msg[11] ^ hash[11]) + (msg[13] ^ hash[13]);
	q[6] = (SHR(tmp, 1) ^ SHL(tmp, 2) ^ ROL(tmp, 13) ^ ROL(tmp, 43)) + hash[7];
	tmp = (msg[ 1] ^ hash[ 1]) - (msg[ 4] ^ hash[ 4]) - (msg[ 5] ^ hash[ 5]) - (msg[12] ^ hash[12]) - (msg[14] ^ hash[14]);
	q[7] = (SHR(tmp, 2) ^ SHL(tmp, 1) ^ ROL(tmp, 19) ^ ROL(tmp, 53)) + hash[8];
	tmp = (msg[ 2] ^ hash[ 2]) - (msg[ 5] ^ hash[ 5]) - (msg[ 6] ^ hash[ 6]) + (msg[13] ^ hash[13]) - (msg[15] ^ hash[15]);
	q[8] = (SHR(tmp, 2) ^ SHL(tmp, 2) ^ ROL(tmp, 28) ^ ROL(tmp, 59)) + hash[9];
	tmp = (msg[ 0] ^ hash[ 0]) - (msg[ 3] ^ hash[ 3]) + (msg[ 6] ^ hash[ 6]) - (msg[ 7] ^ hash[ 7]) + (msg[14] ^ hash[14]);
	q[9] = (SHR(tmp, 1) ^ tmp) + hash[10];
	tmp = (msg[ 8] ^ hash[ 8]) - (msg[ 1] ^ hash[ 1]) - (msg[ 4] ^ hash[ 4]) - (msg[ 7] ^ hash[ 7]) + (msg[15] ^ hash[15]);
	q[10] = (SHR(tmp, 1) ^ SHL(tmp, 3) ^ ROL(tmp,  4) ^ ROL(tmp, 37)) + hash[11];
	tmp = (msg[ 8] ^ hash[ 8]) - (msg[ 0] ^ hash[ 0]) - (msg[ 2] ^ hash[ 2]) - (msg[ 5] ^ hash[ 5]) + (msg[ 9] ^ hash[ 9]);
	q[11] = (SHR(tmp, 1) ^ SHL(tmp, 2) ^ ROL(tmp, 13) ^ ROL(tmp, 43)) + hash[12];
	tmp = (msg[ 1] ^ hash[ 1]) + (msg[ 3] ^ hash[ 3]) - (msg[ 6] ^ hash[ 6]) - (msg[ 9] ^ hash[ 9]) + (msg[10] ^ hash[10]);
	q[12] = (SHR(tmp, 2) ^ SHL(tmp, 1) ^ ROL(tmp, 19) ^ ROL(tmp, 53)) + hash[13];
	tmp = (msg[ 2] ^ hash[ 2]) + (msg[ 4] ^ hash[ 4]) + (msg[ 7] ^ hash[ 7]) + (msg[10] ^ hash[10]) + (msg[11] ^ hash[11]);
	q[13] = (SHR(tmp, 2) ^ SHL(tmp, 2) ^ ROL(tmp, 28) ^ ROL(tmp, 59)) + hash[14];
	tmp = (msg[ 3] ^ hash[ 3]) - (msg[ 5] ^ hash[ 5]) + (msg[ 8] ^ hash[ 8]) - (msg[11] ^ hash[11]) - (msg[12] ^ hash[12]);
	q[14] = (SHR(tmp, 1) ^ tmp) + hash[15];
	tmp = (msg[12] ^ hash[12]) - (msg[ 4] ^ hash[ 4]) - (msg[ 6] ^ hash[ 6]) - (msg[ 9] ^ hash[ 9]) + (msg[13] ^ hash[13]);
	q[15] = (SHR(tmp, 1) ^ SHL(tmp, 3) ^ ROL(tmp, 4) ^ ROL(tmp, 37)) + hash[0];

	q[0+16] =
		(SHR(q[0], 1) ^ SHL(q[0], 2) ^ ROL(q[0], 13) ^ ROL(q[0], 43)) +
		(SHR(q[0+1], 2) ^ SHL(q[0+1], 1) ^ ROL(q[0+1], 19) ^ ROL(q[0+1], 53)) +
		(SHR(q[0+2], 2) ^ SHL(q[0+2], 2) ^ ROL(q[0+2], 28) ^ ROL(q[0+2], 59)) +
		(SHR(q[0+3], 1) ^ SHL(q[0+3], 3) ^ ROL(q[0+3],  4) ^ ROL(q[0+3], 37)) +
		(SHR(q[0+4], 1) ^ SHL(q[0+4], 2) ^ ROL(q[0+4], 13) ^ ROL(q[0+4], 43)) +
		(SHR(q[0+5], 2) ^ SHL(q[0+5], 1) ^ ROL(q[0+5], 19) ^ ROL(q[0+5], 53)) +
		(SHR(q[0+6], 2) ^ SHL(q[0+6], 2) ^ ROL(q[0+6], 28) ^ ROL(q[0+6], 59)) +
		(SHR(q[0+7], 1) ^ SHL(q[0+7], 3) ^ ROL(q[0+7],  4) ^ ROL(q[0+7], 37)) +
		(SHR(q[0+8], 1) ^ SHL(q[0+8], 2) ^ ROL(q[0+8], 13) ^ ROL(q[0+8], 43)) +
		(SHR(q[0+9], 2) ^ SHL(q[0+9], 1) ^ ROL(q[0+9], 19) ^ ROL(q[0+9], 53)) +
		(SHR(q[0+10], 2) ^ SHL(q[0+10], 2) ^ ROL(q[0+10], 28) ^ ROL(q[0+10], 59)) +
		(SHR(q[0+11], 1) ^ SHL(q[0+11], 3) ^ ROL(q[0+11],  4) ^ ROL(q[0+11], 37)) +
		(SHR(q[0+12], 1) ^ SHL(q[0+12], 2) ^ ROL(q[0+12], 13) ^ ROL(q[0+12], 43)) +
		(SHR(q[0+13], 2) ^ SHL(q[0+13], 1) ^ ROL(q[0+13], 19) ^ ROL(q[0+13], 53)) +
		(SHR(q[0+14], 2) ^ SHL(q[0+14], 2) ^ ROL(q[0+14], 28) ^ ROL(q[0+14], 59)) +
		(SHR(q[0+15], 1) ^ SHL(q[0+15], 3) ^ ROL(q[0+15],  4) ^ ROL(q[0+15], 37)) +
		((make_uint2(0x55555550ul, 0x55555555) + ROL(msg[0], 0 + 1) +
		ROL(msg[0+3], 0+4) - ROL(msg[0+10], 0+11) ) ^ hash[0+7]);

	q[1 + 16] =
		(SHR(q[1], 1) ^ SHL(q[1], 2) ^ ROL(q[1], 13) ^ ROL(q[1], 43)) +
		(SHR(q[1 + 1], 2) ^ SHL(q[1 + 1], 1) ^ ROL(q[1 + 1], 19) ^ ROL(q[1 + 1], 53)) +
		(SHR(q[1 + 2], 2) ^ SHL(q[1 + 2], 2) ^ ROL(q[1 + 2], 28) ^ ROL(q[1 + 2], 59)) +
		(SHR(q[1 + 3], 1) ^ SHL(q[1 + 3], 3) ^ ROL(q[1 + 3], 4) ^ ROL(q[1 + 3], 37)) +
		(SHR(q[1 + 4], 1) ^ SHL(q[1 + 4], 2) ^ ROL(q[1 + 4], 13) ^ ROL(q[1 + 4], 43)) +
		(SHR(q[1 + 5], 2) ^ SHL(q[1 + 5], 1) ^ ROL(q[1 + 5], 19) ^ ROL(q[1 + 5], 53)) +
		(SHR(q[1 + 6], 2) ^ SHL(q[1 + 6], 2) ^ ROL(q[1 + 6], 28) ^ ROL(q[1 + 6], 59)) +
		(SHR(q[1 + 7], 1) ^ SHL(q[1 + 7], 3) ^ ROL(q[1 + 7], 4) ^ ROL(q[1 + 7], 37)) +
		(SHR(q[1 + 8], 1) ^ SHL(q[1 + 8], 2) ^ ROL(q[1 + 8], 13) ^ ROL(q[1 + 8], 43)) +
		(SHR(q[1 + 9], 2) ^ SHL(q[1 + 9], 1) ^ ROL(q[1 + 9], 19) ^ ROL(q[1 + 9], 53)) +
		(SHR(q[1 + 10], 2) ^ SHL(q[1 + 10], 2) ^ ROL(q[1 + 10], 28) ^ ROL(q[1 + 10], 59)) +
		(SHR(q[1 + 11], 1) ^ SHL(q[1 + 11], 3) ^ ROL(q[1 + 11], 4) ^ ROL(q[1 + 11], 37)) +
		(SHR(q[1 + 12], 1) ^ SHL(q[1 + 12], 2) ^ ROL(q[1 + 12], 13) ^ ROL(q[1 + 12], 43)) +
		(SHR(q[1 + 13], 2) ^ SHL(q[1 + 13], 1) ^ ROL(q[1 + 13], 19) ^ ROL(q[1 + 13], 53)) +
		(SHR(q[1 + 14], 2) ^ SHL(q[1 + 14], 2) ^ ROL(q[1 + 14], 28) ^ ROL(q[1 + 14], 59)) +
		(SHR(q[1 + 15], 1) ^ SHL(q[1 + 15], 3) ^ ROL(q[1 + 15], 4) ^ ROL(q[1 + 15], 37)) +
		((make_uint2(0xAAAAAAA5, 0x5AAAAAAA) + ROL(msg[1], 1 + 1) +
		ROL(msg[1 + 3], 1 + 4) - ROL(msg[1 + 10], 1 + 11)) ^ hash[1 + 7]);

	q[2 + 16] = CONST_EXP2(2) +
		((make_uint2(0xFFFFFFFA, 0x5FFFFFFF) + ROL(msg[2], 2 + 1) +
		ROL(msg[2+3], 2+4) - ROL(msg[2+10], 2+11) ) ^ hash[2+7]);
	q[3 + 16] = CONST_EXP2(3) +
		((make_uint2(0x5555554F, 0x65555555) + ROL(msg[3], 3 + 1) +
		ROL(msg[3 + 3], 3 + 4) - ROL(msg[3 + 10], 3 + 11)) ^ hash[3 + 7]);
	q[4 + 16] = CONST_EXP2(4) +
		((make_uint2(0xAAAAAAA4, 0x6AAAAAAA) + ROL(msg[4], 4 + 1) +
		ROL(msg[4 + 3], 4 + 4) - ROL(msg[4 + 10], 4 + 11)) ^ hash[4 + 7]);
	q[5 + 16] = CONST_EXP2(5) +
		((make_uint2(0xFFFFFFF9, 0x6FFFFFFF) + ROL(msg[5], 5 + 1) +
		ROL(msg[5 + 3], 5 + 4) - ROL(msg[5 + 10], 5 + 11)) ^ hash[5 + 7]);
	q[6 + 16] = CONST_EXP2(6) +
		((make_uint2(0x5555554E, 0x75555555)+ ROL(msg[6], 6 + 1) +
		ROL(msg[6 + 3], 6 + 4) - ROL(msg[6 - 6], (6 - 6) + 1)) ^ hash[6 + 7]);
	q[7 + 16] = CONST_EXP2(7) +
		((make_uint2(0xAAAAAAA3, 0x7AAAAAAA) + ROL(msg[7], 7 + 1) +
		ROL(msg[7 + 3], 7 + 4) - ROL(msg[7 - 6], (7 - 6) + 1)) ^ hash[7 + 7]);
	q[8 + 16] = CONST_EXP2(8) +
		((make_uint2(0xFFFFFFF8, 0x7FFFFFFF) + ROL(msg[8], 8 + 1) +
		ROL(msg[8 + 3], 8 + 4) - ROL(msg[8 - 6], (8 - 6) + 1)) ^ hash[8 + 7]);
	q[9 + 16] = CONST_EXP2(9) +
		((make_uint2(0x5555554D, 0x85555555) + ROL(msg[9], 9 + 1) +
		ROL(msg[9 + 3], 9 + 4) - ROL(msg[9 - 6], (9 - 6) + 1)) ^ hash[9 - 9]);
	q[10 + 16] = CONST_EXP2(10) +
		((make_uint2(0xAAAAAAA2, 0x8AAAAAAA) + ROL(msg[10], 10 + 1) +
		ROL(msg[10 + 3], 10 + 4) - ROL(msg[10 - 6], (10 - 6) + 1)) ^ hash[10 - 9]);
	q[11 + 16] = CONST_EXP2(11) +
		((make_uint2(0xFFFFFFF7, 0x8FFFFFFF) + ROL(msg[11], 11 + 1) +
		ROL(msg[11 + 3], 11 + 4) - ROL(msg[11 - 6], (11 - 6) + 1)) ^ hash[11 - 9]);
	q[12 + 16] = CONST_EXP2(12) +
		((make_uint2(0x5555554C, 0x95555555) + ROL(msg[12], 12 + 1) +
		ROL(msg[12 + 3], 12 + 4) - ROL(msg[12 - 6], (12 - 6) + 1)) ^ hash[12 - 9]);
	q[13 + 16] = CONST_EXP2(13) +
		((make_uint2(0xAAAAAAA1, 0x9AAAAAAA) + ROL(msg[13], 13 + 1) +
		ROL(msg[13 - 13], (13 - 13) + 1) - ROL(msg[13 - 6], (13 - 6) + 1)) ^ hash[13 - 9]);
	q[14 + 16] = CONST_EXP2(14) +
		((make_uint2(0xFFFFFFF6, 0x9FFFFFFF) + ROL(msg[14], 14 + 1) +
		ROL(msg[14 - 13], (14 - 13) + 1) - ROL(msg[14 - 6], (14 - 6) + 1)) ^ hash[14 - 9]);
	q[15 + 16] = CONST_EXP2(15) +
		((make_uint2(0x5555554B, 0xA5555555) + ROL(msg[15], 15 + 1) +
		ROL(msg[15 - 13], (15 - 13) + 1) - ROL(msg[15 - 6], (15 - 6) + 1)) ^ hash[15 - 9]);

	uint2 XL64 = q[16]^q[17]^q[18]^q[19]^q[20]^q[21]^q[22]^q[23];
	uint2 XH64 = XL64^q[24] ^ q[25] ^ q[26] ^ q[27] ^ q[28] ^ q[29] ^ q[30] ^ q[31];

	hash[0] = (SHL(XH64, 5) ^ SHR(q[16],5) ^ msg[ 0]) + (XL64 ^ q[24] ^ q[ 0]);
	hash[1] = (SHR(XH64, 7) ^ SHL(q[17],8) ^ msg[ 1]) + (XL64 ^ q[25] ^ q[ 1]);
	hash[2] = (SHR(XH64, 5) ^ SHL(q[18],5) ^ msg[ 2]) + (XL64 ^ q[26] ^ q[ 2]);
	hash[3] = (SHR(XH64, 1) ^ SHL(q[19],5) ^ msg[ 3]) + (XL64 ^ q[27] ^ q[ 3]);
	hash[4] = (SHR(XH64, 3) ^     q[20]    ^ msg[ 4]) + (XL64 ^ q[28] ^ q[ 4]);
	hash[5] = (SHL(XH64, 6) ^ SHR(q[21],6) ^ msg[ 5]) + (XL64 ^ q[29] ^ q[ 5]);
	hash[6] = (SHR(XH64, 4) ^ SHL(q[22],6) ^ msg[ 6]) + (XL64 ^ q[30] ^ q[ 6]);
	hash[7] = (SHR(XH64,11) ^ SHL(q[23],2) ^ msg[ 7]) + (XL64 ^ q[31] ^ q[ 7]);

	hash[ 8] = ROL(hash[4], 9) + (XH64 ^ q[24] ^ msg[ 8]) + (SHL(XL64,8) ^ q[23] ^ q[ 8]);
	hash[ 9] = ROL(hash[5],10) + (XH64 ^ q[25] ^ msg[ 9]) + (SHR(XL64,6) ^ q[16] ^ q[ 9]);
	hash[10] = ROL(hash[6],11) + (XH64 ^ q[26] ^ msg[10]) + (SHL(XL64,6) ^ q[17] ^ q[10]);
	hash[11] = ROL(hash[7],12) + (XH64 ^ q[27] ^ msg[11]) + (SHL(XL64,4) ^ q[18] ^ q[11]);
	hash[12] = ROL(hash[0],13) + (XH64 ^ q[28] ^ msg[12]) + (SHR(XL64,3) ^ q[19] ^ q[12]);
	hash[13] = ROL(hash[1],14) + (XH64 ^ q[29] ^ msg[13]) + (SHR(XL64,4) ^ q[20] ^ q[13]);
	hash[14] = ROL(hash[2],15) + (XH64 ^ q[30] ^ msg[14]) + (SHR(XL64,7) ^ q[21] ^ q[14]);
	hash[15] = ROL(hash[3],16) + (XH64 ^ q[31] ^ msg[15]) + (SHR(XL64, 2) ^ q[22] ^ q[15]);
}

/* resNonce[0] counts candidates, resNonce[1..BMW512_MAX_RESULTS-1] hold them */
__global__ __launch_bounds__(128, 4)
void bmw512_gpu_hash_80_final(uint32_t threads, uint32_t startNonce, uint32_t *resNonce, uint64_t target)
{
	const uint32_t thread = blockDim.x * blockIdx.x + threadIdx.x;
	if (thread >= threads)
		return;

	const uint32_t nonce = startNonce + thread;
	uint2 h[16], message[16];
	#pragma unroll
	for (int i = 0; i < 16; i++) {
		h[i] = vectorize(0x8081828384858687ull + 0x0808080808080808ull * i);
		message[i] = vectorize(c_PaddedMessage80[i]);
	}
	message[9].y = cuda_swab32(nonce);
	Compression512(message, h);
	#pragma unroll
	for (int i = 0; i < 16; i++)
		message[i] = make_uint2(0xaaaaaaa0 + i, 0xaaaaaaaa);
	Compression512(h, message);
	const uint64_t high = devectorize(message[8 + 3]);
	// top 64 bits of the 256-bit hash against target words 6..7
	if (high <= target) {
		const uint32_t pos = atomicInc(&resNonce[0], UINT32_MAX) + 1;
		if (pos < BMW512_MAX_RESULTS)
			resNonce[pos] = nonce;
	}
}

__host__
void bmw512_cpu_setBlock_80(const void *pdata)
{
	uint64_t message[16] = { 0 };
	memcpy(message, pdata, 80);
	message[10] = 0x80;
	message[15] = 640; // bits
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_PaddedMessage80, message, sizeof(message), 0, cudaMemcpyHostToDevice));
}

/* one launch over [startNonce, startNonce + threads), returns the candidate count */
__host__
uint32_t bmw512_scan(uint32_t threads, uint32_t startNonce, uint64_t target,
	uint32_t *d_res, uint32_t *h_res)
{
	const uint32_t tpb = 128;
	dim3 grid((threads + tpb - 1) / tpb);
	dim3 block(tpb);
	cudaMemset(d_res, 0, sizeof(uint32_t));
	bmw512_gpu_hash_80_final <<<grid, block>>> (threads, startNonce, d_res, target);
	cudaMemcpy(h_res, d_res, BMW512_MAX_RESULTS * sizeof(uint32_t), cudaMemcpyDeviceToHost);
	return h_res[0];
}
