// SPDX-License-Identifier: GPL-3.0-or-later
/*
 * heavyhash (Optical Bitcoin): SHA3-256 of the 80-byte header, a 64x64 4-bit
 * matrix times the 64 nibbles of that hash, SHA3-256 of the hash XOR the
 * product nibbles (heavyhash_cpu.c). One thread per nonce; the matrix of the
 * job sits in constant memory. sm_61 multiplies with __dp4a (4 nibbles per
 * word); older GPUs keep two rows in the 16-bit halves of one accumulator (a
 * row sum is at most 64 * 225 = 14400). The kernel screens the top 64 bits;
 * every candidate is checked on the CPU.
 */
#include <stdio.h>
#include <memory.h>

#include "miner.h"
#include "cuda_helper.h"
#include "cuda/selftest_gate.cuh"
#include "heavyhash_cpu.h"

#define TPB		256
#define MAX_RESULTS	16

static __constant__ uint2 c_header[10];		/* header lanes, nonce half of lane 9 zero */
static __constant__ uint32_t c_mat4[64 * 16];	/* row i, word w: m[i][4w..4w+3], one per byte */
static __constant__ uint32_t c_mat2[64 * 32];	/* column j, pair p: m[2p][j] | m[2p+1][j] << 16 */

static __constant__ uint2 c_rc[24] = {
	{ 0x00000001, 0x00000000 }, { 0x00008082, 0x00000000 }, { 0x0000808a, 0x80000000 }, { 0x80008000, 0x80000000 },
	{ 0x0000808b, 0x00000000 }, { 0x80000001, 0x00000000 }, { 0x80008081, 0x80000000 }, { 0x00008009, 0x80000000 },
	{ 0x0000008a, 0x00000000 }, { 0x00000088, 0x00000000 }, { 0x80008009, 0x00000000 }, { 0x8000000a, 0x00000000 },
	{ 0x8000808b, 0x00000000 }, { 0x0000008b, 0x80000000 }, { 0x00008089, 0x80000000 }, { 0x00008003, 0x80000000 },
	{ 0x00008002, 0x80000000 }, { 0x00000080, 0x80000000 }, { 0x0000800a, 0x00000000 }, { 0x8000000a, 0x80000000 },
	{ 0x80008081, 0x80000000 }, { 0x00008080, 0x80000000 }, { 0x80000001, 0x00000000 }, { 0x80008008, 0x80000000 }
};

static uint32_t *d_resNonce[MAX_GPUS];
static uint32_t *h_resNonce[MAX_GPUS];

/* the matrix and header of the job each thread last uploaded */
static uint8_t hh_matrix[MAX_GPUS][64][64];
static uint8_t hh_prev[MAX_GPUS][32];
static bool hh_have[MAX_GPUS];

/* Keccak-f[1600], as sha3t */
__device__ __forceinline__ static void keccakf(uint2 st[25])
{
	uint2 t[5], u[5], v, w;

	#pragma unroll
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

/* the hash's top 64 bits in fulltest order (((uint64_t *)vhash)[3]) */
__device__ __forceinline__ static uint64_t heavyhash_top(const uint32_t nonce)
{
	uint2 st[25];
	uint32_t h[8], p[8];

	/* SHA3-256 of the 80 header bytes: 10 lanes, padding at bytes 80 and 135 */
	#pragma unroll
	for (int i = 0; i < 9; i++)
		st[i] = c_header[i];
	st[9] = make_uint2(c_header[9].x, cuda_swab32(nonce));
	st[10] = make_uint2(0x06, 0);
	#pragma unroll
	for (int i = 11; i < 25; i++)
		st[i] = make_uint2(0, 0);
	st[16].y = 0x80000000u;
	keccakf(st);

	#pragma unroll
	for (int i = 0; i < 4; i++) {
		h[2 * i] = st[i].x;
		h[2 * i + 1] = st[i].y;
	}

#if __CUDA_ARCH__ >= 610
	/* word w holds nibbles 4w..4w+3: the high and low nibble of bytes 2w, 2w+1 */
	uint32_t vec[16];
	#pragma unroll
	for (int w = 0; w < 16; w++) {
		const uint32_t b = h[w >> 1] >> ((w & 1) * 16);
		vec[w] = ((b >> 4) & 0x0f) | ((b & 0x0f) << 8) | ((b >> 12) & 0x0f) << 16 | ((b >> 8) & 0x0f) << 24;
	}
	#pragma unroll
	for (int i = 0; i < 8; i++)
		p[i] = 0;
	#pragma unroll
	for (int m = 0; m < 32; m++) {
		uint32_t s0 = 0, s1 = 0;
		#pragma unroll
		for (int w = 0; w < 16; w++) {
			s0 = __dp4a(c_mat4[(2 * m) * 16 + w], vec[w], s0);
			s1 = __dp4a(c_mat4[(2 * m + 1) * 16 + w], vec[w], s1);
		}
		p[m >> 2] |= (((s0 >> 10) << 4) | (s1 >> 10)) << (8 * (m & 3));
	}
#else
	uint32_t acc[32];
	#pragma unroll
	for (int m = 0; m < 32; m++)
		acc[m] = 0;
	#pragma unroll
	for (int j = 0; j < 64; j++) {
		/* nibble j: byte j / 2, high nibble first */
		const uint32_t v = (h[j >> 3] >> (8 * ((j >> 1) & 3) + ((j & 1) ? 0 : 4))) & 0x0f;
		#pragma unroll
		for (int m = 0; m < 32; m++)
			acc[m] += c_mat2[j * 32 + m] * v;
	}
	#pragma unroll
	for (int i = 0; i < 8; i++)
		p[i] = 0;
	#pragma unroll
	for (int m = 0; m < 32; m++)
		p[m >> 2] |= ((((acc[m] & 0xffff) >> 10) << 4) | (acc[m] >> 26)) << (8 * (m & 3));
#endif

	/* SHA3-256 of the 32 bytes h ^ p: 4 lanes, padding at byte 32 */
	#pragma unroll
	for (int i = 0; i < 4; i++)
		st[i] = make_uint2(h[2 * i] ^ p[2 * i], h[2 * i + 1] ^ p[2 * i + 1]);
	st[4] = make_uint2(0x06, 0);
	#pragma unroll
	for (int i = 5; i < 25; i++)
		st[i] = make_uint2(0, 0);
	st[16].y = 0x80000000u;
	keccakf(st);

	return ((uint64_t) st[3].y << 32) | st[3].x;
}

/* resNonce[0] counts candidates, resNonce[1..MAX_RESULTS-1] hold them */
__global__ __launch_bounds__(TPB)
void heavyhash_gpu_hash(const uint32_t threads, const uint32_t startNonce, uint32_t *resNonce, const uint64_t highTarget)
{
	const uint32_t thread = blockDim.x * blockIdx.x + threadIdx.x;
	if (thread >= threads)
		return;

	const uint32_t nonce = startNonce + thread;
	if (heavyhash_top(nonce) <= highTarget) {
		const uint32_t pos = atomicInc(&resNonce[0], UINT32_MAX) + 1;
		if (pos < MAX_RESULTS)
			resNonce[pos] = nonce;
	}
}

/* header from work->data (big-endian words); builds and uploads the matrix
 * when the previous block hash changed */
static void heavyhash_setBlock(int thr_id, const uint32_t *pdata)
{
	uint8_t hdr[80];
	for (int i = 0; i < 20; i++)
		be32enc(hdr + 4 * i, pdata[i]);

	if (!hh_have[thr_id] || memcmp(hh_prev[thr_id], hdr + 4, 32)) {
		uint32_t mat4[64 * 16], mat2[64 * 32];
		uint8_t (*m)[64] = hh_matrix[thr_id];

		heavyhash_matrix(m, hdr + 4);
		for (int i = 0; i < 64; i++)
			for (int w = 0; w < 16; w++)
				mat4[i * 16 + w] = m[i][4 * w] | m[i][4 * w + 1] << 8 |
					m[i][4 * w + 2] << 16 | (uint32_t) m[i][4 * w + 3] << 24;
		for (int j = 0; j < 64; j++)
			for (int p = 0; p < 32; p++)
				mat2[j * 32 + p] = m[2 * p][j] | (uint32_t) m[2 * p + 1][j] << 16;
		CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_mat4, mat4, sizeof(mat4), 0, cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_mat2, mat2, sizeof(mat2), 0, cudaMemcpyHostToDevice));
		memcpy(hh_prev[thr_id], hdr + 4, 32);
		hh_have[thr_id] = true;
	}

	memset(hdr + 76, 0, 4);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_header, hdr, 80, 0, cudaMemcpyHostToDevice));
}

/* the 80 header bytes, with the matrix of the uploaded job */
static void heavyhash_job_hash(int thr_id, void *output, const uint8_t *hdr)
{
	heavyhash_with_matrix(hh_matrix[thr_id], hdr, 80, output);
}

/* one launch over [start, start + count): the candidate count, the slots in res */
static uint32_t heavyhash_scan(int thr_id, uint32_t start, uint32_t count, uint64_t highTarget, uint32_t *res)
{
	cudaMemset(d_resNonce[thr_id], 0, sizeof(uint32_t));
	heavyhash_gpu_hash <<<(count + TPB - 1) / TPB, TPB>>> (count, start, d_resNonce[thr_id], highTarget);
	cudaMemcpy(res, d_resNonce[thr_id], MAX_RESULTS * sizeof(uint32_t), cudaMemcpyDeviceToHost);
	return res[0];
}

/* ---- init-time self-test, fail-closed (cuda/selftest_gate.cuh) ---- */

/* the Optical Bitcoin main and test network genesis headers and their hashes
 * (the block hash reversed) */
static const char *hh_block_hdr[2] = {
	"010000000000000000000000000000000000000000000000000000000000000000000000f5febbad19864a6900b6ce84"
	"287511e6e746229ce6239ff3df7481654778a4c4d3e15d60ffff001c0747d042",
	"010000000000000000000000000000000000000000000000000000000000000000000000f5febbad19864a6900b6ce84"
	"287511e6e746229ce6239ff3df7481654778a4c4e5f0bf60ff3f001d1bcb807b"
};
static const char *hh_block_pow[2] = {
	"c0958f404197924672e7436eca537952de96ee775df63f7e7a5c110000000000",
	"563261f133edc4556b6732d1ab257e49d34b2aa27094bd8df0cb8e2400000000"
};

static bool heavyhash_device_selftest(int thr_id)
{
	static bool tested[MAX_GPUS] = { 0 };
	const int dev_id = device_map[thr_id];
	if (tested[dev_id]) return true;
	tested[dev_id] = true;

	uint32_t _ALIGN(64) pdata[20], dig[8];
	uint32_t *res = h_resNonce[thr_id];
	uint8_t hdr[80];
	char hex[65];
	bool cpu_ok = true, scan_ok = true, miss_ok = true, neg_ok = true;

	for (int b = 0; b < 2; b++) {
		/* the CPU hash gives the block hash */
		hex2bin(hdr, hh_block_hdr[b], 80);
		heavyhash_hash(dig, hdr);
		for (int i = 0; i < 32; i++)
			sprintf(hex + 2 * i, "%02x", ((uint8_t *)dig)[i]);
		cpu_ok &= strcmp(hex, hh_block_pow[b]) == 0;

		/* the kernel finds the block nonce at a target equal to its top
		 * 64 bits, not one below */
		for (int i = 0; i < 20; i++)
			pdata[i] = be32dec(hdr + 4 * i);
		const uint64_t top = ((uint64_t) dig[7] << 32) | dig[6];
		heavyhash_setBlock(thr_id, pdata);
		scan_ok &= heavyhash_scan(thr_id, pdata[19], 1, top, res) == 1 && res[1] == pdata[19];
		miss_ok &= heavyhash_scan(thr_id, pdata[19], 1, top - 1, res) == 0;

		/* a flipped header bit must lose the hit */
		pdata[17] ^= 0x00000100u;
		heavyhash_setBlock(thr_id, pdata);
		neg_ok &= heavyhash_scan(thr_id, pdata[19], 1, top, res) == 0;
	}

	/* 1024 nonces of the synthetic header (byte i = i, another matrix) at a
	 * top-byte-0 target: the kernel reports exactly the nonces the CPU finds
	 * (about 4) */
	bool span_ok = true;
	const uint32_t start = 0x5a1d0c3bu;
	for (int i = 0; i < 80; i++)
		hdr[i] = (uint8_t) i;
	for (int i = 0; i < 20; i++)
		pdata[i] = be32dec(hdr + 4 * i);
	heavyhash_setBlock(thr_id, pdata);
	const uint32_t n = heavyhash_scan(thr_id, start, 1024, 0x00ffffffffffffffULL, res);
	uint32_t found = 0;
	for (uint32_t i = 0; i < 1024; i++) {
		be32enc(hdr + 76, start + i);
		heavyhash_job_hash(thr_id, dig, hdr);
		if (dig[7] <= 0x00ffffffu) {
			bool listed = false;
			for (uint32_t k = 1; k <= n && k < MAX_RESULTS; k++)
				listed |= res[k] == start + i;
			span_ok &= listed;
			found++;
		}
	}
	span_ok &= n == found && found < MAX_RESULTS;
	hh_have[thr_id] = false;

	const cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		return selftest_gate(thr_id, "heavyhash", selftest_cuda_error(thr_id, "heavyhash", err));

	const bool passed = cpu_ok && scan_ok && miss_ok && neg_ok && span_ok;
	if (!passed)
		gpulog(LOG_ERR, thr_id, "heavyhash self-test FAILED (cpu %d scan %d miss %d neg %d span %d)",
			(int)cpu_ok, (int)scan_ok, (int)miss_ok, (int)neg_ok, (int)span_ok);
	return selftest_gate(thr_id, "heavyhash", passed);
}

/* ---- scan ---- */

static bool init[MAX_GPUS] = { 0 };

extern "C" int scanhash_heavyhash(int thr_id, struct work *work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint8_t hdr[80];
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];

	const int dev_id = device_map[thr_id];
	const uint32_t throughput = cuda_default_throughput(thr_id, device_sm[dev_id] < 500 ? 1U << 20 : 1U << 22);

	if (opt_benchmark) {
		// both words the kernel compares
		ptarget[7] = 0x03;
		ptarget[6] = 0xffffffffu;
	}
	const uint64_t highTarget = ((uint64_t) ptarget[7] << 32) | ptarget[6];

	if (!init[thr_id]) {
		cudaSetDevice(dev_id);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
			CUDA_LOG_ERROR();
		}
		gpulog(LOG_INFO, thr_id, "Intensity set to %g, %u cuda threads", throughput2intensity(throughput), throughput);

		cuda_get_arch(thr_id);
		CUDA_CALL_OR_RET_X(cudaMalloc(&d_resNonce[thr_id], MAX_RESULTS * sizeof(uint32_t)), -1);
		CUDA_CALL_OR_RET_X(cudaMallocHost(&h_resNonce[thr_id], MAX_RESULTS * sizeof(uint32_t)), -1);
		hh_have[thr_id] = false;

		heavyhash_device_selftest(thr_id);
		init[thr_id] = true;
	}

	for (int i = 0; i < 20; i++)
		be32enc(hdr + 4 * i, pdata[i]);
	heavyhash_setBlock(thr_id, pdata);

	do {
		// the last launch covers only the rest of the range
		const uint32_t batch = (uint32_t) min((uint64_t) throughput, (uint64_t) max_nonce - pdata[19]);
		if (!batch)
			break;

		uint32_t *res = h_resNonce[thr_id];
		uint32_t count = heavyhash_scan(thr_id, pdata[19], batch, highTarget, res);
		*hashes_done = pdata[19] - first_nonce + batch;

		if (count) {
			if (count >= MAX_RESULTS) {
				gpulog(LOG_WARNING, thr_id, "candidates flood: %u", count);
				count = MAX_RESULTS - 1;
			}
			// CPU-check every candidate, keep the two lowest valid ones
			uint32_t valid[2];
			int nvalid = 0;
			for (uint32_t n = 1; n <= count; n++) {
				uint32_t _ALIGN(64) vhash[8];
				be32enc(hdr + 76, res[n]);
				heavyhash_job_hash(thr_id, vhash, hdr);
				if (!fulltest(vhash, ptarget)) {
					const uint64_t top = ((uint64_t) vhash[7] << 32) | vhash[6];
					if (top > highTarget) {
						gpu_increment_reject(thr_id);
						if (!opt_quiet)
							gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU!", res[n]);
					}
					continue;
				}
				if (nvalid < 2)
					valid[nvalid++] = res[n];
				else if (res[n] < valid[1])
					valid[1] = res[n];
				if (nvalid == 2 && valid[1] < valid[0])
					xchg(valid[0], valid[1]);
			}

			if (nvalid) {
				uint32_t _ALIGN(64) vhash[8];
				work->valid_nonces = nvalid;
				for (int i = 0; i < nvalid; i++) {
					work->nonces[i] = valid[i];
					be32enc(hdr + 76, valid[i]);
					heavyhash_job_hash(thr_id, vhash, hdr);
					if (i == 0)
						work_set_target_ratio(work, vhash);
					else
						bn_set_target_ratio(work, vhash, 1);
				}
				// resume after the last reported nonce
				pdata[19] = valid[nvalid - 1] + 1;
				return nvalid;
			}
		}

		if ((uint64_t) batch + pdata[19] >= max_nonce) {
			pdata[19] = max_nonce;
			break;
		}
		pdata[19] += batch;

	} while (!work_restart[thr_id].restart);

	*hashes_done = pdata[19] - first_nonce;
	return 0;
}

extern "C" void free_heavyhash(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaDeviceSynchronize();
	cudaFreeHost(h_resNonce[thr_id]);
	cudaFree(d_resNonce[thr_id]);
	hh_have[thr_id] = false;
	init[thr_id] = false;
	cudaDeviceSynchronize();
}
