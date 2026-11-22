// SPDX-License-Identifier: GPL-3.0-or-later
/*
 * sha3t (SHA3-256t): SHA3-256 three times over the 80-byte header (FIPS 202
 * padding), the PoW of Bitcoin III up to block 64360 and of other coins. One
 * thread per nonce, three Keccak-f[1600] permutations; the kernel screens the
 * top 64 bits, every candidate is checked on the CPU (sph/sha3.c).
 */
#include <stdio.h>
#include <memory.h>

#include "miner.h"
#include "cuda_helper.h"
#include "cuda/selftest_gate.cuh"
#include "sph/sha3.h"

#define TPB		128
#define MAX_RESULTS	16

static __constant__ uint2 c_header[10];		/* header lanes, nonce half of lane 9 zero */

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

/* Keccak-f[1600], as verthash */
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
__device__ __forceinline__ static uint64_t sha3t_top(const uint32_t nonce)
{
	uint2 st[25];

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

	/* twice SHA3-256 of the 32-byte digest: 4 lanes, padding at byte 32 */
	#pragma unroll
	for (int k = 0; k < 2; k++) {
		st[4] = make_uint2(0x06, 0);
		#pragma unroll
		for (int i = 5; i < 25; i++)
			st[i] = make_uint2(0, 0);
		st[16].y = 0x80000000u;
		keccakf(st);
	}

	return ((uint64_t) st[3].y << 32) | st[3].x;
}

/* resNonce[0] counts candidates, resNonce[1..MAX_RESULTS-1] hold them */
__global__ __launch_bounds__(TPB)
void sha3t_gpu_hash(const uint32_t threads, const uint32_t startNonce, uint32_t *resNonce, const uint64_t highTarget)
{
	const uint32_t thread = blockDim.x * blockIdx.x + threadIdx.x;
	if (thread >= threads)
		return;

	const uint32_t nonce = startNonce + thread;
	if (sha3t_top(nonce) <= highTarget) {
		const uint32_t pos = atomicInc(&resNonce[0], UINT32_MAX) + 1;
		if (pos < MAX_RESULTS)
			resNonce[pos] = nonce;
	}
}

/* the 80 header bytes */
extern "C" void sha3t256_hash(void *state, const void *input)
{
	uint8_t a[32], b[32];
	sha3(input, 80, a, 32);
	sha3(a, 32, b, 32);
	sha3(b, 32, state, 32);
}

static void sha3t_setBlock(const uint32_t *pdata)
{
	uint32_t hdr[20];
	for (int i = 0; i < 20; i++)
		be32enc(&hdr[i], pdata[i]);
	hdr[19] = 0;
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_header, hdr, 80, 0, cudaMemcpyHostToDevice));
}

/* one launch over [start, start + count): the candidate count, the slots in res */
static uint32_t sha3t_scan(int thr_id, uint32_t start, uint32_t count, uint64_t highTarget, uint32_t *res)
{
	cudaMemset(d_resNonce[thr_id], 0, sizeof(uint32_t));
	sha3t_gpu_hash <<<(count + TPB - 1) / TPB, TPB>>> (count, start, d_resNonce[thr_id], highTarget);
	cudaMemcpy(res, d_resNonce[thr_id], MAX_RESULTS * sizeof(uint32_t), cudaMemcpyDeviceToHost);
	return res[0];
}

/* ---- init-time self-test, fail-closed (cuda/selftest_gate.cuh) ---- */

/* Bitcoin III blocks 64000 and 60000 (SHA3 version bit) and their hashes (the
 * block hash reversed) */
static const char *sha3t_block_hdr[2] = {
	"00100020d205b884704611704b309257c244df50499766d82ca7675bf523000000000000116c7d2c1f50f4971848439f"
	"3e36d86c81dc043dbf470ade6495598f489eed136866bb6a05db421a6538df44",
	"0010002064a6ffc22ad8c07e6573beb8ffa90aefeab3a6f5d41dd91de30f0000000000000a97cb61fd5e5f3f05686"
	"0c8134f6d749e45130f951b341811e14515e965a57a32679d6a804d6a1a837c0169"
};
static const char *sha3t_block_pow[2] = {
	"ee8eb9dfe549eb23706a801294121de989d33dbbbc6ab3627d06000000000000",
	"1499edcea0f4170f8ad68c2b0c8635ce8422c7c01da4d901712b000000000000"
};

static bool sha3t_device_selftest(int thr_id)
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
		hex2bin(hdr, sha3t_block_hdr[b], 80);
		sha3t256_hash(dig, hdr);
		for (int i = 0; i < 32; i++)
			sprintf(hex + 2 * i, "%02x", ((uint8_t *)dig)[i]);
		cpu_ok &= strcmp(hex, sha3t_block_pow[b]) == 0;

		/* the kernel finds the block nonce at a target equal to its top
		 * 64 bits, not one below */
		for (int i = 0; i < 20; i++)
			pdata[i] = be32dec(hdr + 4 * i);
		const uint64_t top = ((uint64_t) dig[7] << 32) | dig[6];
		sha3t_setBlock(pdata);
		scan_ok &= sha3t_scan(thr_id, pdata[19], 1, top, res) == 1 && res[1] == pdata[19];
		miss_ok &= sha3t_scan(thr_id, pdata[19], 1, top - 1, res) == 0;

		/* a flipped header bit must lose the hit */
		pdata[3] ^= 0x00000100u;
		sha3t_setBlock(pdata);
		neg_ok &= sha3t_scan(thr_id, pdata[19], 1, top, res) == 0;
	}

	/* 1024 nonces at a top-byte-0 target: the kernel reports exactly the nonces
	 * the CPU finds (about 4) */
	bool span_ok = true;
	const uint32_t start = 0x5a1d0c3bu;
	const uint32_t n = sha3t_scan(thr_id, start, 1024, 0x00ffffffffffffffULL, res);
	uint32_t found = 0;
	for (int i = 0; i < 20; i++)
		be32enc(hdr + 4 * i, pdata[i]);
	for (uint32_t i = 0; i < 1024; i++) {
		be32enc(hdr + 76, start + i);
		sha3t256_hash(dig, hdr);
		if (dig[7] <= 0x00ffffffu) {
			bool listed = false;
			for (uint32_t k = 1; k <= n && k < MAX_RESULTS; k++)
				listed |= res[k] == start + i;
			span_ok &= listed;
			found++;
		}
	}
	span_ok &= n == found && found < MAX_RESULTS;

	const cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		return selftest_gate(thr_id, "sha3t", selftest_cuda_error(thr_id, "sha3t", err));

	const bool passed = cpu_ok && scan_ok && miss_ok && neg_ok && span_ok;
	if (!passed)
		gpulog(LOG_ERR, thr_id, "sha3t self-test FAILED (cpu %d scan %d miss %d neg %d span %d)",
			(int)cpu_ok, (int)scan_ok, (int)miss_ok, (int)neg_ok, (int)span_ok);
	return selftest_gate(thr_id, "sha3t", passed);
}

/* ---- scan ---- */

static bool init[MAX_GPUS] = { 0 };

extern "C" int scanhash_sha3t(int thr_id, struct work *work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint8_t hdr[80];
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];

	const int dev_id = device_map[thr_id];
	// 2^21 at about 8 MH/s (GTX 660M) is 0.27 s per launch
	const uint32_t throughput = cuda_default_throughput(thr_id, device_sm[dev_id] < 500 ? 1U << 21 : 1U << 24);

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

		sha3t_device_selftest(thr_id);
		init[thr_id] = true;
	}

	for (int i = 0; i < 20; i++)
		be32enc(hdr + 4 * i, pdata[i]);
	sha3t_setBlock(pdata);

	do {
		// the last launch covers only the rest of the range
		const uint32_t batch = (uint32_t) min((uint64_t) throughput, (uint64_t) max_nonce - pdata[19]);
		if (!batch)
			break;

		uint32_t *res = h_resNonce[thr_id];
		uint32_t count = sha3t_scan(thr_id, pdata[19], batch, highTarget, res);
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
				sha3t256_hash(vhash, hdr);
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
					sha3t256_hash(vhash, hdr);
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

extern "C" void free_sha3t(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaDeviceSynchronize();
	cudaFreeHost(h_resNonce[thr_id]);
	cudaFree(d_resNonce[thr_id]);
	init[thr_id] = false;
	cudaDeviceSynchronize();
}
