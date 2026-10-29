// SPDX-License-Identifier: GPL-3.0-or-later
/*
 * Decred: BLAKE3-256 of the 180-byte header (DCP-0011). The host compresses
 * header blocks 1-2 once per job, the kernel only the final block.
 */
#include <stdio.h>
#include <memory.h>

#include "miner.h"
#include "cuda_helper.h"
#include "cuda/blake3_device.cuh"
#include "cuda/selftest_gate.cuh"

#define TPB		256
#define MAX_RESULTS	16

/* nonce word in work->data (byte 140) */
#define DCR_NONCE_OFT32	35

__constant__ static uint32_t c_cv[8];	/* midstate of header bytes 0..127 */
__constant__ static uint32_t c_m[16];	/* final block words */

static uint32_t *d_resNonce[MAX_GPUS];
static uint32_t *h_resNonce[MAX_GPUS];

/* resNonce[0] counts candidates, resNonce[1..MAX_RESULTS-1] hold them */
__global__ __launch_bounds__(TPB, 1)
void decred_blake3_gpu_hash(const uint32_t threads, const uint32_t startNonce,
	uint32_t *resNonce, const uint64_t highTarget)
{
	const uint32_t thread = blockDim.x * blockIdx.x + threadIdx.x;
	if (thread >= threads)
		return;

	const uint32_t nonce = startNonce + thread;

	uint32_t cv[8], m[16];
	#pragma unroll
	for (int i = 0; i < 8; i++)
		cv[i] = c_cv[i];
	#pragma unroll
	for (int i = 0; i < 16; i++)
		m[i] = c_m[i];

	uint8_t dig[32];
	decred_blake3_nonce(cv, m, nonce, dig);

	/* top 64 bits against target words 6..7 */
	const uint64_t high = ((const uint64_t *)dig)[3];
	if (high <= highTarget) {
		const uint32_t pos = atomicInc(&resNonce[0], UINT32_MAX) + 1;
		if (pos < MAX_RESULTS)
			resNonce[pos] = nonce;
	}
}

/* work->data holds the header bytes in wire order */
extern "C" void decred_hash(void *output, const void *input)
{
	blake3_256((const uint8_t *)input, DECRED_HDR_LEN, (uint8_t *)output);
}

__host__
static void decred_blake3_cpu_setBlock(const uint32_t *pdata)
{
	uint32_t cv[8], m[16];
	decred_blake3_prepare((const uint8_t *)pdata, cv, m);

	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_cv, cv, sizeof(cv), 0, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_m, m, sizeof(m), 0, cudaMemcpyHostToDevice));
}

/* one launch over [start, start + count), returns the candidate count */
__host__
static uint32_t decred_blake3_scan(int thr_id, uint32_t start, uint32_t count,
	uint64_t highTarget, uint32_t *res)
{
	const dim3 grid((count + TPB - 1) / TPB);
	const dim3 block(TPB);

	cudaMemset(d_resNonce[thr_id], 0x00, sizeof(uint32_t));
	decred_blake3_gpu_hash <<<grid, block>>> (count, start, d_resNonce[thr_id], highTarget);
	cudaMemcpy(res, d_resNonce[thr_id], MAX_RESULTS * sizeof(uint32_t), cudaMemcpyDeviceToHost);
	return res[0];
}

/* init-time self-test, fail-closed (cuda/selftest_gate.cuh) */

__global__ void decred_blake3_selftest_kernel(const uint32_t nonce, uint8_t *out)
{
	uint32_t cv[8], m[16];
	for (int i = 0; i < 8; i++)  cv[i] = c_cv[i];
	for (int i = 0; i < 16; i++) m[i] = c_m[i];
	decred_blake3_nonce(cv, m, nonce, out);
}

static bool decred_selftest_device(const uint32_t *pdata, uint32_t nonce, uint8_t *out32)
{
	decred_blake3_cpu_setBlock(pdata);

	uint8_t *d_out = NULL;
	if (cudaMalloc(&d_out, 32) != cudaSuccess)
		return selftest_cuda_fault();

	decred_blake3_selftest_kernel <<< 1, 1 >>> (nonce, d_out);

	const bool ok = cudaDeviceSynchronize() == cudaSuccess
	             && cudaMemcpy(out32, d_out, 32, cudaMemcpyDeviceToHost) == cudaSuccess;
	cudaFree(d_out);

	return ok ? true : selftest_cuda_fault();
}

static void to_hex(char *hex, const uint8_t *bin, int len)
{
	for (int i = 0; i < len; i++)
		sprintf(hex + 2 * i, "%02x", bin[i]);
	hex[2 * len] = 0;
}

/* mainnet block 794368 (first BLAKE3 block) and its PoW hash */
static const char *dcr_block_hdr =
	"0a000000f04e765c994d860e77afcd25ea4704965ed00974c6d893c200000000"
	"00000000609e379d61a43226fc875bf232c5406085109932a13e3d226e1f6d03"
	"62772abe5ff6678090180b3534830a6daae8b02ccec81c628fca089907720ca1"
	"5363838f01009f1e7b7d200705000b007d9f0000a6a5001b097f8c8205000000"
	"001f0c00cf7e00008af1ed64bd193281b6233e65040b01a00000000000000000"
	"000000000000000000000000000000000a000000";
static const char *dcr_block_pow =
	"ccd1e73bbe0fd433f2ce6f5614dd44851b1d034c4fc98af94683000000000000";

/* mining kernel on one nonce: target == hash top must hit, one below must not */
static bool decred_selftest_scan(int thr_id, const uint32_t *pdata, uint64_t high,
	bool *hit, bool *miss)
{
	uint32_t *res = h_resNonce[thr_id];
	const uint32_t nonce = pdata[DCR_NONCE_OFT32];

	decred_blake3_cpu_setBlock(pdata);
	*hit = decred_blake3_scan(thr_id, nonce, 1, high, res) == 1 && res[1] == nonce;
	*miss = decred_blake3_scan(thr_id, nonce, 1, high - 1, res) != 0;
	if (cudaGetLastError() != cudaSuccess)
		return selftest_cuda_fault();
	return true;
}

__host__
static bool decred_blake3_device_selftest(int thr_id)
{
	static bool tested[MAX_GPUS] = { 0 };
	const int dev_id = device_map[thr_id];
	if (tested[dev_id]) return true;
	tested[dev_id] = true;

	/* published BLAKE3("") vector */
	static const char *kat_empty =
		"af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262";
	uint8_t dig[32];
	char hex[65];
	blake3_256((const uint8_t *)"", 0, dig);
	to_hex(hex, dig, 32);
	const bool kat_ok = strcmp(hex, kat_empty) == 0;

	uint32_t _ALIGN(64) pdata[48];
	hex2bin((uchar *)pdata, dcr_block_hdr, DECRED_HDR_LEN);
	decred_hash(dig, pdata);
	to_hex(hex, dig, 32);
	const bool block_ok = strcmp(hex, dcr_block_pow) == 0;

	bool hit = false, miss = true;
	const bool scan_ran = decred_selftest_scan(thr_id, pdata, ((uint64_t *)dig)[3], &hit, &miss);
	const bool scan_ok = scan_ran && hit && !miss;

	/* synthetic header: full CPU hash == device midstate == host midstate */
	for (int i = 0; i < 45; i++)
		pdata[i] = 0x01020304u * (uint32_t)(i + 1);
	const uint32_t nonce = 0x7a3b1c5du;

	uint8_t full[32], dev[32];
	pdata[DCR_NONCE_OFT32] = nonce;
	decred_hash(full, pdata);

	const bool ran = decred_selftest_device(pdata, nonce, dev);
	const bool dev_ok = ran && memcmp(full, dev, 32) == 0;

	uint32_t cv[8], m[16];
	uint8_t mid[32];
	decred_blake3_prepare((const uint8_t *)pdata, cv, m);
	decred_blake3_nonce(cv, m, nonce, mid);
	const bool mid_ok = memcmp(full, mid, 32) == 0;

	/* a flipped header bit must change the device hash */
	uint8_t dev2[32];
	pdata[0] ^= 1u;
	const bool ran2 = decred_selftest_device(pdata, nonce, dev2);
	const bool neg_ok = ran2 && memcmp(dev, dev2, 32) != 0;

	const bool passed = kat_ok && block_ok && scan_ok && dev_ok && mid_ok && neg_ok;
	if (!passed)
		gpulog(LOG_ERR, thr_id, "decred self-test FAILED"
			" (kat %d block %d scan %d/%d dev %d mid %d neg %d)",
			(int)kat_ok, (int)block_ok, (int)hit, (int)miss,
			(int)dev_ok, (int)mid_ok, (int)neg_ok);

	return selftest_gate(thr_id, "decred", passed);
}

static bool init[MAX_GPUS] = { 0 };

extern "C" int scanhash_decred(int thr_id, struct work *work, uint32_t max_nonce,
	unsigned long *hashes_done)
{
	uint32_t _ALIGN(64) hdr[48];
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	uint32_t *pnonce = &pdata[DCR_NONCE_OFT32];

	const uint32_t first_nonce = *pnonce;

	if (opt_benchmark) {
		// both words the kernel compares
		ptarget[7] = 0x03;
		ptarget[6] = 0xffffffffu;
	}
	const uint64_t targetHigh = ((uint64_t *)ptarget)[3];

	const int dev_id = device_map[thr_id];
	int intensity = (device_sm[dev_id] > 500 && !is_windows()) ? 29 : 25;
	if (device_sm[dev_id] < 500) intensity = 24;

	const uint32_t throughput = cuda_default_throughput(thr_id, 1U << intensity);

	if (!init[thr_id]) {
		cudaSetDevice(dev_id);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
			cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
			CUDA_LOG_ERROR();
		}
		gpulog(LOG_INFO, thr_id, "Intensity set to %g, %u cuda threads",
			throughput2intensity(throughput), throughput);

		cuda_get_arch(thr_id);

		CUDA_CALL_OR_RET_X(cudaMalloc(&d_resNonce[thr_id],
			MAX_RESULTS * sizeof(uint32_t)), -1);
		CUDA_CALL_OR_RET_X(cudaMallocHost(&h_resNonce[thr_id],
			MAX_RESULTS * sizeof(uint32_t)), -1);

		decred_blake3_device_selftest(thr_id);
		init[thr_id] = true;
	}

	memcpy(hdr, pdata, DECRED_HDR_LEN);
	decred_blake3_cpu_setBlock(hdr);

	do {
		// the last launch covers only the rest of the range
		const uint32_t batch = (uint32_t) min((uint64_t) throughput, (uint64_t) max_nonce - *pnonce);
		if (!batch)
			break;

		uint32_t *res = h_resNonce[thr_id];
		uint32_t count = decred_blake3_scan(thr_id, *pnonce, batch, targetHigh, res);
		*hashes_done = *pnonce - first_nonce + batch;

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
				hdr[DCR_NONCE_OFT32] = res[n];
				decred_hash(vhash, hdr);
				if (!fulltest(vhash, ptarget)) {
					gpu_increment_reject(thr_id);
					if (!opt_quiet)
						gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU!", res[n]);
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
					hdr[DCR_NONCE_OFT32] = valid[i];
					decred_hash(vhash, hdr);
					if (i == 0)
						work_set_target_ratio(work, vhash);
					else
						bn_set_target_ratio(work, vhash, 1);
				}
				// resume after the last reported nonce
				*pnonce = valid[nvalid - 1] + 1;
				return nvalid;
			}
		}

		if ((uint64_t) batch + *pnonce >= max_nonce) {
			*pnonce = max_nonce;
			break;
		}

		*pnonce += batch;

	} while (!work_restart[thr_id].restart);

	*hashes_done = *pnonce - first_nonce;
	return 0;
}

extern "C" void free_decred(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaDeviceSynchronize();
	cudaFreeHost(h_resNonce[thr_id]);
	cudaFree(d_resNonce[thr_id]);
	init[thr_id] = false;
	cudaDeviceSynchronize();
}
