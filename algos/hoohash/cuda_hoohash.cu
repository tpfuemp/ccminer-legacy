// SPDX-License-Identifier: GPL-3.0-or-later
/*
 * HoohashV110 (PEPEPOW): BLAKE3 of the 80-byte header, then a 64x64 double matrix
 * pass seeded by the header with the nonce zeroed (once per job) and BLAKE3 again.
 * Built with --fmad=false (algo.mk, ccminer.props). Every candidate is checked
 * with the node's reference code (ref/hoohash.c) on the CPU.
 */
#include <stdio.h>
#include <memory.h>

#include "miner.h"
#include "cuda_helper.h"
#include "cuda/selftest_gate.cuh"
#include "hoohash_device.cuh"
#include "ref/hoohash.h"

#define TPB		64
#define MAX_RESULTS	16

__constant__ static uint8_t c_header[80];   /* wire order, nonce bytes rewritten per thread */
__device__ static double d_mat[64][64];
__device__ static double d_matdiv[64][64];

static uint32_t *d_resNonce[MAX_GPUS];
static uint32_t *h_resNonce[MAX_GPUS];

/* the matrix seed is the header with the nonce zeroed: one thread, once per job */
__global__ void hoohash_gen_matrix()
{
	uint8_t m[80], seed[32];
	for (int i = 0; i < 80; i++)
		m[i] = c_header[i];
	m[76] = m[77] = m[78] = m[79] = 0;
	blake3_256(m, 80, seed);
	hoo_generateMatrix(seed, d_mat, d_matdiv);
}

/* digest of one nonce, in ref/hoohash.c output order (byte 31 most significant);
 * the nonce is written big-endian, as be32enc(&data[19]) */
__device__ __forceinline__ void hoohash_nonce(const uint32_t nonce, uint8_t out[32])
{
	uint8_t h[80], first[32], dig[32];
	for (int i = 0; i < 76; i++)
		h[i] = c_header[i];
	h[76] = (uint8_t)(nonce >> 24);
	h[77] = (uint8_t)(nonce >> 16);
	h[78] = (uint8_t)(nonce >> 8);
	h[79] = (uint8_t)nonce;
	blake3_256(h, 80, first);
	const uint64_t n = (uint64_t)h[76] | ((uint64_t)h[77] << 8) | ((uint64_t)h[78] << 16) | ((uint64_t)h[79] << 24);
	hoo_matmul(d_mat, d_matdiv, first, dig, n);
	for (int i = 0; i < 32; i++)
		out[i] = dig[31 - i];
}

/* resNonce[0] counts candidates, resNonce[1..MAX_RESULTS-1] hold them */
__global__ __launch_bounds__(TPB)
void hoohash_gpu_hash(const uint32_t threads, const uint32_t startNonce,
	uint32_t *resNonce, const uint64_t highTarget)
{
	const uint32_t thread = blockDim.x * blockIdx.x + threadIdx.x;
	if (thread >= threads)
		return;

	const uint32_t nonce = startNonce + thread;
	uint8_t dig[32];
	hoohash_nonce(nonce, dig);

	/* top 64 bits against target words 6..7 */
	uint64_t high = 0;
	for (int i = 31; i >= 24; i--)
		high = (high << 8) | dig[i];
	if (high <= highTarget) {
		const uint32_t pos = atomicInc(&resNonce[0], UINT32_MAX) + 1;
		if (pos < MAX_RESULTS)
			resNonce[pos] = nonce;
	}
}

__global__ void hoohash_selftest_kernel(const uint32_t nonce, uint8_t *out)
{
	hoohash_nonce(nonce, out);
}

/* the 80 header bytes in wire order */
extern "C" void hoohash_hash(void *output, const void *input)
{
	hoohashv110(input, 80, (uint8_t *)output);
}

/* header bytes from work->data (as stratum fills it) and the job's matrix */
static void hoohash_setBlock(const uint32_t *pdata, uint8_t *hdr80)
{
	for (int i = 0; i < 20; i++)
		be32enc(hdr80 + 4 * i, pdata[i]);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_header, hdr80, 80, 0, cudaMemcpyHostToDevice));
	hoohash_gen_matrix <<<1, 1>>> ();
}

/* one launch over [start, start + count), returns the candidate count */
static uint32_t hoohash_scan(int thr_id, uint32_t start, uint32_t count, uint64_t highTarget, uint32_t *res)
{
	cudaMemset(d_resNonce[thr_id], 0x00, sizeof(uint32_t));
	hoohash_gpu_hash <<<(count + TPB - 1) / TPB, TPB>>> (count, start, d_resNonce[thr_id], highTarget);
	cudaMemcpy(res, d_resNonce[thr_id], MAX_RESULTS * sizeof(uint32_t), cudaMemcpyDeviceToHost);
	return res[0];
}

/* ---- init-time self-test, fail-closed (cuda/selftest_gate.cuh) ---- */

/* PEPEPOW block 4666589 (0x4734dd): header and its PoW hash, ref/hoohash.c order */
static const char *hoo_block_hdr =
	"00400020df13f8c7243b6b226c3399f485e902347e41ba370e0c8fea75672f45"
	"0100000022da1946e7dba65352ae0f65ce77dcff5105e2f46a443e3a86bb359b"
	"b6788bf962a2416a33ce011d4d94e755";
static const char *hoo_block_pow =
	"5d84176db65eb694a9d6269f1bbc06af14c503330efd791ed7aa743e01000000";

static void to_hex(char *hex, const uint8_t *bin, int len)
{
	for (int i = 0; i < len; i++)
		sprintf(hex + 2 * i, "%02x", bin[i]);
	hex[2 * len] = 0;
}

/* the device digest of one nonce under the header of pdata */
static bool hoohash_selftest_device(int thr_id, const uint32_t *pdata, uint8_t *out32)
{
	uint8_t hdr[80];
	hoohash_setBlock(pdata, hdr);

	uint8_t *d_out = NULL;
	cudaError_t err = cudaMalloc(&d_out, 32);
	if (err != cudaSuccess)
		return selftest_cuda_error(thr_id, "hoohash", err);
	hoohash_selftest_kernel <<<1, 1>>> (pdata[19], d_out);
	err = cudaDeviceSynchronize();
	if (err == cudaSuccess)
		err = cudaMemcpy(out32, d_out, 32, cudaMemcpyDeviceToHost);
	cudaFree(d_out);
	return err == cudaSuccess ? true : selftest_cuda_error(thr_id, "hoohash", err);
}

static bool hoohash_device_selftest(int thr_id)
{
	static bool tested[MAX_GPUS] = { 0 };
	const int dev_id = device_map[thr_id];
	if (tested[dev_id]) return true;
	tested[dev_id] = true;

	uint8_t hdr[80], dig[32], dev[32], dev2[32];
	uint32_t _ALIGN(64) pdata[20];
	char hex[65];

	/* the CPU reference gives the block's hash */
	hex2bin(hdr, hoo_block_hdr, 80);
	hoohash_hash(dig, hdr);
	to_hex(hex, dig, 32);
	const bool cpu_ok = strcmp(hex, hoo_block_pow) == 0;

	/* the device digest of the block nonce equals it */
	for (int i = 0; i < 20; i++)
		pdata[i] = be32dec(hdr + 4 * i);
	const bool ran = hoohash_selftest_device(thr_id, pdata, dev);
	const bool dev_ok = ran && memcmp(dev, dig, 32) == 0;

	/* the mining kernel on that nonce: target == hash top must hit, one below must not */
	uint64_t high = 0;
	for (int i = 31; i >= 24; i--)
		high = (high << 8) | dig[i];
	uint32_t *res = h_resNonce[thr_id];
	const bool hit = hoohash_scan(thr_id, pdata[19], 1, high, res) == 1 && res[1] == pdata[19];
	const bool miss = hoohash_scan(thr_id, pdata[19], 1, high - 1, res) != 0;
	cudaError_t err = cudaGetLastError();
	const bool scan_ok = err == cudaSuccess ? hit && !miss : selftest_cuda_error(thr_id, "hoohash", err);

	/* a flipped header bit must change the device digest */
	pdata[3] ^= 0x08000000u;
	const bool ran2 = hoohash_selftest_device(thr_id, pdata, dev2);
	const bool neg_ok = ran2 && memcmp(dev, dev2, 32) != 0;

	const bool passed = cpu_ok && dev_ok && scan_ok && neg_ok;
	if (!passed)
		gpulog(LOG_ERR, thr_id, "hoohash self-test FAILED (cpu %d dev %d scan %d/%d neg %d)",
			(int)cpu_ok, (int)dev_ok, (int)hit, (int)miss, (int)neg_ok);
	return selftest_gate(thr_id, "hoohash", passed);
}

/* ---- scan ---- */

static bool init[MAX_GPUS] = { 0 };

extern "C" int scanhash_hoohash(int thr_id, struct work *work, uint32_t max_nonce,
	unsigned long *hashes_done)
{
	uint8_t hdr[80];
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];

	if (opt_benchmark) {
		// both words the kernel compares
		ptarget[7] = 0x00ff;
		ptarget[6] = 0xffffffffu;
	}
	const uint64_t targetHigh = ((uint64_t)ptarget[7] << 32) | ptarget[6];

	// FP64-bound at 30-45 kH/s per multiprocessor (GTX 660M, GTX 1080 Ti): 2^15
	// nonces per multiprocessor keep a launch near 1 s, below the 2 s display watchdog
	const int dev_id = device_map[thr_id];
	const uint32_t throughput = cuda_default_throughput(thr_id, (uint32_t) max((int) device_mpcount[dev_id], 1) << 15);

	if (!init[thr_id]) {
		cudaSetDevice(dev_id);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
			CUDA_LOG_ERROR();
		}
		gpulog(LOG_INFO, thr_id, "Intensity set to %g, %u cuda threads",
			throughput2intensity(throughput), throughput);

		cuda_get_arch(thr_id);

		CUDA_CALL_OR_RET_X(cudaMalloc(&d_resNonce[thr_id], MAX_RESULTS * sizeof(uint32_t)), -1);
		CUDA_CALL_OR_RET_X(cudaMallocHost(&h_resNonce[thr_id], MAX_RESULTS * sizeof(uint32_t)), -1);

		hoohash_device_selftest(thr_id);
		init[thr_id] = true;
	}

	hoohash_setBlock(pdata, hdr);

	do {
		// the last launch covers only the rest of the range
		const uint32_t batch = (uint32_t) min((uint64_t) throughput, (uint64_t) max_nonce - pdata[19]);
		if (!batch)
			break;

		uint32_t *res = h_resNonce[thr_id];
		uint32_t count = hoohash_scan(thr_id, pdata[19], batch, targetHigh, res);
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
				hoohash_hash(vhash, hdr);
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
					be32enc(hdr + 76, valid[i]);
					hoohash_hash(vhash, hdr);
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

extern "C" void free_hoohash(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaDeviceSynchronize();
	cudaFreeHost(h_resNonce[thr_id]);
	cudaFree(d_resNonce[thr_id]);
	init[thr_id] = false;
	cudaDeviceSynchronize();
}
