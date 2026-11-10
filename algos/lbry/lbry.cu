/**
 * LBRY: SHA-256d of the 112-byte header, SHA-512, RIPEMD-160 of both 32-byte halves,
 * SHA-256d of the two digests. tpruvot and Provos Alexis, 2016-2017.
 */
#include <string.h>

extern "C" {
#include "sph/sph_sha2.h"
#include "sph/sph_ripemd.h"
}
#include "miner.h"
#include "cuda_helper.h"
#include "lbry.h"

extern "C" void lbry_hash(void *output, const void *input)
{
	uint32_t _ALIGN(64) hashA[16];
	uint32_t _ALIGN(64) hashB[8];
	uint32_t _ALIGN(64) hashC[8];

	sph_sha256_context ctx_sha256;
	sph_sha512_context ctx_sha512;
	sph_ripemd160_context ctx_ripemd;

	sph_sha256_init(&ctx_sha256);
	sph_sha256(&ctx_sha256, input, 112);
	sph_sha256_close(&ctx_sha256, hashA);
	sph_sha256(&ctx_sha256, hashA, 32);
	sph_sha256_close(&ctx_sha256, hashA);

	sph_sha512_init(&ctx_sha512);
	sph_sha512(&ctx_sha512, hashA, 32);
	sph_sha512_close(&ctx_sha512, hashA);

	sph_ripemd160_init(&ctx_ripemd);
	sph_ripemd160(&ctx_ripemd, hashA, 32);
	sph_ripemd160_close(&ctx_ripemd, hashB);
	sph_ripemd160(&ctx_ripemd, &hashA[8], 32);
	sph_ripemd160_close(&ctx_ripemd, hashC);

	sph_sha256(&ctx_sha256, hashB, 20);
	sph_sha256(&ctx_sha256, hashC, 20);
	sph_sha256_close(&ctx_sha256, hashA);
	sph_sha256(&ctx_sha256, hashA, 32);
	sph_sha256_close(&ctx_sha256, hashA);

	memcpy(output, hashA, 32);
}

void lbry_setBlock(bool merged, uint32_t *endiandata)
{
	if (merged)
		lbry_sha256_setBlock_112_merged(endiandata);
	else
		lbry_sha256_setBlock_112(endiandata);
}

uint32_t lbry_scan(bool merged, uint32_t threads, uint32_t startNonce, uint64_t target,
	uint32_t *d_hash, uint32_t *d_res, uint32_t *h_res)
{
	cudaMemset(d_res, 0, sizeof(uint32_t));
	if (merged) {
		lbry_merged(startNonce, threads, d_res, target);
	} else {
		lbry_sha256d_hash_112(threads, startNonce, d_hash);
		lbry_sha512_hash_32(threads, d_hash);
		lbry_sha256d_hash_final(threads, startNonce, d_hash, d_res, target);
	}
	cudaMemcpy(h_res, d_res, LBRY_MAX_RESULTS * sizeof(uint32_t), cudaMemcpyDeviceToHost);
	return h_res[0];
}

static bool init[MAX_GPUS] = { 0 };
static bool s_merged[MAX_GPUS];
static uint32_t *d_hash[MAX_GPUS];
static uint32_t *d_resNonce[MAX_GPUS];
static uint32_t *h_resNonce[MAX_GPUS];

extern "C" int scanhash_lbry(int thr_id, struct work *work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t _ALIGN(64) endiandata[28];
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[LBRY_NONCE_WORD];

	if (opt_benchmark) {
		// both words the kernels compare
		ptarget[7] = 0x03;
		ptarget[6] = 0xffffffff;
	}
	const uint64_t target64 = ((uint64_t*)ptarget)[3];

	const int dev_id = device_map[thr_id];
	int intensity = (device_sm[dev_id] > 500 && !is_windows()) ? 22 : 20;
	if (device_sm[dev_id] >= 600) intensity = 23;
	if (device_sm[dev_id] < 350) intensity = 18;
	const uint32_t throughput = cuda_default_throughput(thr_id, 1U << intensity);

	if (!init[thr_id])
	{
		cudaSetDevice(dev_id);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			// reduce cpu usage
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
			cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
			CUDA_LOG_ERROR();
		}
		gpulog(LOG_INFO, thr_id, "Intensity set to %g, %u cuda threads", throughput2intensity(throughput), throughput);

		cuda_get_arch(thr_id);
		s_merged[thr_id] = device_sm[dev_id] > 500;
		d_hash[thr_id] = NULL;
		if (!s_merged[thr_id])
			CUDA_CALL_OR_RET_X(cudaMalloc(&d_hash[thr_id], (size_t) 64 * throughput), -1);
		CUDA_CALL_OR_RET_X(cudaMalloc(&d_resNonce[thr_id], LBRY_MAX_RESULTS * sizeof(uint32_t)), -1);
		CUDA_CALL_OR_RET_X(cudaMallocHost(&h_resNonce[thr_id], LBRY_MAX_RESULTS * sizeof(uint32_t)), -1);
		lbry_device_selftest(thr_id, d_resNonce[thr_id], h_resNonce[thr_id]);

		init[thr_id] = true;
	}

	for (int k = 0; k < LBRY_NONCE_WORD; k++)
		be32enc(&endiandata[k], pdata[k]);
	lbry_setBlock(s_merged[thr_id], endiandata);

	do {
		// the last launch covers only the rest of the range
		const uint32_t batch = (uint32_t) min((uint64_t) throughput, (uint64_t) max_nonce - pdata[LBRY_NONCE_WORD]);
		if (!batch)
			break;

		uint32_t *res = h_resNonce[thr_id];
		uint32_t count = lbry_scan(s_merged[thr_id], batch, pdata[LBRY_NONCE_WORD], target64,
			d_hash[thr_id], d_resNonce[thr_id], res);
		*hashes_done = pdata[LBRY_NONCE_WORD] - first_nonce + batch;

		if (count) {
			if (count >= LBRY_MAX_RESULTS) {
				gpulog(LOG_WARNING, thr_id, "candidates flood: %u", count);
				count = LBRY_MAX_RESULTS - 1;
			}
			// CPU-check every candidate, keep the two lowest valid ones
			uint32_t valid[2];
			int nvalid = 0;
			for (uint32_t n = 1; n <= count; n++) {
				uint32_t _ALIGN(64) vhash[8];
				be32enc(&endiandata[LBRY_NONCE_WORD], res[n]);
				lbry_hash(vhash, endiandata);
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
					be32enc(&endiandata[LBRY_NONCE_WORD], valid[i]);
					lbry_hash(vhash, endiandata);
					if (i == 0)
						work_set_target_ratio(work, vhash);
					else
						bn_set_target_ratio(work, vhash, 1);
				}
				// resume after the last reported nonce
				pdata[LBRY_NONCE_WORD] = valid[nvalid - 1] + 1;
				return nvalid;
			}
		}

		if ((uint64_t) batch + pdata[LBRY_NONCE_WORD] >= max_nonce) {
			pdata[LBRY_NONCE_WORD] = max_nonce;
			break;
		}

		pdata[LBRY_NONCE_WORD] += batch;

	} while (!work_restart[thr_id].restart);

	*hashes_done = pdata[LBRY_NONCE_WORD] - first_nonce;
	return 0;
}

// cleanup
extern "C" void free_lbry(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaDeviceSynchronize();
	cudaFreeHost(h_resNonce[thr_id]);
	cudaFree(d_resNonce[thr_id]);
	if (d_hash[thr_id])
		cudaFree(d_hash[thr_id]);
	init[thr_id] = false;
	cudaDeviceSynchronize();
}
