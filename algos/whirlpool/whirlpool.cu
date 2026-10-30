/**
 * Whirlpool (whirlcoin): 4 chained Whirlpool-T hashes of the 80-byte header
 */
extern "C" {
#include "sph/sph_whirlpool.h"
}
#include "miner.h"
#include "cuda_helper.h"
#include "whirlpool.h"

static uint32_t *d_resNonce[MAX_GPUS];
static uint32_t *h_resNonce[MAX_GPUS];

extern "C" void wcoinhash(void *state, const void *input)
{
	sph_whirlpool_context ctx;
	unsigned char hashA[64], hashB[64];

	sph_whirlpool1_init(&ctx);
	sph_whirlpool1(&ctx, input, 80);
	sph_whirlpool1_close(&ctx, hashA);

	sph_whirlpool1_init(&ctx);
	sph_whirlpool1(&ctx, hashA, 64);
	sph_whirlpool1_close(&ctx, hashB);

	sph_whirlpool1_init(&ctx);
	sph_whirlpool1(&ctx, hashB, 64);
	sph_whirlpool1_close(&ctx, hashA);

	sph_whirlpool1_init(&ctx);
	sph_whirlpool1(&ctx, hashA, 64);
	sph_whirlpool1_close(&ctx, hashB);

	memcpy(state, hashB, 32);
}

static bool init[MAX_GPUS] = { 0 };

extern "C" int scanhash_whirl(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t _ALIGN(64) endiandata[20];
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];

	if (opt_benchmark) {
		// both words the kernel compares
		ptarget[7] = 0x03;
		ptarget[6] = 0xffffffff;
	}
	const uint64_t target64 = ((uint64_t*)ptarget)[3];

	const int dev_id = device_map[thr_id];
	const uint32_t defcount = (device_sm[dev_id] < 500) ? (1U << 18) : (1U << 19);
	const uint32_t throughput = cuda_default_throughput(thr_id, defcount);

	if (!init[thr_id])
	{
		cudaSetDevice(dev_id);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			// reduce cpu usage
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
			CUDA_LOG_ERROR();
		}
		gpulog(LOG_INFO, thr_id, "Intensity set to %g, %u cuda threads", throughput2intensity(throughput), throughput);

		cuda_get_arch(thr_id);
		whirlpool_cpu_init();
		CUDA_CALL_OR_RET_X(cudaMalloc(&d_resNonce[thr_id], WHIRLPOOL_MAX_RESULTS * sizeof(uint32_t)), -1);
		CUDA_CALL_OR_RET_X(cudaMallocHost(&h_resNonce[thr_id], WHIRLPOOL_MAX_RESULTS * sizeof(uint32_t)), -1);
		whirlpool_device_selftest(thr_id, d_resNonce[thr_id], h_resNonce[thr_id]);

		init[thr_id] = true;
	}

	for (int k = 0; k < 20; k++)
		be32enc(&endiandata[k], pdata[k]);
	whirlpool512_setBlock_80(endiandata);

	do {
		// the last launch covers only the rest of the range
		const uint32_t batch = (uint32_t) min((uint64_t) throughput, (uint64_t) max_nonce - pdata[19]);
		if (!batch)
			break;

		uint32_t *res = h_resNonce[thr_id];
		uint32_t count = whirlpool512_scan(batch, pdata[19], target64, d_resNonce[thr_id], res);
		*hashes_done = pdata[19] - first_nonce + batch;

		if (count) {
			if (count >= WHIRLPOOL_MAX_RESULTS) {
				gpulog(LOG_WARNING, thr_id, "candidates flood: %u", count);
				count = WHIRLPOOL_MAX_RESULTS - 1;
			}
			// CPU-check every candidate, keep the two lowest valid ones
			uint32_t valid[2];
			int nvalid = 0;
			for (uint32_t n = 1; n <= count; n++) {
				uint32_t _ALIGN(64) vhash[8];
				be32enc(&endiandata[19], res[n]);
				wcoinhash(vhash, endiandata);
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
					be32enc(&endiandata[19], valid[i]);
					wcoinhash(vhash, endiandata);
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

// cleanup
extern "C" void free_whirl(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaDeviceSynchronize();
	cudaFreeHost(h_resNonce[thr_id]);
	cudaFree(d_resNonce[thr_id]);
	init[thr_id] = false;
	cudaDeviceSynchronize();
}
