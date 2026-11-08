/**
 * skein2 (Woodcoin): Skein-512 of the 80-byte header, Skein-512 of that, first 256 bits
 */
extern "C" {
#include "sph/sph_skein.h"
}
#include "miner.h"
#include "cuda_helper.h"
#include "algos/x11/cuda_x11.h"
#include "skein2.h"

#define SKEIN2_MAX_CAND 8

static uint32_t *d_hash[MAX_GPUS];

extern "C" void skein2hash(void *output, const void *input)
{
	uint32_t _ALIGN(64) hash[16];
	sph_skein512_context ctx_skein;

	sph_skein512_init(&ctx_skein);
	sph_skein512(&ctx_skein, input, 80);
	sph_skein512_close(&ctx_skein, hash);

	sph_skein512_init(&ctx_skein);
	sph_skein512(&ctx_skein, hash, 64);
	sph_skein512_close(&ctx_skein, hash);

	memcpy(output, hash, 32);
}

void skein2_setBlock(int thr_id, uint32_t *endiandata, const uint32_t *ptarget)
{
	skein512_cpu_setBlock_80(endiandata);
	cuda_check_cpu_setTarget(ptarget);
}

uint32_t skein2_scan(int thr_id, uint32_t threads, uint32_t startNonce,
	uint32_t *out, uint32_t max_out, uint32_t *total)
{
	uint32_t *h = d_hash[thr_id];
	skein512_cpu_hash_80(thr_id, threads, startNonce, h, 1);
	quark_skein512_cpu_hash_64(thr_id, threads, startNonce, NULL, h, 0);
	return cuda_check_hash_suppl_all(thr_id, threads, startNonce, h, out, max_out, total);
}

static bool init[MAX_GPUS] = { 0 };

extern "C" int scanhash_skein2(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t _ALIGN(64) endiandata[20];
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];

	if (opt_benchmark) {
		ptarget[7] = 0x03;
		ptarget[6] = 0xffffffff;
	}

	const int dev_id = device_map[thr_id];
	const int intensity = (device_sm[dev_id] < 500) ? 20 : (!is_windows() ? 22 : 21);
	const uint32_t throughput = cuda_default_throughput(thr_id, 1U << intensity);

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
		quark_skein512_cpu_init(thr_id, throughput);
		cuda_check_cpu_init(thr_id, throughput);
		CUDA_CALL_OR_RET_X(cudaMalloc(&d_hash[thr_id], (size_t) 64 * throughput), -1);
		skein2_device_selftest(thr_id);

		init[thr_id] = true;
	}

	for (int k = 0; k < 20; k++)
		be32enc(&endiandata[k], pdata[k]);
	skein2_setBlock(thr_id, endiandata, ptarget);

	do {
		// the last launch covers only the rest of the range
		const uint32_t batch = (uint32_t) min((uint64_t) throughput, (uint64_t) max_nonce - pdata[19]);
		if (!batch)
			break;

		uint32_t cand[SKEIN2_MAX_CAND], total = 0;
		const uint32_t count = skein2_scan(thr_id, batch, pdata[19], cand, SKEIN2_MAX_CAND, &total);
		*hashes_done = pdata[19] - first_nonce + batch;

		if (total > count)
			gpulog(LOG_WARNING, thr_id, "candidates flood: %u", total);

		// CPU-check every candidate, keep the two lowest valid ones
		uint32_t valid[2];
		int nvalid = 0;
		for (uint32_t n = 0; n < count && nvalid < 2; n++) {
			uint32_t _ALIGN(64) vhash[8];
			be32enc(&endiandata[19], cand[n]);
			skein2hash(vhash, endiandata);
			if (!fulltest(vhash, ptarget)) {
				gpu_increment_reject(thr_id);
				if (!opt_quiet)
					gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU!", cand[n]);
				continue;
			}
			valid[nvalid++] = cand[n];
		}

		if (nvalid) {
			uint32_t _ALIGN(64) vhash[8];
			work->valid_nonces = nvalid;
			for (int i = 0; i < nvalid; i++) {
				work->nonces[i] = valid[i];
				be32enc(&endiandata[19], valid[i]);
				skein2hash(vhash, endiandata);
				if (i == 0)
					work_set_target_ratio(work, vhash);
				else
					bn_set_target_ratio(work, vhash, 1);
			}
			// resume after the last reported nonce
			pdata[19] = valid[nvalid - 1] + 1;
			return nvalid;
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
extern "C" void free_skein2(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaDeviceSynchronize();
	cudaFree(d_hash[thr_id]);
	cuda_check_cpu_free(thr_id);
	init[thr_id] = false;
	cudaDeviceSynchronize();
}
