/**
 * Lyra2z330 (Gravity): Lyra2 with the 80-byte header as password and salt, time cost 2,
 * 330 rows of 256 blocks (7.7 MiB per hash), no prehash
 */
extern "C" {
#include "algos/lyra2/Lyra2Z.h"
}
#include "miner.h"
#include "cuda_helper.h"
#include "algos/lyra2/cuda_lyra2z330.h"
#include "lyra2z330.h"

// launches cover whole warps: the matrix kernels shuffle across 4 or 16 lanes of a hash
#define LYRA2Z330_PAD 32
#define LYRA2Z330_MAX_CAND 8

static uint32_t *d_resNonce[MAX_GPUS];
static uint32_t *h_resNonce[MAX_GPUS];
static uint32_t s_throughput[MAX_GPUS];

extern "C" void lyra2z330_hash(void *state, const void *input)
{
	uint32_t _ALIGN(64) hash[8];

	LYRA2Z(hash, 32, input, 80, input, 80, LYRA2Z330_TIMECOST, LYRA2Z330_ROWS, LYRA2Z330_COLS);
	memcpy(state, hash, 32);
}

void lyra2z330_setTarget(int thr_id, uint32_t *pdata, const uint32_t *ptarget)
{
	uint32_t _ALIGN(64) endiandata[20];
	for (int k = 0; k < 20; k++)
		be32enc(&endiandata[k], pdata[k]);
	lyra2z330_setBlock(endiandata, ((const uint64_t*) ptarget)[3]);
}

uint32_t lyra2z330_scan(int thr_id, uint32_t threads, uint32_t startNonce,
	uint32_t *out, uint32_t max_out, uint32_t *total)
{
	const uint32_t n = (threads + LYRA2Z330_PAD - 1) / LYRA2Z330_PAD * LYRA2Z330_PAD;
	uint32_t *res = h_resNonce[thr_id];

	lyra2z330_cpu_hash(thr_id, n, threads, startNonce, d_resNonce[thr_id]);

	cudaMemcpy(res, d_resNonce[thr_id], LYRA2Z330_MAX_RESULTS * sizeof(uint32_t), cudaMemcpyDeviceToHost);
	uint32_t found = res[0];
	if (found > LYRA2Z330_MAX_RESULTS - 1)
		found = LYRA2Z330_MAX_RESULTS - 1;

	// ascending; the kernel reports only nonces of the batch, not the padding
	uint32_t count = 0;
	for (uint32_t i = 1; i <= found; i++) {
		const uint32_t nonce = res[i];
		if (count < max_out) {
			uint32_t j = count++;
			while (j > 0 && out[j - 1] > nonce) { out[j] = out[j - 1]; j--; }
			out[j] = nonce;
		}
	}
	// a flood keeps only the first slots; report it as more than were kept
	if (total)
		*total = res[0];
	return count;
}

static bool init[MAX_GPUS] = { 0 };

extern "C" int scanhash_lyra2z330(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t _ALIGN(64) endiandata[20];
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];

	if (opt_benchmark) {
		// both words the kernel compares
		ptarget[7] = 0x00ff;
		ptarget[6] = 0xffffffff;
	}

	const int dev_id = device_map[thr_id];

	if (!init[thr_id])
	{
		cudaSetDevice(dev_id);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			// reduce cpu usage
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
			CUDA_LOG_ERROR();
		}
		cuda_get_arch(thr_id);

		// threads per CUDA core: 3 on sm_50+, 4 on Kepler (192 cores per SM)
		cudaDeviceProp props;
		cudaGetDeviceProperties(&props, dev_id);
		const bool sm5 = cuda_arch[dev_id] >= 500;
		const uint32_t cores = props.multiProcessorCount * (props.major < 5 ? 192 : props.major == 6 && props.minor == 0 ? 64 : 128);
		uint32_t throughput = cuda_default_throughput(thr_id, cores * (sm5 ? 3 : 4));

		// matrix of one sub-launch (throughput / lanes hashes) in 3/4 of the free memory
		const uint32_t lanes = sm5 ? 16 : 4;
		size_t mfree = 0, mtotal = 0;
		cudaMemGetInfo(&mfree, &mtotal);
		const uint64_t fit = (uint64_t) (mfree / 4 * 3) / (LYRA2Z330_MATRIX_BYTES / lanes + 128);
		if (throughput > fit)
			throughput = (uint32_t) fit;
		throughput = throughput / LYRA2Z330_PAD * LYRA2Z330_PAD;
		if (throughput < LYRA2Z330_PAD)
			throughput = LYRA2Z330_PAD;
		// the scan syncs after each matrix kernel, so it is no single long launch
		cuda_tdr_own_check(thr_id, true);
		api_set_throughput(thr_id, throughput);
		gpulog(LOG_INFO, thr_id, "Intensity set to %g, %u cuda threads", throughput2intensity(throughput), throughput);

		if (!lyra2z330_cpu_init(thr_id, throughput)) {
			gpulog(LOG_ERR, thr_id, "lyra2z330: %u threads need %u MiB, init failed", throughput,
				(uint32_t) (((uint64_t) throughput / lanes * LYRA2Z330_MATRIX_BYTES) >> 20));
			lyra2z330_cpu_free(thr_id);
			proper_exit(EXIT_CODE_CUDA_ERROR);
		}
		CUDA_CALL_OR_RET_X(cudaMalloc(&d_resNonce[thr_id], LYRA2Z330_MAX_RESULTS * sizeof(uint32_t)), -1);
		CUDA_CALL_OR_RET_X(cudaMallocHost(&h_resNonce[thr_id], LYRA2Z330_MAX_RESULTS * sizeof(uint32_t)), -1);
		lyra2z330_device_selftest(thr_id);

		s_throughput[thr_id] = throughput;
		init[thr_id] = true;
	}
	const uint32_t throughput = s_throughput[thr_id];

	for (int k = 0; k < 20; k++)
		be32enc(&endiandata[k], pdata[k]);
	lyra2z330_setTarget(thr_id, pdata, ptarget);

	do {
		// the last launch covers only the rest of the range
		const uint32_t batch = (uint32_t) min((uint64_t) throughput, (uint64_t) max_nonce - pdata[19]);
		if (!batch)
			break;

		uint32_t cand[LYRA2Z330_MAX_CAND], total = 0;
		const uint32_t count = lyra2z330_scan(thr_id, batch, pdata[19], cand, LYRA2Z330_MAX_CAND, &total);
		*hashes_done = pdata[19] - first_nonce + batch;

		if (total > count)
			gpulog(LOG_WARNING, thr_id, "candidates flood: %u", total);

		// CPU-check every candidate, keep the two lowest valid ones
		uint32_t valid[2];
		int nvalid = 0;
		for (uint32_t n = 0; n < count && nvalid < 2; n++) {
			uint32_t _ALIGN(64) vhash[8];
			be32enc(&endiandata[19], cand[n]);
			lyra2z330_hash(vhash, endiandata);
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
				lyra2z330_hash(vhash, endiandata);
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
extern "C" void free_lyra2z330(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaDeviceSynchronize();
	lyra2z330_cpu_free(thr_id);
	cudaFree(d_resNonce[thr_id]);
	cudaFreeHost(h_resNonce[thr_id]);
	cuda_tdr_own_check(thr_id, false);
	init[thr_id] = false;
	cudaDeviceSynchronize();
}
