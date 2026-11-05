/**
 * Allium (Garlicoin): BLAKE-256 (14 rounds), Keccak-256, Lyra2, CubeHash-256,
 * Lyra2, Skein-256 and Groestl-256 over the 80-byte header
 */
extern "C" {
#include "sph/sph_blake.h"
#include "sph/sph_keccak.h"
#include "sph/sph_cubehash.h"
#include "sph/sph_skein.h"
#include "sph/sph_groestl.h"
#include "algos/lyra2/Lyra2.h"
}
#include "miner.h"
#include "cuda_helper.h"
#include "algos/lyra2/cuda_lyra2.h"
#include "allium.h"

// launches cover whole warps: Lyra2 shuffles across the 4 lanes of a hash
#define ALLIUM_PAD 32
#define ALLIUM_MAX_CAND 8

static uint64_t *d_hash[MAX_GPUS];
static uint64_t *d_matrix[MAX_GPUS];
static uint32_t *d_resNonce[MAX_GPUS];
static uint32_t *h_resNonce[MAX_GPUS];
static bool gtx750ti[MAX_GPUS];

extern "C" void allium_hash(void *state, const void *input)
{
	uint32_t hashA[8], hashB[8];

	sph_blake256_context ctx_blake;
	sph_keccak256_context ctx_keccak;
	sph_cubehash256_context ctx_cube;
	sph_skein256_context ctx_skein;
	sph_groestl256_context ctx_groestl;

	sph_blake256_set_rounds(14);

	sph_blake256_init(&ctx_blake);
	sph_blake256(&ctx_blake, input, 80);
	sph_blake256_close(&ctx_blake, hashA);

	sph_keccak256_init(&ctx_keccak);
	sph_keccak256(&ctx_keccak, hashA, 32);
	sph_keccak256_close(&ctx_keccak, hashB);

	LYRA2(hashA, 32, hashB, 32, hashB, 32, 1, 8, 8);

	sph_cubehash256_init(&ctx_cube);
	sph_cubehash256(&ctx_cube, hashA, 32);
	sph_cubehash256_close(&ctx_cube, hashB);

	LYRA2(hashA, 32, hashB, 32, hashB, 32, 1, 8, 8);

	sph_skein256_init(&ctx_skein);
	sph_skein256(&ctx_skein, hashA, 32);
	sph_skein256_close(&ctx_skein, hashB);

	sph_groestl256_init(&ctx_groestl);
	sph_groestl256(&ctx_groestl, hashB, 32);
	sph_groestl256_close(&ctx_groestl, hashA);

	memcpy(state, hashA, 32);
}

void allium_setBlock(int thr_id, uint32_t *pdata, const uint32_t *ptarget)
{
	blake256_cpu_setBlock_80(pdata);
	groestl256_setTarget(ptarget);
}

uint32_t allium_scan(int thr_id, uint32_t threads, uint32_t startNonce,
	uint32_t *out, uint32_t max_out, uint32_t *total)
{
	const uint32_t n = (threads + ALLIUM_PAD - 1) / ALLIUM_PAD * ALLIUM_PAD;
	uint64_t *h = d_hash[thr_id];
	uint32_t *res = h_resNonce[thr_id];

	blakeKeccak256_cpu_hash_80(thr_id, n, startNonce, h, 0);
	lyra2_cpu_hash_32(thr_id, n, h, gtx750ti[thr_id]);
	cubehash256_cpu_hash_32(thr_id, n, startNonce, h, 0);
	lyra2_cpu_hash_32(thr_id, n, h, gtx750ti[thr_id]);
	skein256_cpu_hash_32(thr_id, n, startNonce, h, 0);
	groestl256_cpu_hash_32(thr_id, n, threads, startNonce, h, d_resNonce[thr_id]);

	cudaMemcpy(res, d_resNonce[thr_id], GROESTL256_MAX_RESULTS * sizeof(uint32_t), cudaMemcpyDeviceToHost);
	uint32_t found = res[0];
	if (found > GROESTL256_MAX_RESULTS - 1)
		found = GROESTL256_MAX_RESULTS - 1;

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

extern "C" int scanhash_allium(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
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
	int intensity = (device_sm[dev_id] >= 500 && !is_windows()) ? 17 : 16;
	if (device_sm[dev_id] == 500)
		intensity = 15;
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

		cudaDeviceProp props;
		cudaGetDeviceProperties(&props, dev_id);
		gtx750ti[thr_id] = (strstr(props.name, "750 Ti") != NULL);

		cuda_get_arch(thr_id);
		blake256_cpu_init(thr_id, throughput);
		skein256_cpu_init(thr_id, throughput);

		const size_t states = (throughput + ALLIUM_PAD - 1) / ALLIUM_PAD * ALLIUM_PAD;
		if (device_sm[dev_id] >= 500) {
			const size_t words = device_sm[dev_id] > 500 ? 4 * 4 : 8 * 8 * 3 * 4;
			CUDA_CALL_OR_RET_X(cudaMalloc(&d_matrix[thr_id], sizeof(uint64_t) * words * states), -1);
			lyra2_cpu_init(thr_id, (uint32_t) states, d_matrix[thr_id]);
		}
		CUDA_CALL_OR_RET_X(cudaMalloc(&d_hash[thr_id], (size_t) 32 * states), -1);
		CUDA_CALL_OR_RET_X(cudaMalloc(&d_resNonce[thr_id], GROESTL256_MAX_RESULTS * sizeof(uint32_t)), -1);
		CUDA_CALL_OR_RET_X(cudaMallocHost(&h_resNonce[thr_id], GROESTL256_MAX_RESULTS * sizeof(uint32_t)), -1);
		allium_device_selftest(thr_id);

		init[thr_id] = true;
	}

	for (int k = 0; k < 20; k++)
		be32enc(&endiandata[k], pdata[k]);
	allium_setBlock(thr_id, pdata, ptarget);

	do {
		// the last launch covers only the rest of the range
		const uint32_t batch = (uint32_t) min((uint64_t) throughput, (uint64_t) max_nonce - pdata[19]);
		if (!batch)
			break;

		uint32_t cand[ALLIUM_MAX_CAND], total = 0;
		const uint32_t count = allium_scan(thr_id, batch, pdata[19], cand, ALLIUM_MAX_CAND, &total);
		*hashes_done = pdata[19] - first_nonce + batch;

		if (total > count)
			gpulog(LOG_WARNING, thr_id, "candidates flood: %u", total);

		// CPU-check every candidate, keep the two lowest valid ones
		uint32_t valid[2];
		int nvalid = 0;
		for (uint32_t n = 0; n < count && nvalid < 2; n++) {
			uint32_t _ALIGN(64) vhash[8];
			be32enc(&endiandata[19], cand[n]);
			allium_hash(vhash, endiandata);
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
				allium_hash(vhash, endiandata);
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
extern "C" void free_allium(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaDeviceSynchronize();
	cudaFree(d_hash[thr_id]);
	cudaFree(d_matrix[thr_id]);
	cudaFree(d_resNonce[thr_id]);
	cudaFreeHost(h_resNonce[thr_id]);
	d_matrix[thr_id] = NULL;
	init[thr_id] = false;
	cudaDeviceSynchronize();
}
