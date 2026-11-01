/**
 * Quark: BLAKE, BMW, Groestl, JH, Keccak and Skein, 512 bits each, over the
 * 80-byte header; three of the nine steps pick their function by bit 3 of the
 * first hash byte
 */
extern "C" {
#include "sph/sph_blake.h"
#include "sph/sph_bmw.h"
#include "sph/sph_groestl.h"
#include "sph/sph_skein.h"
#include "sph/sph_jh.h"
#include "sph/sph_keccak.h"
}
#include "miner.h"
#include "cuda_helper.h"
#include "algos/x11/cuda_x11.h"
#include "quark.h"

// Groestl runs its 4-lane groups up to the next 256-hash block
#define QUARK_PAD 256
#define QUARK_MAX_CAND 8

static uint32_t *d_hash[MAX_GPUS];
static uint32_t *d_list1[MAX_GPUS];
static uint32_t *d_list2[MAX_GPUS];

extern "C" void quarkhash(void *state, const void *input)
{
	unsigned char _ALIGN(128) hash[64];

	sph_blake512_context ctx_blake;
	sph_bmw512_context ctx_bmw;
	sph_groestl512_context ctx_groestl;
	sph_jh512_context ctx_jh;
	sph_keccak512_context ctx_keccak;
	sph_skein512_context ctx_skein;

	sph_blake512_init(&ctx_blake);
	sph_blake512(&ctx_blake, input, 80);
	sph_blake512_close(&ctx_blake, (void*) hash);

	sph_bmw512_init(&ctx_bmw);
	sph_bmw512(&ctx_bmw, (const void*) hash, 64);
	sph_bmw512_close(&ctx_bmw, (void*) hash);

	if (hash[0] & 0x8) {
		sph_groestl512_init(&ctx_groestl);
		sph_groestl512(&ctx_groestl, (const void*) hash, 64);
		sph_groestl512_close(&ctx_groestl, (void*) hash);
	} else {
		sph_skein512_init(&ctx_skein);
		sph_skein512(&ctx_skein, (const void*) hash, 64);
		sph_skein512_close(&ctx_skein, (void*) hash);
	}

	sph_groestl512_init(&ctx_groestl);
	sph_groestl512(&ctx_groestl, (const void*) hash, 64);
	sph_groestl512_close(&ctx_groestl, (void*) hash);

	sph_jh512_init(&ctx_jh);
	sph_jh512(&ctx_jh, (const void*) hash, 64);
	sph_jh512_close(&ctx_jh, (void*) hash);

	if (hash[0] & 0x8) {
		sph_blake512_init(&ctx_blake);
		sph_blake512(&ctx_blake, (const void*) hash, 64);
		sph_blake512_close(&ctx_blake, (void*) hash);
	} else {
		sph_bmw512_init(&ctx_bmw);
		sph_bmw512(&ctx_bmw, (const void*) hash, 64);
		sph_bmw512_close(&ctx_bmw, (void*) hash);
	}

	sph_keccak512_init(&ctx_keccak);
	sph_keccak512(&ctx_keccak, (const void*) hash, 64);
	sph_keccak512_close(&ctx_keccak, (void*) hash);

	sph_skein512_init(&ctx_skein);
	sph_skein512(&ctx_skein, (const void*) hash, 64);
	sph_skein512_close(&ctx_skein, (void*) hash);

	if (hash[0] & 0x8) {
		sph_keccak512_init(&ctx_keccak);
		sph_keccak512(&ctx_keccak, (const void*) hash, 64);
		sph_keccak512_close(&ctx_keccak, (void*) hash);
	} else {
		sph_jh512_init(&ctx_jh);
		sph_jh512(&ctx_jh, (const void*) hash, 64);
		sph_jh512_close(&ctx_jh, (void*) hash);
	}

	memcpy(state, hash, 32);
}

void quark_setBlock(int thr_id, uint32_t *endiandata, const uint32_t *ptarget)
{
	quark_blake512_cpu_setBlock_80(thr_id, endiandata);
	cuda_check_cpu_setTarget(ptarget);
}

/* every nonce takes its own branch, so all hashes of the batch are final and
 * the target check covers the whole batch */
uint32_t quark_scan(int thr_id, uint32_t threads, uint32_t startNonce,
	uint32_t *out, uint32_t max_out, uint32_t *total)
{
	uint32_t *h = d_hash[thr_id], *l1 = d_list1[thr_id], *l2 = d_list2[thr_id];
	uint32_t n1, n2;

	quark_blake512_cpu_hash_80(thr_id, threads, startNonce, h);
	quark_bmw512_cpu_hash_64(thr_id, threads, startNonce, NULL, h, 0);

	quark_split_cpu(thr_id, threads, startNonce, h, l1, &n1, l2, &n2);
	if (n1) quark_groestl512_cpu_hash_64(thr_id, n1, startNonce, l1, h, 0);
	if (n2) quark_skein512_cpu_hash_64(thr_id, n2, startNonce, l2, h, 0);

	quark_groestl512_cpu_hash_64(thr_id, threads, startNonce, NULL, h, 0);
	quark_jh512_cpu_hash_64(thr_id, threads, startNonce, NULL, h, 0);

	quark_split_cpu(thr_id, threads, startNonce, h, l1, &n1, l2, &n2);
	if (n1) quark_blake512_cpu_hash_64(thr_id, n1, startNonce, l1, h, 0);
	if (n2) quark_bmw512_cpu_hash_64(thr_id, n2, startNonce, l2, h, 0);

	quark_keccak512_cpu_hash_64(thr_id, threads, startNonce, NULL, h, 0);
	quark_skein512_cpu_hash_64(thr_id, threads, startNonce, NULL, h, 0);

	quark_split_cpu(thr_id, threads, startNonce, h, l1, &n1, l2, &n2);
	if (n1) quark_keccak512_cpu_hash_64(thr_id, n1, startNonce, l1, h, 0);
	if (n2) quark_jh512_cpu_hash_64(thr_id, n2, startNonce, l2, h, 0);

	return cuda_check_hash_suppl_all(thr_id, threads, startNonce, h, out, max_out, total);
}

static bool init[MAX_GPUS] = { 0 };

extern "C" int scanhash_quark(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
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
	// Kepler: smaller batches keep one launch well below the 2 s WDDM TDR
	const int intensity = (device_sm[dev_id] < 500) ? 18 : 20;
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
		quark_blake512_cpu_init(thr_id, throughput);
		quark_bmw512_cpu_init(thr_id, throughput);
		quark_groestl512_cpu_init(thr_id, throughput);
		quark_skein512_cpu_init(thr_id, throughput);
		quark_jh512_cpu_init(thr_id, throughput);
		quark_keccak512_cpu_init(thr_id, throughput);
		quark_split_cpu_init(thr_id);
		cuda_check_cpu_init(thr_id, throughput);

		const size_t states = (throughput + QUARK_PAD - 1) / QUARK_PAD * QUARK_PAD;
		CUDA_CALL_OR_RET_X(cudaMalloc(&d_hash[thr_id], (size_t) 64 * states), -1);
		CUDA_CALL_OR_RET_X(cudaMalloc(&d_list1[thr_id], sizeof(uint32_t) * throughput), -1);
		CUDA_CALL_OR_RET_X(cudaMalloc(&d_list2[thr_id], sizeof(uint32_t) * throughput), -1);
		quark_device_selftest(thr_id);

		init[thr_id] = true;
	}

	for (int k = 0; k < 20; k++)
		be32enc(&endiandata[k], pdata[k]);
	quark_setBlock(thr_id, endiandata, ptarget);

	do {
		// the last launch covers only the rest of the range
		const uint32_t batch = (uint32_t) min((uint64_t) throughput, (uint64_t) max_nonce - pdata[19]);
		if (!batch)
			break;

		uint32_t cand[QUARK_MAX_CAND], total = 0;
		const uint32_t count = quark_scan(thr_id, batch, pdata[19], cand, QUARK_MAX_CAND, &total);
		*hashes_done = pdata[19] - first_nonce + batch;

		if (total > count)
			gpulog(LOG_WARNING, thr_id, "candidates flood: %u", total);

		// CPU-check every candidate, keep the two lowest valid ones
		uint32_t valid[2];
		int nvalid = 0;
		for (uint32_t n = 0; n < count && nvalid < 2; n++) {
			uint32_t _ALIGN(64) vhash[8];
			be32enc(&endiandata[19], cand[n]);
			quarkhash(vhash, endiandata);
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
				quarkhash(vhash, endiandata);
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
extern "C" void free_quark(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaDeviceSynchronize();
	cudaFree(d_hash[thr_id]);
	cudaFree(d_list1[thr_id]);
	cudaFree(d_list2[thr_id]);
	quark_blake512_cpu_free(thr_id);
	quark_groestl512_cpu_free(thr_id);
	quark_split_cpu_free(thr_id);
	cuda_check_cpu_free(thr_id);
	init[thr_id] = false;
	cudaDeviceSynchronize();
}
