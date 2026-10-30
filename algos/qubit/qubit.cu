/**
 * Qubit: ECHO-512(SIMD-512(SHAvite-512(CubeHash-512(Luffa-512(80-byte header)))))
 */
extern "C" {
#include "sph/sph_luffa.h"
#include "sph/sph_cubehash.h"
#include "sph/sph_shavite.h"
#include "sph/sph_simd.h"
#include "sph/sph_echo.h"
}
#include "miner.h"
#include "cuda_helper.h"
#include "algos/x11/cuda_x11.h"
#include "qubit.h"

#define QUBIT_TPB 128
#define QUBIT_MAX_CAND 8

static uint32_t *d_hash[MAX_GPUS];

extern "C" void qubithash(void *state, const void *input)
{
	uint8_t _ALIGN(128) hash[64];

	sph_luffa512_context ctx_luffa;
	sph_cubehash512_context ctx_cubehash;
	sph_shavite512_context ctx_shavite;
	sph_simd512_context ctx_simd;
	sph_echo512_context ctx_echo;

	sph_luffa512_init(&ctx_luffa);
	sph_luffa512(&ctx_luffa, input, 80);
	sph_luffa512_close(&ctx_luffa, (void*) hash);

	sph_cubehash512_init(&ctx_cubehash);
	sph_cubehash512(&ctx_cubehash, (const void*) hash, 64);
	sph_cubehash512_close(&ctx_cubehash, (void*) hash);

	sph_shavite512_init(&ctx_shavite);
	sph_shavite512(&ctx_shavite, (const void*) hash, 64);
	sph_shavite512_close(&ctx_shavite, (void*) hash);

	sph_simd512_init(&ctx_simd);
	sph_simd512(&ctx_simd, (const void*) hash, 64);
	sph_simd512_close(&ctx_simd, (void*) hash);

	sph_echo512_init(&ctx_echo);
	sph_echo512(&ctx_echo, (const void*) hash, 64);
	sph_echo512_close(&ctx_echo, (void*) hash);

	memcpy(state, hash, 32);
}

void qubit_setBlock(uint32_t *endiandata, const uint32_t *ptarget)
{
	qubit_luffa512_cpu_setBlock_80((void*) endiandata);
	cuda_check_cpu_setTarget(ptarget);
}

uint32_t qubit_scan(int thr_id, uint32_t threads, uint32_t startNonce,
	uint32_t *out, uint32_t max_out, uint32_t *total)
{
	qubit_luffa512_cpu_hash_80(thr_id, threads, startNonce, d_hash[thr_id], 0);
	x11_cubehash512_cpu_hash_64(thr_id, threads, startNonce, NULL, d_hash[thr_id], 0);
	x11_shavite512_cpu_hash_64(thr_id, threads, startNonce, NULL, d_hash[thr_id], 0);
	x11_simd512_cpu_hash_64(thr_id, threads, startNonce, NULL, d_hash[thr_id], 0);
	x11_echo512_cpu_hash_64(thr_id, threads, startNonce, NULL, d_hash[thr_id], 0);
	return cuda_check_hash_suppl_all(thr_id, threads, startNonce, d_hash[thr_id], out, max_out, total);
}

static bool init[MAX_GPUS] = { 0 };

extern "C" int scanhash_qubit(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
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
	const uint32_t throughput = cuda_default_throughput(thr_id, 1U << 19);

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
		qubit_luffa512_cpu_init(thr_id, throughput);
		x11_cubehash512_cpu_init(thr_id, throughput);
		x11_shavite512_cpu_init(thr_id, throughput);
		if (x11_simd512_cpu_init(thr_id, throughput) != 0)
			return -1;
		x11_echo512_cpu_init(thr_id, throughput);
		cuda_check_cpu_init(thr_id, throughput);

		// SIMD runs its 8-lane groups up to the next block boundary
		const size_t states = (throughput + QUBIT_TPB - 1) / QUBIT_TPB * QUBIT_TPB;
		CUDA_CALL_OR_RET_X(cudaMalloc(&d_hash[thr_id], (size_t) 64 * states), -1);
		qubit_device_selftest(thr_id);

		init[thr_id] = true;
	}

	for (int k = 0; k < 20; k++)
		be32enc(&endiandata[k], pdata[k]);
	qubit_setBlock(endiandata, ptarget);

	do {
		// the last launch covers only the rest of the range
		const uint32_t batch = (uint32_t) min((uint64_t) throughput, (uint64_t) max_nonce - pdata[19]);
		if (!batch)
			break;

		uint32_t cand[QUBIT_MAX_CAND], total = 0;
		const uint32_t count = qubit_scan(thr_id, batch, pdata[19], cand, QUBIT_MAX_CAND, &total);
		*hashes_done = pdata[19] - first_nonce + batch;

		if (total > count)
			gpulog(LOG_WARNING, thr_id, "candidates flood: %u", total);

		// CPU-check every candidate, keep the two lowest valid ones
		uint32_t valid[2];
		int nvalid = 0;
		for (uint32_t n = 0; n < count && nvalid < 2; n++) {
			uint32_t _ALIGN(64) vhash[8];
			be32enc(&endiandata[19], cand[n]);
			qubithash(vhash, endiandata);
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
				qubithash(vhash, endiandata);
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
extern "C" void free_qubit(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaDeviceSynchronize();
	cudaFree(d_hash[thr_id]);
	x11_simd512_cpu_free(thr_id);
	cuda_check_cpu_free(thr_id);
	init[thr_id] = false;
	cudaDeviceSynchronize();
}
