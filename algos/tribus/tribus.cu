/**
 * Tribus (Denarius): ECHO-512(Keccak-512(JH-512(80-byte header)))
 */
extern "C" {
#include "sph/sph_jh.h"
#include "sph/sph_keccak.h"
#include "sph/sph_echo.h"
}
#include "miner.h"
#include "cuda_helper.h"
#include "algos/x11/cuda_x11.h"
#include "tribus.h"

static uint32_t *d_hash[MAX_GPUS];
static uint32_t *d_resNonce[MAX_GPUS];
static uint32_t *h_resNonce[MAX_GPUS];
// sm < 50: x11 ECHO kernel + shared target check, else the fused final ECHO kernel
static bool use_compat_kernels[MAX_GPUS] = { 0 };

extern "C" void tribus_hash(void *state, const void *input)
{
	uint8_t _ALIGN(64) hash[64];
	sph_jh512_context ctx_jh;
	sph_keccak512_context ctx_keccak;
	sph_echo512_context ctx_echo;

	sph_jh512_init(&ctx_jh);
	sph_jh512(&ctx_jh, input, 80);
	sph_jh512_close(&ctx_jh, (void*) hash);

	sph_keccak512_init(&ctx_keccak);
	sph_keccak512(&ctx_keccak, (const void*) hash, 64);
	sph_keccak512_close(&ctx_keccak, (void*) hash);

	sph_echo512_init(&ctx_echo);
	sph_echo512(&ctx_echo, (const void*) hash, 64);
	sph_echo512_close(&ctx_echo, (void*) hash);

	memcpy(state, hash, 32);
}

void tribus_setBlock(int thr_id, uint32_t *endiandata, const uint32_t *ptarget)
{
	jh512_setBlock_80(thr_id, endiandata);
	if (use_compat_kernels[thr_id])
		cuda_check_cpu_setTarget(ptarget);
}

uint32_t tribus_scan(int thr_id, uint32_t threads, uint32_t startNonce, uint64_t target,
	uint32_t *out, uint32_t max_out, uint32_t *total)
{
	jh512_cuda_hash_80(thr_id, threads, startNonce, d_hash[thr_id]);
	quark_keccak512_cpu_hash_64(thr_id, threads, startNonce, NULL, d_hash[thr_id], 0);

	if (use_compat_kernels[thr_id]) {
		x11_echo512_cpu_hash_64(thr_id, threads, startNonce, NULL, d_hash[thr_id], 0);
		return cuda_check_hash_suppl_all(thr_id, threads, startNonce, d_hash[thr_id], out, max_out, total);
	}

	uint32_t *res = h_resNonce[thr_id];
	cudaMemset(d_resNonce[thr_id], 0, sizeof(uint32_t));
	tribus_echo512_final(thr_id, threads, startNonce, d_hash[thr_id], d_resNonce[thr_id], target);
	cudaMemcpy(res, d_resNonce[thr_id], TRIBUS_MAX_RESULTS * sizeof(uint32_t), cudaMemcpyDeviceToHost);

	*total = res[0];
	uint32_t stored = min(res[0], (uint32_t) TRIBUS_MAX_RESULTS - 1);
	if (stored > max_out)
		stored = max_out;
	for (uint32_t i = 0; i < stored; i++) {
		// ascending, as the shared check returns them
		uint32_t v = res[i + 1], j = i;
		while (j > 0 && out[j - 1] > v) { out[j] = out[j - 1]; j--; }
		out[j] = v;
	}
	return stored;
}

static bool init[MAX_GPUS] = { 0 };

extern "C" int scanhash_tribus(int thr_id, struct work *work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t _ALIGN(64) endiandata[20];
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];

	if (opt_benchmark) {
		// both words the final kernel compares
		ptarget[7] = 0x03;
		ptarget[6] = 0xffffffff;
	}
	const uint64_t target64 = ((uint64_t*)ptarget)[3];

	const int dev_id = device_map[thr_id];
	const int intensity = (device_sm[dev_id] < 500 || is_windows()) ? 20 : 23;
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
		use_compat_kernels[thr_id] = (cuda_arch[dev_id] < 500);

		quark_jh512_cpu_init(thr_id, throughput);
		quark_keccak512_cpu_init(thr_id, throughput);
		if (use_compat_kernels[thr_id]) {
			x11_echo512_cpu_init(thr_id, throughput);
			cuda_check_cpu_init(thr_id, throughput);
		}
		CUDA_CALL_OR_RET_X(cudaMalloc(&d_hash[thr_id], (size_t) 64 * throughput), -1);
		CUDA_CALL_OR_RET_X(cudaMalloc(&d_resNonce[thr_id], TRIBUS_MAX_RESULTS * sizeof(uint32_t)), -1);
		CUDA_CALL_OR_RET_X(cudaMallocHost(&h_resNonce[thr_id], TRIBUS_MAX_RESULTS * sizeof(uint32_t)), -1);
		tribus_device_selftest(thr_id);

		init[thr_id] = true;
	}

	for (int k = 0; k < 20; k++)
		be32enc(&endiandata[k], pdata[k]);
	tribus_setBlock(thr_id, endiandata, ptarget);

	do {
		// the last launch covers only the rest of the range
		const uint32_t batch = (uint32_t) min((uint64_t) throughput, (uint64_t) max_nonce - pdata[19]);
		if (!batch)
			break;

		uint32_t cand[TRIBUS_MAX_RESULTS], total = 0;
		const uint32_t count = tribus_scan(thr_id, batch, pdata[19], target64, cand, TRIBUS_MAX_RESULTS, &total);
		*hashes_done = pdata[19] - first_nonce + batch;

		if (total > count)
			gpulog(LOG_WARNING, thr_id, "candidates flood: %u", total);

		// CPU-check every candidate, keep the two lowest valid ones
		uint32_t valid[2];
		int nvalid = 0;
		for (uint32_t n = 0; n < count && nvalid < 2; n++) {
			uint32_t _ALIGN(64) vhash[8];
			be32enc(&endiandata[19], cand[n]);
			tribus_hash(vhash, endiandata);
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
				tribus_hash(vhash, endiandata);
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
extern "C" void free_tribus(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaDeviceSynchronize();
	cudaFree(d_hash[thr_id]);
	cudaFree(d_resNonce[thr_id]);
	cudaFreeHost(h_resNonce[thr_id]);
	if (use_compat_kernels[thr_id])
		cuda_check_cpu_free(thr_id);
	init[thr_id] = false;
	cudaDeviceSynchronize();
}
