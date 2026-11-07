/**
 * yescrypt family: yescrypt (GlobalBoost-Y, or --yescrypt-param / --yescrypt-key),
 * yescryptr8, r16, r16v2, r24, r32 (fixed parameters and client keys)
 */
extern "C" {
#include "yescrypt.h"
}
#include "miner.h"
#include "cuda_helper.h"
#include "cuda_yescrypt.h"
#include "yescrypt_algo.h"

// launches cover whole warps: sMix runs 4 and 16 lanes per hash
#define YESCRYPT_PAD 32
#define YESCRYPT_MAX_CAND 8

extern char *yescrypt_key;
extern size_t yescrypt_key_len;
extern uint32_t yescrypt_param_N, yescrypt_param_r, yescrypt_param_p;

const struct yescrypt_variant yescrypt_variants[YESCRYPT_VARIANTS] = {
	{ "yescrypt",      2048,  8, 1, NULL },
	{ "yescryptr8",    2048,  8, 1, "Client Key" },
	{ "yescryptr16",   4096, 16, 1, "Client Key" },
	{ "yescryptr16v2", 4096, 16, 4, "PPTPPubKey" },
	{ "yescryptr24",   4096, 24, 1, "Jagaricoin" },
	{ "yescryptr32",   4096, 32, 1, "WaviBanana" },
};

/* the parameters a variant mines with; v 0 takes the command line over its defaults */
struct yescrypt_variant yescrypt_params(int v)
{
	struct yescrypt_variant y = yescrypt_variants[v];
	if (v == 0) {
		if (yescrypt_param_N) y.N = yescrypt_param_N;
		if (yescrypt_param_r) y.r = yescrypt_param_r;
		if (yescrypt_param_p) y.p = yescrypt_param_p;
		if (yescrypt_key) y.key = yescrypt_key;
	}
	return y;
}

bool yescrypt_is_default(int v)
{
	const struct yescrypt_variant y = yescrypt_params(v);
	const struct yescrypt_variant d = yescrypt_variants[v];
	return y.N == d.N && y.r == d.r && y.p == d.p && y.key == d.key;
}

void yescrypt_hash_params(const struct yescrypt_variant *y, void *state, const void *input)
{
	yescrypt_bsty((const uint8_t*) input, 80, (const uint8_t*) input, 80, y->N, y->r, y->p,
		y->key, y->key ? strlen(y->key) : 0, (uint8_t*) state, 32);
}

static void hash_v(int v, void *state, const void *input)
{
	const struct yescrypt_variant y = yescrypt_params(v);
	yescrypt_hash_params(&y, state, input);
}

extern "C" void yescrypt_hash(void *state, const void *input) { hash_v(0, state, input); }
extern "C" void yescryptr8_hash(void *state, const void *input) { hash_v(1, state, input); }
extern "C" void yescryptr16_hash(void *state, const void *input) { hash_v(2, state, input); }
extern "C" void yescryptr16v2_hash(void *state, const void *input) { hash_v(3, state, input); }
extern "C" void yescryptr24_hash(void *state, const void *input) { hash_v(4, state, input); }
extern "C" void yescryptr32_hash(void *state, const void *input) { hash_v(5, state, input); }

static uint32_t *d_resNonce[MAX_GPUS];
static uint32_t *h_resNonce[MAX_GPUS];
static uint32_t s_throughput[MAX_GPUS];
static struct yescrypt_variant s_params[MAX_GPUS];
static int s_variant[MAX_GPUS] = { 0 };
static bool init[MAX_GPUS] = { 0 };

void yescrypt_setTarget(int thr_id, const uint32_t *pdata, const uint32_t *ptarget)
{
	const struct yescrypt_variant *y = &s_params[thr_id];
	yescrypt_setBlock(pdata, y->key, y->key ? (uint32_t) strlen(y->key) : 0, ((const uint64_t*) ptarget)[3]);
}

uint32_t yescrypt_scan(int thr_id, uint32_t threads, uint32_t startNonce,
	uint32_t *out, uint32_t max_out, uint32_t *total)
{
	const uint32_t n = (threads + YESCRYPT_PAD - 1) / YESCRYPT_PAD * YESCRYPT_PAD;
	const struct yescrypt_variant *y = &s_params[thr_id];
	uint32_t *res = h_resNonce[thr_id];

	yescrypt_cpu_hash(thr_id, n, threads, startNonce, y->N, y->r, y->p, d_resNonce[thr_id]);

	cudaMemcpy(res, d_resNonce[thr_id], YESCRYPT_MAX_RESULTS * sizeof(uint32_t), cudaMemcpyDeviceToHost);
	uint32_t found = res[0];
	if (found > YESCRYPT_MAX_RESULTS - 1)
		found = YESCRYPT_MAX_RESULTS - 1;

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

/* kernel limits: power-of-2 N, r <= 128, key <= 119 bytes */
static bool params_ok(int thr_id, const struct yescrypt_variant *y)
{
	const bool ok = y->N >= 16 && y->N <= (1U << 24) && !(y->N & (y->N - 1)) &&
		y->r >= 1 && y->r <= 128 && y->p >= 1 && y->p <= 64 &&
		(!y->key || strlen(y->key) <= 119);
	if (!ok)
		gpulog(LOG_ERR, thr_id, "%s: unsupported parameters N %u r %u p %u%s", y->name,
			y->N, y->r, y->p, y->key && strlen(y->key) > 119 ? ", key longer than 119 bytes" : "");
	return ok;
}

static void free_gpu(int thr_id)
{
	cudaDeviceSynchronize();
	yescrypt_cpu_free(thr_id);
	cudaFree(d_resNonce[thr_id]);
	cudaFreeHost(h_resNonce[thr_id]);
	cuda_tdr_own_check(thr_id, false);
	init[thr_id] = false;
}

static int scanhash_v(int v, int thr_id, struct work *work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t _ALIGN(64) endiandata[20];
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];
	const int dev_id = device_map[thr_id];

	if (opt_benchmark) {
		// both words the kernel compares
		ptarget[7] = 0x00ff;
		ptarget[6] = 0xffffffff;
	}

	// a pool may switch between the variants: the buffers follow the parameters
	if (init[thr_id] && s_variant[thr_id] != v)
		free_gpu(thr_id);

	if (!init[thr_id])
	{
		const struct yescrypt_variant y = yescrypt_params(v);
		if (!params_ok(thr_id, &y))
			proper_exit(EXIT_CODE_USAGE);

		cudaSetDevice(dev_id);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			// reduce cpu usage
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
			CUDA_LOG_ERROR();
		}
		cuda_get_arch(thr_id);

		// threads per CUDA core: 3 on sm_50+, 1 on Kepler (192 cores per SM)
		cudaDeviceProp props;
		cudaGetDeviceProperties(&props, dev_id);
		const uint32_t cores = props.multiProcessorCount * (props.major < 5 ? 192 : props.major == 6 && props.minor == 0 ? 64 : 128);
		uint32_t throughput = cuda_default_throughput(thr_id, cores * (props.major < 5 ? 1 : 3));

		// scratch in 3/4 of the free memory; the V index of one sub-launch stays 32-bit
		const uint64_t per = YESCRYPT_WORDS(y.N, y.r, y.p) * 4;
		size_t mfree = 0, mtotal = 0;
		cudaMemGetInfo(&mfree, &mtotal);
		uint64_t fit = (uint64_t) (mfree / 4 * 3) / per;
		const uint64_t vfit = (1ULL << 32) / ((uint64_t) y.N * y.r * 2) & ~31ULL;
		if (fit > vfit) fit = vfit;
		if (throughput > fit)
			throughput = (uint32_t) fit;
		throughput = throughput / YESCRYPT_PAD * YESCRYPT_PAD;
		if (throughput < YESCRYPT_PAD)
			throughput = YESCRYPT_PAD;
		api_set_throughput(thr_id, throughput);
		gpulog(LOG_INFO, thr_id, "Intensity set to %g, %u cuda threads", throughput2intensity(throughput), throughput);

		if (!yescrypt_cpu_init(thr_id, throughput, y.N, y.r, y.p)) {
			gpulog(LOG_ERR, thr_id, "%s: %u threads need %u MiB, init failed", y.name, throughput,
				(uint32_t) ((per * throughput) >> 20));
			yescrypt_cpu_free(thr_id);
			proper_exit(EXIT_CODE_CUDA_ERROR);
		}
		CUDA_CALL_OR_RET_X(cudaMalloc(&d_resNonce[thr_id], YESCRYPT_MAX_RESULTS * sizeof(uint32_t)), -1);
		CUDA_CALL_OR_RET_X(cudaMallocHost(&h_resNonce[thr_id], YESCRYPT_MAX_RESULTS * sizeof(uint32_t)), -1);
		// the scan syncs after each launch on a display GPU, so it is no single long launch
		cuda_tdr_own_check(thr_id, true);

		s_params[thr_id] = y;
		s_variant[thr_id] = v;
		s_throughput[thr_id] = throughput;
		init[thr_id] = true;
		yescrypt_device_selftest(thr_id, v);
	}
	const uint32_t throughput = s_throughput[thr_id];

	for (int k = 0; k < 20; k++)
		be32enc(&endiandata[k], pdata[k]);
	yescrypt_setTarget(thr_id, pdata, ptarget);

	do {
		// the last launch covers only the rest of the range
		const uint32_t batch = (uint32_t) min((uint64_t) throughput, (uint64_t) max_nonce - pdata[19]);
		if (!batch)
			break;

		uint32_t cand[YESCRYPT_MAX_CAND], total = 0;
		const uint32_t count = yescrypt_scan(thr_id, batch, pdata[19], cand, YESCRYPT_MAX_CAND, &total);
		*hashes_done = pdata[19] - first_nonce + batch;

		if (total > count)
			gpulog(LOG_WARNING, thr_id, "candidates flood: %u", total);

		// CPU-check every candidate, keep the two lowest valid ones
		uint32_t valid[2];
		int nvalid = 0;
		for (uint32_t n = 0; n < count && nvalid < 2; n++) {
			uint32_t _ALIGN(64) vhash[8];
			be32enc(&endiandata[19], cand[n]);
			yescrypt_hash_params(&s_params[thr_id], vhash, endiandata);
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
				yescrypt_hash_params(&s_params[thr_id], vhash, endiandata);
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

extern "C" int scanhash_yescrypt(int thr_id, struct work *work, uint32_t max_nonce, unsigned long *hashes_done)
{ return scanhash_v(0, thr_id, work, max_nonce, hashes_done); }
extern "C" int scanhash_yescryptr8(int thr_id, struct work *work, uint32_t max_nonce, unsigned long *hashes_done)
{ return scanhash_v(1, thr_id, work, max_nonce, hashes_done); }
extern "C" int scanhash_yescryptr16(int thr_id, struct work *work, uint32_t max_nonce, unsigned long *hashes_done)
{ return scanhash_v(2, thr_id, work, max_nonce, hashes_done); }
extern "C" int scanhash_yescryptr16v2(int thr_id, struct work *work, uint32_t max_nonce, unsigned long *hashes_done)
{ return scanhash_v(3, thr_id, work, max_nonce, hashes_done); }
extern "C" int scanhash_yescryptr24(int thr_id, struct work *work, uint32_t max_nonce, unsigned long *hashes_done)
{ return scanhash_v(4, thr_id, work, max_nonce, hashes_done); }
extern "C" int scanhash_yescryptr32(int thr_id, struct work *work, uint32_t max_nonce, unsigned long *hashes_done)
{ return scanhash_v(5, thr_id, work, max_nonce, hashes_done); }

// cleanup, the same for all six
extern "C" void free_yescrypt(int thr_id)
{
	if (init[thr_id])
		free_gpu(thr_id);
}
