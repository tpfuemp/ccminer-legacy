/**
 * HMQ1725: 25 chained 512-bit hashes of the 80-byte header; at 8 points the next block
 * is picked per nonce by bits 3 and 4 of the first hash byte. tpruvot, 2017.
 */
extern "C" {
#include "sph/sph_blake.h"
#include "sph/sph_bmw.h"
#include "sph/sph_groestl.h"
#include "sph/sph_skein.h"
#include "sph/sph_jh.h"
#include "sph/sph_keccak.h"
#include "sph/sph_luffa.h"
#include "sph/sph_cubehash.h"
#include "sph/sph_shavite.h"
#include "sph/sph_simd.h"
#include "sph/sph_echo.h"
#include "sph/sph_hamsi.h"
#include "sph/sph_fugue.h"
#include "sph/sph_shabal.h"
#include "sph/sph_whirlpool.h"
#include "sph/sph_sha2.h"
#include "sph/sph_haval.h"
}

#include "miner.h"
#include "cuda_helper.h"
#include "algos/x11/cuda_x11.h"
#include "algos/x16/cuda_x16.h"
#include "algos/quark/quark.h"
#include "hmq1725.h"

/* the 64-byte kernels run in blocks of up to 256 threads */
#define HMQ_PAD 256
#define HMQ_MAX_CAND 8
#define HMQ_MASK 24

#define CPU_BLOCK(type, name) do { type c; name##_init(&c); name(&c, h, 64); name##_close(&c, h); } while (0)

extern "C" void hmq1725hash(void *output, const void *input)
{
	uint32_t _ALIGN(64) h[16];
	sph_bmw512_context bmw;
	sph_bmw512_init(&bmw);
	sph_bmw512(&bmw, input, 80);
	sph_bmw512_close(&bmw, h);

#define BLAKE    CPU_BLOCK(sph_blake512_context, sph_blake512)
#define BMW      CPU_BLOCK(sph_bmw512_context, sph_bmw512)
#define GROESTL  CPU_BLOCK(sph_groestl512_context, sph_groestl512)
#define SKEIN    CPU_BLOCK(sph_skein512_context, sph_skein512)
#define JH       CPU_BLOCK(sph_jh512_context, sph_jh512)
#define KECCAK   CPU_BLOCK(sph_keccak512_context, sph_keccak512)
#define LUFFA    CPU_BLOCK(sph_luffa512_context, sph_luffa512)
#define CUBEHASH CPU_BLOCK(sph_cubehash512_context, sph_cubehash512)
#define SHAVITE  CPU_BLOCK(sph_shavite512_context, sph_shavite512)
#define SIMD     CPU_BLOCK(sph_simd512_context, sph_simd512)
#define ECHO     CPU_BLOCK(sph_echo512_context, sph_echo512)
#define HAMSI    CPU_BLOCK(sph_hamsi512_context, sph_hamsi512)
#define FUGUE    CPU_BLOCK(sph_fugue512_context, sph_fugue512)
#define SHABAL   CPU_BLOCK(sph_shabal512_context, sph_shabal512)
#define WHIRL    CPU_BLOCK(sph_whirlpool_context, sph_whirlpool)
#define SHA512   CPU_BLOCK(sph_sha512_context, sph_sha512)
#define HAVAL    do { CPU_BLOCK(sph_haval256_5_context, sph_haval256_5); memset(&h[8], 0, 32); } while (0)
#define BRANCH(a, b) do { if (h[0] & HMQ_MASK) { a; } else { b; } } while (0)

	WHIRL;
	BRANCH(GROESTL, SKEIN);
	JH;
	KECCAK;
	BRANCH(BLAKE, BMW);
	LUFFA;
	CUBEHASH;
	BRANCH(KECCAK, JH);
	SHAVITE;
	SIMD;
	BRANCH(WHIRL, HAVAL);
	ECHO;
	BLAKE;
	BRANCH(SHAVITE, LUFFA);
	HAMSI;
	FUGUE;
	BRANCH(ECHO, SIMD);
	SHABAL;
	WHIRL;
	BRANCH(FUGUE, SHA512);
	GROESTL;
	SHA512;
	BRANCH(HAVAL, WHIRL);
	BMW;

	memcpy(output, h, 32);
}

static uint32_t *d_hash[MAX_GPUS];
static uint32_t *d_list1[MAX_GPUS], *d_list2[MAX_GPUS];
static bool s_echo64_x16[MAX_GPUS]; /* sm_50+ echo kernel */

enum hmq_block { BLAKE_, BMW_, GROESTL_, SKEIN_, JH_, KECCAK_, LUFFA_, SHAVITE_, SIMD_, ECHO_,
	FUGUE_, WHIRL_, SHA512_, HAVAL_ };

/* nv: list of nonces to hash (their hashes sit at nonce - start), NULL = all of the batch */
static void hash_64(int thr_id, int block, uint32_t threads, uint32_t start, uint32_t *h, uint32_t *nv = NULL)
{
	switch (block) {
	case BLAKE_: quark_blake512_cpu_hash_64(thr_id, threads, start, nv, h, 0); break;
	case BMW_: quark_bmw512_cpu_hash_64(thr_id, threads, start, nv, h, 0); break;
	case GROESTL_: quark_groestl512_cpu_hash_64(thr_id, threads, start, nv, h, 0); break;
	case SKEIN_: quark_skein512_cpu_hash_64(thr_id, threads, start, nv, h, 0); break;
	case JH_: quark_jh512_cpu_hash_64(thr_id, threads, start, nv, h, 0); break;
	case KECCAK_: quark_keccak512_cpu_hash_64(thr_id, threads, start, nv, h, 0); break;
	case LUFFA_: x11_luffa512_cpu_hash_64(thr_id, threads, start, nv, h, 0); break;
	case SHAVITE_: x11_shavite512_cpu_hash_64(thr_id, threads, start, nv, h, 0); break;
	case SIMD_: x11_simd512_cpu_hash_64(thr_id, threads, start, nv, h, 0); break;
	case ECHO_:
		if (s_echo64_x16[thr_id])
			x16_echo512_cpu_hash_64(thr_id, threads, h, start, nv);
		else
			x11_echo512_cpu_hash_64(thr_id, threads, start, nv, h, 0);
		break;
	case FUGUE_: x13_fugue512_cpu_hash_64(thr_id, threads, start, nv, h, 0); break;
	case WHIRL_: x15_whirlpool_cpu_hash_64(thr_id, threads, start, nv, h, 0); break;
	case SHA512_: x17_sha512_cpu_hash_64(thr_id, threads, start, h, nv); break;
	case HAVAL_: x17_haval256_cpu_hash_64(thr_id, threads, start, h, 512, nv); break;
	}
}

/* each branch block runs in place on its own nonces only: bit set -> a, else b */
static void split(int thr_id, int a, int b, uint32_t threads, uint32_t start)
{
	uint32_t n1, n2;
	quark_split_cpu(thr_id, threads, start, d_hash[thr_id], HMQ_MASK, d_list1[thr_id], &n1,
		d_list2[thr_id], &n2);
	if (n1) hash_64(thr_id, a, n1, start, d_hash[thr_id], d_list1[thr_id]);
	if (n2) hash_64(thr_id, b, n2, start, d_hash[thr_id], d_list2[thr_id]);
}

void hmq1725_setBlock(uint32_t *endiandata, const uint32_t *ptarget)
{
	quark_bmw512_cpu_setBlock_80(endiandata);
	cuda_check_cpu_setTarget(ptarget);
}

uint32_t hmq1725_scan(int thr_id, uint32_t threads, uint32_t startNonce,
	uint32_t *out, uint32_t max_out, uint32_t *total)
{
	uint32_t *h = d_hash[thr_id];
	const uint32_t s = startNonce;
	quark_bmw512_cpu_hash_80(thr_id, threads, s, h, 0);
	hash_64(thr_id, WHIRL_, threads, s, h);
	split(thr_id, GROESTL_, SKEIN_, threads, s);
	hash_64(thr_id, JH_, threads, s, h);
	hash_64(thr_id, KECCAK_, threads, s, h);
	split(thr_id, BLAKE_, BMW_, threads, s);
	x11_luffaCubehash512_cpu_hash_64(thr_id, threads, h, 0);
	split(thr_id, KECCAK_, JH_, threads, s);
	hash_64(thr_id, SHAVITE_, threads, s, h);
	hash_64(thr_id, SIMD_, threads, s, h);
	split(thr_id, WHIRL_, HAVAL_, threads, s);
	hash_64(thr_id, ECHO_, threads, s, h);
	hash_64(thr_id, BLAKE_, threads, s, h);
	split(thr_id, SHAVITE_, LUFFA_, threads, s);
	x13_hamsi512_cpu_hash_64(thr_id, threads, s, NULL, h, 0);
	hash_64(thr_id, FUGUE_, threads, s, h);
	split(thr_id, ECHO_, SIMD_, threads, s);
	x14_shabal512_cpu_hash_64(thr_id, threads, s, NULL, h, 0);
	hash_64(thr_id, WHIRL_, threads, s, h);
	split(thr_id, FUGUE_, SHA512_, threads, s);
	hash_64(thr_id, GROESTL_, threads, s, h);
	hash_64(thr_id, SHA512_, threads, s, h);
	split(thr_id, HAVAL_, WHIRL_, threads, s);
	hash_64(thr_id, BMW_, threads, s, h);
	return cuda_check_hash_suppl_all(thr_id, threads, startNonce, h, out, max_out, total);
}

static bool init[MAX_GPUS] = { 0 };

extern "C" int scanhash_hmq1725(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t _ALIGN(64) endiandata[20];
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];

	if (opt_benchmark)
		ptarget[7] = 0x003f;

	const int dev_id = device_map[thr_id];
	// Kepler: smaller batches keep one launch chain well below the 2 s WDDM TDR
	const int intensity = (device_sm[dev_id] < 500) ? 17 : 19;
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
		s_echo64_x16[thr_id] = cuda_arch[dev_id] >= 500;
		quark_blake512_cpu_init(thr_id, throughput);
		quark_bmw512_cpu_init(thr_id, throughput);
		quark_groestl512_cpu_init(thr_id, throughput);
		quark_skein512_cpu_init(thr_id, throughput);
		quark_jh512_cpu_init(thr_id, throughput);
		quark_keccak512_cpu_init(thr_id, throughput);
		x11_luffaCubehash512_cpu_init(thr_id, throughput);
		x11_luffa512_cpu_init(thr_id, throughput);
		x11_shavite512_cpu_init(thr_id, throughput);
		if (x11_simd512_cpu_init(thr_id, throughput) != 0)
			return -1;
		x11_echo512_cpu_init(thr_id, throughput);
		x16_echo512_cuda_init(thr_id, throughput);
		x13_hamsi512_cpu_init(thr_id, throughput);
		x13_fugue512_cpu_init(thr_id, throughput);
		x14_shabal512_cpu_init(thr_id, throughput);
		x16_whirlpool512_init(thr_id, throughput);
		x17_sha512_cpu_init(thr_id, throughput);
		cuda_check_cpu_init(thr_id, throughput);

		const size_t states = (throughput + HMQ_PAD - 1) / HMQ_PAD * HMQ_PAD;
		CUDA_CALL_OR_RET_X(cudaMalloc(&d_hash[thr_id], (size_t) 64 * states), -1);
		CUDA_CALL_OR_RET_X(cudaMalloc(&d_list1[thr_id], sizeof(uint32_t) * states), -1);
		CUDA_CALL_OR_RET_X(cudaMalloc(&d_list2[thr_id], sizeof(uint32_t) * states), -1);
		quark_split_cpu_init(thr_id);
		hmq1725_device_selftest(thr_id);

		init[thr_id] = true;
	}

	for (int k = 0; k < 20; k++)
		be32enc(&endiandata[k], pdata[k]);
	hmq1725_setBlock(endiandata, ptarget);

	do {
		// the last launch covers only the rest of the range
		const uint32_t batch = (uint32_t) min((uint64_t) throughput, (uint64_t) max_nonce - pdata[19]);
		if (!batch)
			break;

		uint32_t cand[HMQ_MAX_CAND], total = 0;
		const uint32_t count = hmq1725_scan(thr_id, batch, pdata[19], cand, HMQ_MAX_CAND, &total);
		*hashes_done = pdata[19] - first_nonce + batch;

		if (total > count)
			gpulog(LOG_WARNING, thr_id, "candidates flood: %u", total);

		// CPU-check every candidate, keep the two lowest valid ones
		uint32_t valid[2];
		int nvalid = 0;
		for (uint32_t n = 0; n < count && nvalid < 2; n++) {
			uint32_t _ALIGN(64) vhash[8];
			be32enc(&endiandata[19], cand[n]);
			hmq1725hash(vhash, endiandata);
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
				hmq1725hash(vhash, endiandata);
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
extern "C" void free_hmq1725(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaDeviceSynchronize();
	cudaFree(d_hash[thr_id]);
	cudaFree(d_list1[thr_id]);
	cudaFree(d_list2[thr_id]);
	quark_split_cpu_free(thr_id);
	quark_blake512_cpu_free(thr_id);
	quark_groestl512_cpu_free(thr_id);
	x11_simd512_cpu_free(thr_id);
	x13_fugue512_cpu_free(thr_id);
	cuda_check_cpu_free(thr_id);
	init[thr_id] = false;
	cudaDeviceSynchronize();
}
