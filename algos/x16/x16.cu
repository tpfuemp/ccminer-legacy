/**
 * X16R, X16S and X16RV2: 16 chained 512-bit hashes of the 80-byte header in an order
 * picked from the previous block hash. X16RV2 hashes Tiger-192 (zero-padded to 64 bytes)
 * before Keccak, Luffa and SHA-512.
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
#include "sph/sph_tiger.h"
}
#include "miner.h"
#include "cuda_helper.h"
#include "algos/x11/cuda_x11.h"
#include "algos/qubit/qubit.h"
#include "cuda_x16.h"
#include "x16.h"

// Groestl runs its 4-lane groups up to the next 256-hash block
#define X16_PAD 256
#define X16_MAX_CAND 8

enum Algo {
	BLAKE = 0, BMW, GROESTL, JH, KECCAK, SKEIN, LUFFA, CUBEHASH,
	SHAVITE, SIMD, ECHO, HAMSI, FUGUE, SHABAL, WHIRLPOOL, SHA512,
	HASH_FUNC_COUNT
};

static const char *const algo_names[HASH_FUNC_COUNT] = {
	"blake", "bmw512", "groestl", "jh512", "keccak", "skein", "luffa", "cube",
	"shavite", "simd", "echo", "hamsi", "fugue", "shabal", "whirlpool", "sha512"
};

void x16_order(int variant, const uint8_t *header80, uint8_t order[16])
{
	const uint8_t *prev = header80 + 4;
	uint8_t list[HASH_FUNC_COUNT];
	for (int i = 0; i < HASH_FUNC_COUNT; i++)
		list[i] = (uint8_t) i;
	for (int i = 0; i < HASH_FUNC_COUNT; i++) {
		// 16 hex digits of the first 8 prevhash bytes, last byte first
		const uint8_t b = prev[(15 - i) >> 1];
		const uint8_t digit = (i & 1) ? (b & 0xF) : (b >> 4);
		if (variant != X16_S) {
			order[i] = digit;
			continue;
		}
		// x16s: move the element at position digit to the front
		const uint8_t v = list[digit];
		for (int j = digit; j > 0; j--)
			list[j] = list[j - 1];
		list[0] = v;
	}
	if (variant == X16_S)
		for (int i = 0; i < HASH_FUNC_COUNT; i++)
			order[i] = list[i];
}

static void tiger_pad64(const void *in, size_t len, uint8_t *out)
{
	sph_tiger_context ctx;
	sph_tiger_init(&ctx);
	sph_tiger(&ctx, in, len);
	sph_tiger_close(&ctx, out);
	memset(out + 24, 0, 40);
}

void x16_hash(int variant, void *output, const void *input)
{
	uint8_t _ALIGN(64) hash[64], tmp[64];
	uint8_t order[HASH_FUNC_COUNT];
	x16_order(variant, (const uint8_t *) input, order);

	const void *in = input;
	size_t size = 80;
	for (int i = 0; i < HASH_FUNC_COUNT; i++) {
		if (variant == X16_RV2 && (order[i] == KECCAK || order[i] == LUFFA || order[i] == SHA512)) {
			tiger_pad64(in, size, tmp);
			in = tmp;
			size = 64;
		}
		switch (order[i]) {
#define SPH(name, ctxt) { ctxt c; sph_##name##_init(&c); sph_##name(&c, in, size); sph_##name##_close(&c, hash); } break
		case BLAKE: SPH(blake512, sph_blake512_context);
		case BMW: SPH(bmw512, sph_bmw512_context);
		case GROESTL: SPH(groestl512, sph_groestl512_context);
		case JH: SPH(jh512, sph_jh512_context);
		case KECCAK: SPH(keccak512, sph_keccak512_context);
		case SKEIN: SPH(skein512, sph_skein512_context);
		case LUFFA: SPH(luffa512, sph_luffa512_context);
		case CUBEHASH: SPH(cubehash512, sph_cubehash512_context);
		case SHAVITE: SPH(shavite512, sph_shavite512_context);
		case SIMD: SPH(simd512, sph_simd512_context);
		case ECHO: SPH(echo512, sph_echo512_context);
		case HAMSI: SPH(hamsi512, sph_hamsi512_context);
		case FUGUE: SPH(fugue512, sph_fugue512_context);
		case SHABAL: SPH(shabal512, sph_shabal512_context);
		case WHIRLPOOL: SPH(whirlpool, sph_whirlpool_context);
		case SHA512: SPH(sha512, sph_sha512_context);
#undef SPH
		}
		in = hash;
		size = 64;
	}
	memcpy(output, hash, 32);
}

extern "C" void x16r_hash(void *output, const void *input) { x16_hash(X16_R, output, input); }
extern "C" void x16s_hash(void *output, const void *input) { x16_hash(X16_S, output, input); }
extern "C" void x16rv2_hash(void *output, const void *input) { x16_hash(X16_RV2, output, input); }

static uint32_t *d_hash[MAX_GPUS];
static int s_variant[MAX_GPUS];
static uint8_t s_order[MAX_GPUS][HASH_FUNC_COUNT];
static bool s_echo64_x16[MAX_GPUS]; /* sm_50+ echo kernel */

void x16_setBlock(int thr_id, int variant, uint32_t *endiandata, const uint32_t *ptarget)
{
	s_variant[thr_id] = variant;
	x16_order(variant, (const uint8_t *) endiandata, s_order[thr_id]);

	const uint8_t first = s_order[thr_id][0];
	const bool tiger = variant == X16_RV2 && (first == KECCAK || first == LUFFA || first == SHA512);
	if (tiger) {
		tiger192_setBlock_80(endiandata);
	} else switch (first) {
	case BLAKE: quark_blake512_cpu_setBlock_80(thr_id, endiandata); break;
	case BMW: quark_bmw512_cpu_setBlock_80(endiandata); break;
	case GROESTL: groestl512_setBlock_80(thr_id, endiandata); break;
	case JH: jh512_setBlock_80(thr_id, endiandata); break;
	case KECCAK: keccak512_setBlock_80(thr_id, endiandata); break;
	case SKEIN: skein512_cpu_setBlock_80(endiandata); break;
	case LUFFA: qubit_luffa512_cpu_setBlock_80(endiandata); break;
	case CUBEHASH: cubehash512_setBlock_80(thr_id, endiandata); break;
	case SHAVITE: x11_shavite512_setBlock_80(endiandata); break;
	case SIMD: x16_simd512_setBlock_80(endiandata); break;
	case ECHO: x16_echo512_setBlock_80(endiandata); break;
	case HAMSI: x16_hamsi512_setBlock_80(endiandata); break;
	case FUGUE: {
		uint32_t pdata[20]; /* host word order */
		for (int i = 0; i < 20; i++)
			pdata[i] = swab32(endiandata[i]);
		x16_fugue512_setBlock_80(pdata);
		break;
	}
	case SHABAL: x16_shabal512_setBlock_80(endiandata); break;
	case WHIRLPOOL: x16_whirlpool512_setBlock_80(endiandata); break;
	case SHA512: x16_sha512_setBlock_80(endiandata); break;
	}
	cuda_check_cpu_setTarget(ptarget);
}

static void hash_64(int thr_id, uint8_t algo, uint32_t threads, uint32_t start, uint32_t *h);

static void hash_80(int thr_id, int variant, uint8_t algo, uint32_t threads, uint32_t start, uint32_t *h)
{
	if (variant == X16_RV2 && (algo == KECCAK || algo == LUFFA || algo == SHA512)) {
		// Tiger-192 of the header, then the block itself over its 64-byte output
		tiger192_cpu_hash_80(thr_id, (int) threads, start, h);
		hash_64(thr_id, algo, threads, start, h);
		return;
	}
	switch (algo) {
	case BLAKE: quark_blake512_cpu_hash_80(thr_id, threads, start, h); break;
	case BMW: quark_bmw512_cpu_hash_80(thr_id, threads, start, h, 0); break;
	case GROESTL: groestl512_cuda_hash_80(thr_id, threads, start, h); break;
	case JH: jh512_cuda_hash_80(thr_id, threads, start, h); break;
	case KECCAK: keccak512_cuda_hash_80(thr_id, threads, start, h); break;
	case SKEIN: skein512_cpu_hash_80(thr_id, threads, start, h, 1); break;
	case LUFFA: qubit_luffa512_cpu_hash_80(thr_id, threads, start, h, 0); break;
	case CUBEHASH: cubehash512_cuda_hash_80(thr_id, threads, start, h); break;
	case SHAVITE: x11_shavite512_cpu_hash_80(thr_id, threads, start, h, 0); break;
	case SIMD: x16_simd512_cuda_hash_80(thr_id, threads, start, h); break;
	case ECHO: x16_echo512_cuda_hash_80(thr_id, threads, start, h); break;
	case HAMSI: x16_hamsi512_cuda_hash_80(thr_id, threads, start, h); break;
	case FUGUE: x16_fugue512_cuda_hash_80(thr_id, threads, start, h); break;
	case SHABAL: x16_shabal512_cuda_hash_80(thr_id, threads, start, h); break;
	case WHIRLPOOL: x16_whirlpool512_hash_80(thr_id, threads, start, h); break;
	case SHA512: x16_sha512_cuda_hash_80(thr_id, threads, start, h); break;
	}
}

static void hash_64(int thr_id, uint8_t algo, uint32_t threads, uint32_t start, uint32_t *h)
{
	switch (algo) {
	case BLAKE: quark_blake512_cpu_hash_64(thr_id, threads, start, NULL, h, 0); break;
	case BMW: quark_bmw512_cpu_hash_64(thr_id, threads, start, NULL, h, 0); break;
	case GROESTL: quark_groestl512_cpu_hash_64(thr_id, threads, start, NULL, h, 0); break;
	case JH: quark_jh512_cpu_hash_64(thr_id, threads, start, NULL, h, 0); break;
	case KECCAK: quark_keccak512_cpu_hash_64(thr_id, threads, start, NULL, h, 0); break;
	case SKEIN: quark_skein512_cpu_hash_64(thr_id, threads, start, NULL, h, 0); break;
	case LUFFA: x11_luffa512_cpu_hash_64(thr_id, threads, start, NULL, h, 0); break;
	case CUBEHASH: x11_cubehash512_cpu_hash_64(thr_id, threads, start, NULL, h, 0); break;
	case SHAVITE: x11_shavite512_cpu_hash_64(thr_id, threads, start, NULL, h, 0); break;
	case SIMD: x11_simd512_cpu_hash_64(thr_id, threads, start, NULL, h, 0); break;
	case ECHO:
		if (s_echo64_x16[thr_id])
			x16_echo512_cpu_hash_64(thr_id, threads, h);
		else
			x11_echo512_cpu_hash_64(thr_id, threads, start, NULL, h, 0);
		break;
	case HAMSI: x13_hamsi512_cpu_hash_64(thr_id, threads, start, NULL, h, 0); break;
	case FUGUE: x13_fugue512_cpu_hash_64(thr_id, threads, start, NULL, h, 0); break;
	case SHABAL: x14_shabal512_cpu_hash_64(thr_id, threads, start, NULL, h, 0); break;
	case WHIRLPOOL: x15_whirlpool_cpu_hash_64(thr_id, threads, start, NULL, h, 0); break;
	case SHA512: x17_sha512_cpu_hash_64(thr_id, threads, start, h); break;
	}
}

uint32_t x16_scan(int thr_id, uint32_t threads, uint32_t startNonce,
	uint32_t *out, uint32_t max_out, uint32_t *total)
{
	uint32_t *h = d_hash[thr_id];
	const int variant = s_variant[thr_id];
	const uint8_t *order = s_order[thr_id];
	hash_80(thr_id, variant, order[0], threads, startNonce, h);
	for (int i = 1; i < HASH_FUNC_COUNT; i++) {
		const uint8_t a = order[i];
		if (variant == X16_RV2 && (a == KECCAK || a == LUFFA || a == SHA512))
			tiger192_cpu_hash_64(thr_id, (int) threads, 1, h);
		hash_64(thr_id, a, threads, startNonce, h);
	}
	return cuda_check_hash_suppl_all(thr_id, threads, startNonce, h, out, max_out, total);
}

static bool init[MAX_GPUS] = { 0 };

static int scanhash_x16(int variant, int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t _ALIGN(64) endiandata[20];
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];

	if (opt_benchmark) {
		ptarget[7] = 0x003f;
		// benchmark order as the reference miner: x16r/x16rv2 ECHO then 15x BLAKE, x16s the
		// shuffle of a zero prevhash
		if (variant != X16_S)
			((uint8_t*) pdata)[8] = 0xAA;
	}

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
		qubit_luffa512_cpu_init(thr_id, throughput);
		x11_luffa512_cpu_init(thr_id, throughput);
		x11_cubehash512_cpu_init(thr_id, throughput);
		x11_shavite512_cpu_init(thr_id, throughput);
		if (x11_simd512_cpu_init(thr_id, throughput) != 0)
			return -1;
		x11_echo512_cpu_init(thr_id, throughput);
		x16_echo512_cuda_init(thr_id, throughput);
		x13_hamsi512_cpu_init(thr_id, throughput);
		x13_fugue512_cpu_init(thr_id, throughput);
		x16_fugue512_cpu_init(thr_id, throughput);
		x14_shabal512_cpu_init(thr_id, throughput);
		x16_whirlpool512_init(thr_id, throughput);
		x17_sha512_cpu_init(thr_id, throughput);
		cuda_check_cpu_init(thr_id, throughput);

		const size_t states = (throughput + X16_PAD - 1) / X16_PAD * X16_PAD;
		CUDA_CALL_OR_RET_X(cudaMalloc(&d_hash[thr_id], (size_t) 64 * states), -1);
		x16_device_selftest(thr_id);

		init[thr_id] = true;
	}

	for (int k = 0; k < 20; k++)
		be32enc(&endiandata[k], pdata[k]);
	x16_setBlock(thr_id, variant, endiandata, ptarget);
	if (opt_debug && !thr_id) {
		char s[HASH_FUNC_COUNT + 1];
		for (int i = 0; i < HASH_FUNC_COUNT; i++)
			s[i] = "0123456789ABCDEF"[s_order[thr_id][i]];
		s[HASH_FUNC_COUNT] = '\0';
		applog(LOG_DEBUG, "hash order %s (first %s)", s, algo_names[s_order[thr_id][0]]);
	}

	do {
		// the last launch covers only the rest of the range
		const uint32_t batch = (uint32_t) min((uint64_t) throughput, (uint64_t) max_nonce - pdata[19]);
		if (!batch)
			break;

		uint32_t cand[X16_MAX_CAND], total = 0;
		const uint32_t count = x16_scan(thr_id, batch, pdata[19], cand, X16_MAX_CAND, &total);
		*hashes_done = pdata[19] - first_nonce + batch;

		if (total > count)
			gpulog(LOG_WARNING, thr_id, "candidates flood: %u", total);

		// CPU-check every candidate, keep the two lowest valid ones
		uint32_t valid[2];
		int nvalid = 0;
		for (uint32_t n = 0; n < count && nvalid < 2; n++) {
			uint32_t _ALIGN(64) vhash[8];
			be32enc(&endiandata[19], cand[n]);
			x16_hash(variant, vhash, endiandata);
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
				x16_hash(variant, vhash, endiandata);
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

extern "C" int scanhash_x16r(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
{
	return scanhash_x16(X16_R, thr_id, work, max_nonce, hashes_done);
}

extern "C" int scanhash_x16s(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
{
	return scanhash_x16(X16_S, thr_id, work, max_nonce, hashes_done);
}

extern "C" int scanhash_x16rv2(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
{
	return scanhash_x16(X16_RV2, thr_id, work, max_nonce, hashes_done);
}

// cleanup, shared by the three algos
static void free_x16(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaDeviceSynchronize();
	cudaFree(d_hash[thr_id]);
	quark_blake512_cpu_free(thr_id);
	quark_groestl512_cpu_free(thr_id);
	x11_simd512_cpu_free(thr_id);
	x13_fugue512_cpu_free(thr_id);
	x16_fugue512_cpu_free(thr_id);
	cuda_check_cpu_free(thr_id);
	init[thr_id] = false;
	cudaDeviceSynchronize();
}

extern "C" void free_x16r(int thr_id) { free_x16(thr_id); }
extern "C" void free_x16s(int thr_id) { free_x16(thr_id); }
extern "C" void free_x16rv2(int thr_id) { free_x16(thr_id); }
