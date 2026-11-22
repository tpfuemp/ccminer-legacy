/**
 * Anime (Animecoin): the quark functions in another order, starting with BMW-512
 * over the 80-byte header; three of the nine steps pick their function by bit 3
 * of the first hash byte. Kernels and the branch split are quark's.
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
#include "cuda/selftest_gate.cuh"
#include "algos/x11/cuda_x11.h"
#include "quark.h"

// Groestl runs its 4-lane groups up to the next 256-hash block
#define ANIME_PAD 256
#define ANIME_MAX_CAND 8

static uint32_t *d_hash[MAX_GPUS];
static uint32_t *d_list1[MAX_GPUS];
static uint32_t *d_list2[MAX_GPUS];

extern "C" void animehash(void *state, const void *input)
{
	uint32_t _ALIGN(128) hash[16];

	sph_blake512_context ctx_blake;
	sph_bmw512_context ctx_bmw;
	sph_groestl512_context ctx_groestl;
	sph_jh512_context ctx_jh;
	sph_keccak512_context ctx_keccak;
	sph_skein512_context ctx_skein;

	sph_bmw512_init(&ctx_bmw);
	sph_bmw512(&ctx_bmw, input, 80);
	sph_bmw512_close(&ctx_bmw, hash);

	sph_blake512_init(&ctx_blake);
	sph_blake512(&ctx_blake, hash, 64);
	sph_blake512_close(&ctx_blake, hash);

	if (hash[0] & 0x8) {
		sph_groestl512_init(&ctx_groestl);
		sph_groestl512(&ctx_groestl, hash, 64);
		sph_groestl512_close(&ctx_groestl, hash);
	} else {
		sph_skein512_init(&ctx_skein);
		sph_skein512(&ctx_skein, hash, 64);
		sph_skein512_close(&ctx_skein, hash);
	}

	sph_groestl512_init(&ctx_groestl);
	sph_groestl512(&ctx_groestl, hash, 64);
	sph_groestl512_close(&ctx_groestl, hash);

	sph_jh512_init(&ctx_jh);
	sph_jh512(&ctx_jh, hash, 64);
	sph_jh512_close(&ctx_jh, hash);

	if (hash[0] & 0x8) {
		sph_blake512_init(&ctx_blake);
		sph_blake512(&ctx_blake, hash, 64);
		sph_blake512_close(&ctx_blake, hash);
	} else {
		sph_bmw512_init(&ctx_bmw);
		sph_bmw512(&ctx_bmw, hash, 64);
		sph_bmw512_close(&ctx_bmw, hash);
	}

	sph_keccak512_init(&ctx_keccak);
	sph_keccak512(&ctx_keccak, hash, 64);
	sph_keccak512_close(&ctx_keccak, hash);

	sph_skein512_init(&ctx_skein);
	sph_skein512(&ctx_skein, hash, 64);
	sph_skein512_close(&ctx_skein, hash);

	if (hash[0] & 0x8) {
		sph_keccak512_init(&ctx_keccak);
		sph_keccak512(&ctx_keccak, hash, 64);
		sph_keccak512_close(&ctx_keccak, hash);
	} else {
		sph_jh512_init(&ctx_jh);
		sph_jh512(&ctx_jh, hash, 64);
		sph_jh512_close(&ctx_jh, hash);
	}

	memcpy(state, hash, 32);
}

static void anime_setBlock(uint32_t *endiandata, const uint32_t *ptarget)
{
	quark_bmw512_cpu_setBlock_80(endiandata);
	cuda_check_cpu_setTarget(ptarget);
}

/* every nonce takes its own branch, so all hashes of the batch are final and
 * the target check covers the whole batch */
static uint32_t anime_scan(int thr_id, uint32_t threads, uint32_t startNonce,
	uint32_t *out, uint32_t max_out, uint32_t *total)
{
	uint32_t *h = d_hash[thr_id], *l1 = d_list1[thr_id], *l2 = d_list2[thr_id];
	uint32_t n1, n2;

	quark_bmw512_cpu_hash_80(thr_id, threads, startNonce, h, 0);
	quark_blake512_cpu_hash_64(thr_id, threads, startNonce, NULL, h, 0);

	quark_split_cpu(thr_id, threads, startNonce, h, 0x8, l1, &n1, l2, &n2);
	if (n1) quark_groestl512_cpu_hash_64(thr_id, n1, startNonce, l1, h, 0);
	if (n2) quark_skein512_cpu_hash_64(thr_id, n2, startNonce, l2, h, 0);

	quark_groestl512_cpu_hash_64(thr_id, threads, startNonce, NULL, h, 0);
	quark_jh512_cpu_hash_64(thr_id, threads, startNonce, NULL, h, 0);

	quark_split_cpu(thr_id, threads, startNonce, h, 0x8, l1, &n1, l2, &n2);
	if (n1) quark_blake512_cpu_hash_64(thr_id, n1, startNonce, l1, h, 0);
	if (n2) quark_bmw512_cpu_hash_64(thr_id, n2, startNonce, l2, h, 0);

	quark_keccak512_cpu_hash_64(thr_id, threads, startNonce, NULL, h, 0);
	quark_skein512_cpu_hash_64(thr_id, threads, startNonce, NULL, h, 0);

	quark_split_cpu(thr_id, threads, startNonce, h, 0x8, l1, &n1, l2, &n2);
	if (n1) quark_keccak512_cpu_hash_64(thr_id, n1, startNonce, l1, h, 0);
	if (n2) quark_jh512_cpu_hash_64(thr_id, n2, startNonce, l2, h, 0);

	return cuda_check_hash_suppl_all(thr_id, threads, startNonce, h, out, max_out, total);
}

/* ---- init-time self-test, fail-closed (cuda/selftest_gate.cuh) ---- */

/* a target equal to the hash top 64 bits must report the nonce, one below not */
static bool anime_scan_one(int thr_id, const uint8_t *header, uint64_t high, bool *found)
{
	uint32_t _ALIGN(64) data[20], target[8], out[ANIME_MAX_CAND], total = 0;
	memcpy(data, header, 80);
	const uint32_t nonce = be32dec(header + 76);

	memset(target, 0xff, 24);
	target[6] = (uint32_t) high;
	target[7] = (uint32_t) (high >> 32);
	anime_setBlock(data, target);
	const uint32_t count = anime_scan(thr_id, 1, nonce, out, ANIME_MAX_CAND, &total);
	const cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		return selftest_cuda_error(thr_id, "anime", err);
	*found = (count == 1 && total == 1 && out[0] == nonce);
	return true;
}

/* the second --test-vectors vector and the synthetic header (byte i = i) */
static bool anime_device_selftest(int thr_id)
{
	static bool tested[MAX_GPUS] = { 0 };
	const int dev_id = device_map[thr_id];
	if (tested[dev_id]) return true;
	tested[dev_id] = true;

	uint8_t header[2][80], hash[2][32];
	bool passed = hex2bin(header[0],
			"000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f"
			"202122232425262728292a2b2c2d2e2f303132333435363738393a3b3c3d3e3f"
			"404142434445464748494a4b00000092", 80) &&
		hex2bin(hash[0], "8e20eefcd3601a422284f2d63eef36bd2ab0f501c932cc34e87edecf7fa92f00", 32) &&
		hex2bin(hash[1], "17263b57e6e2bb93b78027c88b8cc02d5296c278c7dc9f8a6d3b1d55a7824a21", 32);
	for (int i = 0; i < 80; i++)
		header[1][i] = (uint8_t) i;

	for (int v = 0; passed && v < 2; v++) {
		uint8_t cpu[32];
		animehash(cpu, header[v]);
		if (memcmp(cpu, hash[v], 32)) {
			gpulog(LOG_ERR, thr_id, "anime self-test FAILED: CPU reference, vector %d", v + 1);
			passed = false;
			break;
		}
		const uint64_t high = le32dec(hash[v] + 24) | ((uint64_t) le32dec(hash[v] + 28) << 32);
		bool hit = false, miss = true;
		if (!anime_scan_one(thr_id, header[v], high, &hit) ||
		    !anime_scan_one(thr_id, header[v], high - 1, &miss) || !hit || miss) {
			gpulog(LOG_ERR, thr_id, "anime self-test FAILED: GPU, vector %d (hit %d, below %d)",
				v + 1, (int) hit, (int) miss);
			passed = false;
		}
	}
	return selftest_gate(thr_id, "anime", passed);
}

/* ---- scan ---- */

static bool init[MAX_GPUS] = { 0 };

extern "C" int scanhash_anime(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
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

	if (!init[thr_id]) {
		cudaSetDevice(dev_id);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
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

		const size_t states = (throughput + ANIME_PAD - 1) / ANIME_PAD * ANIME_PAD;
		CUDA_CALL_OR_RET_X(cudaMalloc(&d_hash[thr_id], (size_t) 64 * states), -1);
		CUDA_CALL_OR_RET_X(cudaMalloc(&d_list1[thr_id], sizeof(uint32_t) * throughput), -1);
		CUDA_CALL_OR_RET_X(cudaMalloc(&d_list2[thr_id], sizeof(uint32_t) * throughput), -1);
		anime_device_selftest(thr_id);

		init[thr_id] = true;
	}

	for (int k = 0; k < 20; k++)
		be32enc(&endiandata[k], pdata[k]);
	anime_setBlock(endiandata, ptarget);

	do {
		// the last launch covers only the rest of the range
		const uint32_t batch = (uint32_t) min((uint64_t) throughput, (uint64_t) max_nonce - pdata[19]);
		if (!batch)
			break;

		uint32_t cand[ANIME_MAX_CAND], total = 0;
		const uint32_t count = anime_scan(thr_id, batch, pdata[19], cand, ANIME_MAX_CAND, &total);
		*hashes_done = pdata[19] - first_nonce + batch;

		if (total > count)
			gpulog(LOG_WARNING, thr_id, "candidates flood: %u", total);

		// CPU-check every candidate, keep the two lowest valid ones
		uint32_t valid[2];
		int nvalid = 0;
		for (uint32_t n = 0; n < count && nvalid < 2; n++) {
			uint32_t _ALIGN(64) vhash[8];
			be32enc(&endiandata[19], cand[n]);
			animehash(vhash, endiandata);
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
				animehash(vhash, endiandata);
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

extern "C" void free_anime(int thr_id)
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
