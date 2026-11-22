/*
 * whirlpoolx2 (CapStash): scan loop, CPU hash and init self-test. The PoW hash is
 * one Whirlpool-512 over the 80-byte header, folded to 256 bits by XORing the
 * two 32-byte halves (CapStash-Core CBlockHeader::GetPoWHash).
 */
extern "C" {
#include "sph/sph_whirlpool.h"
}

#include "miner.h"
#include "cuda_helper.h"
#include "cuda/selftest_gate.cuh"

#define MAX_RESULTS 16

void whirlpoolx2_cpu_init(int thr_id, uint32_t threads);
void whirlpoolx2_cpu_free(int thr_id);
void whirlpoolx2_setBlock_80(const void *pdata, const void *ptarget);
void whirlpoolx2_precompute(int thr_id);
uint32_t whirlpoolx2_cpu_hash(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *res);
uint32_t *whirlpoolx2_host_slots(int thr_id);

/* the 80 header bytes */
extern "C" void whirlpoolx2_hash(void *state, const void *input)
{
	sph_whirlpool_context ctx;
	unsigned char hash[64];

	sph_whirlpool_init(&ctx);
	sph_whirlpool(&ctx, input, 80);
	sph_whirlpool_close(&ctx, hash);
	for (int i = 0; i < 32; i++)
		((unsigned char *)state)[i] = hash[i] ^ hash[i + 32];
}

/* header bytes from work->data, nonce bytes rewritten by the caller */
static void wx2_header(uint8_t hdr[80], const uint32_t *pdata)
{
	for (int k = 0; k < 20; k++)
		be32enc(hdr + 4 * k, pdata[k]);
}

static uint32_t wx2_scan(int thr_id, const uint32_t *pdata, const uint32_t *ptarget,
	uint32_t start, uint32_t count, uint32_t *res)
{
	uint8_t hdr[80];
	wx2_header(hdr, pdata);
	whirlpoolx2_setBlock_80(hdr, ptarget);
	whirlpoolx2_precompute(thr_id);
	return whirlpoolx2_cpu_hash(thr_id, count, start, res);
}

/* ---- init-time self-test, fail-closed (cuda/selftest_gate.cuh) ---- */

/* CapStash blocks 100000 and 409009 and their PoW hashes */
static const char *wx2_block_hdr[2] = {
	"000000209afc05d8f4b3a1d38f48b821e770623cac110820e48eeeda1b3ab8d5022fb38ddf60d6f449ca5d67bf250b"
	"e04c089fd0ff3bbe1374c1dc5430dac0f983329b887cbce3696c840a1be1fa4bc4",
	"00000020dc1b81b51cc084166b777d949c59161825a5747fe76d87fb3f909108b8808c55b7f7bd20ca59f6ae48b14f"
	"31f38b145f10b836ed60d2287fe241be5568736cb0e621026b83aa001cc57c054d"
};
static const char *wx2_block_pow[2] = {
	"4680b1e3d3f992ded68ce20739798d9896c1d7500805cb580a0f090000000000",
	"f0beb450ff48f6297c2cb3dd4676cb32ca580a928085daef3b92440000000000"
};

static bool wx2_device_selftest(int thr_id)
{
	static bool tested[MAX_GPUS] = { 0 };
	const int dev_id = device_map[thr_id];
	if (tested[dev_id]) return true;
	tested[dev_id] = true;

	uint32_t _ALIGN(64) pdata[20], target[8], dig[8];
	uint32_t *res = whirlpoolx2_host_slots(thr_id);
	uint8_t hdr[80];
	char hex[65];
	bool cpu_ok = true, scan_ok = true, miss_ok = true, neg_ok = true;

	for (int b = 0; b < 2; b++) {
		/* the CPU hash gives the block's PoW hash */
		hex2bin(hdr, wx2_block_hdr[b], 80);
		whirlpoolx2_hash(dig, hdr);
		for (int i = 0; i < 32; i++)
			sprintf(hex + 2 * i, "%02x", ((uint8_t *)dig)[i]);
		cpu_ok &= strcmp(hex, wx2_block_pow[b]) == 0;

		/* the kernel finds the block nonce at a target equal to its top word,
		 * not one below */
		for (int i = 0; i < 20; i++)
			pdata[i] = be32dec(hdr + 4 * i);
		memset(target, 0xff, sizeof(target));
		target[6] = dig[6];
		target[7] = dig[7];
		scan_ok &= wx2_scan(thr_id, pdata, target, pdata[19], 1, res) == 1 && res[1] == pdata[19];
		if (target[6]-- == 0) target[7]--;
		miss_ok &= wx2_scan(thr_id, pdata, target, pdata[19], 1, res) == 0;

		/* a flipped header bit must lose the hit */
		target[6] = dig[6];
		target[7] = dig[7];
		pdata[3] ^= 0x00000100u;
		neg_ok &= wx2_scan(thr_id, pdata, target, pdata[19], 1, res) == 0;
	}

	/* 1024 nonces at a top-byte-0 target: the kernel reports exactly the nonces
	 * the CPU finds (about 4) */
	bool span_ok = true;
	const uint32_t start = 0x5a1d0c3bu;
	memset(target, 0xff, sizeof(target));
	target[7] = 0x00ffffffu;
	const uint32_t n = wx2_scan(thr_id, pdata, target, start, 1024, res);
	uint32_t found = 0;
	wx2_header(hdr, pdata);
	for (uint32_t i = 0; i < 1024; i++) {
		be32enc(hdr + 76, start + i);
		whirlpoolx2_hash(dig, hdr);
		if (dig[7] <= target[7]) {
			bool listed = false;
			for (uint32_t k = 1; k <= n && k < MAX_RESULTS; k++)
				listed |= res[k] == start + i;
			span_ok &= listed;
			found++;
		}
	}
	span_ok &= n == found && found < MAX_RESULTS;

	const cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		return selftest_gate(thr_id, "whirlpoolx2", selftest_cuda_error(thr_id, "whirlpoolx2", err));

	const bool passed = cpu_ok && scan_ok && miss_ok && neg_ok && span_ok;
	if (!passed)
		gpulog(LOG_ERR, thr_id, "whirlpoolx2 self-test FAILED (cpu %d scan %d miss %d neg %d span %d)",
			(int)cpu_ok, (int)scan_ok, (int)miss_ok, (int)neg_ok, (int)span_ok);
	return selftest_gate(thr_id, "whirlpoolx2", passed);
}

/* ---- scan ---- */

static bool init[MAX_GPUS] = { 0 };

extern "C" int scanhash_whirlpoolx2(int thr_id, struct work *work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];
	uint8_t hdr[80];

	const int dev_id = device_map[thr_id];
	// 2^22 at about 30 MH/s (GTX 660M) is 0.14 s per launch
	const uint32_t throughput = cuda_default_throughput(thr_id, device_sm[dev_id] < 500 ? 1U << 22 : 1U << 23);

	if (opt_benchmark) {
		// both words the kernel compares
		ptarget[7] = 0x000f;
		ptarget[6] = 0xffffffffu;
	}

	if (!init[thr_id]) {
		cudaSetDevice(dev_id);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
			CUDA_LOG_ERROR();
		}
		gpulog(LOG_INFO, thr_id, "Intensity set to %g, %u cuda threads", throughput2intensity(throughput), throughput);

		cuda_get_arch(thr_id);
		whirlpoolx2_cpu_init(thr_id, throughput);
		wx2_device_selftest(thr_id);
		init[thr_id] = true;
	}

	wx2_header(hdr, pdata);
	whirlpoolx2_setBlock_80(hdr, ptarget);
	whirlpoolx2_precompute(thr_id);

	do {
		// the last launch covers only the rest of the range
		const uint32_t batch = (uint32_t) min((uint64_t) throughput, (uint64_t) max_nonce - pdata[19]);
		if (!batch)
			break;

		uint32_t *res = whirlpoolx2_host_slots(thr_id);
		uint32_t count = whirlpoolx2_cpu_hash(thr_id, batch, pdata[19], res);
		*hashes_done = pdata[19] - first_nonce + batch;

		if (count) {
			if (count >= MAX_RESULTS) {
				gpulog(LOG_WARNING, thr_id, "candidates flood: %u", count);
				count = MAX_RESULTS - 1;
			}
			// CPU-check every candidate, keep the two lowest valid ones
			uint32_t valid[2];
			int nvalid = 0;
			for (uint32_t n = 1; n <= count; n++) {
				uint32_t _ALIGN(64) vhash[8];
				be32enc(hdr + 76, res[n]);
				whirlpoolx2_hash(vhash, hdr);
				if (!fulltest(vhash, ptarget)) {
					if (vhash[7] > ptarget[7] || (vhash[7] == ptarget[7] && vhash[6] > ptarget[6])) {
						gpu_increment_reject(thr_id);
						if (!opt_quiet)
							gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU!", res[n]);
					}
					continue;
				}
				if (nvalid < 2)
					valid[nvalid++] = res[n];
				else if (res[n] < valid[1])
					valid[1] = res[n];
				if (nvalid == 2 && valid[1] < valid[0])
					xchg(valid[0], valid[1]);
			}

			if (nvalid) {
				uint32_t _ALIGN(64) vhash[8];
				work->valid_nonces = nvalid;
				for (int i = 0; i < nvalid; i++) {
					work->nonces[i] = valid[i];
					be32enc(hdr + 76, valid[i]);
					whirlpoolx2_hash(vhash, hdr);
					if (i == 0)
						work_set_target_ratio(work, vhash);
					else
						bn_set_target_ratio(work, vhash, 1);
				}
				// resume after the last reported nonce
				pdata[19] = valid[nvalid - 1] + 1;
				return nvalid;
			}
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

extern "C" void free_whirlpoolx2(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaDeviceSynchronize();
	whirlpoolx2_cpu_free(thr_id);
	init[thr_id] = false;
	cudaDeviceSynchronize();
}
