// SPDX-License-Identifier: GPL-3.0-or-later
/*
 * Init-time self-test of the blake mining kernels (14 and 8 rounds), once per
 * GPU, fail-closed (cuda/selftest_gate.cuh). The kernels only report hashes
 * with h7 == 0, so the vectors are mined ones. For each vector the kernel scans
 * only its nonce: a target equal to h6 must report it, one below must not.
 */
#include <string.h>

#include <miner.h>
#include <cuda_helper.h>
#include "cuda/selftest_gate.cuh"

extern "C" void blake256hash(void *output, const void *input, int8_t rounds);
extern void blake256mid(uint32_t *output, const uint32_t *input, int8_t rounds);
extern void blake256_cpu_setBlock_16(uint32_t *penddata, const uint32_t *midstate, const uint32_t *ptarget);
extern uint32_t blake256_cpu_hash_16(const int thr_id, const uint32_t threads, const uint32_t startNonce,
	const uint64_t highTarget, const int8_t rounds);

struct blake_kat {
	int8_t rounds;
	uint8_t header[80];
	uint8_t hash[32];
};

static bool hex_to_bin(uint8_t *out, const char *hex, size_t len)
{
	for (size_t i = 0; i < len; i++) {
		unsigned int b;
		if (sscanf(hex + 2 * i, "%2x", &b) != 1) return false;
		out[i] = (uint8_t) b;
	}
	return true;
}

/* the --test-vectors vectors, digests checked against the BLAKE-256 spec KATs */
static bool blake_kats(struct blake_kat k[2])
{
	k[0].rounds = 14;
	bool ok = hex_to_bin(k[0].header,
		"0100000000000000000000000000000000000000000000000000000000000000"
		"000000003ba3edfd7a7b12b27ac72c3e67768f617fc81bc3888a51323a9fb8aa"
		"4b1e5e4a2aab5f49ffff001d87cd94f3", 80);
	ok = ok && hex_to_bin(k[0].hash,
		"683434bc9ade2ec5635d549bc983c0098224e21ba3ece2d8a404aa0d00000000", 32);
	k[1].rounds = 8;
	ok = ok && hex_to_bin(k[1].header,
		"0100000000000000000000000000000000000000000000000000000000000000"
		"000000003ba3edfd7a7b12b27ac72c3e67768f617fc81bc3888a51323a9fb8aa"
		"4b1e5e4a2cab5f49ffff001da928b973", 80);
	ok = ok && hex_to_bin(k[1].hash,
		"80975e07f5b029ff4f8a366cb7c67fd8c066d4f449b06e137da44a5800000000", 32);
	return ok;
}

/* scan only the vector's nonce against the h6 target; true if it ran */
static bool blake_scan_one(int thr_id, const struct blake_kat *k, uint64_t target, bool *found)
{
	uint32_t _ALIGN(64) words[20];
	uint32_t _ALIGN(64) pdata[20];
	uint32_t _ALIGN(64) midstate[8];
	memcpy(words, k->header, 80);
	for (int i = 0; i < 20; i++)
		pdata[i] = be32dec(k->header + 4 * i);

	blake256mid(midstate, words, k->rounds);
	blake256_cpu_setBlock_16(&pdata[16], midstate, NULL);
	const uint32_t res = blake256_cpu_hash_16(thr_id, 1, pdata[19], target, k->rounds);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		gpulog(LOG_ERR, thr_id, "blake self-test: %s", cudaGetErrorString(err));
		if (err == cudaErrorMemoryAllocation)
			selftest_cuda_fault(); // could not run, not a wrong hash
		return false;
	}
	*found = (res == pdata[19]);
	return true;
}

/* call after the result buffers are allocated, before mining */
bool blake256_device_selftest(int thr_id)
{
	static bool tested[MAX_GPUS] = { 0 };
	const int dev_id = device_map[thr_id];
	if (tested[dev_id]) return true;
	tested[dev_id] = true;

	struct blake_kat k[2];
	bool passed = blake_kats(k);
	for (int v = 0; passed && v < 2; v++) {
		uint8_t cpu[32];
		blake256hash(cpu, k[v].header, k[v].rounds);
		if (memcmp(cpu, k[v].hash, 32)) {
			gpulog(LOG_ERR, thr_id, "blake self-test FAILED: CPU reference, %d rounds", (int) k[v].rounds);
			passed = false;
			break;
		}
		const uint64_t h6 = le32dec(k[v].hash + 24);
		bool hit = false, miss = false;
		if (!blake_scan_one(thr_id, &k[v], h6, &hit) ||
		    !blake_scan_one(thr_id, &k[v], h6 - 1, &miss) || !hit || miss) {
			gpulog(LOG_ERR, thr_id, "blake self-test FAILED: GPU, %d rounds (hit %d, below %d)",
				(int) k[v].rounds, (int) hit, (int) miss);
			passed = false;
		}
	}
	return selftest_gate(thr_id, "blake", passed);
}
