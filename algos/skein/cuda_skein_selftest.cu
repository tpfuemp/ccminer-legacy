// SPDX-License-Identifier: GPL-3.0-or-later
/*
 * Init-time self-test of the skein mining kernel, once per GPU, fail-closed
 * (cuda/selftest_gate.cuh). For each vector the kernel scans only its nonce:
 * a target equal to the top 64 hash bits must report it, one below must not.
 */
#include <string.h>

#include <miner.h>
#include <cuda_helper.h>
#include "cuda/selftest_gate.cuh"

extern "C" void skeincoinhash(void *output, const void *input);
extern void skeincoin_setBlock_80(int thr_id, void *pdata);
extern uint32_t skeincoin_hash_sm5(int thr_id, uint32_t threads, uint32_t startNounce, int swap, uint64_t target64, uint32_t *secNonce);

struct skein_kat {
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

/* digests from an independent Skein-512 implementation (spec KAT checked) */
static bool skein_kats(struct skein_kat k[2])
{
	// the --test-vectors vector
	bool ok = hex_to_bin(k[0].header,
		"0100000000000000000000000000000000000000000000000000000000000000"
		"000000003ba3edfd7a7b12b27ac72c3e67768f617fc81bc3888a51323a9fb8aa"
		"4b1e5e4a2eab5f49ffff001dd5df6acb", 80);
	ok = ok && hex_to_bin(k[0].hash,
		"bc520ff9f7fabcc4b5773dcfd883455d865ef7adbb4768da051f647e00000000", 32);
	// synthetic header, byte i = i
	for (int i = 0; i < 80; i++) k[1].header[i] = (uint8_t) i;
	ok = ok && hex_to_bin(k[1].hash,
		"4de841423eddf5bd7faa05470262cfa33a74caa8081254e412091a657a96bbe8", 32);
	return ok;
}

/* scan only the vector's nonce against the 64-bit target; true if it ran */
static bool skein_scan_one(int thr_id, const uint8_t header[80], uint64_t target, bool *found)
{
	uint32_t _ALIGN(64) data[20];
	memcpy(data, header, 80);
	const uint32_t nonce = be32dec(header + 76);
	uint32_t second = UINT32_MAX;

	skeincoin_setBlock_80(thr_id, data);
	const uint32_t res = skeincoin_hash_sm5(thr_id, 1, nonce, 1, target, &second);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		gpulog(LOG_ERR, thr_id, "skein self-test: %s", cudaGetErrorString(err));
		if (err == cudaErrorMemoryAllocation)
			selftest_cuda_fault(); // could not run, not a wrong hash
		return false;
	}
	*found = (res == nonce);
	return true;
}

/* call after skeincoin_init(), before mining */
bool skein_device_selftest(int thr_id)
{
	static bool tested[MAX_GPUS] = { 0 };
	const int dev_id = device_map[thr_id];
	if (tested[dev_id]) return true;
	tested[dev_id] = true;

	struct skein_kat k[2];
	bool passed = skein_kats(k);
	for (int v = 0; passed && v < 2; v++) {
		uint8_t cpu[32];
		skeincoinhash(cpu, k[v].header);
		if (memcmp(cpu, k[v].hash, 32)) {
			gpulog(LOG_ERR, thr_id, "skein self-test FAILED: CPU reference, vector %d", v + 1);
			passed = false;
			break;
		}
		uint64_t top;
		memcpy(&top, k[v].hash + 24, 8);
		bool hit = false, miss = false;
		if (!skein_scan_one(thr_id, k[v].header, top, &hit) ||
		    !skein_scan_one(thr_id, k[v].header, top - 1, &miss) || !hit || miss) {
			gpulog(LOG_ERR, thr_id, "skein self-test FAILED: GPU, vector %d (hit %d, below %d)",
				v + 1, (int) hit, (int) miss);
			passed = false;
		}
	}
	return selftest_gate(thr_id, "skein", passed);
}
