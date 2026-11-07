// SPDX-License-Identifier: GPL-3.0-or-later
/*
 * Init-time self-test of the lyra2z330 kernel, once per GPU, fail-closed
 * (cuda/selftest_gate.cuh). Each scan covers one nonce: a target equal to the
 * hash top 64 bits must report it, one below must not.
 */
#include <string.h>

#include <miner.h>
#include <cuda_helper.h>
#include "cuda/selftest_gate.cuh"
#include "lyra2z330.h"

struct lyra2z330_kat {
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

static bool scan_one(int thr_id, const struct lyra2z330_kat *k, uint64_t high, bool *found)
{
	uint32_t _ALIGN(64) pdata[20];
	uint32_t _ALIGN(64) target[8];
	// work data words, as the stratum code fills them
	for (int i = 0; i < 20; i++)
		pdata[i] = be32dec(k->header + 4 * i);
	const uint32_t nonce = pdata[19];

	// words 0..5 open, so the top 64 bits decide
	memset(target, 0xff, 24);
	target[6] = (uint32_t) high;
	target[7] = (uint32_t) (high >> 32);

	uint32_t out[8], total = 0;
	lyra2z330_setTarget(thr_id, pdata, target);
	const uint32_t count = lyra2z330_scan(thr_id, 1, nonce, out, 8, &total);
	const cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		return selftest_cuda_error(thr_id, "lyra2z330", err);
	*found = (count == 1 && total == 1 && out[0] == nonce);
	return true;
}

/* the --test-vectors vector and a synthetic header (byte i = i) */
bool lyra2z330_device_selftest(int thr_id)
{
	static bool tested[MAX_GPUS] = { 0 };
	const int dev_id = device_map[thr_id];
	if (tested[dev_id]) return true;
	tested[dev_id] = true;

	struct lyra2z330_kat k[2];
	bool passed = hex_to_bin(k[0].header,
			"0100000000000000000000000000000000000000000000000000000000000000"
			"000000003ba3edfd7a7b12b27ac72c3e67768f617fc81bc3888a51323a9fb8aa"
			"4b1e5e4a3aab5f49ffff001d0000004b", 80) &&
		hex_to_bin(k[0].hash, "126de1ac244cfed0253ab8fe9158a0a15aa58f3e95532ed5ca08e0d5a113e800", 32) &&
		hex_to_bin(k[1].hash, "bbc07308856eef2305237fd2aa662c6573d2e173fddee568788bc30048d54ab0", 32);
	for (int i = 0; i < 80; i++) k[1].header[i] = (uint8_t) i;

	for (int v = 0; passed && v < 2; v++) {
		uint8_t cpu[32];
		lyra2z330_hash(cpu, k[v].header);
		if (memcmp(cpu, k[v].hash, 32)) {
			gpulog(LOG_ERR, thr_id, "lyra2z330 self-test FAILED: CPU reference, vector %d", v + 1);
			passed = false;
			break;
		}
		const uint64_t high = le32dec(k[v].hash + 24) | ((uint64_t) le32dec(k[v].hash + 28) << 32);
		bool hit = false, miss = true;
		if (!scan_one(thr_id, &k[v], high, &hit) ||
		    !scan_one(thr_id, &k[v], high - 1, &miss) || !hit || miss) {
			gpulog(LOG_ERR, thr_id, "lyra2z330 self-test FAILED: GPU, vector %d (hit %d, below %d)",
				v + 1, (int) hit, (int) miss);
			passed = false;
		}
	}
	return selftest_gate(thr_id, "lyra2z330", passed);
}
