// SPDX-License-Identifier: GPL-3.0-or-later
/*
 * Init-time self-test of the tribus kernel chain, once per GPU, fail-closed
 * (cuda/selftest_gate.cuh). Each scan covers one nonce: a target equal to the
 * hash top 64 bits must report it, one below must not.
 */
#include <string.h>

#include <miner.h>
#include <cuda_helper.h>
#include "cuda/selftest_gate.cuh"
#include "tribus.h"

struct tribus_kat {
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

static bool scan_one(int thr_id, const struct tribus_kat *k, uint64_t high, bool *found)
{
	uint32_t _ALIGN(64) data[20];
	uint32_t _ALIGN(64) target[8];
	memcpy(data, k->header, 80);
	const uint32_t nonce = be32dec(k->header + 76);

	// words 0..5 open, so the top 64 bits decide on both target checks
	memset(target, 0xff, 24);
	target[6] = (uint32_t) high;
	target[7] = (uint32_t) (high >> 32);

	uint32_t out[TRIBUS_MAX_RESULTS], total = 0;
	tribus_setBlock(thr_id, data, target);
	const uint32_t count = tribus_scan(thr_id, 1, nonce, high, out, TRIBUS_MAX_RESULTS, &total);
	if (cudaGetLastError() != cudaSuccess)
		return selftest_cuda_fault();
	*found = (count == 1 && total == 1 && out[0] == nonce);
	return true;
}

/* the --test-vectors vector and a synthetic header (byte i = i) */
bool tribus_device_selftest(int thr_id)
{
	static bool tested[MAX_GPUS] = { 0 };
	const int dev_id = device_map[thr_id];
	if (tested[dev_id]) return true;
	tested[dev_id] = true;

	struct tribus_kat k[2];
	bool passed = hex_to_bin(k[0].header,
			"0100000000000000000000000000000000000000000000000000000000000000"
			"000000003ba3edfd7a7b12b27ac72c3e67768f617fc81bc3888a51323a9fb8aa"
			"4b1e5e4a35ab5f49ffff001d014633c7", 80) &&
		hex_to_bin(k[0].hash, "01267fe436afab515dae226cb430bd2fabf70978d8fa448a9adf406977000000", 32) &&
		hex_to_bin(k[1].hash, "83dd7a543b16df4c3de48c531e7dd4295aef287376caacdae5df8c5d84ff7068", 32);
	for (int i = 0; i < 80; i++) k[1].header[i] = (uint8_t) i;

	for (int v = 0; passed && v < 2; v++) {
		uint8_t cpu[32];
		tribus_hash(cpu, k[v].header);
		if (memcmp(cpu, k[v].hash, 32)) {
			gpulog(LOG_ERR, thr_id, "tribus self-test FAILED: CPU reference, vector %d", v + 1);
			passed = false;
			break;
		}
		const uint64_t high = le32dec(k[v].hash + 24) | ((uint64_t) le32dec(k[v].hash + 28) << 32);
		bool hit = false, miss = true;
		if (!scan_one(thr_id, &k[v], high, &hit) ||
		    !scan_one(thr_id, &k[v], high - 1, &miss) || !hit || miss) {
			gpulog(LOG_ERR, thr_id, "tribus self-test FAILED: GPU, vector %d (hit %d, below %d)",
				v + 1, (int) hit, (int) miss);
			passed = false;
		}
	}
	return selftest_gate(thr_id, "tribus", passed);
}
