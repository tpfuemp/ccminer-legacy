// SPDX-License-Identifier: GPL-3.0-or-later
/*
 * Init-time self-test of the x13 kernel chain, once per GPU, fail-closed
 * (cuda/selftest_gate.cuh). Each scan covers one nonce: a target equal to the
 * hash top 64 bits must report it, one below must not.
 */
#include <string.h>

#include <miner.h>
#include <cuda_helper.h>
#include "cuda/selftest_gate.cuh"
#include "x13.h"

struct x13_kat {
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

static bool scan_one(int thr_id, const struct x13_kat *k, uint64_t high, bool *found)
{
	uint32_t _ALIGN(64) data[20];
	uint32_t _ALIGN(64) target[8];
	memcpy(data, k->header, 80);
	const uint32_t nonce = be32dec(k->header + 76);

	// words 0..5 open, so the top 64 bits decide
	memset(target, 0xff, 24);
	target[6] = (uint32_t) high;
	target[7] = (uint32_t) (high >> 32);

	uint32_t out[8], total = 0;
	x13_setBlock(thr_id, data, target);
	const uint32_t count = x13_scan(thr_id, 1, nonce, out, 8, &total);
	if (cudaGetLastError() != cudaSuccess)
		return selftest_cuda_fault();
	*found = (count == 1 && total == 1 && out[0] == nonce);
	return true;
}

/* the --test-vectors vector and a synthetic header (byte i = i) */
bool x13_device_selftest(int thr_id)
{
	static bool tested[MAX_GPUS] = { 0 };
	const int dev_id = device_map[thr_id];
	if (tested[dev_id]) return true;
	tested[dev_id] = true;

	struct x13_kat k[2];
	bool passed = hex_to_bin(k[0].header,
			"0100000000000000000000000000000000000000000000000000000000000000"
			"000000003ba3edfd7a7b12b27ac72c3e67768f617fc81bc3888a51323a9fb8aa"
			"4b1e5e4a37ab5f49ffff001d00df1087", 80) &&
		hex_to_bin(k[0].hash, "be2ae451c663abe9046cbfaa142a723bdaf29f4d4577174d997e74af12000000", 32) &&
		hex_to_bin(k[1].hash, "44e39c878aa74badd072b531b2fdf181051431053f486fc36435b347e3986dac", 32);
	for (int i = 0; i < 80; i++) k[1].header[i] = (uint8_t) i;

	for (int v = 0; passed && v < 2; v++) {
		uint8_t cpu[32];
		x13hash(cpu, k[v].header);
		if (memcmp(cpu, k[v].hash, 32)) {
			gpulog(LOG_ERR, thr_id, "x13 self-test FAILED: CPU reference, vector %d", v + 1);
			passed = false;
			break;
		}
		const uint64_t high = le32dec(k[v].hash + 24) | ((uint64_t) le32dec(k[v].hash + 28) << 32);
		bool hit = false, miss = true;
		if (!scan_one(thr_id, &k[v], high, &hit) ||
		    !scan_one(thr_id, &k[v], high - 1, &miss) || !hit || miss) {
			gpulog(LOG_ERR, thr_id, "x13 self-test FAILED: GPU, vector %d (hit %d, below %d)",
				v + 1, (int) hit, (int) miss);
			passed = false;
		}
	}
	return selftest_gate(thr_id, "x13", passed);
}
