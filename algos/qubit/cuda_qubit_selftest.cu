// SPDX-License-Identifier: GPL-3.0-or-later
/*
 * Init-time self-test of the qubit kernel chain, once per GPU, fail-closed
 * (cuda/selftest_gate.cuh). Each scan covers one nonce: a target equal to the
 * hash top 64 bits must report it, one below must not.
 */
#include <string.h>

#include <miner.h>
#include <cuda_helper.h>
#include "cuda/selftest_gate.cuh"
#include "qubit.h"

struct qubit_kat {
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

static bool scan_one(int thr_id, const struct qubit_kat *k, uint64_t high, bool *found)
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
	qubit_setBlock(data, target);
	const uint32_t count = qubit_scan(thr_id, 1, nonce, out, 8, &total);
	if (cudaGetLastError() != cudaSuccess)
		return selftest_cuda_fault();
	*found = (count == 1 && total == 1 && out[0] == nonce);
	return true;
}

/* the --test-vectors vector and a synthetic header (byte i = i) */
bool qubit_device_selftest(int thr_id)
{
	static bool tested[MAX_GPUS] = { 0 };
	const int dev_id = device_map[thr_id];
	if (tested[dev_id]) return true;
	tested[dev_id] = true;

	struct qubit_kat k[2];
	bool passed = hex_to_bin(k[0].header,
			"0100000000000000000000000000000000000000000000000000000000000000"
			"000000003ba3edfd7a7b12b27ac72c3e67768f617fc81bc3888a51323a9fb8aa"
			"4b1e5e4a36ab5f49ffff001d00a31512", 80) &&
		hex_to_bin(k[0].hash, "03ccee9d0e2678a80a41b995facd0a1a1c1fb6556b3c89756c9b99ad10000000", 32) &&
		hex_to_bin(k[1].hash, "5f445fc9277a5af9e10c6d580bcec9e801a241e75e9734ce2acd3143db177c15", 32);
	for (int i = 0; i < 80; i++) k[1].header[i] = (uint8_t) i;

	for (int v = 0; passed && v < 2; v++) {
		uint8_t cpu[32];
		qubithash(cpu, k[v].header);
		if (memcmp(cpu, k[v].hash, 32)) {
			gpulog(LOG_ERR, thr_id, "qubit self-test FAILED: CPU reference, vector %d", v + 1);
			passed = false;
			break;
		}
		const uint64_t high = le32dec(k[v].hash + 24) | ((uint64_t) le32dec(k[v].hash + 28) << 32);
		bool hit = false, miss = true;
		if (!scan_one(thr_id, &k[v], high, &hit) ||
		    !scan_one(thr_id, &k[v], high - 1, &miss) || !hit || miss) {
			gpulog(LOG_ERR, thr_id, "qubit self-test FAILED: GPU, vector %d (hit %d, below %d)",
				v + 1, (int) hit, (int) miss);
			passed = false;
		}
	}
	return selftest_gate(thr_id, "qubit", passed);
}
