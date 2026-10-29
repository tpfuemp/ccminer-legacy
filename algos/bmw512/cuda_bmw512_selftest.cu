// SPDX-License-Identifier: GPL-3.0-or-later
/*
 * Init-time self-test of the bmw512 kernel, once per GPU, fail-closed
 * (cuda/selftest_gate.cuh). Each scan covers one nonce: a target equal to the
 * hash top 64 bits must report it, one below must not.
 */
#include <string.h>

#include <miner.h>
#include <cuda_helper.h>
#include "cuda/selftest_gate.cuh"
#include "bmw512.h"

struct bmw512_kat {
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

static bool scan_one(const struct bmw512_kat *k, uint64_t target, uint32_t *d_res,
	uint32_t *h_res, bool *found)
{
	uint32_t _ALIGN(64) data[20];
	memcpy(data, k->header, 80);
	const uint32_t nonce = be32dec(k->header + 76);

	bmw512_cpu_setBlock_80(data);
	const uint32_t count = bmw512_scan(1, nonce, target, d_res, h_res);
	if (cudaGetLastError() != cudaSuccess)
		return selftest_cuda_fault();
	*found = (count == 1 && h_res[1] == nonce);
	return true;
}

/* the --test-vectors vector and a synthetic header (byte i = i) */
bool bmw512_device_selftest(int thr_id, uint32_t *d_res, uint32_t *h_res)
{
	static bool tested[MAX_GPUS] = { 0 };
	const int dev_id = device_map[thr_id];
	if (tested[dev_id]) return true;
	tested[dev_id] = true;

	struct bmw512_kat k[2];
	bool passed = hex_to_bin(k[0].header,
			"0100000000000000000000000000000000000000000000000000000000000000"
			"000000003ba3edfd7a7b12b27ac72c3e67768f617fc81bc3888a51323a9fb8aa"
			"4b1e5e4a33ab5f49ffff001d0f1c9b85", 80) &&
		hex_to_bin(k[0].hash, "d190f9055a4e21748a5c633ff545bc80125deb499232cdad547d48b300000000", 32) &&
		hex_to_bin(k[1].hash, "c2d90cdec45e5c6ad8a5bcb775f982db1e80903cf7166f10303b2cb2cd4abb5b", 32);
	for (int i = 0; i < 80; i++) k[1].header[i] = (uint8_t) i;

	for (int v = 0; passed && v < 2; v++) {
		uint8_t cpu[32];
		bmw512_hash(cpu, k[v].header);
		if (memcmp(cpu, k[v].hash, 32)) {
			gpulog(LOG_ERR, thr_id, "bmw512 self-test FAILED: CPU reference, vector %d", v + 1);
			passed = false;
			break;
		}
		const uint64_t high = le32dec(k[v].hash + 24) | ((uint64_t) le32dec(k[v].hash + 28) << 32);
		bool hit = false, miss = true;
		if (!scan_one(&k[v], high, d_res, h_res, &hit) ||
		    !scan_one(&k[v], high - 1, d_res, h_res, &miss) || !hit || miss) {
			gpulog(LOG_ERR, thr_id, "bmw512 self-test FAILED: GPU, vector %d (hit %d, below %d)",
				v + 1, (int) hit, (int) miss);
			passed = false;
		}
	}
	return selftest_gate(thr_id, "bmw512", passed);
}
