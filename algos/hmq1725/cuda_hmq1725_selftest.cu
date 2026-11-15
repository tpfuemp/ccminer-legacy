// SPDX-License-Identifier: GPL-3.0-or-later
/*
 * Init-time self-test of the hmq1725 kernel chain, once per GPU, fail-closed
 * (cuda/selftest_gate.cuh). Each scan covers one nonce: a target equal to the
 * hash top 64 bits must report it, one below must not. Two known-answer vectors,
 * then the synthetic header with nonces 0..6 against the CPU reference: together
 * they take both sides of all eight branch points.
 */
#include <string.h>

#include <miner.h>
#include <cuda_helper.h>
#include "cuda/selftest_gate.cuh"
#include "hmq1725.h"

#define COVER_NONCES 7

struct hmq_kat {
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

static bool scan_one(int thr_id, const struct hmq_kat *k, uint64_t high, bool *found)
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
	hmq1725_setBlock(data, target);
	const uint32_t count = hmq1725_scan(thr_id, 1, nonce, out, 8, &total);
	const cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		return selftest_cuda_error(thr_id, "hmq1725", err);
	*found = (count == 1 && total == 1 && out[0] == nonce);
	return true;
}

static bool check(int thr_id, const struct hmq_kat *k, const char *what, int n)
{
	const uint64_t high = le32dec(k->hash + 24) | ((uint64_t) le32dec(k->hash + 28) << 32);
	bool hit = false, miss = true;
	if (!scan_one(thr_id, k, high, &hit) || !scan_one(thr_id, k, high - 1, &miss) || !hit || miss) {
		gpulog(LOG_ERR, thr_id, "hmq1725 self-test FAILED: GPU, %s %d (hit %d, below %d)",
			what, n, (int) hit, (int) miss);
		return false;
	}
	return true;
}

/* the --test-vectors vector (mined) and a synthetic header (byte i = i) */
bool hmq1725_device_selftest(int thr_id)
{
	static bool tested[MAX_GPUS] = { 0 };
	const int dev_id = device_map[thr_id];
	if (tested[dev_id]) return true;
	tested[dev_id] = true;

	struct hmq_kat k[2];
	bool passed = hex_to_bin(k[0].header,
			"0100000000000000000000000000000000000000000000000000000000000000"
			"000000003ba3edfd7a7b12b27ac72c3e67768f617fc81bc3888a51323a9fb8aa"
			"4b1e5e4a33ab5f49ffff001d006de1cd", 80) &&
		hex_to_bin(k[0].hash, "f3f40619e3a7ce3e1aa5335c761378bd30b4c81af27fc2291f05951806000000", 32) &&
		hex_to_bin(k[1].hash, "8ac907d3b6ad9ede574347816462e19a60e2f44249d2b922cd78d689a7f8136d", 32);
	for (int i = 0; i < 80; i++) k[1].header[i] = (uint8_t) i;

	for (int v = 0; passed && v < 2; v++) {
		uint8_t cpu[32];
		hmq1725hash(cpu, k[v].header);
		if (memcmp(cpu, k[v].hash, 32)) {
			gpulog(LOG_ERR, thr_id, "hmq1725 self-test FAILED: CPU reference, vector %d", v + 1);
			passed = false;
			break;
		}
		passed = check(thr_id, &k[v], "vector", v + 1);
	}

	for (uint32_t n = 0; passed && n < COVER_NONCES; n++) {
		struct hmq_kat c = k[1];
		be32enc(c.header + 76, n);
		hmq1725hash(c.hash, c.header);
		passed = check(thr_id, &c, "branch nonce", (int) n);
	}
	return selftest_gate(thr_id, "hmq1725", passed);
}
