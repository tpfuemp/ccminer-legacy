// SPDX-License-Identifier: GPL-3.0-or-later
/*
 * Init-time self-test of both lbry kernel paths (merged and split), once per GPU,
 * fail-closed (cuda/selftest_gate.cuh). Each scan covers one nonce: a target equal to
 * the hash top 64 bits must report it, one below must not.
 */
#include <string.h>

#include <miner.h>
#include <cuda_helper.h>
#include "cuda/selftest_gate.cuh"
#include "lbry.h"

struct lbry_kat {
	uint8_t header[112];
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

static bool scan_one(const struct lbry_kat *k, bool merged, uint64_t target, uint32_t *d_hash,
	uint32_t *d_res, uint32_t *h_res, bool *found)
{
	uint32_t _ALIGN(64) data[28];
	memcpy(data, k->header, 112);
	const uint32_t nonce = be32dec(k->header + 4 * LBRY_NONCE_WORD);

	lbry_setBlock(merged, data);
	const uint32_t count = lbry_scan(merged, 1, nonce, target, d_hash, d_res, h_res);
	const cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		return selftest_cuda_error(-1, "lbry", err);
	*found = (count == 1 && h_res[1] == nonce);
	return true;
}

/* the --test-vectors vector (mined) and a synthetic header (byte i = i) */
bool lbry_device_selftest(int thr_id, uint32_t *d_res, uint32_t *h_res)
{
	static bool tested[MAX_GPUS] = { 0 };
	const int dev_id = device_map[thr_id];
	if (tested[dev_id]) return true;
	tested[dev_id] = true;

	struct lbry_kat k[2];
	bool passed = hex_to_bin(k[0].header,
			"0100000000000000000000000000000000000000000000000000000000000000"
			"00000000cc59e59ff97ac092b55e423aa5495151ed6fb80570a5bb78cd5bd1c3"
			"821c21b801000000000000000000000000000000000000000000000000000000"
			"0000000034193156ffff001faa14d202", 112) &&
		hex_to_bin(k[0].hash, "8d8d8538e0e83e25c11943dd615acc7c27cd3a8cf7ddc393d999118e31000000", 32) &&
		hex_to_bin(k[1].hash, "5896632aee5dc847a81deb60ba5a44faaedbee357ad44cf48f3f149809bc2cc3", 32);
	for (int i = 0; i < 112; i++) k[1].header[i] = (uint8_t) i;

	uint32_t *d_hash = NULL;
	if (passed) {
		const cudaError_t err = cudaMalloc(&d_hash, 64);
		if (err != cudaSuccess)
			passed = selftest_cuda_error(thr_id, "lbry", err);
	}

	for (int v = 0; passed && v < 2; v++) {
		uint8_t cpu[32];
		lbry_hash(cpu, k[v].header);
		if (memcmp(cpu, k[v].hash, 32)) {
			gpulog(LOG_ERR, thr_id, "lbry self-test FAILED: CPU reference, vector %d", v + 1);
			passed = false;
			break;
		}
		const uint64_t high = le32dec(k[v].hash + 24) | ((uint64_t) le32dec(k[v].hash + 28) << 32);
		for (int m = 0; passed && m < 2; m++) {
			const bool merged = (m == 0);
			bool hit = false, miss = true;
			if (!scan_one(&k[v], merged, high, d_hash, d_res, h_res, &hit) ||
			    !scan_one(&k[v], merged, high - 1, d_hash, d_res, h_res, &miss) || !hit || miss) {
				gpulog(LOG_ERR, thr_id, "lbry self-test FAILED: GPU %s kernel, vector %d (hit %d, below %d)",
					merged ? "merged" : "split", v + 1, (int) hit, (int) miss);
				passed = false;
			}
		}
	}
	if (d_hash)
		cudaFree(d_hash);
	return selftest_gate(thr_id, "lbry", passed);
}
