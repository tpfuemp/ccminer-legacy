// SPDX-License-Identifier: GPL-3.0-or-later
/*
 * Init-time self-test of the neoscrypt kernels, once per GPU, fail-closed
 * (cuda/selftest_gate.cuh). A target equal to the hash top 64 bits must report the nonce,
 * one below must not; one nonce, and for the mined vector also inside a 389-nonce batch
 * (several blocks).
 */
#include <string.h>

#include <miner.h>
#include <cuda_helper.h>
#include "cuda/selftest_gate.cuh"
#include "neoscrypt.h"

extern "C" void neoscrypt_hash(void *output, const void *input);

struct neoscrypt_kat {
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

/* scan [nonce - before, nonce - before + threads), found = only the nonce reported */
static bool scan_at(int thr_id, const struct neoscrypt_kat *k, uint32_t before, uint32_t threads,
	uint64_t target, uint32_t *d_res, uint32_t *h_res, bool *found)
{
	uint32_t _ALIGN(64) data[20];
	memcpy(data, k->header, 80);
	const uint32_t nonce = be32dec(k->header + 76);

	neoscrypt_setBlock(data);
	const uint32_t count = neoscrypt_scan(thr_id, threads, nonce - before, target, d_res, h_res);
	if (cudaGetLastError() != cudaSuccess)
		return selftest_cuda_fault();
	*found = (count == 1 && h_res[1] == nonce);
	return true;
}

/* the --test-vectors vector and a synthetic header (byte i = i) */
bool neoscrypt_device_selftest(int thr_id, uint32_t *d_res, uint32_t *h_res)
{
	static bool tested[MAX_GPUS] = { 0 };
	const int dev_id = device_map[thr_id];
	if (tested[dev_id]) return true;
	tested[dev_id] = true;

	struct neoscrypt_kat k[2];
	bool passed = hex_to_bin(k[0].header,
			"0100000000000000000000000000000000000000000000000000000000000000"
			"000000003ba3edfd7a7b12b27ac72c3e67768f617fc81bc3888a51323a9fb8aa"
			"4b1e5e4a29ab5f49ffff001d0000dc53", 80) &&
		hex_to_bin(k[0].hash, "dc1ceebf50f0f8330862f697a5e9b8212ac9ea6ff101c02489a8ca09b8430000", 32) &&
		hex_to_bin(k[1].hash, "7258961afb33fd12d00cacb8d63f4f4f52bb6917043865dd24a08f578853122d", 32);
	for (int i = 0; i < 80; i++) k[1].header[i] = (uint8_t) i;

	for (int v = 0; passed && v < 2; v++) {
		uint8_t cpu[32];
		neoscrypt_hash(cpu, k[v].header);
		if (memcmp(cpu, k[v].hash, 32)) {
			gpulog(LOG_ERR, thr_id, "neoscrypt self-test FAILED: CPU reference, vector %d", v + 1);
			passed = false;
			break;
		}
		const uint64_t high = le32dec(k[v].hash + 24) | ((uint64_t) le32dec(k[v].hash + 28) << 32);
		for (int batch = 0; passed && batch < (v ? 1 : 2); batch++) {
			const uint32_t before = batch ? 300 : 0, threads = batch ? 389 : 1;
			bool hit = false, miss = true;
			if (!scan_at(thr_id, &k[v], before, threads, high, d_res, h_res, &hit) ||
			    !scan_at(thr_id, &k[v], before, threads, high - 1, d_res, h_res, &miss) || !hit || miss) {
				gpulog(LOG_ERR, thr_id, "neoscrypt self-test FAILED: GPU, vector %d, %u nonces (hit %d, below %d)",
					v + 1, threads, (int) hit, (int) miss);
				passed = false;
			}
		}
	}
	return selftest_gate(thr_id, "neoscrypt", passed);
}
