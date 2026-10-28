// SPDX-License-Identifier: GPL-3.0-or-later
/*
 * Init-time self-test of both blake2s kernels (target and zero top word), once
 * per GPU, fail-closed (cuda/selftest_gate.cuh). Each scan covers one nonce.
 */
#include <string.h>

#include <miner.h>
#include <cuda_helper.h>
#include "cuda/selftest_gate.cuh"

extern "C" void blake2s_hash(void *output, const void *input);
extern void blake2s_setBlock(const uint32_t* input, const uint32_t ptarget7);
extern uint32_t blake2s_hash_nonces(int thr_id, uint32_t threads, uint32_t startNonce, uint32_t ptarget7, uint32_t *resNonces);

struct blake2s_kat {
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

/* digests from an independent BLAKE2s implementation */
static bool blake2s_kats(struct blake2s_kat k[2])
{
	// the --test-vectors vector
	bool ok = hex_to_bin(k[0].header,
		"0100000000000000000000000000000000000000000000000000000000000000"
		"000000003ba3edfd7a7b12b27ac72c3e67768f617fc81bc3888a51323a9fb8aa"
		"4b1e5e4a30ab5f49ffff001d0b2961c4", 80);
	ok = ok && hex_to_bin(k[0].hash,
		"ec538163285a1a07c96d90d8636706c6b66e603e788cfa3678d16e3300000000", 32);
	// synthetic header, byte i = i
	for (int i = 0; i < 80; i++) k[1].header[i] = (uint8_t) i;
	ok = ok && hex_to_bin(k[1].hash,
		"aff3b75f3f581264d7661662b92f5ad37c1d32bd45ff81a4ed8adc9ef30dd989", 32);
	return ok;
}

/* scan only the vector's nonce; true if it ran */
static bool blake2s_scan_one(int thr_id, const uint8_t header[80], uint32_t ptarget7, bool *found)
{
	uint32_t _ALIGN(64) data[20];
	uint32_t res[16];
	memcpy(data, header, 80);
	const uint32_t nonce = be32dec(header + 76);

	blake2s_setBlock(data, ptarget7);
	const uint32_t count = blake2s_hash_nonces(thr_id, 1, nonce, ptarget7, res);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		gpulog(LOG_ERR, thr_id, "blake2s self-test: %s", cudaGetErrorString(err));
		if (err == cudaErrorMemoryAllocation)
			selftest_cuda_fault(); // could not run, not a wrong hash
		return false;
	}
	*found = (count == 1 && swab32(res[1]) == nonce);
	return true;
}

/* call after the result buffers are allocated, before mining */
bool blake2s_device_selftest(int thr_id)
{
	static bool tested[MAX_GPUS] = { 0 };
	const int dev_id = device_map[thr_id];
	if (tested[dev_id]) return true;
	tested[dev_id] = true;

	struct blake2s_kat k[2];
	bool passed = blake2s_kats(k);
	for (int v = 0; passed && v < 2; v++) {
		uint8_t cpu[32];
		blake2s_hash(cpu, k[v].header);
		if (memcmp(cpu, k[v].hash, 32)) {
			gpulog(LOG_ERR, thr_id, "blake2s self-test FAILED: CPU reference, vector %d", v + 1);
			passed = false;
		}
	}
	const uint32_t top = le32dec(k[1].hash + 28);
	bool zero_hit = false, hit = false, below = false, zero_synth = false;
	if (passed && (!blake2s_scan_one(thr_id, k[0].header, 0, &zero_hit) ||
	    !blake2s_scan_one(thr_id, k[1].header, top, &hit) ||
	    !blake2s_scan_one(thr_id, k[1].header, top - 1, &below) ||
	    !blake2s_scan_one(thr_id, k[1].header, 0, &zero_synth) ||
	    !zero_hit || !hit || below || zero_synth)) {
		gpulog(LOG_ERR, thr_id, "blake2s self-test FAILED: GPU (zero kernel %d/%d, target kernel %d/%d)",
			(int) zero_hit, (int) zero_synth, (int) hit, (int) below);
		passed = false;
	}
	return selftest_gate(thr_id, "blake2s", passed);
}
