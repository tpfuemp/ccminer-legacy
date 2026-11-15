// SPDX-License-Identifier: GPL-3.0-or-later
/*
 * Init-time self-test of the verthash kernels, once per GPU, fail-closed
 * (cuda/selftest_gate.cuh). The known-answer vector must be reported at a target
 * equal to its hash word 7 and not one below; the hashes of 256 nonces of a synthetic
 * header must equal the CPU reference.
 */
#include <stdlib.h>
#include <string.h>

#include <miner.h>
#include <cuda_helper.h>
#include "cuda/selftest_gate.cuh"
#include "verthash.h"

#define CMP_NONCES 256

static bool hex_to_bin(uint8_t *out, const char *hex, size_t len)
{
	for (size_t i = 0; i < len; i++) {
		unsigned int b;
		if (sscanf(hex + 2 * i, "%2x", &b) != 1) return false;
		out[i] = (uint8_t) b;
	}
	return true;
}

static bool scan_one(int thr_id, const uint8_t *header, uint32_t target7, bool *found)
{
	uint32_t _ALIGN(64) data[20];
	memcpy(data, header, 80);
	const uint32_t nonce = le32dec(header + 76);
	uint32_t out[4], total = 0;
	verthash_setBlock(thr_id, data);
	const uint32_t count = verthash_scan(thr_id, 1, nonce, target7, out, 4, &total);
	const cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		return selftest_cuda_error(thr_id, "verthash", err);
	*found = (count == 1 && total == 1 && out[0] == nonce);
	return true;
}

/* the --test-vectors vector (mined) */
static bool check_vector(int thr_id)
{
	uint8_t header[80], hash[32], cpu[32];
	if (!hex_to_bin(header, VERTHASH_TV_HEADER, 80) || !hex_to_bin(hash, VERTHASH_TV_HASH, 32))
		return false;
	verthash_hash(cpu, header);
	if (memcmp(cpu, hash, 32)) {
		gpulog(LOG_ERR, thr_id, "verthash self-test FAILED: CPU reference");
		return false;
	}
	const uint32_t high = le32dec(hash + 28);
	bool hit = false, miss = true;
	if (!scan_one(thr_id, header, high, &hit) || !scan_one(thr_id, header, high - 1, &miss) || !hit || miss) {
		gpulog(LOG_ERR, thr_id, "verthash self-test FAILED: GPU, vector (hit %d, below %d)",
			(int) hit, (int) miss);
		return false;
	}
	return true;
}

/* full GPU hashes of nonces 0..CMP_NONCES-1 of a synthetic header (byte i = 3i + 1) */
static bool check_hashes(int thr_id)
{
	uint8_t header[80];
	for (int i = 0; i < 80; i++)
		header[i] = (uint8_t) (3 * i + 1);
	uint32_t _ALIGN(64) data[20];
	memcpy(data, header, 80);

	uint32_t out[4], total = 0;
	verthash_setBlock(thr_id, data);
	verthash_scan(thr_id, CMP_NONCES, 0, 0, out, 4, &total);
	const cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		return selftest_cuda_error(thr_id, "verthash", err);

	uint32_t *gpu = (uint32_t *) malloc(CMP_NONCES * 32);
	if (!gpu)
		return false;
	verthash_read_hashes(thr_id, CMP_NONCES, gpu);
	int bad = 0;
	for (uint32_t n = 0; n < CMP_NONCES; n++) {
		uint8_t cpu[32];
		le32enc(header + 76, n);
		verthash_hash(cpu, header);
		if (memcmp(cpu, gpu + 8 * n, 32))
			bad++;
	}
	free(gpu);
	if (bad)
		gpulog(LOG_ERR, thr_id, "verthash self-test FAILED: GPU, %d of %d hashes differ", bad, CMP_NONCES);
	return !bad;
}

bool verthash_device_selftest(int thr_id)
{
	static bool tested[MAX_GPUS] = { 0 };
	const int dev_id = device_map[thr_id];
	if (tested[dev_id]) return true;
	tested[dev_id] = true;

	const bool passed = check_vector(thr_id) && check_hashes(thr_id);
	return selftest_gate(thr_id, "verthash", passed);
}
