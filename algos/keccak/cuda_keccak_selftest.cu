// SPDX-License-Identifier: GPL-3.0-or-later
/*
 * Init-time self-test of the keccak mining kernel, once per GPU, fail-closed
 * (cuda/selftest_gate.cuh). For each vector the kernel scans only its nonce:
 * a target equal to the top 64 hash bits must report it, one below must not.
 */
#include <string.h>

#include <miner.h>
#include <cuda_helper.h>
#include "cuda/selftest_gate.cuh"

extern "C" void keccak256_hash(void *state, const void *input);
extern void keccak256_setBlock_80(uint64_t *endiandata);
extern void keccak256_setOutput(int thr_id);
extern void keccak256_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNonce, uint32_t* resNonces, const uint2 highTarget);

struct keccak_kat {
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

/* digests from an independent Keccak-256 implementation */
static bool keccak_kats(struct keccak_kat k[2])
{
	// the --test-vectors vector: genesis header, nonce 15c6eb93
	bool ok = hex_to_bin(k[0].header,
		"0100000000000000000000000000000000000000000000000000000000000000"
		"000000003ba3edfd7a7b12b27ac72c3e67768f617fc81bc3888a51323a9fb8aa"
		"4b1e5e4a29ab5f49ffff001d15c6eb93", 80);
	ok = ok && hex_to_bin(k[0].hash,
		"1e7a7cd7824dcc74a3efe99681117ad410b561654e975464acc35b0500000000", 32);
	// synthetic header, byte i = i
	for (int i = 0; i < 80; i++) k[1].header[i] = (uint8_t) i;
	ok = ok && hex_to_bin(k[1].hash,
		"f0fe5c66fa31e6089ce5553a1bee59a71251a9801e1cbcd133353ce8079e085f", 32);
	return ok;
}

/* scan only the vector's nonce against the 64-bit target; true if reported */
static bool keccak_scan_one(int thr_id, const uint8_t header[80], uint64_t target, bool *found)
{
	uint64_t endiandata[10];
	memcpy(endiandata, header, 80);
	const uint32_t nonce = be32dec(header + 76);
	uint32_t res[2] = { UINT32_MAX, UINT32_MAX };

	keccak256_setBlock_80(endiandata);
	keccak256_setOutput(thr_id);
	keccak256_cpu_hash_80(thr_id, 1, nonce, res, make_uint2((uint32_t) target, (uint32_t) (target >> 32)));

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		gpulog(LOG_ERR, thr_id, "keccak self-test: %s", cudaGetErrorString(err));
		if (err == cudaErrorMemoryAllocation)
			selftest_cuda_fault(); // could not run, not a wrong hash
		return false;
	}
	*found = (res[0] == nonce || res[1] == nonce);
	return true;
}

/* call after keccak256_cpu_init(), before mining */
bool keccak256_device_selftest(int thr_id)
{
	static bool tested[MAX_GPUS] = { 0 };
	const int dev_id = device_map[thr_id];
	if (tested[dev_id]) return true;
	tested[dev_id] = true;

	struct keccak_kat k[2];
	bool passed = keccak_kats(k);
	for (int v = 0; passed && v < 2; v++) {
		uint8_t cpu[32];
		keccak256_hash(cpu, k[v].header);
		if (memcmp(cpu, k[v].hash, 32)) {
			gpulog(LOG_ERR, thr_id, "keccak self-test FAILED: CPU reference, vector %d", v + 1);
			passed = false;
			break;
		}
		uint64_t top;
		memcpy(&top, k[v].hash + 24, 8);
		bool hit = false, miss = false;
		if (!keccak_scan_one(thr_id, k[v].header, top, &hit) ||
		    !keccak_scan_one(thr_id, k[v].header, top - 1, &miss) || !hit || miss) {
			gpulog(LOG_ERR, thr_id, "keccak self-test FAILED: GPU, vector %d (hit %d, below %d)",
				v + 1, (int) hit, (int) miss);
			passed = false;
		}
	}
	return selftest_gate(thr_id, "keccak", passed);
}
