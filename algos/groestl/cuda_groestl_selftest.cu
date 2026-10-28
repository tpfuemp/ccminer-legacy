// SPDX-License-Identifier: GPL-3.0-or-later
/*
 * Init-time self-tests of the groestl and myr-gr kernels, once per GPU,
 * fail-closed (cuda/selftest_gate.cuh). Each scan covers one nonce: a target
 * equal to the hash must report it, one below must not.
 */
#include <string.h>

#include <miner.h>
#include <cuda_helper.h>
#include "cuda/selftest_gate.cuh"
#include "cuda_groestlcoin.h"

void groestlhash(void *state, const void *input);
void myriadhash(void *state, const void *input);
void myriadgroestl_cpu_setBlock(int thr_id, void *data, uint32_t *target);
void myriadgroestl_cpu_hash(int thr_id, uint32_t threads, uint32_t startNonce, uint32_t *resNonces);

struct groestl_kat {
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

/* the --test-vectors vector and a synthetic header (byte i = i); digests from
 * an independent Groestl-512 implementation (spec KAT checked) */
static bool load_kats(struct groestl_kat k[2], const char *header, const char *hash, const char *synth)
{
	bool ok = hex_to_bin(k[0].header, header, 80) && hex_to_bin(k[0].hash, hash, 32);
	for (int i = 0; i < 80; i++) k[1].header[i] = (uint8_t) i;
	return ok && hex_to_bin(k[1].hash, synth, 32);
}

static bool cuda_ok(int thr_id, const char *name)
{
	cudaError_t err = cudaGetLastError();
	if (err == cudaSuccess)
		return true;
	gpulog(LOG_ERR, thr_id, "%s self-test: %s", name, cudaGetErrorString(err));
	if (err == cudaErrorMemoryAllocation)
		selftest_cuda_fault(); // could not run, not a wrong hash
	return false;
}

/* target words of a hash, optionally one below it (256-bit decrement) */
static void hash_target(uint32_t target[8], const uint8_t hash[32], bool below)
{
	for (int i = 0; i < 8; i++)
		target[i] = le32dec(hash + 4 * i);
	for (int i = 0; below && i < 8; i++)
		if (target[i]-- != 0)
			break;
}

static bool groestl_scan_one(int thr_id, const struct groestl_kat *k, bool below, bool *found)
{
	uint32_t _ALIGN(64) data[20];
	uint32_t target[8];
	uint32_t res = UINT32_MAX;
	memcpy(data, k->header, 80);
	hash_target(target, k->hash, below);
	const uint32_t nonce = be32dec(k->header + 76);

	groestlcoin_cpu_setBlock(thr_id, data, target);
	groestlcoin_cpu_hash(thr_id, 1, nonce, &res);
	if (!cuda_ok(thr_id, "groestl"))
		return false;
	*found = (res == nonce);
	return true;
}

static bool myriad_scan_one(int thr_id, const struct groestl_kat *k, bool below, bool *found)
{
	uint32_t _ALIGN(64) data[20];
	uint32_t target[8];
	uint32_t res[2] = { UINT32_MAX, UINT32_MAX };
	memcpy(data, k->header, 80);
	hash_target(target, k->hash, false);
	if (below)
		target[7]--; // the kernel compares words 7 and 6 separately
	const uint32_t nonce = be32dec(k->header + 76);

	myriadgroestl_cpu_setBlock(thr_id, data, target);
	myriadgroestl_cpu_hash(thr_id, 1, nonce, res);
	if (!cuda_ok(thr_id, "myr-gr"))
		return false;
	*found = (res[0] == nonce);
	return true;
}

typedef bool (*scan_fn)(int thr_id, const struct groestl_kat *k, bool below, bool *found);
typedef void (*hash_fn)(void *state, const void *input);

static bool run_selftest(int thr_id, const char *name, struct groestl_kat k[2], hash_fn hash,
	scan_fn scan)
{
	bool passed = true;
	for (int v = 0; passed && v < 2; v++) {
		uint8_t cpu[32];
		hash(cpu, k[v].header);
		if (memcmp(cpu, k[v].hash, 32)) {
			gpulog(LOG_ERR, thr_id, "%s self-test FAILED: CPU reference, vector %d", name, v + 1);
			passed = false;
			break;
		}
		bool hit = false, miss = false;
		if (!scan(thr_id, &k[v], false, &hit) ||
		    !scan(thr_id, &k[v], true, &miss) || !hit || miss) {
			gpulog(LOG_ERR, thr_id, "%s self-test FAILED: GPU, vector %d (hit %d, below %d)",
				name, v + 1, (int) hit, (int) miss);
			passed = false;
		}
	}
	return selftest_gate(thr_id, name, passed);
}

/* call after groestlcoin_cpu_init(), before mining */
bool groestl_device_selftest(int thr_id)
{
	static bool tested[MAX_GPUS] = { 0 };
	const int dev_id = device_map[thr_id];
	if (tested[dev_id]) return true;
	tested[dev_id] = true;

	struct groestl_kat k[2];
	bool loaded = load_kats(k,
		"0100000000000000000000000000000000000000000000000000000000000000"
		"000000003ba3edfd7a7b12b27ac72c3e67768f617fc81bc3888a51323a9fb8aa"
		"4b1e5e4a31ab5f49ffff001d01037018",
		"91f6d3afdc7e5dad8d99bf5b03e19f286df3c09fd81490c9a4a4cb21ad000000",
		"e273974c11312120ced9b3f366ac07f991fc79c87325309de3a28926c48e9165");
	if (!loaded)
		return selftest_gate(thr_id, "groestl", false);
	return run_selftest(thr_id, "groestl", k, groestlhash, groestl_scan_one);
}

/* call after myriadgroestl_cpu_init(), before mining */
bool myriad_device_selftest(int thr_id)
{
	static bool tested[MAX_GPUS] = { 0 };
	const int dev_id = device_map[thr_id];
	if (tested[dev_id]) return true;
	tested[dev_id] = true;

	struct groestl_kat k[2];
	bool loaded = load_kats(k,
		"0100000000000000000000000000000000000000000000000000000000000000"
		"000000003ba3edfd7a7b12b27ac72c3e67768f617fc81bc3888a51323a9fb8aa"
		"4b1e5e4a32ab5f49ffff001d01515b64",
		"afa7c406935ebd74987e665754567dcd513a61e6b5c8f0deb8416d82d2000000",
		"f28596c452bcec2325350f94e909ef885404430394681803ac167499fd717416");
	if (!loaded)
		return selftest_gate(thr_id, "myr-gr", false);
	return run_selftest(thr_id, "myr-gr", k, myriadhash, myriad_scan_one);
}
