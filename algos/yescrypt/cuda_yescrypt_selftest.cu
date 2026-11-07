// SPDX-License-Identifier: GPL-3.0-or-later
/*
 * Init-time self-test of the yescrypt kernels, once per GPU and variant, fail-closed
 * (cuda/selftest_gate.cuh). Each scan covers one nonce: a target equal to the hash top
 * 64 bits must report it, one below must not. Changed yescrypt parameters have no
 * known answers: the CPU hash of the synthetic header is the reference then.
 */
#include <string.h>

#include <miner.h>
#include <cuda_helper.h>
#include "cuda/selftest_gate.cuh"
#include "yescrypt_algo.h"

struct yescrypt_kat {
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

/* per variant: the --test-vectors nonce (genesis header, time byte 39) and the hashes of
 * that header and of the synthetic one (byte i = i) */
static const struct { const char *nonce, *mined, *synth; } kats[YESCRYPT_VARIANTS] = {
	{ "0000011d", "134bc18e0ed45d53de93c9a0efb5488d5831adf08ad5ed33eeb55003c58d0c00",
	  "ae955413ae874374e49bcce4d5476fde4903b10a1402377f8ed70adf5968d9f7" },
	{ "000000ed", "ad8f50af99a0c8f7e6a9a778023ad89a5991832d70ab0b4de86f81095b4adb00",
	  "727148f2c0e778655717a92db8831aa1b241aefd7d1a51b4c7294c264ae82162" },
	{ "00000019", "a1fb72c76f30b71d3459051c617ba33690a0d86a65fc75936cb4a9646067b500",
	  "03695508103955e53e70aa9914f5635c26e4a7321f25e2d800b3ca6695d7dd68" },
	{ "00000190", "40114b5fa2687aba1f23f67af89f22327e063f1cea609e394eb8c79c7b3bbd00",
	  "393f06241acca96647b039c3562dbcc35b673664337eb14861e68b2f42c6a437" },
	{ "0000019f", "6c536e20b8e02559082f0b9fa2b43f7763f1919acc847e4244f387b881727e00",
	  "9dd06c40016a34ee554907305b11541555198d76180a3e781814f042fa0744cc" },
	{ "00000086", "7348bc946135a11ed64908632238c775a47bf834c99db9eb6677202a34d5de00",
	  "b9c8ba8527fb47306ff753329a8c25c421450a24cec7df77b470f8ecc31d49ba" },
};

static bool scan_one(int thr_id, const struct yescrypt_kat *k, uint64_t high, bool *found)
{
	uint32_t _ALIGN(64) pdata[20];
	uint32_t _ALIGN(64) target[8];
	// work data words, as the stratum code fills them
	for (int i = 0; i < 20; i++)
		pdata[i] = be32dec(k->header + 4 * i);
	const uint32_t nonce = pdata[19];

	// words 0..5 open, so the top 64 bits decide
	memset(target, 0xff, 24);
	target[6] = (uint32_t) high;
	target[7] = (uint32_t) (high >> 32);

	uint32_t out[8], total = 0;
	yescrypt_setTarget(thr_id, pdata, target);
	const uint32_t count = yescrypt_scan(thr_id, 1, nonce, out, 8, &total);
	const cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		return selftest_cuda_error(thr_id, "yescrypt", err);
	*found = (count == 1 && total == 1 && out[0] == nonce);
	return true;
}

bool yescrypt_device_selftest(int thr_id, int v)
{
	static bool tested[MAX_GPUS][YESCRYPT_VARIANTS] = { { 0 } };
	const int dev_id = device_map[thr_id];
	if (tested[dev_id][v]) return true;
	tested[dev_id][v] = true;

	const struct yescrypt_variant y = yescrypt_params(v);
	const bool known = yescrypt_is_default(v);
	struct yescrypt_kat k[2];
	bool passed = hex_to_bin(k[0].header,
			"0100000000000000000000000000000000000000000000000000000000000000"
			"000000003ba3edfd7a7b12b27ac72c3e67768f617fc81bc3888a51323a9fb8aa"
			"4b1e5e4a39ab5f49ffff001d", 76) &&
		hex_to_bin(k[0].header + 76, kats[v].nonce, 4) &&
		hex_to_bin(k[0].hash, kats[v].mined, 32) &&
		hex_to_bin(k[1].hash, kats[v].synth, 32);
	for (int i = 0; i < 80; i++) k[1].header[i] = (uint8_t) i;

	// changed parameters: synthetic header only, CPU hash as the reference
	const int first = known ? 0 : 1;
	if (!known) {
		yescrypt_hash_params(&y, k[1].hash, k[1].header);
		gpulog(LOG_INFO, thr_id, "%s: changed parameters, self-test against the CPU hash", y.name);
	}

	for (int t = first; passed && t < 2; t++) {
		uint8_t cpu[32];
		yescrypt_hash_params(&y, cpu, k[t].header);
		if (memcmp(cpu, k[t].hash, 32)) {
			gpulog(LOG_ERR, thr_id, "%s self-test FAILED: CPU reference, vector %d", y.name, t + 1);
			passed = false;
			break;
		}
		const uint64_t high = le32dec(k[t].hash + 24) | ((uint64_t) le32dec(k[t].hash + 28) << 32);
		bool hit = false, miss = true;
		if (!scan_one(thr_id, &k[t], high, &hit) ||
		    !scan_one(thr_id, &k[t], high - 1, &miss) || !hit || miss) {
			gpulog(LOG_ERR, thr_id, "%s self-test FAILED: GPU, vector %d (hit %d, below %d)",
				y.name, t + 1, (int) hit, (int) miss);
			passed = false;
		}
	}
	return selftest_gate(thr_id, y.name, passed);
}
