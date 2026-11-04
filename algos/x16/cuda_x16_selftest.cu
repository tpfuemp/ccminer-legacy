// SPDX-License-Identifier: GPL-3.0-or-later
/*
 * Init-time self-test of the x16 chains, once per GPU, fail-closed
 * (cuda/selftest_gate.cuh). 16 headers, header byte 11 picks the first block of
 * the x16r order, so every 80-byte kernel runs first once; each header under x16r,
 * x16s, x16rv2 and x16rt. Each scan covers one nonce: a target equal to the top 64 bits
 * of the CPU hash must report it, one below must not.
 */
#include <string.h>

#include <miner.h>
#include <cuda_helper.h>
#include "cuda/selftest_gate.cuh"
#include "x16.h"

static bool scan_one(int thr_id, int variant, const uint8_t *header, uint64_t high, bool *found)
{
	uint32_t _ALIGN(64) data[20];
	uint32_t _ALIGN(64) target[8];
	memcpy(data, header, 80);
	const uint32_t nonce = be32dec(header + 76);

	// words 0..5 open, so the top 64 bits decide
	memset(target, 0xff, 24);
	target[6] = (uint32_t) high;
	target[7] = (uint32_t) (high >> 32);

	uint32_t out[8], total = 0;
	x16_setBlock(thr_id, variant, data, target);
	const uint32_t count = x16_scan(thr_id, 1, nonce, out, 8, &total);
	if (cudaGetLastError() != cudaSuccess)
		return selftest_cuda_fault();
	*found = (count == 1 && total == 1 && out[0] == nonce);
	return true;
}

bool x16_device_selftest(int thr_id)
{
	static bool tested[MAX_GPUS] = { 0 };
	const int dev_id = device_map[thr_id];
	if (tested[dev_id]) return true;
	tested[dev_id] = true;

	static const char *const names[X16_VARIANTS] = { "x16r", "x16s", "x16rv2", "x16rt" };
	bool passed = true;
	for (int first = 0; passed && first < 16; first++) {
		uint8_t header[80];
		for (int i = 0; i < 80; i++)
			header[i] = (uint8_t) (i * 29 + first * 7 + 3);
		header[11] = (uint8_t) ((first << 4) | (header[11] & 0x0F));
		for (int v = 0; passed && v < X16_VARIANTS; v++) {
			uint8_t order[16], cpu[32];
			x16_order(v, header, order);
			if ((v == X16_R || v == X16_RV2) && order[0] != first) {
				gpulog(LOG_ERR, thr_id, "x16 self-test FAILED: order, first %d", first);
				passed = false;
				break;
			}
			x16_hash(v, cpu, header);
			const uint64_t high = le32dec(cpu + 24) | ((uint64_t) le32dec(cpu + 28) << 32);
			bool hit = false, miss = true;
			if (!scan_one(thr_id, v, header, high, &hit) ||
			    !scan_one(thr_id, v, header, high - 1, &miss) || !hit || miss) {
				gpulog(LOG_ERR, thr_id, "%s self-test FAILED: GPU, first block %d (hit %d, below %d)",
					names[v], first, (int) hit, (int) miss);
				passed = false;
			}
		}
	}
	return selftest_gate(thr_id, "x16", passed);
}
