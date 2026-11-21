// SPDX-License-Identifier: GPL-3.0-or-later
/*
 * RinHash (Rincoin): BLAKE3-256 of the 80-byte header, Argon2d of that digest
 * (salt "RinCoinSalt", t_cost 2, m_cost 64 KiB, 1 lane, v1.3), then SHA3-256.
 * Three kernels per launch, after ccminer-tpfuemp: rin_initialize (BLAKE3, the
 * Argon2 pre-hash H0, blocks 0 and 1), argon2_fill (the other 62 blocks, 32
 * threads per nonce), rin_finalize (Blake2b-long tag, SHA3-256, target).
 * Every candidate is checked on the CPU (rinhash_cpu.cpp).
 */
#include <stdio.h>
#include <memory.h>

#include "miner.h"
#include "cuda_helper.h"
#include "cuda/blake3_device.cuh"
#include "cuda/selftest_gate.cuh"
#include "argon2_device.cuh"

#define RIN_LANES		1u
#define RIN_PASSES		2u
#define RIN_MCOST		64u				/* KiB = blocks */
#define RIN_SEGMENT_BLOCKS	(RIN_MCOST / (4u * RIN_LANES))	/* 16 */
#define RIN_JOBS_PER_BLOCK	16u		/* nonces per CUDA block in the end kernels */
#define RIN_MAX_JOBS		65520u		/* gridDim.z limit, a multiple of 16 */
#define MAX_RESULTS		16

extern "C" void rinhash_hash(void *output, const void *input);

__constant__ static uint32_t c_header[20];	/* header bytes, nonce word replaced */
__constant__ static uint32_t c_target[8];

static __constant__ uint2 c_rc[24] = {
	{ 0x00000001, 0x00000000 }, { 0x00008082, 0x00000000 }, { 0x0000808a, 0x80000000 }, { 0x80008000, 0x80000000 },
	{ 0x0000808b, 0x00000000 }, { 0x80000001, 0x00000000 }, { 0x80008081, 0x80000000 }, { 0x00008009, 0x80000000 },
	{ 0x0000008a, 0x00000000 }, { 0x00000088, 0x00000000 }, { 0x80008009, 0x00000000 }, { 0x8000000a, 0x00000000 },
	{ 0x8000808b, 0x00000000 }, { 0x0000008b, 0x80000000 }, { 0x00008089, 0x80000000 }, { 0x00008003, 0x80000000 },
	{ 0x00008002, 0x80000000 }, { 0x00000080, 0x80000000 }, { 0x0000800a, 0x00000000 }, { 0x8000000a, 0x80000000 },
	{ 0x80008081, 0x80000000 }, { 0x00008080, 0x80000000 }, { 0x80000001, 0x00000000 }, { 0x80008008, 0x80000000 }
};

/* Keccak-f[1600], as verthash */
__device__ __forceinline__ static void keccakf(uint2 st[25])
{
	uint2 t[5], u[5], v, w;

	#pragma unroll 1
	for (int r = 0; r < 24; r++) {
		#pragma unroll
		for (int i = 0; i < 5; i++)
			t[i] = st[i] ^ st[i + 5] ^ st[i + 10] ^ st[i + 15] ^ st[i + 20];
		#pragma unroll
		for (int i = 0; i < 5; i++)
			u[i] = ROL2(t[(i + 1) % 5], 1) ^ t[(i + 4) % 5];
		#pragma unroll
		for (int i = 0; i < 25; i++)
			st[i] ^= u[i % 5];

		v = st[1];
		st[1]  = ROL2(st[6], 44);
		st[6]  = ROL2(st[9], 20);
		st[9]  = ROL2(st[22], 61);
		st[22] = ROL2(st[14], 39);
		st[14] = ROL2(st[20], 18);
		st[20] = ROL2(st[2], 62);
		st[2]  = ROL2(st[12], 43);
		st[12] = ROL2(st[13], 25);
		st[13] = ROL2(st[19], 8);
		st[19] = ROL2(st[23], 56);
		st[23] = ROL2(st[15], 41);
		st[15] = ROL2(st[4], 27);
		st[4]  = ROL2(st[24], 14);
		st[24] = ROL2(st[21], 2);
		st[21] = ROL2(st[8], 55);
		st[8]  = ROL2(st[16], 45);
		st[16] = ROL2(st[5], 36);
		st[5]  = ROL2(st[3], 28);
		st[3]  = ROL2(st[18], 21);
		st[18] = ROL2(st[17], 15);
		st[17] = ROL2(st[11], 10);
		st[11] = ROL2(st[7], 6);
		st[7]  = ROL2(st[10], 3);
		st[10] = ROL2(v, 1);

		#pragma unroll
		for (int j = 0; j < 25; j += 5) {
			v = st[j]; w = st[j + 1];
			st[j]     ^= ~w & st[j + 2];
			st[j + 1] ^= ~st[j + 2] & st[j + 3];
			st[j + 2] ^= ~st[j + 3] & st[j + 4];
			st[j + 3] ^= ~st[j + 4] & v;
			st[j + 4] ^= ~v & w;
		}
		st[0] ^= c_rc[r];
	}
}

/* SHA3-256 (FIPS 202) of 32 bytes given as 8 little-endian words */
__device__ __forceinline__ static void sha3_256_32(const uint32_t in[8], uint32_t out[8])
{
	uint2 st[25];
	#pragma unroll
	for (int i = 0; i < 25; i++)
		st[i] = make_uint2(0, 0);
	#pragma unroll
	for (int i = 0; i < 4; i++)
		st[i] = make_uint2(in[2 * i], in[2 * i + 1]);
	st[4].x = 0x06;			/* byte 32: domain padding */
	st[16].y = 0x80000000u;		/* byte 135: last bit of the rate */
	keccakf(st);
	#pragma unroll
	for (int i = 0; i < 4; i++) {
		out[2 * i] = st[i].x;
		out[2 * i + 1] = st[i].y;
	}
}

/* H0 = Blake2b-512 of the Argon2 pre-hash input (83 bytes, one block):
 * LE32 lanes, outlen, m_cost, t_cost, version, type, pwdlen | pwd (32) |
 * LE32 saltlen | salt (11) | LE32 secretlen 0 | LE32 adlen 0.
 * Leaves H0 in buffer words 1..16, the rest zero, as a2_fill_first_block wants. */
__device__ static void rin_initial_hash(uint32_t nonce, uint32_t buffer[32])
{
	uint32_t hdr[20];
	#pragma unroll
	for (int i = 0; i < 19; i++)
		hdr[i] = c_header[i];
	hdr[19] = nonce;

	uint8_t pwd[32];
	blake3_256((const uint8_t *)hdr, 80, pwd);

	uint8_t *p = (uint8_t *)buffer;
	#pragma unroll
	for (int i = 0; i < 32; i++)
		buffer[i] = 0;
	buffer[0] = RIN_LANES;
	buffer[1] = 32;			/* tag length */
	buffer[2] = RIN_MCOST;
	buffer[3] = RIN_PASSES;
	buffer[4] = ARGON2_VERSION_13;
	buffer[5] = ARGON2_D;
	buffer[6] = 32;			/* password length */
	#pragma unroll
	for (int i = 0; i < 32; i++)
		p[28 + i] = pwd[i];
	buffer[15] = 11;		/* salt length, byte 60 */
	p[64] = 'R'; p[65] = 'i'; p[66] = 'n'; p[67] = 'C'; p[68] = 'o'; p[69] = 'i';
	p[70] = 'n'; p[71] = 'S'; p[72] = 'a'; p[73] = 'l'; p[74] = 't';

	uint64x8 state;
	state.s0 = a2_blake2b_init64[0]; state.s1 = a2_blake2b_init64[1];
	state.s2 = a2_blake2b_init64[2]; state.s3 = a2_blake2b_init64[3];
	state.s4 = a2_blake2b_init64[4]; state.s5 = a2_blake2b_init64[5];
	state.s6 = a2_blake2b_init64[6]; state.s7 = a2_blake2b_init64[7];
	a2_blake2b_compress_1w(&state, (const uint64_t *)buffer, 1, true, 83);

	#pragma unroll
	for (int i = 0; i < 32; i++)
		buffer[i] = 0;
	uint64_t *b64 = (uint64_t *)&buffer[1];
	b64[0] = state.s0; b64[1] = state.s1; b64[2] = state.s2; b64[3] = state.s3;
	b64[4] = state.s4; b64[5] = state.s5; b64[6] = state.s6; b64[7] = state.s7;
}

/* blocks 0 and 1 of each nonce: threadIdx.x is the block, threadIdx.y the nonce */
__global__ void rin_initialize(block_g *memory, uint32_t start_nonce)
{
	uint32_t buffer[32];
	rin_initial_hash(start_nonce + blockIdx.x * blockDim.y + threadIdx.y, buffer);
	a2_fill_first_block(memory, buffer, RIN_LANES, RIN_MCOST);
}

/* The tag: Blake2b-long(32) of block 0 (argon2_fill leaves the final block
 * there) = Blake2b-256(LE32(32) || block), 1028 bytes: eight full compressions
 * and a 4-byte tail, four threads per nonce. Then SHA3-256 and the target.
 * Nonces past span were filled to round the launch up: they never report.
 * digests (optional): every nonce's hash, for the self-test. */
__global__ void rin_finalize(block_g *memory, uint32_t start_nonce, uint32_t span,
	uint32_t *resNonce, uint32_t *digests)
{
	extern __shared__ uint32_t s_buf[];
	uint32_t *input = &s_buf[threadIdx.y * 258];
	uint64_t *input_64 = (uint64_t *)input;

	const uint32_t idx = threadIdx.x;
	const uint32_t job = blockIdx.x * blockDim.y + threadIdx.y;
	const uint32_t *src = (const uint32_t *)(memory + (size_t)job * RIN_MCOST);

	for (uint32_t i = 0; i < 64; i++)
		input[1 + idx + i * 4] = src[idx + i * 4];
	input[0] = 32;
	WARP_SYNC();

	partialState state;
	state.a = a2_blake2b_init32[idx];
	state.b = a2_blake2b_init32[idx + 4];
	for (uint32_t step = 1; step <= 8; step++)
		a2_blake2b_compress_4w(&state, &input_64[(step - 1) * 16], step, idx);

	/* the 4-byte tail goes to word 0 of a cleared buffer */
	const uint32_t tail = input[256];
	WARP_SYNC();
	for (uint32_t i = 0; i < 8; i++)
		input[idx + i * 4] = 0;
	input[0] = tail;
	WARP_SYNC();
	a2_blake2b_compress_4w(&state, &input_64[0], 9, idx, true, 4);

	WARP_SYNC();
	input_64[idx] = state.a;	/* tag = state words 0..3 */
	WARP_SYNC();

	if (idx != 0)
		return;

	uint32_t hash[8];
	sha3_256_32(input, hash);

	if (digests)
		for (int i = 0; i < 8; i++)
			digests[job * 8 + i] = hash[i];

	if (job >= span)
		return;
	bool meets = true;
	for (int i = 7; i >= 0; i--) {
		if (hash[i] != c_target[i]) {
			meets = hash[i] < c_target[i];
			break;
		}
	}
	if (meets) {
		const uint32_t pos = atomicInc(&resNonce[0], UINT32_MAX) + 1;
		if (pos < MAX_RESULTS)
			resNonce[pos] = start_nonce + job;
	}
}

/* ---- host ---- */

static block_g *d_memory[MAX_GPUS];
static uint32_t *d_resNonce[MAX_GPUS];
static uint32_t *h_resNonce[MAX_GPUS];
static uint32_t *d_digests[MAX_GPUS];
static uint32_t max_jobs[MAX_GPUS];

/* Hash [start, start + count) under the header in c_header (count <= max_jobs,
 * rounded up to whole 16-nonce blocks); returns the candidate count, the slots
 * in res. digests: device buffer for every nonce's hash, or NULL. */
static uint32_t rin_scan(int thr_id, uint32_t start, uint32_t count, uint32_t *res, uint32_t *digests)
{
	const uint32_t jobs = (count + RIN_JOBS_PER_BLOCK - 1) / RIN_JOBS_PER_BLOCK * RIN_JOBS_PER_BLOCK;
	const uint32_t blocks = jobs / RIN_JOBS_PER_BLOCK;

	cudaMemset(d_resNonce[thr_id], 0, sizeof(uint32_t));
	rin_initialize <<<blocks, dim3(2 * RIN_LANES, RIN_JOBS_PER_BLOCK)>>> (d_memory[thr_id], start);
	argon2_fill <<<dim3(1, 1, jobs), dim3(THREADS_PER_LANE, RIN_LANES, 1),
		RIN_LANES * ARGON2_SHARED_BLOCKS_PER_LANE * sizeof(block_g)>>> (
		d_memory[thr_id], RIN_PASSES, RIN_LANES, RIN_SEGMENT_BLOCKS, ARGON2_VERSION_13, ARGON2_D);
	rin_finalize <<<blocks, dim3(4, RIN_JOBS_PER_BLOCK), RIN_JOBS_PER_BLOCK * 258 * sizeof(uint32_t)>>> (
		d_memory[thr_id], start, count, d_resNonce[thr_id], digests);
	cudaMemcpy(res, d_resNonce[thr_id], MAX_RESULTS * sizeof(uint32_t), cudaMemcpyDeviceToHost);
	return res[0];
}

static void rin_set_block(const uint32_t *pdata, const uint32_t *ptarget)
{
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_header, pdata, 80, 0, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_target, ptarget, 32, 0, cudaMemcpyHostToDevice));
}

/* ---- init-time self-test, fail-closed (cuda/selftest_gate.cuh) ---- */

/* Rincoin blocks 800000 and 828000: header (data words = header bytes) and the
 * RinHash digest (the block hash reversed) */
static const char *rin_block_hdr[2] = {
	"00000020ecbdc1deca7adee6b25b4995d75f1cb05eb3628fb7c11bda9d2c201d03000000"
	"b6cef4d970ee5733a04b6e7daab497f0cfac881c6e80e0dee92b7d85ca0afc649453e56a"
	"ca8c031da69f1140",
	"0000002056c562d56c8c0a923ac7211340607d37b2c3f8ccbde02c3cd3faa65102000000"
	"387bbbcf7440b0d6f158268124a36bd8c62aeae88c00d8b1670f52ae495d8088f806016b"
	"b464061d3f970740"
};
static const char *rin_block_pow[2] = {
	"96b334f6fc423b00afe0814c85ab76fe36ac44848b12cf69b107e72d02000000",
	"de211e255aca85200639fd297b6b46adb6156615fc2e6031280eb9d502000000"
};

static void to_hex(char *hex, const uint8_t *bin, int len)
{
	for (int i = 0; i < len; i++)
		sprintf(hex + 2 * i, "%02x", bin[i]);
	hex[2 * len] = 0;
}

/* the device digests of nonces [start, start + n), n <= 64 */
static bool rin_selftest_device(int thr_id, const uint32_t *pdata, uint32_t start, uint32_t n, uint32_t *out)
{
	uint32_t target[8], res[MAX_RESULTS];
	memset(target, 0, sizeof(target));
	rin_set_block(pdata, target);
	rin_scan(thr_id, start, n, res, d_digests[thr_id]);
	cudaError_t err = cudaDeviceSynchronize();
	if (err == cudaSuccess)
		err = cudaMemcpy(out, d_digests[thr_id], n * 32, cudaMemcpyDeviceToHost);
	return err == cudaSuccess ? true : selftest_cuda_error(thr_id, "rinhash", err);
}

static bool rinhash_device_selftest(int thr_id)
{
	static bool tested[MAX_GPUS] = { 0 };
	const int dev_id = device_map[thr_id];
	if (tested[dev_id]) return true;
	tested[dev_id] = true;

	uint32_t _ALIGN(64) pdata[20], dig[8], dev[64 * 8], cpu[8];
	char hex[65];
	bool cpu_ok = true, dev_ok = true, ran = true;

	/* the CPU reference and the device give both block hashes */
	for (int b = 0; b < 2; b++) {
		hex2bin((uchar *)pdata, rin_block_hdr[b], 80);
		rinhash_hash(dig, pdata);
		to_hex(hex, (const uint8_t *)dig, 32);
		cpu_ok &= strcmp(hex, rin_block_pow[b]) == 0;
		ran &= rin_selftest_device(thr_id, pdata, pdata[19], 1, dev);
		dev_ok &= ran && memcmp(dev, dig, 32) == 0;
	}

	/* 64 nonces from an odd start: device == CPU for each */
	const uint32_t start = 0x7a3b1c5du;
	ran &= rin_selftest_device(thr_id, pdata, start, 64, dev);
	bool span_ok = ran;
	for (uint32_t i = 0; i < 64 && span_ok; i++) {
		pdata[19] = start + i;
		rinhash_hash(cpu, pdata);
		span_ok = memcmp(cpu, &dev[i * 8], 32) == 0;
	}

	/* the scan on one nonce: target == hash must hit, one below must miss */
	uint32_t target[8], res[MAX_RESULTS];
	pdata[19] = start;
	memcpy(target, dev, 32);
	rin_set_block(pdata, target);
	const bool hit = rin_scan(thr_id, start, 1, res, NULL) == 1 && res[1] == start;
	for (int i = 0; i < 8 && target[i]-- == 0; i++);
	rin_set_block(pdata, target);
	const bool miss = rin_scan(thr_id, start, 1, res, NULL) != 0;
	cudaError_t err = cudaGetLastError();
	const bool scan_ok = err == cudaSuccess ? hit && !miss : selftest_cuda_error(thr_id, "rinhash", err);

	/* a flipped header bit must change the device digest */
	pdata[3] ^= 0x00000008u;
	const bool ran2 = rin_selftest_device(thr_id, pdata, start, 1, &dev[8]);
	const bool neg_ok = ran2 && memcmp(dev, &dev[8], 32) != 0;

	const bool passed = cpu_ok && dev_ok && span_ok && scan_ok && neg_ok;
	if (!passed)
		gpulog(LOG_ERR, thr_id, "rinhash self-test FAILED (cpu %d dev %d span %d scan %d/%d neg %d)",
			(int)cpu_ok, (int)dev_ok, (int)span_ok, (int)hit, (int)miss, (int)neg_ok);
	return selftest_gate(thr_id, "rinhash", passed);
}

/* ---- scan ---- */

static bool init[MAX_GPUS] = { 0 };

extern "C" int scanhash_rinhash(int thr_id, struct work *work, uint32_t max_nonce,
	unsigned long *hashes_done)
{
	uint32_t _ALIGN(64) hdr[20];
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];

	if (opt_benchmark)
		ptarget[7] = 0x00ff;

	// 256 nonces (16 MB) per multiprocessor
	const int dev_id = device_map[thr_id];
	uint32_t throughput = cuda_default_throughput(thr_id, (uint32_t) max((int) device_mpcount[dev_id], 1) << 8);
	throughput = min(max(throughput & ~(RIN_JOBS_PER_BLOCK - 1), RIN_JOBS_PER_BLOCK), RIN_MAX_JOBS);

	if (!init[thr_id]) {
		cudaSetDevice(dev_id);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
			CUDA_LOG_ERROR();
		}
		gpulog(LOG_INFO, thr_id, "Intensity set to %g, %u cuda threads (%u MB)",
			throughput2intensity(throughput), throughput, throughput / 16);

		cuda_get_arch(thr_id);

		max_jobs[thr_id] = max(throughput, 64u);
		CUDA_CALL_OR_RET_X(cudaMalloc(&d_memory[thr_id], (size_t)max_jobs[thr_id] * RIN_MCOST * sizeof(block_g)), -1);
		CUDA_CALL_OR_RET_X(cudaMalloc(&d_digests[thr_id], (size_t)max_jobs[thr_id] * 32), -1);
		CUDA_CALL_OR_RET_X(cudaMalloc(&d_resNonce[thr_id], MAX_RESULTS * sizeof(uint32_t)), -1);
		CUDA_CALL_OR_RET_X(cudaMallocHost(&h_resNonce[thr_id], MAX_RESULTS * sizeof(uint32_t)), -1);

		rinhash_device_selftest(thr_id);
		init[thr_id] = true;
	}

	memcpy(hdr, pdata, 80);
	rin_set_block(pdata, ptarget);

	do {
		// the last launch covers only the rest of the range
		const uint32_t batch = (uint32_t) min((uint64_t) throughput, (uint64_t) max_nonce - pdata[19]);
		if (!batch)
			break;

		uint32_t *res = h_resNonce[thr_id];
		uint32_t count = rin_scan(thr_id, pdata[19], batch, res, NULL);
		*hashes_done = pdata[19] - first_nonce + batch;

		if (count) {
			if (count >= MAX_RESULTS) {
				gpulog(LOG_WARNING, thr_id, "candidates flood: %u", count);
				count = MAX_RESULTS - 1;
			}
			// CPU-check every candidate, keep the two lowest valid ones
			uint32_t valid[2];
			int nvalid = 0;
			for (uint32_t n = 1; n <= count; n++) {
				uint32_t _ALIGN(64) vhash[8];
				hdr[19] = res[n];
				rinhash_hash(vhash, hdr);
				if (!fulltest(vhash, ptarget)) {
					gpu_increment_reject(thr_id);
					if (!opt_quiet)
						gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU!", res[n]);
					continue;
				}
				if (nvalid < 2)
					valid[nvalid++] = res[n];
				else if (res[n] < valid[1])
					valid[1] = res[n];
				if (nvalid == 2 && valid[1] < valid[0])
					xchg(valid[0], valid[1]);
			}

			if (nvalid) {
				uint32_t _ALIGN(64) vhash[8];
				work->valid_nonces = nvalid;
				for (int i = 0; i < nvalid; i++) {
					work->nonces[i] = valid[i];
					hdr[19] = valid[i];
					rinhash_hash(vhash, hdr);
					if (i == 0)
						work_set_target_ratio(work, vhash);
					else
						bn_set_target_ratio(work, vhash, 1);
				}
				// resume after the last reported nonce
				pdata[19] = valid[nvalid - 1] + 1;
				return nvalid;
			}
		}

		if ((uint64_t) batch + pdata[19] >= max_nonce) {
			pdata[19] = max_nonce;
			break;
		}
		pdata[19] += batch;

	} while (!work_restart[thr_id].restart);

	*hashes_done = pdata[19] - first_nonce;
	return 0;
}

extern "C" void free_rinhash(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaDeviceSynchronize();
	cudaFreeHost(h_resNonce[thr_id]);
	cudaFree(d_resNonce[thr_id]);
	cudaFree(d_digests[thr_id]);
	cudaFree(d_memory[thr_id]);
	init[thr_id] = false;
	cudaDeviceSynchronize();
}
