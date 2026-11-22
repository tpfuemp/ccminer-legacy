// SPDX-License-Identifier: GPL-3.0-or-later
/*
 * sha512256d (Radiant): SHA-512/256 of the SHA-512/256 of the 80-byte header.
 * The header fits one 128-byte SHA-512 block, so a nonce costs two transforms.
 * The kernel screens the top 64 bits of the hash; every candidate is checked
 * on the CPU (sph SHA-512 started from the SHA-512/256 IV).
 */
#include <stdio.h>
#include <memory.h>

extern "C" {
#include "sph/sph_sha2.h"
}

#include "miner.h"
#include "cuda_helper.h"
#include "cuda/selftest_gate.cuh"

#define TPB		256
#define MAX_RESULTS	16

static const uint64_t H512_256[8] = {
	0x22312194FC2BF72CULL, 0x9F555FA3C84C64C2ULL, 0x2393B86B6F53B151ULL, 0x963877195940EABDULL,
	0x96283EE2A88EFFE3ULL, 0xBE5E1E2553863992ULL, 0x2B0199FC2C85B8AAULL, 0x0EB72DDC81C52CA2ULL
};

static __constant__ uint64_t c_H[8];
static __constant__ uint64_t c_K[80];
static __constant__ uint64_t c_header[10];	/* big-endian message words, w9 without the nonce */

static const uint64_t K512[80] = {
	0x428A2F98D728AE22ULL, 0x7137449123EF65CDULL, 0xB5C0FBCFEC4D3B2FULL, 0xE9B5DBA58189DBBCULL,
	0x3956C25BF348B538ULL, 0x59F111F1B605D019ULL, 0x923F82A4AF194F9BULL, 0xAB1C5ED5DA6D8118ULL,
	0xD807AA98A3030242ULL, 0x12835B0145706FBEULL, 0x243185BE4EE4B28CULL, 0x550C7DC3D5FFB4E2ULL,
	0x72BE5D74F27B896FULL, 0x80DEB1FE3B1696B1ULL, 0x9BDC06A725C71235ULL, 0xC19BF174CF692694ULL,
	0xE49B69C19EF14AD2ULL, 0xEFBE4786384F25E3ULL, 0x0FC19DC68B8CD5B5ULL, 0x240CA1CC77AC9C65ULL,
	0x2DE92C6F592B0275ULL, 0x4A7484AA6EA6E483ULL, 0x5CB0A9DCBD41FBD4ULL, 0x76F988DA831153B5ULL,
	0x983E5152EE66DFABULL, 0xA831C66D2DB43210ULL, 0xB00327C898FB213FULL, 0xBF597FC7BEEF0EE4ULL,
	0xC6E00BF33DA88FC2ULL, 0xD5A79147930AA725ULL, 0x06CA6351E003826FULL, 0x142929670A0E6E70ULL,
	0x27B70A8546D22FFCULL, 0x2E1B21385C26C926ULL, 0x4D2C6DFC5AC42AEDULL, 0x53380D139D95B3DFULL,
	0x650A73548BAF63DEULL, 0x766A0ABB3C77B2A8ULL, 0x81C2C92E47EDAEE6ULL, 0x92722C851482353BULL,
	0xA2BFE8A14CF10364ULL, 0xA81A664BBC423001ULL, 0xC24B8B70D0F89791ULL, 0xC76C51A30654BE30ULL,
	0xD192E819D6EF5218ULL, 0xD69906245565A910ULL, 0xF40E35855771202AULL, 0x106AA07032BBD1B8ULL,
	0x19A4C116B8D2D0C8ULL, 0x1E376C085141AB53ULL, 0x2748774CDF8EEB99ULL, 0x34B0BCB5E19B48A8ULL,
	0x391C0CB3C5C95A63ULL, 0x4ED8AA4AE3418ACBULL, 0x5B9CCA4F7763E373ULL, 0x682E6FF3D6B2B8A3ULL,
	0x748F82EE5DEFB2FCULL, 0x78A5636F43172F60ULL, 0x84C87814A1F0AB72ULL, 0x8CC702081A6439ECULL,
	0x90BEFFFA23631E28ULL, 0xA4506CEBDE82BDE9ULL, 0xBEF9A3F7B2C67915ULL, 0xC67178F2E372532BULL,
	0xCA273ECEEA26619CULL, 0xD186B8C721C0C207ULL, 0xEADA7DD6CDE0EB1EULL, 0xF57D4F7FEE6ED178ULL,
	0x06F067AA72176FBAULL, 0x0A637DC5A2C898A6ULL, 0x113F9804BEF90DAEULL, 0x1B710B35131C471BULL,
	0x28DB77F523047D84ULL, 0x32CAAB7B40C72493ULL, 0x3C9EBE0A15C9BEBCULL, 0x431D67C49C100D4CULL,
	0x4CC5D4BECB3E42B6ULL, 0x597F299CFC657E2AULL, 0x5FCB6FAB3AD6FAECULL, 0x6C44198C4A475817ULL
};

static uint32_t *d_resNonce[MAX_GPUS];
static uint32_t *h_resNonce[MAX_GPUS];

#define S512_BSG0(x)	(ROTR64(x, 28) ^ ROTR64(x, 34) ^ ROTR64(x, 39))
#define S512_BSG1(x)	(ROTR64(x, 14) ^ ROTR64(x, 18) ^ ROTR64(x, 41))
#define S512_SSG0(x)	(ROTR64(x, 1) ^ ROTR64(x, 8) ^ ((x) >> 7))
#define S512_SSG1(x)	(ROTR64(x, 19) ^ ROTR64(x, 61) ^ ((x) >> 6))
#define S512_CH(x, y, z)	(((y ^ z) & x) ^ z)
#define S512_MAJ(x, y, z)	((x & y) | ((x | y) & z))

/* one SHA-512 compression of w (16 words, destroyed) into st */
__device__ __forceinline__ static void sha512_transform(uint64_t w[16], uint64_t st[8])
{
	uint64_t a = st[0], b = st[1], c = st[2], d = st[3];
	uint64_t e = st[4], f = st[5], g = st[6], h = st[7];

	#pragma unroll
	for (int i = 0; i < 80; i++) {
		if (i >= 16)
			w[i & 15] += S512_SSG1(w[(i - 2) & 15]) + w[(i - 7) & 15] + S512_SSG0(w[(i - 15) & 15]);
		const uint64_t t1 = h + S512_BSG1(e) + S512_CH(e, f, g) + c_K[i] + w[i & 15];
		const uint64_t t2 = S512_BSG0(a) + S512_MAJ(a, b, c);
		h = g; g = f; f = e; e = d + t1;
		d = c; c = b; b = a; a = t1 + t2;
	}

	st[0] += a; st[1] += b; st[2] += c; st[3] += d;
	st[4] += e; st[5] += f; st[6] += g; st[7] += h;
}

/* the hash's top 64 bits in fulltest order (((uint64_t *)vhash)[3]) */
__device__ __forceinline__ static uint64_t sha512256d_top(const uint32_t nonce)
{
	uint64_t w[16], st[8];

	#pragma unroll
	for (int i = 0; i < 9; i++)
		w[i] = c_header[i];
	w[9] = c_header[9] | nonce;
	w[10] = 0x8000000000000000ULL;
	#pragma unroll
	for (int i = 11; i < 15; i++)
		w[i] = 0;
	w[15] = 640;
	#pragma unroll
	for (int i = 0; i < 8; i++)
		st[i] = c_H[i];
	sha512_transform(w, st);

	#pragma unroll
	for (int i = 0; i < 4; i++)
		w[i] = st[i];
	w[4] = 0x8000000000000000ULL;
	#pragma unroll
	for (int i = 5; i < 15; i++)
		w[i] = 0;
	w[15] = 256;
	#pragma unroll
	for (int i = 0; i < 8; i++)
		st[i] = c_H[i];
	sha512_transform(w, st);

	return cuda_swab64(st[3]);
}

/* resNonce[0] counts candidates, resNonce[1..MAX_RESULTS-1] hold them */
__global__ __launch_bounds__(TPB)
void sha512256d_gpu_hash(const uint32_t threads, const uint32_t startNonce, uint32_t *resNonce, const uint64_t highTarget)
{
	const uint32_t thread = blockDim.x * blockIdx.x + threadIdx.x;
	if (thread >= threads)
		return;

	const uint32_t nonce = startNonce + thread;
	if (sha512256d_top(nonce) <= highTarget) {
		const uint32_t pos = atomicInc(&resNonce[0], UINT32_MAX) + 1;
		if (pos < MAX_RESULTS)
			resNonce[pos] = nonce;
	}
}

/* the 80 header bytes */
extern "C" void sha512256d_hash(void *output, const void *input)
{
	uint64_t _ALIGN(64) hash[8];
	sph_sha512_context ctx;

	sph_sha512_init(&ctx);
	memcpy(ctx.val, H512_256, sizeof(H512_256));
	sph_sha512(&ctx, input, 80);
	sph_sha512_close(&ctx, hash);

	sph_sha512_init(&ctx);
	memcpy(ctx.val, H512_256, sizeof(H512_256));
	sph_sha512(&ctx, hash, 32);
	sph_sha512_close(&ctx, hash);

	memcpy(output, hash, 32);
}

static void sha512256d_setBlock(const uint32_t *pdata)
{
	uint64_t hdr[10];
	for (int i = 0; i < 9; i++)
		hdr[i] = ((uint64_t) pdata[2 * i] << 32) | pdata[2 * i + 1];
	hdr[9] = (uint64_t) pdata[18] << 32;
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_header, hdr, sizeof(hdr), 0, cudaMemcpyHostToDevice));
}

/* one launch over [start, start + count): the candidate count, the slots in res */
static uint32_t sha512256d_scan(int thr_id, uint32_t start, uint32_t count, uint64_t highTarget, uint32_t *res)
{
	cudaMemset(d_resNonce[thr_id], 0, sizeof(uint32_t));
	sha512256d_gpu_hash <<<(count + TPB - 1) / TPB, TPB>>> (count, start, d_resNonce[thr_id], highTarget);
	cudaMemcpy(res, d_resNonce[thr_id], MAX_RESULTS * sizeof(uint32_t), cudaMemcpyDeviceToHost);
	return res[0];
}

/* ---- init-time self-test, fail-closed (cuda/selftest_gate.cuh) ---- */

/* Radiant blocks 100000 and 480000 and their hashes (the block hash reversed) */
static const char *s512_block_hdr[2] = {
	"000000209d6aa4360ed0fc4e095611c704cacef07f200c6957b73a429101000000000000d4d600dc69aabc685139e535"
	"0177ff6bb8673701c0c622aad52535030b09bd0c4e7b2c6448f5041ae6bd023a",
	"00000020844792b4305b94da1501d27417c41644694eb6ce87a91c3c72010000000000000c11c21d78053515e545bf69"
	"4dbe117ac47b63c4e67536252d8061d0a5b47bc937a6f06a219d011a40419341"
};
static const char *s512_block_pow[2] = {
	"089ccf7c6c19aec88fd825a9171e5ab4b478cf11a8a5f6c37d02000000000000",
	"5aa4d8f2784747b91a34ae9168cb772fbddb3e682d055073c700000000000000"
};

static bool sha512256d_device_selftest(int thr_id)
{
	static bool tested[MAX_GPUS] = { 0 };
	const int dev_id = device_map[thr_id];
	if (tested[dev_id]) return true;
	tested[dev_id] = true;

	uint32_t _ALIGN(64) pdata[20], dig[8];
	uint32_t *res = h_resNonce[thr_id];
	uint8_t hdr[80];
	char hex[65];
	bool cpu_ok = true, scan_ok = true, miss_ok = true, neg_ok = true;

	for (int b = 0; b < 2; b++) {
		/* the CPU hash gives the block hash */
		hex2bin(hdr, s512_block_hdr[b], 80);
		sha512256d_hash(dig, hdr);
		for (int i = 0; i < 32; i++)
			sprintf(hex + 2 * i, "%02x", ((uint8_t *)dig)[i]);
		cpu_ok &= strcmp(hex, s512_block_pow[b]) == 0;

		/* the kernel finds the block nonce at a target equal to its top
		 * 64 bits, not one below */
		for (int i = 0; i < 20; i++)
			pdata[i] = be32dec(hdr + 4 * i);
		const uint64_t top = ((uint64_t) dig[7] << 32) | dig[6];
		sha512256d_setBlock(pdata);
		scan_ok &= sha512256d_scan(thr_id, pdata[19], 1, top, res) == 1 && res[1] == pdata[19];
		miss_ok &= sha512256d_scan(thr_id, pdata[19], 1, top - 1, res) == 0;

		/* a flipped header bit must lose the hit */
		pdata[3] ^= 0x00000100u;
		sha512256d_setBlock(pdata);
		neg_ok &= sha512256d_scan(thr_id, pdata[19], 1, top, res) == 0;
	}

	/* 1024 nonces at a top-byte-0 target: the kernel reports exactly the nonces
	 * the CPU finds (about 4) */
	bool span_ok = true;
	const uint32_t start = 0x5a1d0c3bu;
	const uint32_t n = sha512256d_scan(thr_id, start, 1024, 0x00ffffffffffffffULL, res);
	uint32_t found = 0;
	for (int i = 0; i < 20; i++)
		be32enc(hdr + 4 * i, pdata[i]);
	for (uint32_t i = 0; i < 1024; i++) {
		be32enc(hdr + 76, start + i);
		sha512256d_hash(dig, hdr);
		if (dig[7] <= 0x00ffffffu) {
			bool listed = false;
			for (uint32_t k = 1; k <= n && k < MAX_RESULTS; k++)
				listed |= res[k] == start + i;
			span_ok &= listed;
			found++;
		}
	}
	span_ok &= n == found && found < MAX_RESULTS;

	const cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		return selftest_gate(thr_id, "sha512256d", selftest_cuda_error(thr_id, "sha512256d", err));

	const bool passed = cpu_ok && scan_ok && miss_ok && neg_ok && span_ok;
	if (!passed)
		gpulog(LOG_ERR, thr_id, "sha512256d self-test FAILED (cpu %d scan %d miss %d neg %d span %d)",
			(int)cpu_ok, (int)scan_ok, (int)miss_ok, (int)neg_ok, (int)span_ok);
	return selftest_gate(thr_id, "sha512256d", passed);
}

/* ---- scan ---- */

static bool init[MAX_GPUS] = { 0 };

extern "C" int scanhash_sha512256d(int thr_id, struct work *work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint8_t hdr[80];
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];

	const int dev_id = device_map[thr_id];
	const uint32_t throughput = cuda_default_throughput(thr_id, device_sm[dev_id] < 500 ? 1U << 22 : 1U << 24);

	if (opt_benchmark) {
		// both words the kernel compares
		ptarget[7] = 0x03;
		ptarget[6] = 0xffffffffu;
	}
	const uint64_t highTarget = ((uint64_t) ptarget[7] << 32) | ptarget[6];

	if (!init[thr_id]) {
		cudaSetDevice(dev_id);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
			CUDA_LOG_ERROR();
		}
		gpulog(LOG_INFO, thr_id, "Intensity set to %g, %u cuda threads", throughput2intensity(throughput), throughput);

		cuda_get_arch(thr_id);
		CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_H, H512_256, sizeof(H512_256), 0, cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_K, K512, sizeof(K512), 0, cudaMemcpyHostToDevice));
		CUDA_CALL_OR_RET_X(cudaMalloc(&d_resNonce[thr_id], MAX_RESULTS * sizeof(uint32_t)), -1);
		CUDA_CALL_OR_RET_X(cudaMallocHost(&h_resNonce[thr_id], MAX_RESULTS * sizeof(uint32_t)), -1);

		sha512256d_device_selftest(thr_id);
		init[thr_id] = true;
	}

	for (int i = 0; i < 20; i++)
		be32enc(hdr + 4 * i, pdata[i]);
	sha512256d_setBlock(pdata);

	do {
		// the last launch covers only the rest of the range
		const uint32_t batch = (uint32_t) min((uint64_t) throughput, (uint64_t) max_nonce - pdata[19]);
		if (!batch)
			break;

		uint32_t *res = h_resNonce[thr_id];
		uint32_t count = sha512256d_scan(thr_id, pdata[19], batch, highTarget, res);
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
				be32enc(hdr + 76, res[n]);
				sha512256d_hash(vhash, hdr);
				if (!fulltest(vhash, ptarget)) {
					const uint64_t top = ((uint64_t) vhash[7] << 32) | vhash[6];
					if (top > highTarget) {
						gpu_increment_reject(thr_id);
						if (!opt_quiet)
							gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU!", res[n]);
					}
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
					be32enc(hdr + 76, valid[i]);
					sha512256d_hash(vhash, hdr);
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

extern "C" void free_sha512256d(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaDeviceSynchronize();
	cudaFreeHost(h_resNonce[thr_id]);
	cudaFree(d_resNonce[thr_id]);
	init[thr_id] = false;
	cudaDeviceSynchronize();
}
