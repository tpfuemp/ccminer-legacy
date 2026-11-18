// ---------------------------------------------------------------------------
// equi24l -- lean Equihash 144/5 solver for 2 GB cards: about 1.5 GiB of VRAM
// instead of the 2.8 GiB of equi24b, at a lower rate. Same buckets, chunks,
// staging and record contents as equi24b (equi24b_params.h), but references and
// payloads live in separate arrays and a payload is gone once the next round
// has read it:
//
//   * layer 0 is one 4 B word per slot, [leaf index 25 | round-1 chunk 2];
//     round 1 recomputes a leaf's hash (blake2b) while staging it
//   * round 1 runs twice, each pass keeping the items bound for one half of the
//     buckets; round 2 runs on that half right after, so at most half of
//     layer 1 exists
//   * the references of layers 1-4 (source bucket 12 bits + Cantor code of the
//     two origin slots 26 bits) are kept as u32 + u8 arrays until expandSols
//   * expandSols walks them to the leaf indices and recomputes every leaf hash
//     for the full check
//
// Payload stored per layer: the bytes the next round stages (EQ24B_SBYTES),
// the last layer the 4 bytes digitK tests.
// ---------------------------------------------------------------------------
#define EQ_WN 144
#define EQ_WK 5

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "blake2/blake2.h"
#include "equi24b_params.h"

typedef uint32_t u32;
typedef uint64_t u64;
typedef uint16_t u16;
typedef unsigned char uchar;

#define EQ24B_BYTE(w, j) ((u32)((w)[(j) >> 3] >> (((j) & 7) * 8)) & 0xffu)

#define EQL_NB        EQ24B_NBUCKETS
#define EQL_HALF      (EQ24B_NBUCKETS / 2u)
#define EQL_N         ((size_t)EQ24B_NBUCKETS * EQ24B_NSLOTS)
#define EQL_HPB       (512u / EQ_WN)                    // hashes per blake
#define EQL_NHASHES   (2u * (1u << EQ24B_DIGITBITS))
#define EQL_NBLAKES   ((EQL_NHASHES + EQL_HPB - 1u) / EQL_HPB)
#define EQL_IDXMASK   0x1ffffffu                        // 25-bit leaf index
// slot-major record index (as EQ24B_RIDX) over nb buckets
#define EQL_RIDX(nb, b, s) (((size_t)((s) >> 5) * (nb) + (b)) * 32u + ((s) & 31u))
// payload words stored for layer r (1-4) and the bytes round r writes
#define EQL_PW(r)     ((r) == 1u ? 4u : (r) == 2u ? 3u : (r) == 3u ? 2u : 1u)
#define EQL_KEEP(r)   (EQ24B_SBYTES(r) - 3u)
#define EQL_K_SHARED  (EQ24B_NRESTS * 4u + EQ24B_NSLOTS * 3u)
#define EQL_K_END     0x3fffu

static_assert(EQL_KEEP(2u) == EQ24B_SBYTES(3u) && EQL_KEEP(3u) == EQ24B_SBYTES(4u) && EQL_KEEP(4u) == 4u,
	"a layer must store exactly the bytes the next round stages");
static_assert(EQL_KEEP(1u) <= EQL_PW(1u) * 4u && EQL_KEEP(2u) <= EQL_PW(2u) * 4u &&
	EQL_KEEP(3u) <= EQL_PW(3u) * 4u && EQL_KEEP(4u) <= EQL_PW(4u) * 4u, "payload words too few");
static_assert(EQ24B_CHUNKS_R(1u) <= 4u && EQ24B_STAGE_W(1u) == 4u && EQ24B_STAGE_H(1u) == 0u,
	"round 1 stages 16 hash bytes and keeps its chunk in 2 bits");
static_assert(EQL_HPB == 3u && EQL_NHASHES <= EQL_IDXMASK + 1u, "144/5 leaf indices");
static_assert(EQ24B_HEADS(1u) * EQ24B_CHUNKS_R(1u) >= EQ24B_NRESTS, "round-1 chunk of a rest");

namespace equi24l_144 {

// see cuda_equi24b.cuh: these headers define functions with external linkage
#include "blake2b_tromp.cuh"
#include "equi24_warp.cuh"

__device__ u64 eql_mid_h[8];   // blake2b midstate of the current header

struct eql_blake { u64 m0, m1lo, counter; };

// leaf hash (EQ_WN / 8 = 18 bytes) of a blake index, as bytes of hh at offset *off
__device__ __forceinline__ void eql_leaf(const u64 *__restrict__ blake_h, const eql_blake &bl,
                                         u32 index, u64 hh[7], int *off)
{
	u64 h[8];
#pragma unroll
	for (int i = 0; i < 8; i++) h[i] = blake_h[i];
	blake2b_gpu_hash_regs(h, bl.m0, bl.m1lo | ((u64)(index / EQL_HPB) << 32), bl.counter, hh);
	*off = (int)((index % EQL_HPB) * (EQ_WN / 8u));
}

// ---------------------------------------------------------------------------
// digitHL -- blake2b over the nonce space; a layer-0 slot keeps the leaf index
// and the round-1 chunk of its rest, the hash itself is recomputed in round 1.
// ---------------------------------------------------------------------------
__global__ void digitHL(const u64 *__restrict__ blake_h, eql_blake bl,
                        u32 *__restrict__ w0, u32 *__restrict__ counts, u32 *__restrict__ overflow)
{
	const u32 id = blockIdx.x * blockDim.x + threadIdx.x;
	const u32 stride = gridDim.x * blockDim.x;
	u64 h[8], hh[7];
#pragma unroll
	for (int i = 0; i < 8; i++) h[i] = blake_h[i];

	for (u32 blk = id; blk < EQL_NBLAKES; blk += stride) {
		blake2b_gpu_hash_regs(h, bl.m0, bl.m1lo | ((u64)blk << 32), bl.counter, hh);
#pragma unroll
		for (u32 i = 0; i < EQL_HPB; i++) {
			const int b = (EQ_WN / 8) * (int)i;
			const u32 index = blk * EQL_HPB + i;
			if (index >= EQL_NHASHES) continue;   // past the nonce space (see equi24b)
			const u32 d0 = (EQ24B_BYTE(hh, b) << 16) | (EQ24B_BYTE(hh, b + 1) << 8) | EQ24B_BYTE(hh, b + 2);
			const u32 bucket = (d0 >> EQ24B_RESTBITS) & EQ24B_BUCKMASK;
			const u32 chunk = (d0 & EQ24B_RESTMASK) / EQ24B_HEADS(1u);
			const u32 slot = eq24b_scatter_slot(&counts[bucket], bucket, threadIdx.x & 31);
			if (slot >= EQ24B_NSLOTS) { atomicAdd(overflow, 1u); continue; }
			w0[EQL_RIDX(EQL_NB, bucket, slot)] = index | (chunk << 25);
		}
	}
}

// Reference of a pair: source bucket + Cantor code of the two origin slots.
__device__ __forceinline__ u64 eql_ref(u32 bucket, u32 sa, u32 sb)
{
	const u32 a = max(sa, sb), b = min(sa, sb);
	return ((u64)bucket << EQ24B_CANTORBITS) | ((a * (a - 1u)) / 2u + b);
}

// Pass B of a round, shared by digitR1L and digitRL: walk each staged item's
// list and emit its pairs. Payload bytes 0-1 of the xor are zero (same rest),
// bytes 2-4 are the next digit, the next payload starts at byte 3 (equi24b).
// Round 1 keeps only destinations in half `half`.
template <u32 R>
__device__ __forceinline__ void eql_pairs(const u32 *s_pay, const u16 *s_next, const u16 *s_tail, u32 nst,
                                          u32 bucket, u32 half, u32 *__restrict__ payOut,
                                          u32 *__restrict__ refLo, uchar *__restrict__ refHi,
                                          u32 *__restrict__ cntOut, u32 *__restrict__ overflow)
{
	const u32 PW = EQ24B_STAGE_W(R), PH = EQ24B_STAGE_H(R), SB = EQ24B_SBYTES(R);
	const u32 OUTW = EQL_PW(R), KEEP = EQL_KEEP(R);
	const u32 lane = threadIdx.x & 31u;
	for (u32 s = threadIdx.x; s < nst; s += blockDim.x) {
		for (u32 t = s_next[s]; t != EQ24B_LIST_END; t = s_next[t]) {
			u32 x[5];
			for (u32 w = 0; w < PW; w++) x[w] = s_pay[s * PW + w] ^ s_pay[t * PW + w];
			if (PH) x[PW] = (u32)(s_tail[s] ^ s_tail[t]);
			x[0] &= 0xffff0000u;                     // origin slots, not payload
			// a pair whose payload cancels entirely shares leaves (see equi24b)
			u32 nz = 0u;
			for (u32 w = 0; w < PW + PH; w++) nz |= x[w];
			if (!nz) continue;
			const u32 b2 = (x[0] >> 16) & 0xffu, b3 = (x[0] >> 24) & 0xffu;
			const u32 dst = ((b2 << 4) | (b3 >> 4)) & EQ24B_BUCKMASK;
			if (R == 1u && dst / EQL_HALF != half) continue;
			const u32 os = eq24b_scatter_slot(&cntOut[dst], dst, lane);
			if (os >= EQ24B_NSLOTS) { atomicAdd(overflow, 1u); continue; }

			u32 o[4] = { 0u, 0u, 0u, 0u };
			for (u32 w = 0; w < OUTW; w++) {
				u32 v = 0u;
				for (u32 j = 0; j < 4u; j++) {
					const u32 bi = 3u + w * 4u + j;
					// bytes past KEEP stay zero: the next round reads whole words
					if (w * 4u + j < KEEP && bi < SB)
						v |= ((x[bi >> 2] >> ((bi & 3u) * 8u)) & 0xffu) << (8u * j);
				}
				o[w] = v;
			}
			const size_t ri = EQL_RIDX(EQL_NB, dst, os);
			u32 *out = payOut + (R == 1u ? EQL_RIDX(EQL_HALF, dst % EQL_HALF, os) : ri) * OUTW;
			if (OUTW == 4u)      *(uint4 *)out = make_uint4(o[0], o[1], o[2], o[3]);
			else if (OUTW == 2u) *(uint2 *)out = make_uint2(o[0], o[1]);
			else for (u32 w = 0; w < OUTW; w++) out[w] = o[w];

			const u64 ref = eql_ref(bucket, s_pay[s * PW] & 0xffffu, s_pay[t * PW] & 0xffffu);
			refLo[ri] = (u32)ref;
			refHi[ri] = (uchar)(ref >> 32);
		}
	}
}

// ---------------------------------------------------------------------------
// digitR1L -- round 1, one pass per destination half. Pass A collects the
// bucket's items of this chunk (from the chunk bits, no blake), then hashes
// them densely: a blake per staged item, none for the other chunks' items.
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(1024) digitR1L(const u64 *__restrict__ blake_h, eql_blake bl,
                         const u32 *__restrict__ w0, u32 *__restrict__ payOut,
                         u32 *__restrict__ refLo, uchar *__restrict__ refHi,
                         const u32 *__restrict__ cntIn, u32 *__restrict__ cntOut,
                         u32 *__restrict__ overflow, u32 half)
{
	const u32 CH = EQ24B_CHUNKS_R(1u);
	const u32 bucket = blockIdx.x / CH, chunk = blockIdx.x % CH;
	const u32 tid = threadIdx.x, lane = tid & 31u;
	const u32 nin = min(cntIn[bucket], EQ24B_NSLOTS);
	const u32 PW = EQ24B_STAGE_W(1u), SLOTS = EQ24B_STAGE_SLOTS(1u), HEADS = EQ24B_HEADS(1u);

	extern __shared__ u32 sh[];
	u32 *s_pay  = sh;                              // [SLOTS][PW], word 0 low 16 bits = origin slot
	u16 *s_next = (u16 *)(sh + SLOTS * PW);        // [SLOTS], first the origin slots
	u32 *s_head = sh + EQ24B_HEAD_OFF_W(1u);       // [HEADS]
	__shared__ u32 s_n;

	for (u32 i = tid; i < HEADS; i += blockDim.x) s_head[i] = EQ24B_LIST_END;
	if (tid == 0) s_n = 0u;
	__syncthreads();

	for (u32 slot = tid; slot < nin; slot += blockDim.x) {
		if ((w0[EQL_RIDX(EQL_NB, bucket, slot)] >> 25) != chunk) continue;
		const u32 s = eq24b_append(&s_n, chunk, lane);
		if (s >= SLOTS) { if (lane == 0) atomicAdd(overflow + 1, 1u); continue; }
		s_next[s] = (u16)slot;
	}
	__syncthreads();
	const u32 nst = min(s_n, SLOTS);
	for (u32 s = tid; s < nst; s += blockDim.x) {
		const u32 slot = s_next[s];
		u64 hh[7];
		int b;
		eql_leaf(blake_h, bl, w0[EQL_RIDX(EQL_NB, bucket, slot)] & EQL_IDXMASK, hh, &b);
		u32 p[4];
#pragma unroll
		for (u32 k = 0; k < 4u; k++) {           // payload: hash bytes 1-16
			u32 v = 0u;
#pragma unroll
			for (u32 j = 0; j < 4u; j++) v |= EQ24B_BYTE(hh, b + 1 + (int)(k * 4u + j)) << (8u * j);
			p[k] = v;
		}
		const u32 rest = ((p[0] & 0x0fu) << 8) | ((p[0] >> 8) & 0xffu);
		s_pay[s * PW] = (p[0] & 0xffff0000u) | slot;
#pragma unroll
		for (u32 w = 1; w < 4u; w++) s_pay[s * PW + w] = p[w];
		s_next[s] = (u16)atomicExch(&s_head[rest % HEADS], s);
	}
	__syncthreads();
	eql_pairs<1u>(s_pay, s_next, NULL, nst, bucket, half, payOut, refLo, refHi, cntOut, overflow);
}

// ---------------------------------------------------------------------------
// digitRL<R> -- rounds 2-4 as equi24b's digitR on the stored payload words.
// Round 2 runs on the source half `half` only (its layer-1 array is that half).
// ---------------------------------------------------------------------------
template <u32 R>
__global__ void __launch_bounds__(1024) digitRL(const u32 *__restrict__ payIn, u32 *__restrict__ payOut,
                        u32 *__restrict__ refLo, uchar *__restrict__ refHi,
                        const u32 *__restrict__ cntIn, u32 *__restrict__ cntOut,
                        u32 *__restrict__ overflow, u32 half)
{
	const u32 CH = EQ24B_CHUNKS_R(R);
	const u32 lb = blockIdx.x / CH, chunk = blockIdx.x % CH;
	const u32 bucket = R == 2u ? half * EQL_HALF + lb : lb;
	const u32 tid = threadIdx.x, lane = tid & 31u;
	const u32 nin = min(cntIn[bucket], EQ24B_NSLOTS);
	const u32 PW = EQ24B_STAGE_W(R), PH = EQ24B_STAGE_H(R);
	const u32 SLOTS = EQ24B_STAGE_SLOTS(R), HEADS = EQ24B_HEADS(R);
	const u32 INW = EQL_PW(R - 1u);

	extern __shared__ u32 sh[];
	u32 *s_pay  = sh;
	u16 *s_next = (u16 *)(sh + SLOTS * PW);
	u16 *s_tail = s_next + SLOTS;
	u32 *s_head = sh + EQ24B_HEAD_OFF_W(R);
	__shared__ u32 s_n;

	for (u32 i = tid; i < HEADS; i += blockDim.x) s_head[i] = EQ24B_LIST_END;
	if (tid == 0) s_n = 0u;
	__syncthreads();

	for (u32 slot = tid; slot < nin; slot += blockDim.x) {
		const u32 *rec = payIn + (R == 2u ? EQL_RIDX(EQL_HALF, lb, slot) : EQL_RIDX(EQL_NB, bucket, slot)) * INW;
		u32 rw[4];
		if (INW == 4u) {
			const uint4 v = *(const uint4 *)rec;
			rw[0] = v.x; rw[1] = v.y; rw[2] = v.z; rw[3] = v.w;
		} else if (INW == 2u) {
			const uint2 v = *(const uint2 *)rec;
			rw[0] = v.x; rw[1] = v.y; rw[2] = rw[3] = 0u;
		} else {
			for (u32 w = 0; w < 4u; w++) rw[w] = w < INW ? rec[w] : 0u;
		}
		const u32 p0 = rw[0];
		const u32 rest = ((p0 & 0x0fu) << 8) | ((p0 >> 8) & 0xffu);
		if (rest / HEADS != chunk) continue;
		const u32 s = eq24b_append(&s_n, chunk, lane);
		if (s >= SLOTS) { if (lane == 0) atomicAdd(overflow + 1, 1u); continue; }
		s_pay[s * PW] = (p0 & 0xffff0000u) | slot;
		for (u32 w = 1; w < PW; w++) s_pay[s * PW + w] = rw[w];
		if (PH) s_tail[s] = (u16)rw[PW];
		s_next[s] = (u16)atomicExch(&s_head[rest % HEADS], s);
	}
	__syncthreads();
	eql_pairs<R>(s_pay, s_next, s_tail, min(s_n, SLOTS), bucket, 0u, payOut, refLo, refHi, cntOut, overflow);
}

// ---------------------------------------------------------------------------
// digitKL -- the final collision as equi24b's digitK, on the 4-byte last-layer
// payload: the rest, then a 10-bit key, then all 16 kept final-digit bits.
// ---------------------------------------------------------------------------
__global__ void digitKL(const u32 *__restrict__ pay4, const u32 *__restrict__ cntIn,
                        u32 *__restrict__ sols, u32 *__restrict__ nsols, u32 maxsols)
{
	const u32 bucket = blockIdx.x, tid = threadIdx.x;
	const u32 nin = min(cntIn[bucket], EQ24B_NSLOTS);

	extern __shared__ u32 sh[];
	u32   *s_head = sh;                                  // [NRESTS]
	u16   *s_link = (u16 *)(sh + EQ24B_NRESTS);          // [NSLOTS]: link 14 | key bits 8-9
	uchar *s_key  = (uchar *)(s_link + EQ24B_NSLOTS);    // [NSLOTS]: key bits 0-7

	for (u32 i = tid; i < EQ24B_NRESTS; i += blockDim.x) s_head[i] = EQL_K_END;
	__syncthreads();

	for (u32 slot = tid; slot < nin; slot += blockDim.x) {
		const u32 p0 = pay4[EQL_RIDX(EQL_NB, bucket, slot)];
		const u32 rest = ((p0 & 0x0fu) << 8) | ((p0 >> 8) & 0xffu);
		const u32 key = (p0 >> 16) & 0x3ffu;
		s_key[slot] = (uchar)key;
		const u32 prev = atomicExch(&s_head[rest], slot);
		s_link[slot] = (u16)((prev & EQL_K_END) | ((key >> 8) << 14));
	}
	__syncthreads();

	for (u32 sl = tid; sl < nin; sl += blockDim.x) {
		const u32 ls = s_link[sl];
		const u32 ks = s_key[sl] | ((ls >> 14) << 8);
		for (u32 t = ls & EQL_K_END; t != EQL_K_END; ) {
			const u32 lt = s_link[t];
			if ((s_key[t] | ((lt >> 14) << 8)) == ks) {
				const u32 a = pay4[EQL_RIDX(EQL_NB, bucket, sl)];
				const u32 b = pay4[EQL_RIDX(EQL_NB, bucket, t)];
				if (!((a ^ b) & 0xffff0000u)) {
					const u32 at = atomicAdd(nsols, 1u);
					if (at < maxsols) {
						sols[at * 3 + 0] = bucket;
						sols[at * 3 + 1] = sl;
						sols[at * 3 + 2] = t;
					}
				}
			}
			t = lt & EQL_K_END;
		}
	}
}

// ---------------------------------------------------------------------------
// expandSolsL -- references down to the layer-0 words, then the full check:
// the recomputed hashes of all leaves must cancel. Proof order as equi24b.
// ---------------------------------------------------------------------------
struct eql_refs { const u32 *lo[EQ_WK]; const uchar *hi[EQ_WK]; };   // layers 1-4

__global__ void expandSolsL(const u32 *__restrict__ w0, eql_refs refs,
                            const u64 *__restrict__ blake_h, eql_blake bl,
                            const u32 *__restrict__ sols, const u32 *__restrict__ found, u32 maxsol,
                            u32 *__restrict__ out, u32 *__restrict__ nout)
{
	const u32 nfound = min(*found, maxsol);
	for (u32 t = blockIdx.x * blockDim.x + threadIdx.x; t < nfound; t += gridDim.x * blockDim.x) {
		u32 idx[EQ24B_PROOFSIZE], bkt[EQ24B_PROOFSIZE], slt[EQ24B_PROOFSIZE];
		bkt[0] = bkt[1] = sols[t * 3 + 0];
		slt[0] = sols[t * 3 + 1]; slt[1] = sols[t * 3 + 2];
		u32 n = 2;
		for (int layer = (int)EQ_WK - 1; layer >= 1; layer--) {
			for (int i = (int)n - 1; i >= 0; i--) {
				const size_t ri = EQL_RIDX(EQL_NB, bkt[i], slt[i]);
				const u64 ref = ((u64)refs.hi[layer][ri] << 32) | refs.lo[layer][ri];
				const u32 src = (u32)(ref >> EQ24B_CANTORBITS);
				const u64 c = ref & EQ24B_CANTORMASK;
				u32 a = (u32)((1.0 + sqrt(1.0 + 8.0 * (double)c)) * 0.5);
				while (a > 1u && (u64)a * (a - 1u) / 2u > c) a--;
				while ((u64)(a + 1u) * a / 2u <= c) a++;
				bkt[2*i] = src; slt[2*i] = (u32)(c - (u64)a * (a - 1u) / 2u);
				bkt[2*i+1] = src; slt[2*i+1] = a;
			}
			n *= 2;
		}
		bool dup = false;
		for (u32 i = 0; i < n; i++) {
			idx[i] = w0[EQL_RIDX(EQL_NB, bkt[i], slt[i])] & EQL_IDXMASK;
			for (u32 j = 0; j < i; j++) dup |= idx[i] == idx[j];
		}
		if (dup) continue;   // a repeated leaf satisfies every XOR, never a share

		u32 acc[5] = { 0u, 0u, 0u, 0u, 0u };
		for (u32 i = 0; i < n; i++) {
			u64 hh[7];
			int b;
			eql_leaf(blake_h, bl, idx[i], hh, &b);
			for (u32 k = 0; k < EQ_WN / 8u; k++)
				acc[k >> 2] ^= EQ24B_BYTE(hh, b + (int)k) << ((k & 3u) * 8u);
		}
		if (acc[0] | acc[1] | acc[2] | acc[3] | acc[4]) continue;

		for (u32 blk = 2; blk <= EQ24B_PROOFSIZE; blk <<= 1)
			for (u32 base = 0; base < EQ24B_PROOFSIZE; base += blk)
				if (idx[base] > idx[base + blk / 2])
					for (u32 k = 0; k < blk / 2; k++) {
						const u32 tmp = idx[base + k];
						idx[base + k] = idx[base + blk / 2 + k];
						idx[base + blk / 2 + k] = tmp;
					}
		const u32 at = atomicAdd(nout, 1u);
		if (at >= nfound) continue;
		for (u32 i = 0; i < EQ24B_PROOFSIZE; i++) out[at * EQ24B_PROOFSIZE + i] = idx[i];
	}
}

// ---------------------------------------------------------------------------
// host side
// ---------------------------------------------------------------------------
#define EQL_MAXSOL    262144u   // digitK candidates, as equi24b
#define EQL_PIPE_CAP  64u       // proofs copied back per solve

// buffer A: layer-1 half, then layer 3; buffer B: layer 2, then the layer-4 payload
#define EQL_A_BYTES   ((size_t)EQL_HALF * EQ24B_NSLOTS * EQL_PW(1u) * 4u)
#define EQL_B_BYTES   (EQL_N * EQL_PW(2u) * 4u)
static_assert(EQL_A_BYTES >= EQL_N * EQL_PW(3u) * 4u && EQL_B_BYTES >= EQL_N * EQL_PW(4u) * 4u,
	"layers 3 and 4 reuse the buffers of layers 1 and 2");

struct ctx {
	u32 *w0, *bufA, *bufB;
	u32 *lo[EQ_WK];
	uchar *hi[EQ_WK];
	u32 *cnt[EQ_WK];
	u32 *over, *sols, *nsols, *idx;
};

struct pipe {
	u32 *h[2];           // pinned: over[4], found, CAP proofs
	u64 *mid[2];         // pinned blake2b midstate
	cudaEvent_t ev[2];
	unsigned nl, nc;     // solves launched / collected
};

static void setheader(blake2b_state *c, const char *hdr, const char *personal)
{
	char p16[16];
	memcpy(p16, personal, 8);
	u32 n = EQ_WN, k = EQ_WK;
	memcpy(p16 + 8, &n, 4); memcpy(p16 + 12, &k, 4);
	blake2b_param P[1];
	memset(P, 0, sizeof(blake2b_param));
	P->digest_length = (512 / EQ_WN) * EQ_WN / 8;
	P->fanout = P->depth = 1;
	memcpy(P->personal, p16, 16);
	eq_blake2b_init_param(c, P);
	eq_blake2b_update(c, (const uint8_t *)hdr, 140);
}

static unsigned round_shared(unsigned r)
{
	switch (r) {
	case 1: return EQ24B_SHARED_BYTES(1u);
	case 2: return EQ24B_SHARED_BYTES(2u);
	case 3: return EQ24B_SHARED_BYTES(3u);
	case 4: return EQ24B_SHARED_BYTES(4u);
	default: return 0;
	}
}

static int major_cc(void)
{
	int dev = 0, major = 8;
	cudaGetDevice(&dev);
	cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
	return major;
}

// Hand each kernel to the GPU on its own: WDDM batches queued launches and its
// watchdog (about 2 s) covers the whole batch, which a slow card exceeds with
// several rounds. Elsewhere it is a cheap status query.
#define EQL_SUBMIT() ((void)cudaStreamQuery(0))

// Reset the counters and queue digitHL .. expandSolsL for one header on the
// default stream. A launch that never starts is caught by cudaGetLastError.
static int queue(ctx *c, const char *headernonce, const char *personal, u64 *pinned)
{
	blake2b_state st;
	setheader(&st, headernonce, personal);
	if (st.buflen != 12) return -1;
	eql_blake bl;
	bl.m0 = *(const u64 *)(st.buf);
	bl.m1lo = *(const u32 *)(st.buf + 8);
	bl.counter = st.counter + 16;      // 12 buffered + the 4-byte index
	memcpy(pinned, st.h, 8 * sizeof(u64));
	u64 *d_h = NULL;
	if (cudaMemcpyToSymbolAsync(eql_mid_h, pinned, 8 * sizeof(u64), 0, cudaMemcpyHostToDevice, 0) != cudaSuccess ||
	    cudaGetSymbolAddress((void **)&d_h, eql_mid_h) != cudaSuccess)
		return -2;

	cudaMemset(c->over, 0, 4 * sizeof(u32));
	cudaMemset(c->nsols, 0, sizeof(u32));
	for (unsigned r = 0; r < EQ_WK; r++)
		cudaMemset(c->cnt[r], 0, (size_t)EQL_NB * sizeof(u32));

	const int major = major_cc();
	const int tpb = major >= 8 ? 512 : 1024;
	const int hblocks = major == 6 ? 1024 : 2048, htpb = major == 6 ? 512 : 128;   // as equi24b
	digitHL<<<hblocks, htpb>>>(d_h, bl, c->w0, c->cnt[0], c->over);
	EQL_SUBMIT();
	for (u32 half = 0; half < 2u; half++) {
		digitR1L<<<EQL_NB * EQ24B_CHUNKS_R(1u), tpb, round_shared(1)>>>(d_h, bl, c->w0, c->bufA,
			c->lo[1], c->hi[1], c->cnt[0], c->cnt[1], c->over, half);
		EQL_SUBMIT();
		digitRL<2u><<<EQL_HALF * EQ24B_CHUNKS_R(2u), tpb, round_shared(2)>>>(c->bufA, c->bufB,
			c->lo[2], c->hi[2], c->cnt[1], c->cnt[2], c->over, half);
		EQL_SUBMIT();
	}
	digitRL<3u><<<EQL_NB * EQ24B_CHUNKS_R(3u), tpb, round_shared(3)>>>(c->bufB, c->bufA,
		c->lo[3], c->hi[3], c->cnt[2], c->cnt[3], c->over, 0u);
	EQL_SUBMIT();
	digitRL<4u><<<EQL_NB * EQ24B_CHUNKS_R(4u), tpb, round_shared(4)>>>(c->bufA, c->bufB,
		c->lo[4], c->hi[4], c->cnt[3], c->cnt[4], c->over, 0u);
	EQL_SUBMIT();
	digitKL<<<EQL_NB, tpb, EQL_K_SHARED>>>(c->bufB, c->cnt[4], c->sols, c->nsols, EQL_MAXSOL);
	EQL_SUBMIT();
	eql_refs refs;
	for (unsigned r = 0; r < EQ_WK; r++) { refs.lo[r] = c->lo[r]; refs.hi[r] = c->hi[r]; }
	expandSolsL<<<64, 64>>>(c->w0, refs, d_h, bl, c->sols, c->nsols, EQL_MAXSOL, c->idx, c->over + 3);
	return cudaGetLastError() == cudaSuccess ? 0 : -4;
}

// Trivial-solution backstop, as equi24b: emits the proofs without a repeated
// leaf and returns their count.
static int emit_proofs(const u32 *hidx, u32 nok, void (*emit)(void *, const uint32_t *, uint32_t), void *ud)
{
	int emitted = 0;
	for (u32 s = 0; s < nok; s++) {
		const u32 *q = hidx + (size_t)s * EQ24B_PROOFSIZE;
		u32 sorted[EQ24B_PROOFSIZE];
		memcpy(sorted, q, sizeof(sorted));
		for (u32 i = 1; i < EQ24B_PROOFSIZE; i++)
			for (u32 j = i; j && sorted[j-1] > sorted[j]; j--) {
				const u32 tv = sorted[j]; sorted[j] = sorted[j-1]; sorted[j-1] = tv;
			}
		bool dup = false;
		for (u32 i = 1; i < EQ24B_PROOFSIZE && !dup; i++) dup = sorted[i] == sorted[i-1];
		if (dup) continue;
		if (emit) emit(ud, q, EQ24B_PROOFSIZE);
		emitted++;
	}
	return emitted;
}

}  // namespace equi24l_144

using namespace equi24l_144;

extern "C" void equi24l_144_free(void *p)
{
	ctx *c = (ctx *)p;
	if (!c) return;
	void *bufs[] = { c->w0, c->bufA, c->bufB, c->over, c->sols, c->nsols, c->idx };
	for (unsigned i = 0; i < sizeof(bufs) / sizeof(bufs[0]); i++) if (bufs[i]) cudaFree(bufs[i]);
	for (unsigned r = 0; r < EQ_WK; r++) {
		if (c->lo[r]) cudaFree(c->lo[r]);
		if (c->hi[r]) cudaFree(c->hi[r]);
		if (c->cnt[r]) cudaFree(c->cnt[r]);
	}
	::free(c);
}

extern "C" unsigned long long equi24l_144_arena_needed(void)
{
	unsigned long long t = (unsigned long long)EQL_N * 4u + EQL_A_BYTES + EQL_B_BYTES;
	t += (unsigned long long)(EQ_WK - 1u) * EQL_N * 5u;                      // references, layers 1-4
	t += (unsigned long long)EQ_WK * EQL_NB * sizeof(u32);                   // bucket counters
	t += (unsigned long long)EQL_MAXSOL * (3u + EQ24B_PROOFSIZE) * sizeof(u32);
	return t;
}

// NULL when the device cannot provide the arena.
extern "C" void *equi24l_144_init(void)
{
	ctx *c = (ctx *)calloc(1, sizeof(ctx));
	if (!c) return NULL;
	bool ok = cudaMalloc(&c->w0, EQL_N * 4u) == cudaSuccess
		&& cudaMalloc(&c->bufA, EQL_A_BYTES) == cudaSuccess
		&& cudaMalloc(&c->bufB, EQL_B_BYTES) == cudaSuccess;
	for (unsigned r = 0; r < EQ_WK && ok; r++) {
		ok = cudaMalloc(&c->cnt[r], (size_t)EQL_NB * sizeof(u32)) == cudaSuccess;
		if (ok && r >= 1u)
			ok = cudaMalloc(&c->lo[r], EQL_N * 4u) == cudaSuccess
				&& cudaMalloc(&c->hi[r], EQL_N) == cudaSuccess;
	}
	ok = ok && cudaMalloc(&c->over, 4 * sizeof(u32)) == cudaSuccess
		&& cudaMalloc(&c->sols, (size_t)EQL_MAXSOL * 3 * sizeof(u32)) == cudaSuccess
		&& cudaMalloc(&c->nsols, sizeof(u32)) == cudaSuccess
		&& cudaMalloc(&c->idx, (size_t)EQL_MAXSOL * EQ24B_PROOFSIZE * sizeof(u32)) == cudaSuccess;
	if (!ok) { equi24l_144_free(c); cudaGetLastError(); return NULL; }
	return c;
}

// ---- pipelined solve, the same contract as equi24b (cuda_equi24b.h) ----------
extern "C" void equi24l_144_pipe_free(void *pq)
{
	pipe *q = (pipe *)pq;
	if (!q) return;
	for (int i = 0; i < 2; i++) {
		if (q->ev[i]) cudaEventDestroy(q->ev[i]);
		if (q->h[i]) cudaFreeHost(q->h[i]);
		if (q->mid[i]) cudaFreeHost(q->mid[i]);
	}
	::free(q);
}

extern "C" void *equi24l_144_pipe_new(void)
{
	pipe *q = (pipe *)calloc(1, sizeof(pipe));
	if (!q) return NULL;
	for (int i = 0; i < 2; i++)
		if (cudaHostAlloc((void **)&q->h[i], (5u + EQL_PIPE_CAP * EQ24B_PROOFSIZE) * sizeof(u32), cudaHostAllocDefault) != cudaSuccess
		    || cudaHostAlloc((void **)&q->mid[i], 8 * sizeof(u64), cudaHostAllocDefault) != cudaSuccess
		    || cudaEventCreateWithFlags(&q->ev[i], cudaEventDisableTiming) != cudaSuccess) {
			equi24l_144_pipe_free(q);
			return NULL;
		}
	return q;
}

extern "C" void equi24l_144_pipe_reset(void *pq)
{
	pipe *q = (pipe *)pq;
	if (!q) return;
	while (q->nc != q->nl) cudaEventSynchronize(q->ev[q->nc++ & 1u]);
}

extern "C" int equi24l_144_launch(void *p, void *pq, const char *headernonce, const char *personal)
{
	ctx *c = (ctx *)p;
	pipe *q = (pipe *)pq;
	if (!c || !q) return -1;
	if (q->nl - q->nc >= 2u) return -5;   // both slots in flight
	const unsigned k = q->nl & 1u;
	const int rc = queue(c, headernonce, personal, q->mid[k]);
	if (rc) return rc;
	cudaMemcpyAsync(q->h[k], c->over, 4 * sizeof(u32), cudaMemcpyDeviceToHost, 0);
	cudaMemcpyAsync(q->h[k] + 4, c->nsols, sizeof(u32), cudaMemcpyDeviceToHost, 0);
	cudaMemcpyAsync(q->h[k] + 5, c->idx, (size_t)EQL_PIPE_CAP * EQ24B_PROOFSIZE * sizeof(u32),
	                cudaMemcpyDeviceToHost, 0);
	if (cudaEventRecord(q->ev[k], 0) != cudaSuccess) return -4;
	q->nl++;
	return 0;
}

extern "C" int equi24l_144_collect(void *p, void *pq, void (*emit)(void *, const uint32_t *, uint32_t),
                                   void *ud, unsigned *drops4)
{
	ctx *c = (ctx *)p;
	pipe *q = (pipe *)pq;
	if (!c || !q) return -1;
	if (q->nl == q->nc) return -5;        // nothing queued
	const unsigned k = q->nc & 1u;
	q->nc++;
	if (cudaEventSynchronize(q->ev[k]) != cudaSuccess) return -3;
	const u32 *h = q->h[k];
	const u32 found = h[4];
	const u32 store = found < EQL_MAXSOL ? found : EQL_MAXSOL;
	u32 nok = h[3] < store ? h[3] : store;
	const u32 over = nok > EQL_PIPE_CAP ? nok - EQL_PIPE_CAP : 0u;
	if (drops4) {
		drops4[0] = h[0]; drops4[1] = h[1]; drops4[2] = h[2];
		drops4[3] = (found - store) + over;
	}
	if (over) nok = EQL_PIPE_CAP;
	return emit_proofs(h + 5, nok, emit, ud);
}
