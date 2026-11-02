#ifndef CUDA_LYRA2_H
#define CUDA_LYRA2_H

#include <stdint.h>

/* Lyra2 family building blocks on a 32-byte state per thread, stored
 * column-major: word k of thread t at d_hash[k * threads + t] */

/* BLAKE-256 (14 rounds) then Keccak-256 of the 80-byte header */
void blake256_cpu_init(int thr_id, uint32_t threads);
void blake256_cpu_setBlock_80(uint32_t *pdata);
void blakeKeccak256_cpu_hash_80(const int thr_id, const uint32_t threads, const uint32_t startNonce, uint64_t *Hash, int order);

/* Lyra2 v1 (8x8, 1 pass); d_matrix: 16 words per thread on sm_52+, 768 on
 * sm_50, unused below (the Kepler kernel keeps its matrix in local memory) */
void lyra2_cpu_init(int thr_id, uint32_t threads, uint64_t *d_matrix);
void lyra2_cpu_hash_32(int thr_id, uint32_t threads, uint64_t *d_outputHash, bool gtx750ti);

void cubehash256_cpu_hash_32(int thr_id, uint32_t threads, uint32_t startNounce, uint64_t *d_hash, int order);

void skein256_cpu_init(int thr_id, uint32_t threads);
void skein256_cpu_hash_32(int thr_id, uint32_t threads, uint32_t startNonce, uint64_t *d_outputHash, int order);

/* final Groestl-256 with target screen: d_resNonce[0] = count, then nonces */
#define GROESTL256_MAX_RESULTS 16
void groestl256_setTarget(const void *ptarget);
void groestl256_cpu_hash_32(int thr_id, uint32_t threads, uint32_t startNounce, uint64_t *d_outputHash, uint32_t *d_resNonce);

#endif
