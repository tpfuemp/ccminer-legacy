#ifndef CUDA_X16_H
#define CUDA_X16_H
/* x16 family blocks beyond algos/x11: 80-byte first hashes (endiandata layout unless
 * noted) and the 64-byte shabal / whirlpool / sha512 / luffa / echo (sm_50+) blocks */
#include <stdint.h>

void x16_simd512_setBlock_80(void *pdata);
void x16_simd512_cuda_hash_80(int thr_id, const uint32_t threads, const uint32_t startNonce, uint32_t *d_hash);

void x16_echo512_cuda_init(int thr_id, const uint32_t threads);
void x16_echo512_setBlock_80(void *endiandata);
void x16_echo512_cuda_hash_80(int thr_id, const uint32_t threads, const uint32_t startNonce, uint32_t *d_hash);
/* sm_50+ (cuda_arch); older archs use x11_echo512_cpu_hash_64 */
void x16_echo512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash);

/* pdata (host word order), not endiandata */
void x16_fugue512_cpu_init(int thr_id, uint32_t threads);
void x16_fugue512_cpu_free(int thr_id);
void x16_fugue512_setBlock_80(void *pdata);
void x16_fugue512_cuda_hash_80(int thr_id, const uint32_t threads, const uint32_t startNonce, uint32_t *d_hash);

void x16_shabal512_setBlock_80(void *pdata);
void x16_shabal512_cuda_hash_80(int thr_id, const uint32_t threads, const uint32_t startNonce, uint32_t *d_hash);
void x14_shabal512_cpu_init(int thr_id, uint32_t threads);
void x14_shabal512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

void x16_whirlpool512_init(int thr_id, uint32_t threads);
void x16_whirlpool512_setBlock_80(void *pdata);
void x16_whirlpool512_hash_80(int thr_id, const uint32_t threads, const uint32_t startNonce, uint32_t *d_outputHash);
void x15_whirlpool_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

void x17_sha512_cpu_init(int thr_id, uint32_t threads);
void x17_sha512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_hash);
void x16_sha512_setBlock_80(void *pdata);
void x16_sha512_cuda_hash_80(int thr_id, const uint32_t threads, const uint32_t startNounce, uint32_t *d_hash);

void x11_luffa512_cpu_init(int thr_id, uint32_t threads);
void x11_luffa512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

/* x16rv2: Tiger-192, 24 bytes zero-padded to the 64-byte state (blocks of 256 threads) */
void tiger192_setBlock_80(void *pdata);
void tiger192_cpu_hash_80(int thr_id, int threads, uint32_t startNonce, uint32_t *d_hash);
void tiger192_cpu_hash_64(int thr_id, int threads, int zero_pad_64, uint32_t *d_hash);

#endif
