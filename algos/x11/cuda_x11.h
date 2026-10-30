#ifndef CUDA_X11_H
#define CUDA_X11_H

#include <stdint.h>

/* X11 family building blocks: each one hashes the 64-byte state of every
 * thread in d_hash in place (d_nonceVector NULL = all threads) */

void quark_jh512_cpu_init(int thr_id, uint32_t threads);
void quark_jh512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);
/* JH-512 of the 80-byte header, midstate of the first 64 bytes on the host */
void jh512_setBlock_80(int thr_id, uint32_t *endiandata);
void jh512_cuda_hash_80(const int thr_id, const uint32_t threads, const uint32_t startNounce, uint32_t *d_hash);

void quark_keccak512_cpu_init(int thr_id, uint32_t threads);
void quark_keccak512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

void x11_echo512_cpu_init(int thr_id, uint32_t threads);
void x11_echo512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

#endif
