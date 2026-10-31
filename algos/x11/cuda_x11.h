#ifndef CUDA_X11_H
#define CUDA_X11_H

#include <stdint.h>

/* X11 family building blocks: each one hashes the 64-byte state of every
 * thread in d_hash in place (d_nonceVector NULL = all threads) */

/* BLAKE-512 of the 80-byte header (block set by quark_blake512_cpu_setBlock_80) */
void quark_blake512_cpu_init(int thr_id, uint32_t threads);
void quark_blake512_cpu_free(int thr_id);
void quark_blake512_cpu_setBlock_80(int thr_id, uint32_t *endiandata);
void quark_blake512_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_outputHash);
void quark_blake512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_outputHash, int order);

void quark_bmw512_cpu_init(int thr_id, uint32_t threads);
void quark_bmw512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

/* d_hash must hold threads rounded up to 256 states, see quark_groestl512_gpu_hash_64_quad */
void quark_groestl512_cpu_init(int thr_id, uint32_t threads);
void quark_groestl512_cpu_free(int thr_id);
void quark_groestl512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

void quark_skein512_cpu_init(int thr_id, uint32_t threads);
void quark_skein512_cpu_hash_64(int thr_id, const uint32_t threads, const uint32_t startNonce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

/* Luffa-512 then CubeHash-512 in one kernel */
void x11_luffaCubehash512_cpu_init(int thr_id, uint32_t threads);
void x11_luffaCubehash512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash, int order);

void quark_jh512_cpu_init(int thr_id, uint32_t threads);
void quark_jh512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);
/* JH-512 of the 80-byte header, midstate of the first 64 bytes on the host */
void jh512_setBlock_80(int thr_id, uint32_t *endiandata);
void jh512_cuda_hash_80(const int thr_id, const uint32_t threads, const uint32_t startNounce, uint32_t *d_hash);

void quark_keccak512_cpu_init(int thr_id, uint32_t threads);
void quark_keccak512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

void x11_cubehash512_cpu_init(int thr_id, uint32_t threads);
void x11_cubehash512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

void x11_shavite512_cpu_init(int thr_id, uint32_t threads);
void x11_shavite512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

/* d_hash must hold threads rounded up to 128 states, see x11_simd512_gpu_expand_64 */
int  x11_simd512_cpu_init(int thr_id, uint32_t threads);
void x11_simd512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);
void x11_simd512_cpu_free(int thr_id);

void x11_echo512_cpu_init(int thr_id, uint32_t threads);
void x11_echo512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

#endif
