#ifndef CUDA_YESCRYPT_H
#define CUDA_YESCRYPT_H

#include <stdint.h>

/* d_resNonce[0] counts candidates, [1..YESCRYPT_MAX_RESULTS-1] hold them */
#define YESCRYPT_MAX_RESULTS 16

/* scratch words per thread: B, S, V, SHA-256 state */
#define YESCRYPT_WORDS(N, r, p) (520ULL + 2ULL * (r) * ((N) + 16ULL * (p)))

/* threads: a multiple of 32 */
bool yescrypt_cpu_init(int thr_id, uint32_t threads, uint32_t N, uint32_t r, uint32_t p);
void yescrypt_cpu_free(int thr_id);
/* pdata: work data words; key NULL: the header (GlobalBoost-Y), else key_len <= 119 */
void yescrypt_setBlock(const uint32_t *pdata, const char *key, uint32_t key_len, uint64_t target);
/* nonces [startNounce, startNounce + threads), only the first count report */
void yescrypt_cpu_hash(int thr_id, uint32_t threads, uint32_t count, uint32_t startNounce,
	uint32_t N, uint32_t r, uint32_t p, uint32_t *d_resNonce);

#endif
