#ifndef YESCRYPT_ALGO_H
#define YESCRYPT_ALGO_H

#include <stdint.h>

/* 0 yescrypt, 1 r8, 2 r16, 3 r16v2, 4 r24, 5 r32 */
#define YESCRYPT_VARIANTS 6

struct yescrypt_variant {
	const char *name;
	uint32_t N, r, p;
	const char *key; /* NULL: the header (GlobalBoost-Y) */
};

extern const struct yescrypt_variant yescrypt_variants[YESCRYPT_VARIANTS];
struct yescrypt_variant yescrypt_params(int v);
/* the variant mines with its built-in parameters (always, except a changed v 0) */
bool yescrypt_is_default(int v);
void yescrypt_hash_params(const struct yescrypt_variant *y, void *state, const void *input);

/* one launch series over [startNonce, startNonce + threads) for the block and target set
 * by yescrypt_setTarget() (pdata: work data words, not byte-swapped); up to max_out
 * candidates ascending into out, returns how many, *total gets how many were found */
void yescrypt_setTarget(int thr_id, const uint32_t *pdata, const uint32_t *ptarget);
uint32_t yescrypt_scan(int thr_id, uint32_t threads, uint32_t startNonce,
	uint32_t *out, uint32_t max_out, uint32_t *total);

/* call after the buffers are allocated, before mining */
bool yescrypt_device_selftest(int thr_id, int v);

#endif
