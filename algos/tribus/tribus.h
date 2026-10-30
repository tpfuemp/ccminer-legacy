#ifndef TRIBUS_H
#define TRIBUS_H

#include <stdint.h>

/* candidate slots of the final ECHO kernel: [0] counts, [1..] hold nonces */
#define TRIBUS_MAX_RESULTS 16

void tribus_echo512_final(int thr_id, uint32_t threads, uint32_t startNonce, uint32_t *d_hash,
	uint32_t *d_resNonce, const uint64_t target);

/* one launch chain over [startNonce, startNonce + threads) for the block set
 * by tribus_setBlock(); up to max_out candidates ascending into out, returns
 * how many, *total gets how many were found */
void tribus_setBlock(int thr_id, uint32_t *endiandata, const uint32_t *ptarget);
uint32_t tribus_scan(int thr_id, uint32_t threads, uint32_t startNonce, uint64_t target,
	uint32_t *out, uint32_t max_out, uint32_t *total);

/* call after the buffers are allocated, before mining */
bool tribus_device_selftest(int thr_id);

#endif
