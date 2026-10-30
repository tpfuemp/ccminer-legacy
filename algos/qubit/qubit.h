#ifndef QUBIT_H
#define QUBIT_H

#include <stdint.h>

void qubit_luffa512_cpu_init(int thr_id, uint32_t threads);
void qubit_luffa512_cpu_setBlock_80(void *pdata);
void qubit_luffa512_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_hash, int order);

/* one launch chain over [startNonce, startNonce + threads) for the block and
 * target set by qubit_setBlock(); up to max_out candidates ascending into out,
 * returns how many, *total gets how many were found */
void qubit_setBlock(uint32_t *endiandata, const uint32_t *ptarget);
uint32_t qubit_scan(int thr_id, uint32_t threads, uint32_t startNonce,
	uint32_t *out, uint32_t max_out, uint32_t *total);

/* call after the buffers are allocated, before mining */
bool qubit_device_selftest(int thr_id);

#endif
