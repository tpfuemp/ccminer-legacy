#ifndef LYRA2V2_H
#define LYRA2V2_H

#include <stdint.h>

/* one launch chain over [startNonce, startNonce + threads) for the block and
 * target set by lyra2v2_setBlock() (pdata: work data words, not byte-swapped);
 * up to max_out candidates ascending into out, returns how many, *total gets
 * how many were found */
void lyra2v2_setBlock(int thr_id, uint32_t *pdata, const uint32_t *ptarget);
uint32_t lyra2v2_scan(int thr_id, uint32_t threads, uint32_t startNonce,
	uint32_t *out, uint32_t max_out, uint32_t *total);

/* call after the buffers are allocated, before mining */
bool lyra2v2_device_selftest(int thr_id);

#endif
