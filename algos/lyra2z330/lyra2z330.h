#ifndef LYRA2Z330_H
#define LYRA2Z330_H

#include <stdint.h>

extern "C" void lyra2z330_hash(void *output, const void *input);

/* one launch series over [startNonce, startNonce + threads) for the block and target set
 * by lyra2z330_setTarget() (pdata: work data words, not byte-swapped); up to max_out
 * candidates ascending into out, returns how many, *total gets how many were found */
void lyra2z330_setTarget(int thr_id, uint32_t *pdata, const uint32_t *ptarget);
uint32_t lyra2z330_scan(int thr_id, uint32_t threads, uint32_t startNonce,
	uint32_t *out, uint32_t max_out, uint32_t *total);

/* call after the buffers are allocated, before mining */
bool lyra2z330_device_selftest(int thr_id);

#endif
