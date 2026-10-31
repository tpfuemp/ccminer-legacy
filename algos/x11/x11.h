#ifndef X11_H
#define X11_H

#include <stdint.h>

/* one launch chain over [startNonce, startNonce + threads) for the block and
 * target set by x11_setBlock(); up to max_out candidates ascending into out,
 * returns how many, *total gets how many were found */
void x11_setBlock(int thr_id, uint32_t *endiandata, const uint32_t *ptarget);
uint32_t x11_scan(int thr_id, uint32_t threads, uint32_t startNonce,
	uint32_t *out, uint32_t max_out, uint32_t *total);

/* call after the buffers are allocated, before mining */
bool x11_device_selftest(int thr_id);

#endif
