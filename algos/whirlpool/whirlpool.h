#ifndef WHIRLPOOL_H
#define WHIRLPOOL_H

#include <stdint.h>

/* candidate slots: [0] counts, [1..WHIRLPOOL_MAX_RESULTS-1] hold nonces */
#define WHIRLPOOL_MAX_RESULTS 16

/* tables to the current device, once per GPU */
void whirlpool_cpu_init(void);
void whirlpool512_setBlock_80(const void *pdata);
uint32_t whirlpool512_scan(uint32_t threads, uint32_t startNonce, uint64_t target,
	uint32_t *d_res, uint32_t *h_res);

/* call after the result buffers are allocated, before mining */
bool whirlpool_device_selftest(int thr_id, uint32_t *d_res, uint32_t *h_res);

#endif
