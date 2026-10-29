#ifndef BMW512_H
#define BMW512_H

#include <stdint.h>

/* candidate slots: [0] counts, [1..BMW512_MAX_RESULTS-1] hold nonces */
#define BMW512_MAX_RESULTS 16

void bmw512_cpu_setBlock_80(const void *pdata);
uint32_t bmw512_scan(uint32_t threads, uint32_t startNonce, uint64_t target,
	uint32_t *d_res, uint32_t *h_res);

/* call after the result buffers are allocated, before mining */
bool bmw512_device_selftest(int thr_id, uint32_t *d_res, uint32_t *h_res);

#endif
