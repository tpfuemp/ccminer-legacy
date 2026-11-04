#ifndef NEOSCRYPT_H
#define NEOSCRYPT_H

#include <stdint.h>

/* candidate slots: [0] counts, [1..NEOSCRYPT_MAX_RESULTS-1] hold nonces */
#define NEOSCRYPT_MAX_RESULTS 16

/* the mining profile of the CPU reference neoscrypt() (core/miner.h) */
#define NEOSCRYPT_PROFILE 0x80000620U

/* 32 KiB scratch plus 768 bytes per thread; false on a CUDA error */
bool neoscrypt_cpu_init(int thr_id, uint32_t threads);
void neoscrypt_cpu_free(int thr_id);
void neoscrypt_setBlock(const uint32_t *pdata);
uint32_t neoscrypt_scan(int thr_id, uint32_t threads, uint32_t startNonce, uint64_t target,
	uint32_t *d_res, uint32_t *h_res);

/* call after the buffers are allocated, before mining */
bool neoscrypt_device_selftest(int thr_id, uint32_t *d_res, uint32_t *h_res);

#endif
