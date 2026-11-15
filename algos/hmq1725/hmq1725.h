#ifndef HMQ1725_H
#define HMQ1725_H

#include <stddef.h>
#include <stdint.h>

/* HAVAL-256/5 of each 64-byte hash (or of the listed nonces); outlen 512 also zeroes the
 * upper 32 bytes */
void x17_haval256_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_hash, const int outlen,
	uint32_t *d_nonceVector = NULL);

/* one launch chain over [startNonce, startNonce + threads) for the block and
 * target set by hmq1725_setBlock(); up to max_out candidates ascending into out,
 * returns how many, *total gets how many were found */
void hmq1725_setBlock(uint32_t *endiandata, const uint32_t *ptarget);
uint32_t hmq1725_scan(int thr_id, uint32_t threads, uint32_t startNonce,
	uint32_t *out, uint32_t max_out, uint32_t *total);

/* call after the buffers are allocated, before mining */
bool hmq1725_device_selftest(int thr_id);

#endif
