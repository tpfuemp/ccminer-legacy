#ifndef QUARK_H
#define QUARK_H

#include <stdint.h>

/* splits the nonces [startNonce, startNonce + threads) by bit 3 of byte 0 of
 * their hash in d_hash: set -> d_list1 (*n1 entries), clear -> d_list2 (*n2) */
void quark_split_cpu_init(int thr_id);
void quark_split_cpu_free(int thr_id);
void quark_split_cpu(int thr_id, uint32_t threads, uint32_t startNonce, uint32_t *d_hash,
	uint32_t *d_list1, uint32_t *n1, uint32_t *d_list2, uint32_t *n2);

/* one launch chain over [startNonce, startNonce + threads) for the block and
 * target set by quark_setBlock(); up to max_out candidates ascending into out,
 * returns how many, *total gets how many were found */
void quark_setBlock(int thr_id, uint32_t *endiandata, const uint32_t *ptarget);
uint32_t quark_scan(int thr_id, uint32_t threads, uint32_t startNonce,
	uint32_t *out, uint32_t max_out, uint32_t *total);

/* call after the buffers are allocated, before mining */
bool quark_device_selftest(int thr_id);

#endif
