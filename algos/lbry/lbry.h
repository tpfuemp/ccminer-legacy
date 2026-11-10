#ifndef LBRY_H
#define LBRY_H

#include <stdint.h>

/* candidate slots: [0] counts, [1..LBRY_MAX_RESULTS-1] hold nonces */
#define LBRY_MAX_RESULTS 16

/* nonce word of the 112-byte header */
#define LBRY_NONCE_WORD 27

/* merged: one kernel (sm > 500); else SHA-256d, SHA-512, RIPEMD + SHA-256d in three
 * launches over d_hash (64 bytes per thread) */
void lbry_setBlock(bool merged, uint32_t *endiandata);
uint32_t lbry_scan(bool merged, uint32_t threads, uint32_t startNonce, uint64_t target,
	uint32_t *d_hash, uint32_t *d_res, uint32_t *h_res);

/* kernel launchers */
void lbry_sha256_setBlock_112(uint32_t *pdata);
void lbry_sha256d_hash_112(uint32_t threads, uint32_t startNonce, uint32_t *d_outputHash);
void lbry_sha512_hash_32(uint32_t threads, uint32_t *d_hash);
void lbry_sha256d_hash_final(uint32_t threads, uint32_t startNonce, uint32_t *d_inputHash,
	uint32_t *d_resNonce, const uint64_t target64);
void lbry_sha256_setBlock_112_merged(uint32_t *pdata);
void lbry_merged(uint32_t startNonce, uint32_t threads, uint32_t *d_resNonce, const uint64_t target64);

/* both kernel paths; call after the result buffers are allocated, before mining */
bool lbry_device_selftest(int thr_id, uint32_t *d_res, uint32_t *h_res);

#endif
