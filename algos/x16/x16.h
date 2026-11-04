#ifndef X16_H
#define X16_H

#include <stdint.h>

/* chain order variants: x16r (prevhash nibbles), x16s (shuffle by them), x16rv2 (x16r
 * order, Tiger-192 before Keccak, Luffa and SHA-512) */
enum x16_variant { X16_R = 0, X16_S = 1, X16_RV2 = 2 };

/* order[i] = block index 0..15 of chain position i, from header bytes 4..11 */
void x16_order(int variant, const uint8_t *header80, uint8_t order[16]);

/* CPU reference of the 80-byte serialized header, 32 bytes out */
void x16_hash(int variant, void *output, const void *input);

/* one launch chain over [startNonce, startNonce + threads) for the block, order and
 * target set by x16_setBlock(); up to max_out candidates ascending into out, returns how
 * many, *total gets how many were found */
void x16_setBlock(int thr_id, int variant, uint32_t *endiandata, const uint32_t *ptarget);
uint32_t x16_scan(int thr_id, uint32_t threads, uint32_t startNonce,
	uint32_t *out, uint32_t max_out, uint32_t *total);

/* call after the buffers are allocated, before mining; covers all three variants */
bool x16_device_selftest(int thr_id);

#endif
