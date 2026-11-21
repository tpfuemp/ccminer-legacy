/*
 * The BLAKE3 hasher API that ref/hoohash.c (PePe-core, unchanged) includes as
 * "../blake3.h", over the single-chunk blake3_256(): its inputs are 80 and 32
 * bytes.
 */
#ifndef HOOHASH_BLAKE3_SHIM_H
#define HOOHASH_BLAKE3_SHIM_H

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdbool.h>
#include "cuda/blake3_device.cuh"

typedef struct {
	uint8_t buf[BLAKE3_CHUNK_LEN];
	size_t len;
} blake3_hasher;

static inline void blake3_hasher_init(blake3_hasher *h)
{
	h->len = 0;
}

static inline void blake3_hasher_update(blake3_hasher *h, const void *in, size_t len)
{
	if (len > BLAKE3_CHUNK_LEN - h->len)
		len = BLAKE3_CHUNK_LEN - h->len;   /* never reached: at most 80 bytes */
	memcpy(h->buf + h->len, in, len);
	h->len += len;
}

static inline void blake3_hasher_finalize(const blake3_hasher *h, uint8_t *out, size_t outlen)
{
	uint8_t d[BLAKE3_OUT_LEN];
	blake3_256(h->buf, h->len, d);
	memcpy(out, d, outlen < BLAKE3_OUT_LEN ? outlen : BLAKE3_OUT_LEN);
}

#endif
