// SPDX-License-Identifier: GPL-3.0-or-later
/*
 * BLAKE3-256, unkeyed, single chunk (input <= 1024 bytes), one body for host
 * and device. Everything is static, so several translation units may include it.
 */
#ifndef CUDA_BLAKE3_DEVICE_CUH
#define CUDA_BLAKE3_DEVICE_CUH

#include <stdint.h>
#include <stddef.h>

#ifdef __CUDACC__
#define B3_HD __host__ __device__
#define B3_HDI __host__ __device__ __forceinline__
#else
#define B3_HD
#define B3_HDI inline
#endif

#define BLAKE3_OUT_LEN		32
#define BLAKE3_BLOCK_LEN	64
#define BLAKE3_CHUNK_LEN	1024

/* flag bits (the only ones a single chunk needs) */
#define BLAKE3_CHUNK_START	1u
#define BLAKE3_CHUNK_END	2u
#define BLAKE3_ROOT		8u

static B3_HDI uint32_t blake3_rotr(uint32_t w, int c)
{
	return (w >> c) | (w << (32 - c));
}

static B3_HDI void blake3_g(uint32_t *s, int a, int b, int c, int d, uint32_t mx, uint32_t my)
{
	s[a] = s[a] + s[b] + mx;
	s[d] = blake3_rotr(s[d] ^ s[a], 16);
	s[c] = s[c] + s[d];
	s[b] = blake3_rotr(s[b] ^ s[c], 12);
	s[a] = s[a] + s[b] + my;
	s[d] = blake3_rotr(s[d] ^ s[a], 8);
	s[c] = s[c] + s[d];
	s[b] = blake3_rotr(s[b] ^ s[c], 7);
}

static B3_HD void blake3_round(uint32_t *s, const uint32_t *m)
{
	blake3_g(s, 0, 4, 8,  12, m[0],  m[1]);
	blake3_g(s, 1, 5, 9,  13, m[2],  m[3]);
	blake3_g(s, 2, 6, 10, 14, m[4],  m[5]);
	blake3_g(s, 3, 7, 11, 15, m[6],  m[7]);
	blake3_g(s, 0, 5, 10, 15, m[8],  m[9]);
	blake3_g(s, 1, 6, 11, 12, m[10], m[11]);
	blake3_g(s, 2, 7, 8,  13, m[12], m[13]);
	blake3_g(s, 3, 4, 9,  14, m[14], m[15]);
}

static B3_HD const uint32_t *blake3_iv(void)
{
	static const uint32_t IV[8] = {
		0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
		0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u
	};
	return IV;
}

/* one compression; with ROOT, out_cv is the digest */
static B3_HD void blake3_compress(const uint32_t cv[8], const uint32_t block[16],
	uint64_t counter, uint32_t block_len, uint32_t flags, uint32_t out_cv[8])
{
	const uint32_t IV[8] = {
		0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
		0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u
	};
	const uint8_t SIGMA[16] = { 2,6,3,10,7,0,4,13,1,11,12,5,9,14,15,8 };

	uint32_t s[16];
	s[0] = cv[0]; s[1] = cv[1]; s[2] = cv[2]; s[3] = cv[3];
	s[4] = cv[4]; s[5] = cv[5]; s[6] = cv[6]; s[7] = cv[7];
	s[8] = IV[0]; s[9] = IV[1]; s[10] = IV[2]; s[11] = IV[3];
	s[12] = (uint32_t)counter;
	s[13] = (uint32_t)(counter >> 32);
	s[14] = block_len;
	s[15] = flags;

	uint32_t m[16];
	for (int i = 0; i < 16; i++)
		m[i] = block[i];

	for (int r = 0; r < 7; r++) {
		blake3_round(s, m);
		if (r < 6) {
			uint32_t pm[16];
			for (int i = 0; i < 16; i++) pm[i] = m[SIGMA[i]];
			for (int i = 0; i < 16; i++) m[i] = pm[i];
		}
	}

	for (int i = 0; i < 8; i++)
		out_cv[i] = s[i] ^ s[i + 8];
}

static B3_HDI void blake3_words_from_le(const uint8_t *b, uint32_t *w)
{
	for (int i = 0; i < 16; i++)
		w[i] = (uint32_t)b[i*4] | ((uint32_t)b[i*4+1] << 8)
		     | ((uint32_t)b[i*4+2] << 16) | ((uint32_t)b[i*4+3] << 24);
}

/* unkeyed BLAKE3-256 of a single chunk into 32 bytes */
static B3_HD void blake3_256(const uint8_t *input, size_t len, uint8_t *out)
{
	const uint32_t *IV = blake3_iv();

	uint32_t cv[8];
	for (int i = 0; i < 8; i++)
		cv[i] = IV[i];

	size_t pos = 0;
	uint32_t blocks_done = 0;
	for (;;) {
		const size_t remaining = len - pos;
		const uint32_t block_len = remaining >= BLAKE3_BLOCK_LEN
			? (uint32_t)BLAKE3_BLOCK_LEN : (uint32_t)remaining;
		const bool is_last = (remaining <= BLAKE3_BLOCK_LEN);

		uint8_t buf[BLAKE3_BLOCK_LEN];
		for (int i = 0; i < BLAKE3_BLOCK_LEN; i++)
			buf[i] = (i < (int)block_len) ? input[pos + i] : 0;

		uint32_t block[16];
		blake3_words_from_le(buf, block);

		uint32_t flags = 0;
		if (blocks_done == 0) flags |= BLAKE3_CHUNK_START;
		if (is_last)          flags |= BLAKE3_CHUNK_END | BLAKE3_ROOT;

		uint32_t next[8];
		blake3_compress(cv, block, 0, block_len, flags, next);
		for (int i = 0; i < 8; i++)
			cv[i] = next[i];

		blocks_done++;
		pos += block_len;
		if (is_last)
			break;
	}

	for (int i = 0; i < 8; i++) {
		out[i*4]   = (uint8_t)(cv[i]);
		out[i*4+1] = (uint8_t)(cv[i] >> 8);
		out[i*4+2] = (uint8_t)(cv[i] >> 16);
		out[i*4+3] = (uint8_t)(cv[i] >> 24);
	}
}

/* midstate split for a nonce in the last block: the host compresses the blocks
 * before it once per job, the device only the final block */

/* compress nblocks whole blocks, none of them the last of the chunk */
static B3_HD void blake3_prefix(const uint8_t *in, uint32_t nblocks, uint32_t cv[8])
{
	const uint32_t *IV = blake3_iv();
	for (int i = 0; i < 8; i++)
		cv[i] = IV[i];

	for (uint32_t b = 0; b < nblocks; b++) {
		uint32_t block[16], next[8];
		blake3_words_from_le(in + (size_t)b * BLAKE3_BLOCK_LEN, block);
		blake3_compress(cv, block, 0, BLAKE3_BLOCK_LEN,
			b == 0 ? BLAKE3_CHUNK_START : 0u, next);
		for (int i = 0; i < 8; i++)
			cv[i] = next[i];
	}
}

/* final block from a midstate; m: 16 LE words, zero-padded past block_len */
static B3_HD void blake3_final(const uint32_t cv[8], const uint32_t m[16],
	uint32_t block_len, bool first_block, uint8_t *out)
{
	uint32_t flags = BLAKE3_CHUNK_END | BLAKE3_ROOT;
	if (first_block)
		flags |= BLAKE3_CHUNK_START;

	uint32_t next[8];
	blake3_compress(cv, m, 0, block_len, flags, next);

	for (int i = 0; i < 8; i++) {
		out[i*4]   = (uint8_t)(next[i]);
		out[i*4+1] = (uint8_t)(next[i] >> 8);
		out[i*4+2] = (uint8_t)(next[i] >> 16);
		out[i*4+3] = (uint8_t)(next[i] >> 24);
	}
}

/* decred: 180-byte header = 64 + 64 + 52, nonce at byte 140 = word 3 of block 3 */
#define DECRED_HDR_LEN		180
#define DECRED_NONCE_BYTE	140
#define DECRED_PREFIX_BLOCKS	2
#define DECRED_TAIL_LEN		(DECRED_HDR_LEN - DECRED_PREFIX_BLOCKS * BLAKE3_BLOCK_LEN)
#define DECRED_NONCE_WORD	((DECRED_NONCE_BYTE - DECRED_PREFIX_BLOCKS * BLAKE3_BLOCK_LEN) / 4)

/* host, once per job: midstate of bytes 0..127 and the tail words */
static B3_HD void decred_blake3_prepare(const uint8_t *header, uint32_t cv[8], uint32_t m[16])
{
	blake3_prefix(header, DECRED_PREFIX_BLOCKS, cv);

	uint8_t tail[BLAKE3_BLOCK_LEN];
	for (int i = 0; i < BLAKE3_BLOCK_LEN; i++)
		tail[i] = (i < DECRED_TAIL_LEN)
			? header[DECRED_PREFIX_BLOCKS * BLAKE3_BLOCK_LEN + i] : 0;
	blake3_words_from_le(tail, m);
}

/* device, once per nonce */
static B3_HD void decred_blake3_nonce(const uint32_t cv[8], const uint32_t m[16],
	uint32_t nonce, uint8_t *out)
{
	uint32_t mm[16];
	for (int i = 0; i < 16; i++)
		mm[i] = m[i];
	mm[DECRED_NONCE_WORD] = nonce;

	blake3_final(cv, mm, DECRED_TAIL_LEN, false, out);
}

#endif // CUDA_BLAKE3_DEVICE_CUH
