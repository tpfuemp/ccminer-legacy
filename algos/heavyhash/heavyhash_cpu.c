/*
 * HeavyHash (Optical Bitcoin, obtc-core CBlockHeader::GetPoWHash): the 64x64
 * matrix of 4-bit values comes from xoshiro256++ seeded with SHA3-256 of the
 * previous block hash (header bytes 4..35), regenerated until it has full
 * rank. Per nonce: h = SHA3-256(header), v = the 64 nibbles of h (high nibble
 * first), p[i] = (row i . v) >> 10, PoW = SHA3-256(h ^ nibbles of p).
 */
#include <math.h>
#include <string.h>

#include "sph/sha3.h"
#include "heavyhash_cpu.h"

static uint64_t hh_rotl64(uint64_t x, int k)
{
	return (x << k) | (x >> (64 - k));
}

static uint64_t hh_xoshiro(uint64_t s[4])
{
	const uint64_t result = hh_rotl64(s[0] + s[3], 23) + s[0];
	const uint64_t t = s[1] << 17;

	s[2] ^= s[0];
	s[3] ^= s[1];
	s[1] ^= s[2];
	s[0] ^= s[3];
	s[2] ^= t;
	s[3] = hh_rotl64(s[3], 45);
	return result;
}

/* Gaussian elimination in double. The node tests the singular values of an
 * SVD against 1e-12; both agree for every matrix that is not near-singular,
 * and a random 4-bit 64x64 matrix is almost never singular. */
static int hh_full_rank(const uint8_t m[64][64])
{
	static const double eps = 1e-9;
	double b[64][64];
	int row_used[64] = { 0 };
	int rank = 0;

	for (int i = 0; i < 64; i++)
		for (int j = 0; j < 64; j++)
			b[i][j] = m[i][j];

	for (int i = 0; i < 64; i++) {
		int j;
		for (j = 0; j < 64; j++)
			if (!row_used[j] && fabs(b[j][i]) > eps)
				break;
		if (j == 64)
			continue;
		rank++;
		row_used[j] = 1;
		for (int p = i + 1; p < 64; p++)
			b[j][p] /= b[j][i];
		for (int k = 0; k < 64; k++) {
			if (k != j && fabs(b[k][i]) > eps) {
				for (int p = i + 1; p < 64; p++)
					b[k][p] -= b[j][p] * b[k][i];
			}
		}
	}
	return rank == 64;
}

void heavyhash_matrix(uint8_t m[64][64], const void *prevhash)
{
	uint8_t seed[32];
	uint64_t s[4];

	sha3(prevhash, 32, seed, 32);
	for (int i = 0; i < 4; i++) {
		s[i] = 0;
		for (int k = 7; k >= 0; k--)
			s[i] = (s[i] << 8) | seed[8 * i + k];
	}

	do {
		for (int i = 0; i < 64; i++) {
			for (int j = 0; j < 64; j += 16) {
				const uint64_t value = hh_xoshiro(s);
				for (int shift = 0; shift < 16; shift++)
					m[i][j + shift] = (uint8_t) ((value >> (4 * shift)) & 0xF);
			}
		}
	} while (!hh_full_rank(m));
}

void heavyhash_with_matrix(const uint8_t m[64][64], const void *data, size_t len, void *output)
{
	uint8_t h[32], x[32];
	uint32_t v[64];

	sha3(data, len, h, 32);
	for (int i = 0; i < 32; i++) {
		v[2 * i] = h[i] >> 4;
		v[2 * i + 1] = h[i] & 0xF;
	}
	for (int i = 0; i < 32; i++) {
		uint32_t s0 = 0, s1 = 0;
		for (int j = 0; j < 64; j++) {
			s0 += m[2 * i][j] * v[j];
			s1 += m[2 * i + 1][j] * v[j];
		}
		x[i] = h[i] ^ (uint8_t) (((s0 >> 10) << 4) | (s1 >> 10));
	}
	sha3(x, 32, output, 32);
}

/* the 80 header bytes; builds the matrix on every call */
void heavyhash_hash(void *output, const void *input)
{
	uint8_t m[64][64];

	heavyhash_matrix(m, (const uint8_t *) input + 4);
	heavyhash_with_matrix(m, input, 80, output);
}
