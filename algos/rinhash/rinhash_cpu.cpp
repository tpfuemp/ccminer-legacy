/*
 * RinHash on the CPU, as the Rincoin node computes it (src/crypto/rinhash.cpp):
 * BLAKE3-256 of the 80-byte header, Argon2d (password = that digest, salt
 * "RinCoinSalt", t_cost 2, m_cost 64 KiB, 1 lane, version 1.3, 32 bytes) with
 * the reference implementation in argon2/, then SHA3-256.
 */
#include <string.h>
#include <stdint.h>

#include "cuda/blake3_device.cuh"
#include "argon2/argon2.h"
extern "C" {
#include "sph/sha3.h"
}

extern "C" void rinhash_hash(void *output, const void *input)
{
	static const char salt[] = "RinCoinSalt";
	uint8_t b3[32], a2[32];

	blake3_256((const uint8_t *)input, 80, b3);

	argon2_context c;
	memset(&c, 0, sizeof(c));
	c.out = a2;
	c.outlen = sizeof(a2);
	c.pwd = b3;
	c.pwdlen = sizeof(b3);
	c.salt = (uint8_t *)salt;
	c.saltlen = (uint32_t)(sizeof(salt) - 1);
	c.t_cost = 2;
	c.m_cost = 64;
	c.lanes = 1;
	c.threads = 1;
	c.version = ARGON2_VERSION_13;
	c.flags = ARGON2_DEFAULT_FLAGS;
	if (argon2d_ctx(&c) != ARGON2_OK) {
		memset(output, 0xff, 32);   /* fails every target */
		return;
	}
	sha3(a2, sizeof(a2), output, 32);
}
