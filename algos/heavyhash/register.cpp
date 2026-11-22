#include <miner.h>
#include "algo_registry.h"

ALGO_REGISTER(ALGO_HEAVYHASH, scanhash_heavyhash, free_heavyhash, heavyhash_hash,
	"HeavyHash: SHA3-256, 64x64 4-bit matrix, SHA3-256 (Optical Bitcoin)");

// The Optical Bitcoin genesis header and a synthetic one (byte i = i), each with
// the lowest nonce (big-endian at byte 76, as the miner scans) whose hash top
// byte is 0.
static const struct algo_test_vector heavyhash_tests[] = {
	{ "010000000000000000000000000000000000000000000000000000000000000000000000f5febbad19864a6900b6ce84"
	  "287511e6e746229ce6239ff3df7481654778a4c4d3e15d60ffff001c00000369",
	  "3a64e293491e863c3822a00ac0484f3e00a187c9e1fb0e2922e03e63f2981300" },
	{ "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f"
	  "202122232425262728292a2b2c2d2e2f303132333435363738393a3b3c3d3e3f"
	  "404142434445464748494a4b0000000f",
	  "8d8ea39733d68c834e9eebe6238110fb71db6b5beb345c8b793fcd0e07152200" },
};
ALGO_TEST_VECTORS(ALGO_HEAVYHASH, heavyhash_tests);
