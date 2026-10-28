#include <miner.h>
#include "algo_registry.h"

ALGO_REGISTER(ALGO_BLAKE2S, scanhash_blake2s, free_blake2s, blake2s_hash,
	"BLAKE2s-256 (Nevacoin)");

// Bitcoin genesis header with another time and nonce, hash below 2^224
static const struct algo_test_vector blake2s_tests[] = {
	{ "01000000" "0000000000000000000000000000000000000000000000000000000000000000"
	  "3ba3edfd7a7b12b27ac72c3e67768f617fc81bc3888a51323a9fb8aa4b1e5e4a"
	  "30ab5f49" "ffff001d" "0b2961c4",
	  "ec538163285a1a07c96d90d8636706c6b66e603e788cfa3678d16e3300000000" },
};
ALGO_TEST_VECTORS(ALGO_BLAKE2S, blake2s_tests);
