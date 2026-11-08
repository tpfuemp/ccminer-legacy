#include <miner.h>
#include "algo_registry.h"

ALGO_REGISTER(ALGO_SKEIN2, scanhash_skein2, free_skein2, skein2hash,
	"Skein-512 twice (Woodcoin)");

// Bitcoin genesis header with another time and nonce, hash below 2^224
static const struct algo_test_vector skein2_tests[] = {
	{ "01000000" "0000000000000000000000000000000000000000000000000000000000000000"
	  "3ba3edfd7a7b12b27ac72c3e67768f617fc81bc3888a51323a9fb8aa4b1e5e4a"
	  "39ab5f49" "ffff001d" "d2d3cc36",
	  "26963e0215e1a2976dc3f557309ca84d1aebd79f07ef8cc7b0b75ef400000000" },
};
ALGO_TEST_VECTORS(ALGO_SKEIN2, skein2_tests);
