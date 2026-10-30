#include <miner.h>
#include "algo_registry.h"

// whirlcoin is the same hash, core handles the merkle root difference
ALGO_REGISTER(ALGO_WHIRLPOOL, scanhash_whirl, free_whirl, wcoinhash,
	"4x Whirlpool-T (old whirlpool)");
ALGO_REGISTER(ALGO_WHIRLCOIN, scanhash_whirl, free_whirl, wcoinhash,
	"4x Whirlpool-T (Whirlcoin), same as whirlpool");

// Bitcoin genesis header with another time and nonce, hash below 2^224
static const struct algo_test_vector whirlpool_tests[] = {
	{ "01000000" "0000000000000000000000000000000000000000000000000000000000000000"
	  "3ba3edfd7a7b12b27ac72c3e67768f617fc81bc3888a51323a9fb8aa4b1e5e4a"
	  "34ab5f49" "ffff001d" "1cda1811",
	  "503587a6cb4d2b9ee128dcdb743477caa8ee5c8ce20ae2b28d59f98500000000" },
};
ALGO_TEST_VECTORS(ALGO_WHIRLPOOL, whirlpool_tests);
ALGO_TEST_VECTORS(ALGO_WHIRLCOIN, whirlpool_tests);
