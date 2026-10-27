#include <miner.h>
#include "algo_registry.h"

// keccakc is the same hash, only the pool difficulty factor differs
ALGO_REGISTER(ALGO_KECCAK, scanhash_keccak256, free_keccak256, keccak256_hash,
	"Keccak-256 (Maxcoin)");
ALGO_REGISTER(ALGO_KECCAKC, scanhash_keccak256, free_keccak256, keccak256_hash,
	"Keccak-256 (CreativeCoin), pool difficulty factor 256");

static const struct algo_test_vector keccak_tests[] = {
	// Bitcoin genesis header with a nonce giving a keccak256 hash below 2^224
	{ "01000000" "0000000000000000000000000000000000000000000000000000000000000000"
	  "3ba3edfd7a7b12b27ac72c3e67768f617fc81bc3888a51323a9fb8aa4b1e5e4a"
	  "29ab5f49" "ffff001d" "15c6eb93",
	  "1e7a7cd7824dcc74a3efe99681117ad410b561654e975464acc35b0500000000" },
};
ALGO_TEST_VECTORS(ALGO_KECCAK, keccak_tests);
ALGO_TEST_VECTORS(ALGO_KECCAKC, keccak_tests);
