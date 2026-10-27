#include <miner.h>
#include "algo_registry.h"

ALGO_REGISTER(ALGO_SHA256D, scanhash_sha256d, free_sha256d, sha256d_hash,
	"SHA256d (Bitcoin)");

static const struct algo_test_vector sha256d_tests[] = {
	// Bitcoin genesis block, hash 000000000019d668...0a8ce26f
	{ "01000000" "0000000000000000000000000000000000000000000000000000000000000000"
	  "3ba3edfd7a7b12b27ac72c3e67768f617fc81bc3888a51323a9fb8aa4b1e5e4a"
	  "29ab5f49" "ffff001d" "1dac2b7c",
	  "6fe28c0ab6f1b372c1a6a246ae63f74f931e8365e15a089c68d6190000000000" },
};
ALGO_TEST_VECTORS(ALGO_SHA256D, sha256d_tests);
