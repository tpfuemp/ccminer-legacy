#include <miner.h>
#include "algo_registry.h"

ALGO_REGISTER(ALGO_ALLIUM, scanhash_allium, free_allium, allium_hash,
	"BLAKE-256, Keccak-256, Lyra2, CubeHash-256, Lyra2, Skein-256, Groestl-256 (Garlicoin)");

// Bitcoin genesis header with another time and nonce, hash top word below 0x100
static const struct algo_test_vector allium_tests[] = {
	{ "01000000" "0000000000000000000000000000000000000000000000000000000000000000"
	  "3ba3edfd7a7b12b27ac72c3e67768f617fc81bc3888a51323a9fb8aa4b1e5e4a"
	  "39ab5f49" "ffff001d" "004da384",
	  "742f9cb6633fc8a22b87e9a4dbdf72e3ee45eb49c82114364bbaf9cc94000000" },
};
ALGO_TEST_VECTORS(ALGO_ALLIUM, allium_tests);
