#include <miner.h>
#include "algo_registry.h"

ALGO_REGISTER(ALGO_LYRA2v2, scanhash_lyra2v2, free_lyra2v2, lyra2v2_hash,
	"Lyra2REv2: BLAKE-256, Keccak-256, CubeHash-256, Lyra2, Skein-256, CubeHash-256, BMW-256");

// Bitcoin genesis header with another time and nonce, hash top word below 0x100
static const struct algo_test_vector lyra2v2_tests[] = {
	{ "01000000" "0000000000000000000000000000000000000000000000000000000000000000"
	  "3ba3edfd7a7b12b27ac72c3e67768f617fc81bc3888a51323a9fb8aa4b1e5e4a"
	  "39ab5f49" "ffff001d" "00042b41",
	  "9076a6a8a99e905d130fc9b9f47407fa47e0068f71fa2bbe4dea9f39d4000000" },
};
ALGO_TEST_VECTORS(ALGO_LYRA2v2, lyra2v2_tests);
