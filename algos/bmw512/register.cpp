#include <miner.h>
#include "algo_registry.h"

ALGO_REGISTER(ALGO_BMW512, scanhash_bmw512, free_bmw512, bmw512_hash,
	"BMW-512");

// Bitcoin genesis header with another time and nonce, hash below 2^224
static const struct algo_test_vector bmw512_tests[] = {
	{ "01000000" "0000000000000000000000000000000000000000000000000000000000000000"
	  "3ba3edfd7a7b12b27ac72c3e67768f617fc81bc3888a51323a9fb8aa4b1e5e4a"
	  "33ab5f49" "ffff001d" "0f1c9b85",
	  "d190f9055a4e21748a5c633ff545bc80125deb499232cdad547d48b300000000" },
};
ALGO_TEST_VECTORS(ALGO_BMW512, bmw512_tests);
