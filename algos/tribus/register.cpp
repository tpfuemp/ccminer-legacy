#include <miner.h>
#include "algo_registry.h"

ALGO_REGISTER(ALGO_TRIBUS, scanhash_tribus, free_tribus, tribus_hash,
	"JH-512 + Keccak-512 + ECHO-512 (Denarius)");

// Bitcoin genesis header with another time and nonce, hash top word below 0x100
static const struct algo_test_vector tribus_tests[] = {
	{ "01000000" "0000000000000000000000000000000000000000000000000000000000000000"
	  "3ba3edfd7a7b12b27ac72c3e67768f617fc81bc3888a51323a9fb8aa4b1e5e4a"
	  "35ab5f49" "ffff001d" "014633c7",
	  "01267fe436afab515dae226cb430bd2fabf70978d8fa448a9adf406977000000" },
};
ALGO_TEST_VECTORS(ALGO_TRIBUS, tribus_tests);
