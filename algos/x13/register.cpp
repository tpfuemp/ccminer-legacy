#include <miner.h>
#include "algo_registry.h"

ALGO_REGISTER(ALGO_X13, scanhash_x13, free_x13, x13hash,
	"X13, X11 + Hamsi + Fugue, 512 bits each");

// Bitcoin genesis header with another time and nonce, hash top word below 0x100
static const struct algo_test_vector x13_tests[] = {
	{ "01000000" "0000000000000000000000000000000000000000000000000000000000000000"
	  "3ba3edfd7a7b12b27ac72c3e67768f617fc81bc3888a51323a9fb8aa4b1e5e4a"
	  "37ab5f49" "ffff001d" "00df1087",
	  "be2ae451c663abe9046cbfaa142a723bdaf29f4d4577174d997e74af12000000" },
};
ALGO_TEST_VECTORS(ALGO_X13, x13_tests);
