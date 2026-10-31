#include <miner.h>
#include "algo_registry.h"

ALGO_REGISTER(ALGO_X11, scanhash_x11, free_x11, x11hash,
	"X11, 11 chained 512-bit hashes (Dash)");

// Dash mainnet genesis block
static const struct algo_test_vector x11_tests[] = {
	{ "01000000" "0000000000000000000000000000000000000000000000000000000000000000"
	  "c762a6567f3cc092f0684bb62b7e00a84890b990f07cc71a6bb58d64b98e02e0"
	  "022ddb52" "f0ff0f1e" "c23fb901",
	  "b67a40f3cd5804437a108f105533739c37e6229bc1adcab385140b59fd0f0000" },
};
ALGO_TEST_VECTORS(ALGO_X11, x11_tests);
