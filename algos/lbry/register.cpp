#include <miner.h>
#include "algo_registry.h"

ALGO_REGISTER(ALGO_LBRY, scanhash_lbry, free_lbry, lbry_hash,
	"LBRY, SHA-256d + SHA-512 + RIPEMD-160 (LBRY Credits)");

// LBRY genesis header with ntime byte 34 and another nonce, hash below 2^232
static const struct algo_test_vector lbry_tests[] = {
	{ "01000000" "0000000000000000000000000000000000000000000000000000000000000000"
	  "cc59e59ff97ac092b55e423aa5495151ed6fb80570a5bb78cd5bd1c3821c21b8"
	  "0100000000000000000000000000000000000000000000000000000000000000"
	  "34193156" "ffff001f" "aa14d202",
	  "8d8d8538e0e83e25c11943dd615acc7c27cd3a8cf7ddc393d999118e31000000" },
};
ALGO_TEST_VECTORS(ALGO_LBRY, lbry_tests);
