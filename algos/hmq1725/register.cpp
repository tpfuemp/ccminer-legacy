#include <miner.h>
#include "algo_registry.h"

ALGO_REGISTER(ALGO_HMQ1725, scanhash_hmq1725, free_hmq1725, hmq1725hash,
	"HMQ1725, 25 chained 512-bit hashes, 8 picked per nonce by the hash (Espers), alias hmq17");

// Bitcoin genesis header with another time and nonce, hash below 2^232
static const struct algo_test_vector hmq1725_tests[] = {
	{ "01000000" "0000000000000000000000000000000000000000000000000000000000000000"
	  "3ba3edfd7a7b12b27ac72c3e67768f617fc81bc3888a51323a9fb8aa4b1e5e4a"
	  "33ab5f49" "ffff001d" "006de1cd",
	  "f3f40619e3a7ce3e1aa5335c761378bd30b4c81af27fc2291f05951806000000" },
};
ALGO_TEST_VECTORS(ALGO_HMQ1725, hmq1725_tests);
