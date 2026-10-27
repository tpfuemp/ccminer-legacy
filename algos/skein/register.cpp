#include <miner.h>
#include "algo_registry.h"

ALGO_REGISTER(ALGO_SKEIN, scanhash_skeincoin, free_skeincoin, skeincoinhash,
	"Skein-512 + SHA256 (Skeincoin)");

// Bitcoin genesis header with another time and nonce, hash below 2^224
static const struct algo_test_vector skein_tests[] = {
	{ "01000000" "0000000000000000000000000000000000000000000000000000000000000000"
	  "3ba3edfd7a7b12b27ac72c3e67768f617fc81bc3888a51323a9fb8aa4b1e5e4a"
	  "2eab5f49" "ffff001d" "d5df6acb",
	  "bc520ff9f7fabcc4b5773dcfd883455d865ef7adbb4768da051f647e00000000" },
};
ALGO_TEST_VECTORS(ALGO_SKEIN, skein_tests);
