#include <miner.h>
#include "algo_registry.h"

extern "C" void neoscrypt_hash(void *output, const void *input);

// neoscrypt-xaya is the same hash, core handles the header difference
ALGO_REGISTER(ALGO_NEOSCRYPT, scanhash_neoscrypt, free_neoscrypt, neoscrypt_hash,
	"NeoScrypt, memory-hard (Feathercoin and others)");
ALGO_REGISTER(ALGO_XAYA, scanhash_neoscrypt, free_neoscrypt, neoscrypt_hash,
	"NeoScrypt with the XAYA header layout, same hash as neoscrypt");

// Bitcoin genesis header with another nonce, hash below 2^240
static const struct algo_test_vector neoscrypt_tests[] = {
	{ "01000000" "0000000000000000000000000000000000000000000000000000000000000000"
	  "3ba3edfd7a7b12b27ac72c3e67768f617fc81bc3888a51323a9fb8aa4b1e5e4a"
	  "29ab5f49" "ffff001d" "0000dc53",
	  "dc1ceebf50f0f8330862f697a5e9b8212ac9ea6ff101c02489a8ca09b8430000" },
};
ALGO_TEST_VECTORS(ALGO_NEOSCRYPT, neoscrypt_tests);
ALGO_TEST_VECTORS(ALGO_XAYA, neoscrypt_tests);
