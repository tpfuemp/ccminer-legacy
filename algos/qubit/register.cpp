#include <miner.h>
#include "algo_registry.h"

ALGO_REGISTER(ALGO_QUBIT, scanhash_qubit, free_qubit, qubithash,
	"Luffa + CubeHash + SHAvite + SIMD + ECHO, 512 bits each (Qubit)");

// Bitcoin genesis header with another time and nonce, hash top word below 0x100
static const struct algo_test_vector qubit_tests[] = {
	{ "01000000" "0000000000000000000000000000000000000000000000000000000000000000"
	  "3ba3edfd7a7b12b27ac72c3e67768f617fc81bc3888a51323a9fb8aa4b1e5e4a"
	  "36ab5f49" "ffff001d" "00a31512",
	  "03ccee9d0e2678a80a41b995facd0a1a1c1fb6556b3c89756c9b99ad10000000" },
};
ALGO_TEST_VECTORS(ALGO_QUBIT, qubit_tests);
