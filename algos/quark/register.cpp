#include <miner.h>
#include "algo_registry.h"

ALGO_REGISTER(ALGO_QUARK, scanhash_quark, free_quark, quarkhash,
	"Quark, 9 steps of BLAKE/BMW/Groestl/JH/Keccak/Skein-512, 3 of them hash-dependent");

// Bitcoin genesis header with another time and nonce, hash top word below 0x100
static const struct algo_test_vector quark_tests[] = {
	{ "01000000" "0000000000000000000000000000000000000000000000000000000000000000"
	  "3ba3edfd7a7b12b27ac72c3e67768f617fc81bc3888a51323a9fb8aa4b1e5e4a"
	  "38ab5f49" "ffff001d" "005e3595",
	  "0cef9b4931daa401cb1ac71dbe8e66232d6cf56255663878276d6e5bf6000000" },
};
ALGO_TEST_VECTORS(ALGO_QUARK, quark_tests);
