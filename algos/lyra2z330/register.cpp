#include <miner.h>
#include "algo_registry.h"

extern "C" void lyra2z330_hash(void *output, const void *input);

ALGO_REGISTER(ALGO_LYRA2Z330, scanhash_lyra2z330, free_lyra2z330, lyra2z330_hash,
	"Lyra2 over the header, 330 rows x 256 blocks, time cost 2 (Gravity)");

// Bitcoin genesis header with another time and nonce, hash top byte 0 at a small nonce
static const struct algo_test_vector lyra2z330_tests[] = {
	{ "01000000" "0000000000000000000000000000000000000000000000000000000000000000"
	  "3ba3edfd7a7b12b27ac72c3e67768f617fc81bc3888a51323a9fb8aa4b1e5e4a"
	  "3aab5f49" "ffff001d" "0000004b",
	  "126de1ac244cfed0253ab8fe9158a0a15aa58f3e95532ed5ca08e0d5a113e800" },
};
ALGO_TEST_VECTORS(ALGO_LYRA2Z330, lyra2z330_tests);
