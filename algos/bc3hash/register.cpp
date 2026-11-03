#include <miner.h>
#include "algo_registry.h"

ALGO_REGISTER(ALGO_BC3HASH, scanhash_bc3hash, free_bc3hash, bc3hash_hash,
	"BC3Hash (Bitcoin III, from block 64360), alias bc3");

// the coin's first test vector header with nonce 5
static const struct algo_test_vector bc3hash_tests[] = {
	{ "00080020" "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f"
	  "202122232425262728292a2b2c2d2e2f303132333435363738393a3b3c3d3e3f"
	  "00d2496b" "ffff7f20" "00000005",
	  "e60e650d57dc038c1898feb7cd34e928b287c47a668dc71c6bda31fe3e3f0aa1" },
};
ALGO_TEST_VECTORS(ALGO_BC3HASH, bc3hash_tests);
