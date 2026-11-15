#include <miner.h>
#include "algo_registry.h"
#include "verthash.h"

ALGO_REGISTER(ALGO_VERTHASH, scanhash_verthash, free_verthash, verthash_hash,
	"Verthash, SHA-3 + 4096 reads from the 1.28 GB verthash.dat (Vertcoin)");

static const struct algo_test_vector verthash_tests[] = {
	{ VERTHASH_TV_HEADER, VERTHASH_TV_HASH },
};
ALGO_TEST_VECTORS(ALGO_VERTHASH, verthash_tests);
