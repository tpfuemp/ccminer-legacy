#include <miner.h>
#include "algo_registry.h"

// dmd-gr is the same hash, core handles both alike
ALGO_REGISTER(ALGO_GROESTL, scanhash_groestlcoin, free_groestlcoin, groestlhash,
	"Double Groestl-512 (Groestlcoin)");
ALGO_REGISTER(ALGO_DMD_GR, scanhash_groestlcoin, free_groestlcoin, groestlhash,
	"Double Groestl-512 (Diamond), same as groestl");
ALGO_REGISTER(ALGO_MYR_GR, scanhash_myriad, free_myriad, myriadhash,
	"Groestl-512 + SHA256 (Myriad)");

// Bitcoin genesis header with another time and nonce, hash top word below 0x100
static const struct algo_test_vector groestl_tests[] = {
	{ "01000000" "0000000000000000000000000000000000000000000000000000000000000000"
	  "3ba3edfd7a7b12b27ac72c3e67768f617fc81bc3888a51323a9fb8aa4b1e5e4a"
	  "31ab5f49" "ffff001d" "01037018",
	  "91f6d3afdc7e5dad8d99bf5b03e19f286df3c09fd81490c9a4a4cb21ad000000" },
};
static const struct algo_test_vector myriad_tests[] = {
	{ "01000000" "0000000000000000000000000000000000000000000000000000000000000000"
	  "3ba3edfd7a7b12b27ac72c3e67768f617fc81bc3888a51323a9fb8aa4b1e5e4a"
	  "32ab5f49" "ffff001d" "01515b64",
	  "afa7c406935ebd74987e665754567dcd513a61e6b5c8f0deb8416d82d2000000" },
};
ALGO_TEST_VECTORS(ALGO_GROESTL, groestl_tests);
ALGO_TEST_VECTORS(ALGO_DMD_GR, groestl_tests);
ALGO_TEST_VECTORS(ALGO_MYR_GR, myriad_tests);
