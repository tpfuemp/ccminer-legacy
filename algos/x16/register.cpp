#include <miner.h>
#include "algo_registry.h"

ALGO_REGISTER(ALGO_X16R, scanhash_x16r, free_x16r, x16r_hash,
	"X16R, 16 chained 512-bit hashes in prevhash order");
ALGO_REGISTER(ALGO_X16S, scanhash_x16s, free_x16s, x16s_hash,
	"X16S, X16R with the order shuffled by the prevhash");
ALGO_REGISTER(ALGO_X16RV2, scanhash_x16rv2, free_x16rv2, x16rv2_hash,
	"X16RV2, X16R with Tiger-192 before Keccak, Luffa and SHA-512");
ALGO_REGISTER(ALGO_X16RT, scanhash_x16rt, free_x16rt, x16rt_hash,
	"X16RT, X16R with the order taken from the block time");
ALGO_REGISTER(ALGO_TIMETRAVEL, scanhash_timetravel, free_timetravel, timetravel_hash,
	"Timetravel, 8 of the X16R blocks in an order picked by the block time (Machinecoin)");

// Bitcoin genesis header with a fixed prevhash (first block ECHO, SHA-512, Tiger+Keccak;
// x16rt: ECHO from the genesis time), nonce giving a hash below 2^232
#define X16_HDR(prev7) "01000000" "9fc4e90e33587d" prev7 "c7ec11365b80a5caef14395e83a8cdf2173c6186abd0f51a" \
	"3ba3edfd7a7b12b27ac72c3e67768f617fc81bc3888a51323a9fb8aa4b1e5e4a" "29ab5f49" "ffff001d"
static const struct algo_test_vector x16r_tests[] = {
	{ X16_HDR("a2") "00a42a33", "516b5c69280013de874bd2ff1c8aa031cdd247f63f3a578f369e0ded76000000" },
};
static const struct algo_test_vector x16s_tests[] = {
	{ X16_HDR("92") "003ed2fc", "901b8ec9f192a6cb2ec6852f78d8635cd6847737fbe192e57f06dcd63e000000" },
};
static const struct algo_test_vector x16rv2_tests[] = {
	{ X16_HDR("42") "019420de", "45f311febe6d8c6a12ff2389264f397f5b4503e6935f6aef1131820660000000" },
};
static const struct algo_test_vector x16rt_tests[] = {
	{ X16_HDR("a2") "009a0dcd", "1f5d97cc1221fc3c924f540df976156c6578d395a7dfcb132cc4bf66f6000000" },
};
ALGO_TEST_VECTORS(ALGO_X16R, x16r_tests);
ALGO_TEST_VECTORS(ALGO_X16S, x16s_tests);
ALGO_TEST_VECTORS(ALGO_X16RV2, x16rv2_tests);
ALGO_TEST_VECTORS(ALGO_X16RT, x16rt_tests);

// Bitcoin genesis header with ntime 0x69000000 (order BMW first), hash below 2^232
static const struct algo_test_vector timetravel_tests[] = {
	{ "01000000" "0000000000000000000000000000000000000000000000000000000000000000"
	  "3ba3edfd7a7b12b27ac72c3e67768f617fc81bc3888a51323a9fb8aa4b1e5e4a"
	  "00000069" "ffff001d" "00238e49",
	  "b3cef4760ffdaec27b18f0c0665e86420bb04a30d34e9b24ee3f12fba0000000" },
};
ALGO_TEST_VECTORS(ALGO_TIMETRAVEL, timetravel_tests);
