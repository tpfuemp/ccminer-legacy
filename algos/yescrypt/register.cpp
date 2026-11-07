#include <miner.h>
#include "algo_registry.h"

extern "C" void free_yescrypt(int thr_id);

// Bitcoin genesis header with another time and nonce: the lowest nonce whose hash top byte is 0

ALGO_REGISTER(ALGO_YESCRYPT, scanhash_yescrypt, free_yescrypt, yescrypt_hash,
	"yescrypt N 2048 r 8 p 1 (GlobalBoost-Y), or --yescrypt-param / --yescrypt-key");
static const struct algo_test_vector yescrypt_tests[] = {
	{ "01000000" "0000000000000000000000000000000000000000000000000000000000000000"
	  "3ba3edfd7a7b12b27ac72c3e67768f617fc81bc3888a51323a9fb8aa4b1e5e4a"
	  "39ab5f49" "ffff001d" "0000011d",
	  "134bc18e0ed45d53de93c9a0efb5488d5831adf08ad5ed33eeb55003c58d0c00" },
};
ALGO_TEST_VECTORS(ALGO_YESCRYPT, yescrypt_tests);

ALGO_REGISTER(ALGO_YESCRYPTR8, scanhash_yescryptr8, free_yescrypt, yescryptr8_hash,
	"yescrypt N 2048 r 8, client key (BitZeny)");
static const struct algo_test_vector yescryptr8_tests[] = {
	{ "01000000" "0000000000000000000000000000000000000000000000000000000000000000"
	  "3ba3edfd7a7b12b27ac72c3e67768f617fc81bc3888a51323a9fb8aa4b1e5e4a"
	  "39ab5f49" "ffff001d" "000000ed",
	  "ad8f50af99a0c8f7e6a9a778023ad89a5991832d70ab0b4de86f81095b4adb00" },
};
ALGO_TEST_VECTORS(ALGO_YESCRYPTR8, yescryptr8_tests);

ALGO_REGISTER(ALGO_YESCRYPTR16, scanhash_yescryptr16, free_yescrypt, yescryptr16_hash,
	"yescrypt N 4096 r 16, client key (Yenten)");
static const struct algo_test_vector yescryptr16_tests[] = {
	{ "01000000" "0000000000000000000000000000000000000000000000000000000000000000"
	  "3ba3edfd7a7b12b27ac72c3e67768f617fc81bc3888a51323a9fb8aa4b1e5e4a"
	  "39ab5f49" "ffff001d" "00000019",
	  "a1fb72c76f30b71d3459051c617ba33690a0d86a65fc75936cb4a9646067b500" },
};
ALGO_TEST_VECTORS(ALGO_YESCRYPTR16, yescryptr16_tests);

ALGO_REGISTER(ALGO_YESCRYPTR16V2, scanhash_yescryptr16v2, free_yescrypt, yescryptr16v2_hash,
	"yescrypt N 4096 r 16 p 4, client key (PPTP)");
static const struct algo_test_vector yescryptr16v2_tests[] = {
	{ "01000000" "0000000000000000000000000000000000000000000000000000000000000000"
	  "3ba3edfd7a7b12b27ac72c3e67768f617fc81bc3888a51323a9fb8aa4b1e5e4a"
	  "39ab5f49" "ffff001d" "00000190",
	  "40114b5fa2687aba1f23f67af89f22327e063f1cea609e394eb8c79c7b3bbd00" },
};
ALGO_TEST_VECTORS(ALGO_YESCRYPTR16V2, yescryptr16v2_tests);

ALGO_REGISTER(ALGO_YESCRYPTR24, scanhash_yescryptr24, free_yescrypt, yescryptr24_hash,
	"yescrypt N 4096 r 24, client key (JagariCoinR)");
static const struct algo_test_vector yescryptr24_tests[] = {
	{ "01000000" "0000000000000000000000000000000000000000000000000000000000000000"
	  "3ba3edfd7a7b12b27ac72c3e67768f617fc81bc3888a51323a9fb8aa4b1e5e4a"
	  "39ab5f49" "ffff001d" "0000019f",
	  "6c536e20b8e02559082f0b9fa2b43f7763f1919acc847e4244f387b881727e00" },
};
ALGO_TEST_VECTORS(ALGO_YESCRYPTR24, yescryptr24_tests);

ALGO_REGISTER(ALGO_YESCRYPTR32, scanhash_yescryptr32, free_yescrypt, yescryptr32_hash,
	"yescrypt N 4096 r 32, client key (WAVI)");
static const struct algo_test_vector yescryptr32_tests[] = {
	{ "01000000" "0000000000000000000000000000000000000000000000000000000000000000"
	  "3ba3edfd7a7b12b27ac72c3e67768f617fc81bc3888a51323a9fb8aa4b1e5e4a"
	  "39ab5f49" "ffff001d" "00000086",
	  "7348bc946135a11ed64908632238c775a47bf834c99db9eb6677202a34d5de00" },
};
ALGO_TEST_VECTORS(ALGO_YESCRYPTR32, yescryptr32_tests);
