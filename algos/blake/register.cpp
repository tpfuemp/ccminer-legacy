#include <miner.h>
#include "algo_registry.h"

// blake: 14 rounds, blakecoin: 8 rounds
static int scanhash_blake(int thr_id, struct work *work, uint32_t max_nonce, unsigned long *hashes_done)
{
	return scanhash_blake256(thr_id, work, max_nonce, hashes_done, 14);
}

static int scanhash_blakecoin(int thr_id, struct work *work, uint32_t max_nonce, unsigned long *hashes_done)
{
	return scanhash_blake256(thr_id, work, max_nonce, hashes_done, 8);
}

static void blake_hash(void *output, const void *input) { blake256hash(output, input, 14); }
static void blakecoin_hash(void *output, const void *input) { blake256hash(output, input, 8); }

ALGO_REGISTER(ALGO_BLAKE, scanhash_blake, free_blake256, blake_hash,
	"Blake-256, 14 rounds (Saffroncoin)");
ALGO_REGISTER(ALGO_BLAKECOIN, scanhash_blakecoin, free_blake256, blakecoin_hash,
	"Blake-256, 8 rounds (Blakecoin)");

// Bitcoin genesis header with another time and nonce, hash below 2^224
static const struct algo_test_vector blake_tests[] = {
	{ "01000000" "0000000000000000000000000000000000000000000000000000000000000000"
	  "3ba3edfd7a7b12b27ac72c3e67768f617fc81bc3888a51323a9fb8aa4b1e5e4a"
	  "2aab5f49" "ffff001d" "87cd94f3",
	  "683434bc9ade2ec5635d549bc983c0098224e21ba3ece2d8a404aa0d00000000" },
};
static const struct algo_test_vector blakecoin_tests[] = {
	{ "01000000" "0000000000000000000000000000000000000000000000000000000000000000"
	  "3ba3edfd7a7b12b27ac72c3e67768f617fc81bc3888a51323a9fb8aa4b1e5e4a"
	  "2cab5f49" "ffff001d" "a928b973",
	  "80975e07f5b029ff4f8a366cb7c67fd8c066d4f449b06e137da44a5800000000" },
};
ALGO_TEST_VECTORS(ALGO_BLAKE, blake_tests);
ALGO_TEST_VECTORS(ALGO_BLAKECOIN, blakecoin_tests);
