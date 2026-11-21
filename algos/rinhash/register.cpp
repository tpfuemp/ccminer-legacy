#include <miner.h>
#include "algo_registry.h"

extern "C" void rinhash_hash(void *output, const void *input);

ALGO_REGISTER(ALGO_RINHASH, scanhash_rinhash, free_rinhash, rinhash_hash,
	"RinHash: BLAKE3, Argon2d (64 KB per hash), SHA3-256 (Rincoin)");

// The headers of Rincoin blocks 800000 and 828000, each with the lowest nonce
// (little-endian at byte 76, as hashed) whose hash top byte is 0.
static const struct algo_test_vector rinhash_tests[] = {
	{ "00000020ecbdc1deca7adee6b25b4995d75f1cb05eb3628fb7c11bda9d2c201d03000000"
	  "b6cef4d970ee5733a04b6e7daab497f0cfac881c6e80e0dee92b7d85ca0afc649453e56a"
	  "ca8c031da4010000",
	  "16edd95051e6621fdf010534196384b898060024d084fd601ec59d255ee41000" },
	{ "0000002056c562d56c8c0a923ac7211340607d37b2c3f8ccbde02c3cd3faa65102000000"
	  "387bbbcf7440b0d6f158268124a36bd8c62aeae88c00d8b1670f52ae495d8088f806016b"
	  "b464061d88000000",
	  "2ccedc03b30f201062b7e33caae2fc101cc13a11c5d3ce73599776056fe4e800" },
};
ALGO_TEST_VECTORS(ALGO_RINHASH, rinhash_tests);
