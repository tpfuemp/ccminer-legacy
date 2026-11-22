#include <miner.h>
#include "algo_registry.h"

extern "C" void sha512256d_hash(void *output, const void *input);

ALGO_REGISTER(ALGO_SHA512256D, scanhash_sha512256d, free_sha512256d, sha512256d_hash,
	"Double SHA-512/256 (Radiant)");

// The headers of Radiant blocks 100000 and 480000, each with the lowest nonce
// (big-endian at byte 76, as the miner scans) whose hash top byte is 0.
static const struct algo_test_vector sha512256d_tests[] = {
	{ "000000209d6aa4360ed0fc4e095611c704cacef07f200c6957b73a429101000000000000d4d600dc69aabc685139e535"
	  "0177ff6bb8673701c0c622aad52535030b09bd0c4e7b2c6448f5041a00000185",
	  "95034ede049c06ede5124227615dfcf0649c409a8dcf751c597affc5fc587c00" },
	{ "00000020844792b4305b94da1501d27417c41644694eb6ce87a91c3c72010000000000000c11c21d78053515e545bf69"
	  "4dbe117ac47b63c4e67536252d8061d0a5b47bc937a6f06a219d011a0000025b",
	  "4863b1e82b9a9b061bd594bc90978ef0b7a8fee97e9a1a4fb62903faf5f96300" },
};
ALGO_TEST_VECTORS(ALGO_SHA512256D, sha512256d_tests);
