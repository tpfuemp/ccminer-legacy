#include <miner.h>
#include "algo_registry.h"

extern "C" void hoohash_hash(void *output, const void *input);

ALGO_REGISTER(ALGO_HOOHASH, scanhash_hoohash, free_hoohash, hoohash_hash,
	"HoohashV110 (PEPEPOW), FP64-heavy");

// The header of PEPEPOW block 4666589 and a synthetic one, each with the lowest
// nonce (big-endian at byte 76, as the miner scans) whose hash top byte is 0.
static const struct algo_test_vector hoohash_tests[] = {
	{ "00400020df13f8c7243b6b226c3399f485e902347e41ba370e0c8fea75672f45"
	  "0100000022da1946e7dba65352ae0f65ce77dcff5105e2f46a443e3a86bb359b"
	  "b6788bf962a2416a33ce011d0000025a",
	  "485caa7b082d94fdb761b131d97d592f2ad4ab5f1d3108b8a945f179ee06cd00" },
	{ "004000209fc4e90e33587da2c7ec11365b80a5caef14395e83a8cdf2173c6186"
	  "abd0f51a3f6489aed3f81d42678cb1d6fb20456a8fb4d9fe23486d92b7dc0126"
	  "4b7095badf04294e7398bde2000000ba",
	  "dcae0d06466267d11be71408165040ea2c226bb81331692f676fb41f17eaad00" },
};
ALGO_TEST_VECTORS(ALGO_HOOHASH, hoohash_tests);
