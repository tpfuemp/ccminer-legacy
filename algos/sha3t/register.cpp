#include <miner.h>
#include "algo_registry.h"

ALGO_REGISTER(ALGO_SHA3T, scanhash_sha3t, free_sha3t, sha3t256_hash,
	"SHA3-256 three times (SHA3-256t), alias sha3-256t");

// The headers of Bitcoin III blocks 64000 and 60000, each with the lowest nonce
// (big-endian at byte 76, as the miner scans) whose hash top byte is 0.
static const struct algo_test_vector sha3t_tests[] = {
	{ "00100020d205b884704611704b309257c244df50499766d82ca7675bf523000000000000116c7d2c1f50f4971848439f"
	  "3e36d86c81dc043dbf470ade6495598f489eed136866bb6a05db421a00000044",
	  "4c5226845d595994fcd4e66d7a93b64cbecfc86f8cebae0cca35d1b976933500" },
	{ "0010002064a6ffc22ad8c07e6573beb8ffa90aefeab3a6f5d41dd91de30f0000000000000a97cb61fd5e5f3f05686"
	  "0c8134f6d749e45130f951b341811e14515e965a57a32679d6a804d6a1a00000128",
	  "31f333ff4a581979dc26ebe3db5ca54d4e19c9eb02697c6ad5c918dc95a77e00" },
};
ALGO_TEST_VECTORS(ALGO_SHA3T, sha3t_tests);
