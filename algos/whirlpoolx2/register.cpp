#include <miner.h>
#include "algo_registry.h"

ALGO_REGISTER(ALGO_WHIRLPOOLX2, scanhash_whirlpoolx2, free_whirlpoolx2, whirlpoolx2_hash,
	"Whirlpool-512 folded to 256 bits (CapStash)");

// The headers of CapStash blocks 100000 and 409009, each with the lowest nonce
// (big-endian at byte 76, as the miner scans) whose hash top byte is 0.
static const struct algo_test_vector whirlpoolx2_tests[] = {
	{ "000000209afc05d8f4b3a1d38f48b821e770623cac110820e48eeeda1b3ab8d5022fb38ddf60d6f449ca5d67bf250b"
	  "e04c089fd0ff3bbe1374c1dc5430dac0f983329b887cbce3696c840a1b000001b5",
	  "f0c6d22f7edea0ce3a0a1a3d4eabdb5d07c21f4cc892b23057b07815bb5a7800" },
	{ "00000020dc1b81b51cc084166b777d949c59161825a5747fe76d87fb3f909108b8808c55b7f7bd20ca59f6ae48b14f"
	  "31f38b145f10b836ed60d2287fe241be5568736cb0e621026b83aa001c00000015",
	  "d341de2089660b9fec058b44d2d7de77de66aa7a506941484ccdd032e3fe8800" },
};
ALGO_TEST_VECTORS(ALGO_WHIRLPOOLX2, whirlpoolx2_tests);
