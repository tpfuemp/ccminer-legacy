#include <miner.h>
#include "algo_registry.h"

ALGO_REGISTER(ALGO_QUARK, scanhash_quark, free_quark, quarkhash,
	"Quark, 9 steps of BLAKE/BMW/Groestl/JH/Keccak/Skein-512, 3 of them hash-dependent");

// Bitcoin genesis header with another time and nonce, hash top word below 0x100
static const struct algo_test_vector quark_tests[] = {
	{ "01000000" "0000000000000000000000000000000000000000000000000000000000000000"
	  "3ba3edfd7a7b12b27ac72c3e67768f617fc81bc3888a51323a9fb8aa4b1e5e4a"
	  "38ab5f49" "ffff001d" "005e3595",
	  "0cef9b4931daa401cb1ac71dbe8e66232d6cf56255663878276d6e5bf6000000" },
};
ALGO_TEST_VECTORS(ALGO_QUARK, quark_tests);

extern "C" void free_anime(int thr_id);
extern "C" void animehash(void *state, const void *input);

ALGO_REGISTER(ALGO_ANIME, scanhash_anime, free_anime, animehash,
	"Anime, the quark functions in another order (Animecoin)");

// the Bitcoin genesis header and a synthetic one (byte i = i), each with the
// lowest nonce whose hash top byte is 0
static const struct algo_test_vector anime_tests[] = {
	{ "01000000" "0000000000000000000000000000000000000000000000000000000000000000"
	  "3ba3edfd7a7b12b27ac72c3e67768f617fc81bc3888a51323a9fb8aa4b1e5e4a"
	  "29ab5f49" "ffff001d" "000001c8",
	  "f194005c69715c633c7767742e2747c8f9c58225ffdccbc24cb59851bb5b0800" },
	{ "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f"
	  "202122232425262728292a2b2c2d2e2f303132333435363738393a3b3c3d3e3f"
	  "404142434445464748494a4b00000092",
	  "8e20eefcd3601a422284f2d63eef36bd2ab0f501c932cc34e87edecf7fa92f00" },
};
ALGO_TEST_VECTORS(ALGO_ANIME, anime_tests);
