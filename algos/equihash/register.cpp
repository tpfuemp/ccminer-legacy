#include <miner.h>
#include "algo_registry.h"

/* vectors: equihash.cu (equihash_test_vectors), the registry format has no solution */
ALGO_REGISTER(ALGO_EQUIHASH, scanhash_equihash, free_equihash, NULL,
	"Equihash 144/5 (-a equihash144) and 192/7 (-a equihash192), 2.8 (lean 1.6) / 5.4 GB VRAM");
