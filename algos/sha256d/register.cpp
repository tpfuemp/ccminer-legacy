#include <miner.h>
#include "algo_registry.h"

ALGO_REGISTER(ALGO_SHA256D, scanhash_sha256d, free_sha256d, sha256d_hash);
