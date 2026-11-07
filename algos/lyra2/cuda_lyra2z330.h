#ifndef CUDA_LYRA2Z330_H
#define CUDA_LYRA2Z330_H

#include <stdint.h>

#define LYRA2Z330_TIMECOST 2
#define LYRA2Z330_ROWS 330
#define LYRA2Z330_COLS 256
/* one hash: nRows x nCols blocks of 12 words */
#define LYRA2Z330_MATRIX_BYTES ((size_t) LYRA2Z330_ROWS * LYRA2Z330_COLS * 12 * sizeof(uint64_t))

/* d_resNonce[0] counts candidates, [1..LYRA2Z330_MAX_RESULTS-1] hold them */
#define LYRA2Z330_MAX_RESULTS 16

/* threads: a multiple of 32; the matrix buffer is LYRA2Z330_MATRIX_BYTES per hash of
 * one sub-launch, threads / 16 on sm_50+ and threads / 4 below */
bool lyra2z330_cpu_init(int thr_id, uint32_t threads);
void lyra2z330_cpu_free(int thr_id);
void lyra2z330_setBlock(const uint32_t *endiandata, uint64_t target);
/* nonces [startNounce, startNounce + threads), only the first count report */
void lyra2z330_cpu_hash(int thr_id, uint32_t threads, uint32_t count, uint32_t startNounce, uint32_t *d_resNonce);

#endif
