#ifndef HEAVYHASH_CPU_H
#define HEAVYHASH_CPU_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* the full-rank matrix of a job, from the 32 previous-block-hash bytes */
void heavyhash_matrix(uint8_t m[64][64], const void *prevhash);
/* HeavyHash of len bytes with a given matrix; len must be a multiple of 8
 * (sph/sha3.c absorbs whole 64-bit words only) */
void heavyhash_with_matrix(const uint8_t m[64][64], const void *data, size_t len, void *output);
/* the 80 header bytes */
void heavyhash_hash(void *output, const void *input);

#ifdef __cplusplus
}
#endif

#endif
