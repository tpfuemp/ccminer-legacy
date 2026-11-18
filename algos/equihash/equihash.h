#ifndef EQUIHASH_H
#define EQUIHASH_H

#include <stdint.h>

/* 140-byte header, the 32-byte nonce in data[27..34]: the pool's extranonce first,
 * the miner steps word 30 */
#define EQNONCE_OFFSET 30

/* sha256d of header + compactSize + solution, the share hash */
extern "C" void equi_hash(const void *input, void *output, int len);

/* --test-vectors: 144/5 and 192/7 on the CPU and on each GPU, counted into
 * *passed / *failed (a GPU without room for a variant's arena is skipped) */
void equihash_test_vectors(int *passed, int *failed);

#endif
