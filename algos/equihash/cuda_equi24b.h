// equi24b -- Equihash solver for the DIGITBITS=24 family (144/5 and 192/7) on a
// heap of few large buckets; one source, two instantiations keyed on EQ_WN.
#pragma once
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// NULL when the device cannot provide the arena (see *_arena_needed).
void *equi24b_144_init(void);
void  equi24b_144_free(void *ctx);
void *equi24b_192_init(void);
void  equi24b_192_free(void *ctx);

// Device bytes the context will try to allocate. Every layer is retained, so
// this grows with the round count and differs between the variants.
unsigned long long equi24b_144_arena_needed(void);
unsigned long long equi24b_192_arena_needed(void);

// Pipelined solve of one 140-byte header+nonce under `personal`: launch() queues
// a whole solve and returns at once (at most two queued), collect() waits for the
// oldest, emits each distinct non-trivial proof (not verified: the caller does)
// and returns their count, negative on error. out4, when given, gets the loss
// counters of that solve:
//   [0] scatter overflow (destination bucket full, the expected tail)
//   [1] round staging overflow (chunk full)
//   [2] unused, always 0
//   [3] candidates or proofs found but not stored (capacity clamps)
// pipe_reset() drops the queued solves without emitting.
void *equi24b_144_pipe_new(void);
void  equi24b_144_pipe_free(void *pipe);
void  equi24b_144_pipe_reset(void *pipe);
int   equi24b_144_launch(void *ctx, void *pipe, const char *headernonce, const char *personal);
int   equi24b_144_collect(void *ctx, void *pipe, void (*emit)(void *, const uint32_t *, uint32_t),
                          void *ud, unsigned *out4);
// equi24l (cuda_equi24l_144.cu): lean 144/5 for 2 GB cards, the same contract.
void *equi24l_144_init(void);
void  equi24l_144_free(void *ctx);
unsigned long long equi24l_144_arena_needed(void);
void *equi24l_144_pipe_new(void);
void  equi24l_144_pipe_free(void *pipe);
void  equi24l_144_pipe_reset(void *pipe);
int   equi24l_144_launch(void *ctx, void *pipe, const char *headernonce, const char *personal);
int   equi24l_144_collect(void *ctx, void *pipe, void (*emit)(void *, const uint32_t *, uint32_t),
                          void *ud, unsigned *out4);
void *equi24b_192_pipe_new(void);
void  equi24b_192_pipe_free(void *pipe);
void  equi24b_192_pipe_reset(void *pipe);
int   equi24b_192_launch(void *ctx, void *pipe, const char *headernonce, const char *personal);
int   equi24b_192_collect(void *ctx, void *pipe, void (*emit)(void *, const uint32_t *, uint32_t),
                          void *ud, unsigned *out4);

#ifdef __cplusplus
}
#endif
