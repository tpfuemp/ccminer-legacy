/**
 * Quark branch split: the nonces of a batch whose current hash has bit 3 of
 * byte 0 set go to list 1, the others to list 2. One ballot and one atomic per
 * warp; the lists are unordered, every block finds its hash by nonce position.
 */
#include <stdio.h>
#include <memory.h>

#include "cuda_helper.h"
#include "quark.h"

#define SPLIT_TPB 256

static uint32_t *d_count[MAX_GPUS];

__global__ __launch_bounds__(SPLIT_TPB)
void quark_split_gpu(const uint32_t threads, const uint32_t startNonce, const uint32_t *g_hash,
	uint32_t *g_list1, uint32_t *g_list2, uint32_t *g_count)
{
	const uint32_t thread = blockDim.x * blockIdx.x + threadIdx.x;
	const uint32_t lane = threadIdx.x & 31;

	// no early exit: the ballots and shuffles need the whole warp
	const bool valid = thread < threads;
	const bool bit = valid && (g_hash[thread << 4] & 0x8);
	const uint32_t m1 = __ballot(bit);
	const uint32_t m2 = __ballot(valid && !bit);

	uint32_t base1 = 0, base2 = 0;
	if (lane == 0) {
		if (m1) base1 = atomicAdd(&g_count[0], __popc(m1));
		if (m2) base2 = atomicAdd(&g_count[1], __popc(m2));
	}
	base1 = __shfl(base1, 0);
	base2 = __shfl(base2, 0);

	const uint32_t below = (1U << lane) - 1;
	if (bit)
		g_list1[base1 + __popc(m1 & below)] = startNonce + thread;
	else if (valid)
		g_list2[base2 + __popc(m2 & below)] = startNonce + thread;
}

__host__
void quark_split_cpu_init(int thr_id)
{
	CUDA_SAFE_CALL(cudaMalloc(&d_count[thr_id], 2 * sizeof(uint32_t)));
}

__host__
void quark_split_cpu_free(int thr_id)
{
	cudaFree(d_count[thr_id]);
}

__host__
void quark_split_cpu(int thr_id, uint32_t threads, uint32_t startNonce, uint32_t *d_hash,
	uint32_t *d_list1, uint32_t *n1, uint32_t *d_list2, uint32_t *n2)
{
	dim3 grid((threads + SPLIT_TPB - 1) / SPLIT_TPB);
	dim3 block(SPLIT_TPB);
	uint32_t count[2];

	cudaMemset(d_count[thr_id], 0, sizeof(count));
	quark_split_gpu <<<grid, block>>> (threads, startNonce, d_hash, d_list1, d_list2, d_count[thr_id]);
	cudaMemcpy(count, d_count[thr_id], sizeof(count), cudaMemcpyDeviceToHost);
	*n1 = count[0];
	*n2 = count[1];
}
