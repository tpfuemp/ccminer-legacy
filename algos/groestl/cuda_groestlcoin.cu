// Groestl specialised for Groestlcoin, bitsliced

#include <stdio.h>
#include <memory.h>

#include "cuda_helper.h"

#ifdef __INTELLISENSE__
#define __CUDA_ARCH__ 500
#define __byte_perm(x,y,n) x
#endif

#include "miner.h"

__constant__ uint32_t pTarget[8];
__constant__ uint32_t groestlcoin_gpu_msg[32];

static uint32_t *d_resultNonce[MAX_GPUS];

#if __CUDA_ARCH__ >= 300
// 64 Registers Variant for Compute 3.0+
#include "groestl_functions_quad.h"
#include "groestl_transf_quad.h"
#endif

#define SWAB32(x) cuda_swab32(x)

__global__ __launch_bounds__(256, 4)
void groestlcoin_gpu_hash_quad(uint32_t threads, uint32_t startNounce, uint32_t *resNounce)
{
#if __CUDA_ARCH__ >= 300
	// 4 threads compute one hash together
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x) / 4;
	// every lane takes part in the quad shuffles (full-warp mask), only
	// nonces inside the batch report
	{
		// GROESTL
		uint32_t paddedInput[8];

		#pragma unroll 8
		for(int k=0;k<8;k++) paddedInput[k] = groestlcoin_gpu_msg[4*k+threadIdx.x%4];

		uint32_t nounce = startNounce + thread;
		if ((threadIdx.x % 4) == 3)
			paddedInput[4] = SWAB32(nounce);  // 4*4+3 = 19

		uint32_t msgBitsliced[8];
		to_bitslice_quad(paddedInput, msgBitsliced);

		uint32_t state[8];
		for (int round=0; round<2; round++)
		{
			groestl512_progressMessage_quad(state, msgBitsliced);

			if (round < 1)
			{
				// chain the two rounds, including padding
				msgBitsliced[ 0] = __byte_perm(state[ 0], 0x00800100, 0x4341 + ((threadIdx.x%4)==3)*0x2000);
				msgBitsliced[ 1] = __byte_perm(state[ 1], 0x00800100, 0x4341);
				msgBitsliced[ 2] = __byte_perm(state[ 2], 0x00800100, 0x4341);
				msgBitsliced[ 3] = __byte_perm(state[ 3], 0x00800100, 0x4341);
				msgBitsliced[ 4] = __byte_perm(state[ 4], 0x00800100, 0x4341);
				msgBitsliced[ 5] = __byte_perm(state[ 5], 0x00800100, 0x4341);
				msgBitsliced[ 6] = __byte_perm(state[ 6], 0x00800100, 0x4341);
				msgBitsliced[ 7] = __byte_perm(state[ 7], 0x00800100, 0x4341 + ((threadIdx.x%4)==0)*0x0010);
			}
		}

		// only the first of the 4 threads gets the result hash
		uint32_t out_state[16];
		from_bitslice_quad(state, out_state);

		if (threadIdx.x % 4 == 0)
		{
			int i, position = -1;
			bool rc = true;

			#pragma unroll 8
			for (i = 7; i >= 0; i--) {
				if (out_state[i] > pTarget[i]) {
					if(position < i) {
						position = i;
						rc = false;
					}
				 }
				 if (out_state[i] < pTarget[i]) {
					if(position < i) {
						position = i;
						rc = true;
					}
				 }
			}

			if (rc && thread < threads)
				atomicMin(&resNounce[0], nounce);
		}
	}
#endif
}

__host__
void groestlcoin_cpu_init(int thr_id, uint32_t threads)
{
	// to check if the binary supports SM3+
	cuda_get_arch(thr_id);

	CUDA_SAFE_CALL(cudaMalloc(&d_resultNonce[thr_id], sizeof(uint32_t)));
}

__host__
void groestlcoin_cpu_free(int thr_id)
{
	cudaFree(d_resultNonce[thr_id]);
}

__host__
void groestlcoin_cpu_setBlock(int thr_id, void *data, void *pTargetIn)
{
	uint32_t msgBlock[32] = { 0 };

	memcpy(&msgBlock[0], data, 80);

	// pad the 80-byte message to the message block
	msgBlock[20] = 0x80;
	msgBlock[31] = 0x01000000;

	// no CPU precompute: the only round runs on the GPU

	// block header without the nonce
	cudaMemcpyToSymbol(groestlcoin_gpu_msg, msgBlock, 128);

	cudaMemset(d_resultNonce[thr_id], 0xFF, sizeof(uint32_t));
	cudaMemcpyToSymbol(pTarget, pTargetIn, 32);
}

__host__
void groestlcoin_cpu_hash(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *resNonce)
{
	uint32_t threadsperblock = 256;

	// quad variant (warp shuffle, SM 3.0+): 4 threads per hash, so 4x the blocks
	int factor = 4;

		dim3 grid(factor*((threads + threadsperblock-1)/threadsperblock));
	dim3 block(threadsperblock);

	int dev_id = device_map[thr_id];
	if (device_sm[dev_id] < 300 || cuda_arch[dev_id] < 300) {
		gpulog(LOG_ERR, thr_id, "Sorry, This algo is not supported by this GPU arch (SM 3.0 required)");
		proper_exit(EXIT_CODE_CUDA_ERROR);
	}

	cudaMemset(d_resultNonce[thr_id], 0xFF, sizeof(uint32_t));
	groestlcoin_gpu_hash_quad <<<grid, block>>> (threads, startNounce, d_resultNonce[thr_id]);

	cudaMemcpy(resNonce, d_resultNonce[thr_id], sizeof(uint32_t), cudaMemcpyDeviceToHost);
}
