#ifndef CUDA_COMPAT_H
#define CUDA_COMPAT_H

/*
 * Warp intrinsics for CUDA 8 (implicit mask) and CUDA 9+ (*_sync, full-warp
 * mask). The whole warp must be converged at each call. Host pass: stubs.
 */

#include <stdint.h>
#include <cuda_runtime.h>

#if defined(CUDART_VERSION) && CUDART_VERSION >= 9000
#define CUDA_COMPAT_SYNC 1
#define CUDA_FULL_MASK 0xffffffffu
#else
#define CUDA_COMPAT_SYNC 0
#endif

#if defined(__CUDA_ARCH__)
#if CUDA_COMPAT_SYNC
#define CC_SHFL(v, l, w)      __shfl_sync(CUDA_FULL_MASK, v, l, w)
#define CC_SHFL_UP(v, d, w)   __shfl_up_sync(CUDA_FULL_MASK, v, d, w)
#define CC_SHFL_DOWN(v, d, w) __shfl_down_sync(CUDA_FULL_MASK, v, d, w)
#define CC_SHFL_XOR(v, m, w)  __shfl_xor_sync(CUDA_FULL_MASK, v, m, w)
#else
#define CC_SHFL(v, l, w)      __shfl(v, l, w)
#define CC_SHFL_UP(v, d, w)   __shfl_up(v, d, w)
#define CC_SHFL_DOWN(v, d, w) __shfl_down(v, d, w)
#define CC_SHFL_XOR(v, m, w)  __shfl_xor(v, m, w)
#endif
#define CC_DEVICE_OR(dev, host) dev
#else
#define CC_DEVICE_OR(dev, host) host
#endif

static __device__ __forceinline__ int SHFL(int var, int src, int width = 32)
{
	return CC_DEVICE_OR(CC_SHFL(var, src, width), ((void) src, (void) width, var));
}

static __device__ __forceinline__ int SHFL_UP(int var, unsigned int delta, int width = 32)
{
	return CC_DEVICE_OR(CC_SHFL_UP(var, delta, width), ((void) delta, (void) width, var));
}

static __device__ __forceinline__ int SHFL_DOWN(int var, unsigned int delta, int width = 32)
{
	return CC_DEVICE_OR(CC_SHFL_DOWN(var, delta, width), ((void) delta, (void) width, var));
}

static __device__ __forceinline__ int SHFL_XOR(int var, int mask, int width = 32)
{
	return CC_DEVICE_OR(CC_SHFL_XOR(var, mask, width), ((void) mask, (void) width, var));
}

/* unsigned and 64-bit values, as two 32-bit shuffles for uint64_t */
static __device__ __forceinline__ uint32_t SHFL(uint32_t var, int src, int width = 32)
{
	return (uint32_t) SHFL((int) var, src, width);
}

static __device__ __forceinline__ uint32_t SHFL_XOR(uint32_t var, int mask, int width = 32)
{
	return (uint32_t) SHFL_XOR((int) var, mask, width);
}

static __device__ __forceinline__ uint64_t SHFL(uint64_t var, int src, int width = 32)
{
	const uint32_t lo = SHFL((uint32_t) var, src, width);
	const uint32_t hi = SHFL((uint32_t) (var >> 32), src, width);
	return ((uint64_t) hi << 32) | lo;
}

static __device__ __forceinline__ uint64_t SHFL_XOR(uint64_t var, int mask, int width = 32)
{
	const uint32_t lo = SHFL_XOR((uint32_t) var, mask, width);
	const uint32_t hi = SHFL_XOR((uint32_t) (var >> 32), mask, width);
	return ((uint64_t) hi << 32) | lo;
}

/* warp votes */
static __device__ __forceinline__ bool WARP_ANY(bool pred)
{
#if defined(__CUDA_ARCH__) && CUDA_COMPAT_SYNC
	return __any_sync(CUDA_FULL_MASK, pred) != 0;
#elif defined(__CUDA_ARCH__)
	return __any(pred) != 0;
#else
	return pred;
#endif
}

static __device__ __forceinline__ bool WARP_ALL(bool pred)
{
#if defined(__CUDA_ARCH__) && CUDA_COMPAT_SYNC
	return __all_sync(CUDA_FULL_MASK, pred) != 0;
#elif defined(__CUDA_ARCH__)
	return __all(pred) != 0;
#else
	return pred;
#endif
}

static __device__ __forceinline__ uint32_t WARP_BALLOT(bool pred)
{
#if defined(__CUDA_ARCH__) && CUDA_COMPAT_SYNC
	return __ballot_sync(CUDA_FULL_MASK, pred);
#elif defined(__CUDA_ARCH__)
	return __ballot(pred);
#else
	return pred ? 1u : 0u;
#endif
}

/* no-op before CUDA 9: pre-Volta warps run in lockstep */
static __device__ __forceinline__ void WARP_SYNC(void)
{
#if defined(__CUDA_ARCH__) && CUDA_COMPAT_SYNC
	__syncwarp();
#endif
}

#undef CC_SHFL
#undef CC_SHFL_UP
#undef CC_SHFL_DOWN
#undef CC_SHFL_XOR
#undef CC_DEVICE_OR

/* CUDA 9+: legacy names used by older kernels map to the *_sync forms */
#if CUDA_COMPAT_SYNC && defined(__CUDA_ARCH__)
#undef __shfl
#undef __shfl_up
#undef __shfl_down
#undef __shfl_xor
#undef __any
#undef __all
#undef __ballot
#define __shfl(...)      __shfl_sync(CUDA_FULL_MASK, __VA_ARGS__)
#define __shfl_up(...)   __shfl_up_sync(CUDA_FULL_MASK, __VA_ARGS__)
#define __shfl_down(...) __shfl_down_sync(CUDA_FULL_MASK, __VA_ARGS__)
#define __shfl_xor(...)  __shfl_xor_sync(CUDA_FULL_MASK, __VA_ARGS__)
#define __any(p)         __any_sync(CUDA_FULL_MASK, p)
#define __all(p)         __all_sync(CUDA_FULL_MASK, p)
#define __ballot(p)      __ballot_sync(CUDA_FULL_MASK, p)
#endif

#endif /* CUDA_COMPAT_H */
