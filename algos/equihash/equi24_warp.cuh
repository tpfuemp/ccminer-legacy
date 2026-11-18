// Warp helpers shared by equi24b and equi24l. Included INSIDE each solver's
// namespace, like blake2b_tromp.cuh, so the two objects never share a symbol.

// Warp-aggregated append: lanes hitting the same counter elect one leader that
// does a single atomicAdd for the group.
__device__ __forceinline__ u32 eq24b_append(u32 *ctr, u32 key, u32 lane)
{
#if __CUDA_ARCH__ >= 700
	// MUST be the active mask, not 0xffffffff: the call sites sit after a
	// `continue`, so naming absent lanes is undefined behaviour (it can hang).
	const u32 act = __activemask();
	const u32 peers = __match_any_sync(act, key);
	const u32 leader = __ffs(peers) - 1u;
	const u32 rank = __popc(peers & ((1u << lane) - 1u));
	u32 base = 0u;
	if (lane == leader) base = atomicAdd(ctr, (u32)__popc(peers));
	base = __shfl_sync(peers, base, leader);
	return base + rank;
#else
	// No __match_any_sync before sm_70: ballot repeatedly over the distinct keys
	// the warp holds. Lanes may have left (continue), so only the active ones
	// take part: implicit before CUDA 9, the active mask after.
#if CUDART_VERSION >= 9000
	const u32 act = __activemask();
#define EQ24B_BALLOT(p)    __ballot_sync(act, p)
#define EQ24B_SHFL(v, l)   __shfl_sync(act, v, l)
#else
	const u32 act = __ballot(1);
#define EQ24B_BALLOT(p)    __ballot(p)
#define EQ24B_SHFL(v, l)   __shfl(v, l)
#endif
	u32 active = act;
	u32 peers = 0u, base = 0u;
	while (active) {
		const u32 ldr = (u32)__ffs(active) - 1u;
		const u32 k = EQ24B_SHFL(key, ldr);
		const u32 same = EQ24B_BALLOT(k == key);
		if (k == key) peers = same;
		active &= ~same;
	}
	const u32 leader = (u32)__ffs(peers) - 1u;
	const u32 rank = __popc(peers & ((1u << lane) - 1u));
	if (lane == leader) base = atomicAdd(ctr, (u32)__popc(peers));
	base = EQ24B_SHFL(base, leader);
#undef EQ24B_BALLOT
#undef EQ24B_SHFL
	return base + rank;
#endif
}

// Slot in a random destination bucket: a warp's keys are nearly all distinct,
// so before sm_70 (ballot loop above) a plain atomic is cheaper.
#ifndef EQ24B_PLAIN_SCATTER
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700
#define EQ24B_PLAIN_SCATTER 1
#else
#define EQ24B_PLAIN_SCATTER 0
#endif
#endif
__device__ __forceinline__ u32 eq24b_scatter_slot(u32 *ctr, u32 key, u32 lane)
{
#if EQ24B_PLAIN_SCATTER
	(void)key; (void)lane;
	return atomicAdd(ctr, 1u);
#else
	return eq24b_append(ctr, key, lane);
#endif
}
