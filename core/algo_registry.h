#ifndef ALGO_REGISTRY_H
#define ALGO_REGISTRY_H

/*
 * Algo self-registration.
 *
 * Every algo folder has a register.cpp that calls ALGO_REGISTER once per
 * enum sha_algos id it implements. The host code only reaches algos through
 * algo_find(), so the build contains exactly the algos whose files are
 * compiled in; an id without a registration is "not available in this build".
 *
 * Registration runs from static initializers, before main(). The table is a
 * zero-initialized array indexed by id, so it is valid before any dynamic
 * init runs and the order in which objects register does not matter.
 *
 * This header is included from .cu files: keep it plain C++11.
 */

#include <stdint.h>
#include "algos.h"

struct work;

/* scanhash entry point; algos with extra arguments register a wrapper */
typedef int  (*algo_scanhash_fn)(int thr_id, struct work *work, uint32_t max_nonce, unsigned long *hashes_done);
/* releases the thread's device memory; must be safe on an uninitialized thread */
typedef void (*algo_free_fn)(int thr_id);
/* CPU reference hash of an 80-byte header into 32 bytes, for --cputest */
typedef void (*algo_cpu_hash_fn)(void *output, const void *input);

struct algo_desc {
	enum sha_algos id;
	algo_scanhash_fn scanhash;
	algo_free_fn free;
	algo_cpu_hash_fn cpu_hash; /* NULL if the algo has no 80-byte CPU hash */
};

void algo_register(const struct algo_desc *desc);

/* NULL when the algo is not compiled into this build */
const struct algo_desc *algo_find(int id);

/* frees every registered algo on this thread (bench/pool algo switch) */
void algo_free_all(int thr_id);

struct algo_registrar {
	explicit algo_registrar(const struct algo_desc *desc) { algo_register(desc); }
};

#define ALGO_REGISTER(id, scanhash, free, cpu_hash) \
	static const struct algo_desc algo_desc_##id = { id, scanhash, free, cpu_hash }; \
	static struct algo_registrar algo_registrar_##id(&algo_desc_##id)

#endif
