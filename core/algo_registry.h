#ifndef ALGO_REGISTRY_H
#define ALGO_REGISTRY_H

/*
 * Algo self-registration: each algos/<name>/register.cpp calls ALGO_REGISTER
 * per id it implements (static initializers, order independent). An id
 * without registration is not available in this build. Plain C++11 (.cu).
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
	const char *help;          /* one line for --help */
};

void algo_register(const struct algo_desc *desc);

/* NULL when the algo is not compiled into this build */
const struct algo_desc *algo_find(int id);

/* frees every registered algo on this thread (bench/pool algo switch) */
void algo_free_all(int thr_id);

struct algo_registrar {
	explicit algo_registrar(const struct algo_desc *desc) { algo_register(desc); }
};

#define ALGO_REGISTER(id, scanhash, free, cpu_hash, help) \
	static const struct algo_desc algo_desc_##id = { id, scanhash, free, cpu_hash, help }; \
	static struct algo_registrar algo_registrar_##id(&algo_desc_##id)

/*
 * Known-answer vectors for --test-vectors: `header` is an 80-byte header in
 * hex whose nonce (bytes 76..79) is a solution, `hash` its cpu_hash() output.
 */
struct algo_test_vector {
	const char *header;
	const char *hash;
};

void algo_register_tests(int id, const struct algo_test_vector *v, int count);

/* number of vectors of the algo (0 if none), *v set to the first */
int algo_find_tests(int id, const struct algo_test_vector **v);

struct algo_tests_registrar {
	algo_tests_registrar(int id, const struct algo_test_vector *v, int count) {
		algo_register_tests(id, v, count);
	}
};

#define ALGO_TEST_VECTORS(id, vectors) \
	static struct algo_tests_registrar algo_tests_registrar_##id(id, vectors, \
		(int) (sizeof(vectors) / sizeof((vectors)[0])))

#endif
