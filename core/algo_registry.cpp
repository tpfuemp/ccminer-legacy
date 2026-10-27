/**
 * Algo self-registration table, see algo_registry.h
 */

#include <stdio.h>
#include <stdlib.h>

#include "miner.h"
#include "algo_registry.h"

static const struct algo_desc *registry[ALGO_COUNT];

#ifndef WITH_YESPOWER
/* host code switches yespower coin variants on -a, pool and api algo
 * changes; without the algo there are no variants to select */
bool yespower_set_variant(const char* name) { (void) name; return false; }
void yespower_clear_variant(void) { }
const char* yespower_variant_name(void) { return ""; }
#endif

void algo_register(const struct algo_desc *desc)
{
	/* runs before main(): a bad id or a duplicate is a build error, not a
	 * runtime condition, so report on stderr and stop */
	if ((int) desc->id < 0 || desc->id >= ALGO_COUNT || !desc->scanhash) {
		fprintf(stderr, "algo_register: invalid entry for id %d\n", (int) desc->id);
		abort();
	}
	if (registry[desc->id]) {
		fprintf(stderr, "algo_register: '%s' registered twice\n", algo_names[desc->id]);
		abort();
	}
	registry[desc->id] = desc;
}

static struct {
	const struct algo_test_vector *v;
	int count;
} tests[ALGO_COUNT];

void algo_register_tests(int id, const struct algo_test_vector *v, int count)
{
	if (id < 0 || id >= ALGO_COUNT || !v || count <= 0 || tests[id].v) {
		fprintf(stderr, "algo_register_tests: invalid entry for id %d\n", id);
		abort();
	}
	tests[id].v = v;
	tests[id].count = count;
}

int algo_find_tests(int id, const struct algo_test_vector **v)
{
	if (id < 0 || id >= ALGO_COUNT || !tests[id].v)
		return 0;
	*v = tests[id].v;
	return tests[id].count;
}

const struct algo_desc *algo_find(int id)
{
	if (id < 0 || id >= ALGO_COUNT)
		return NULL;
	return registry[id];
}

// required to switch algos; only initialized algos will be freed
void algo_free_all(int thr_id)
{
	for (int i = 0; i < ALGO_COUNT; i++) {
		if (!registry[i] || !registry[i]->free)
			continue;
		// aliases (keccak/keccakc, groestl/dmd-gr, ...) share one free function
		bool seen = false;
		for (int j = 0; j < i && !seen; j++)
			seen = registry[j] && registry[j]->free == registry[i]->free;
		if (!seen)
			registry[i]->free(thr_id);
	}
}
