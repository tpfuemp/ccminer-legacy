/**
 * --test-vectors: known-answer checks of every compiled-in algo on the CPU
 * and each GPU, instead of mining; the result is the exit code.
 */

#include <string.h>

#include "miner.h"
#include "algo_registry.h"

/* nonces scanned before the vector's solution nonce */
#define GPU_SPAN (1U << 24)

/* longest header: decred, 180 bytes */
#define MAX_HEADER 180

/* header layout of an algo: length, nonce word in work.data, byte order */
struct header_layout {
	int len;
	int nonce_word;
	bool le_words; /* data[] holds the raw little-endian words (decred) */
};

static struct header_layout layout_of(int algo_id)
{
	struct header_layout l = { 80, 19, false };
	if (algo_id == ALGO_DECRED) {
		l.len = 180;
		l.nonce_word = 35;
		l.le_words = true;
	}
	return l;
}

static bool parse_vector(const struct algo_test_vector *t, const struct header_layout *l,
	uint8_t header[MAX_HEADER], uint8_t hash[32])
{
	return strlen(t->header) == (size_t) l->len * 2 && strlen(t->hash) == 64 &&
		hex2bin(header, t->header, l->len) && hex2bin(hash, t->hash, 32);
}

static bool cpu_check(const struct algo_desc *algo, const uint8_t header[MAX_HEADER], const uint8_t hash[32])
{
	uint32_t _ALIGN(64) out[16];
	memset(out, 0, sizeof(out));
	algo->cpu_hash(out, header);
	return memcmp(out, hash, 32) == 0;
}

/* scans up to the vector's nonce: it must be found, every hit must verify */
static bool gpu_check(int thr_id, const struct algo_desc *algo, const struct header_layout *l,
	const uint8_t header[MAX_HEADER], const uint8_t hash[32])
{
	struct work work;
	memset(&work, 0, sizeof(work));
	for (int i = 0; i < l->len / 4; i++)
		work.data[i] = l->le_words ? le32dec(header + 4 * i) : be32dec(header + 4 * i);
	if (!l->le_words) {
		// SHA-256 padding of the 80-byte header, as stratum_gen_work() sets it
		work.data[20] = 0x80000000;
		work.data[31] = 0x00000280;
	}
	for (int i = 0; i < 8; i++)
		work.target[i] = UINT32_MAX;
	work.target[7] = le32dec(hash + 28);

	uint32_t *pnonce = &work.data[l->nonce_word];
	const uint32_t nonce = *pnonce;
	if (nonce == UINT32_MAX) {
		gpulog(LOG_ERR, thr_id, "%s: vector nonce must be below ffffffff", algo_names[algo->id]);
		return false;
	}
	*pnonce = nonce >= GPU_SPAN ? nonce - GPU_SPAN : 0;

	const uint32_t rejected = thr_info[thr_id].gpu.rejected;
	bool found = false;
	work_restart[thr_id].restart = 0;
	while (!found && *pnonce <= nonce) {
		unsigned long done = 0;
		int rc = algo->scanhash(thr_id, &work, nonce + 1, &done);
		if (rc <= 0)
			break;
		for (int n = 0; n < rc && n < MAX_NONCES; n++) {
			uint8_t buf[MAX_HEADER];
			uint32_t _ALIGN(64) out[16];
			memcpy(buf, header, l->len);
			if (l->le_words)
				le32enc(buf + 4 * l->nonce_word, work.nonces[n]);
			else
				be32enc(buf + 4 * l->nonce_word, work.nonces[n]);
			algo->cpu_hash(out, buf);
			if (!fulltest(out, work.target)) {
				gpulog(LOG_ERR, thr_id, "%s: GPU nonce %08x does not verify on the CPU",
					algo_names[algo->id], work.nonces[n]);
				return false;
			}
			if (work.nonces[n] == nonce)
				found = true;
		}
	}
	if (thr_info[thr_id].gpu.rejected != rejected)
		return false;
	if (!found)
		gpulog(LOG_ERR, thr_id, "%s: GPU did not find nonce %08x", algo_names[algo->id], nonce);
	return found;
}

int run_test_vectors(void)
{
	int passed = 0, failed = 0;

	for (int id = 0; id < ALGO_COUNT; id++) {
		const struct algo_desc *algo = algo_find(id);
		if (!algo)
			continue;
		const struct algo_test_vector *v;
		const int count = algo_find_tests(id, &v);
		if (!count || !algo->cpu_hash) {
			applog(LOG_WARNING, "%s: no test vectors", algo_names[id]);
			continue;
		}
		const struct header_layout l = layout_of(id);
		for (int i = 0; i < count; i++) {
			uint8_t header[MAX_HEADER], hash[32];
			if (!parse_vector(&v[i], &l, header, hash)) {
				applog(LOG_ERR, "%s: vector %d is malformed", algo_names[id], i + 1);
				failed++;
				continue;
			}
			const bool cpu_ok = cpu_check(algo, header, hash);
			applog(cpu_ok ? LOG_INFO : LOG_ERR, "CPU: %s vector %d %s",
				algo_names[id], i + 1, cpu_ok ? "OK" : "FAILED");
			cpu_ok ? passed++ : failed++;
			if (!cpu_ok)
				continue; // a wrong reference makes the GPU check meaningless

			for (int thr_id = 0; thr_id < opt_n_threads; thr_id++) {
				const bool gpu_ok = gpu_check(thr_id, algo, &l, header, hash);
				gpulog(gpu_ok ? LOG_INFO : LOG_ERR, thr_id, "%s vector %d %s",
					algo_names[id], i + 1, gpu_ok ? "OK" : "FAILED");
				gpu_ok ? passed++ : failed++;
			}
		}
		for (int thr_id = 0; thr_id < opt_n_threads; thr_id++)
			if (algo->free)
				algo->free(thr_id);
	}

	applog(failed ? LOG_ERR : LOG_INFO, "test vectors: %d passed, %d failed", passed, failed);
	return failed || !passed ? EXIT_CODE_SW_INIT_ERROR : EXIT_CODE_OK;
}
