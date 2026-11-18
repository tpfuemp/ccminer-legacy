/**
 * Equihash 144/5 (BitcoinZ, Gemlink, LiteZ) and 192/7 (YCash, ZClassic, Zero): variant
 * selection, scan loop, share hash, init self-test and --test-vectors. Solvers: equi24b
 * (cuda_equi24b.cuh), for 144/5 on smaller cards the lean equi24l (cuda_equi24l_144.cu);
 * scan loop after ccminer-tpfuemp. Zcash 200/9 (-a equihash) is not built.
 */
#include <stdio.h>
#include <string.h>
#include <vector>

#include <sph/sph_sha2.h>
#include <miner.h>
#include <cuda_runtime.h>

#include "cuda/selftest_gate.cuh"
#include "equihash.h"
#include "cuda_equi24b.h"
#include "equi_verify.h"
#include "equi_pack.h"
#include "equi_vectors.h"

#define NONCE_OFT EQNONCE_OFFSET
#define EQ_MAX_PROOF 128   /* 192/7 */
#define EQ_MAX_SOLS 64     /* proofs taken from one solve */

/* a solver: equi24b (fast, every layer kept) or equi24l (144/5 for 2 GB cards) */
struct eq_solver {
	const char *name;
	void *(*init)(void);
	void (*free)(void *ctx);
	unsigned long long (*arena)(void);
	void *(*pipe_new)(void);
	void (*pipe_free)(void *pipe);
	void (*pipe_reset)(void *pipe);
	int (*launch)(void *ctx, void *pipe, const char *hdr, const char *personal);
	int (*collect)(void *ctx, void *pipe, void (*emit)(void *, const uint32_t *, uint32_t),
		void *ud, unsigned *out4);
};

static const struct eq_solver eq_s144 = { "equi24b",
	equi24b_144_init, equi24b_144_free, equi24b_144_arena_needed, equi24b_144_pipe_new,
	equi24b_144_pipe_free, equi24b_144_pipe_reset, equi24b_144_launch, equi24b_144_collect };
static const struct eq_solver eq_s144_lean = { "equi24l (lean)",
	equi24l_144_init, equi24l_144_free, equi24l_144_arena_needed, equi24l_144_pipe_new,
	equi24l_144_pipe_free, equi24l_144_pipe_reset, equi24l_144_launch, equi24l_144_collect };
static const struct eq_solver eq_s192 = { "equi24b",
	equi24b_192_init, equi24b_192_free, equi24b_192_arena_needed, equi24b_192_pipe_new,
	equi24b_192_pipe_free, equi24b_192_pipe_reset, equi24b_192_launch, equi24b_192_collect };

struct eq_variant {
	int wn, wk;
	const char *personal;                    /* default, the pool may send another */
	const char *tv_header, *tv_solution, *tv_hash;
	const struct eq_solver *solver[2];       /* preferred first; the second may be NULL */
	int (*verify)(const char *hdr, const char *personal, const uint32_t *indices);
};

static const struct eq_variant eq_variants[2] = {
	{ 144, 5, "BitcoinZ", EQ_TV144_HEADER, EQ_TV144_SOLUTION, EQ_TV144_HASH,
	  { &eq_s144, &eq_s144_lean }, eq_verify_144 },
	{ 192, 7, "ZcashPoW", EQ_TV192_HEADER, EQ_TV192_SOLUTION, EQ_TV192_HASH,
	  { &eq_s192, NULL }, eq_verify_192 },
};

/* fixed by -a (equihash144 / equihash192); NULL is 200/9, not built */
static const struct eq_variant *eqv = NULL;
static char eq_personal[16] = "ZcashPoW";

static int eq_solsize(int wn, int wk) { return (1 << wk) * (wn / (wk + 1) + 1) / 8; }
static int eq_solprefix(int size) { return size < 253 ? 1 : 3; }

static int eq_cur_wn(void) { return eqv ? eqv->wn : 200; }
static int eq_cur_wk(void) { return eqv ? eqv->wk : 9; }

/* bytes in work->extra: compactSize + solution */
extern "C" int eq_variant_storelen(void)
{
	const int s = eq_solsize(eq_cur_wn(), eq_cur_wk());
	return eq_solprefix(s) + s;
}

extern "C" int eq_variant_wk(void)
{
	return eq_cur_wk();
}

static void eq_select(int i)
{
	eqv = &eq_variants[i];
	snprintf(eq_personal, sizeof(eq_personal), "%s", eqv->personal);
	applog(LOG_NOTICE, "equihash %d/%d, personalization \"%s\" (solution %d bytes)",
		eqv->wn, eqv->wk, eq_personal, eq_solsize(eqv->wn, eqv->wk));
}

extern "C" void eq_set_variant_144(void) { eq_select(0); }
extern "C" void eq_set_variant_192(void) { eq_select(1); }

/* mining.notify may carry "<n>_<k>" and a personalization (zpool): (n,k) is fixed by -a
 * and only checked, the personalization is taken */
extern "C" void eq_set_variant_params(int wn, int wk, const char *personal)
{
	if (wn > 0 && wk > 0 && (wn != eq_cur_wn() || wk != eq_cur_wk())) {
		static bool warned = false;
		if (!warned) {
			applog(LOG_WARNING, "pool sends equihash %d/%d, the miner runs %d/%d (-a): pool"
				" parameters ignored", wn, wk, eq_cur_wn(), eq_cur_wk());
			warned = true;
		}
		return;
	}
	if (personal && *personal) {
		char pers[16];
		snprintf(pers, sizeof(pers), "%.8s", personal);
		if (strncmp(pers, eq_personal, 8) != 0) {
			snprintf(eq_personal, sizeof(eq_personal), "%s", pers);
			applog(LOG_NOTICE, "equihash personalization \"%s\" (from the pool)", eq_personal);
		}
	}
}

extern "C" void equi_hash(const void *input, void *output, int len)
{
	uint8_t _ALIGN(64) hash0[32];
	sph_sha256_context ctx;
	sph_sha256_init(&ctx);
	sph_sha256(&ctx, input, len);
	sph_sha256_close(&ctx, hash0);
	sph_sha256(&ctx, hash0, 32);
	sph_sha256_close(&ctx, output);
}

/* ---- proofs ---- */

struct eq_proofs {
	int n, proofsize;
	uint32_t idx[EQ_MAX_SOLS][EQ_MAX_PROOF];
};

static void eq_emit(void *ud, const uint32_t *idx, uint32_t proofsize)
{
	struct eq_proofs *p = (struct eq_proofs *) ud;
	if (p->n < EQ_MAX_SOLS && proofsize <= EQ_MAX_PROOF) {
		memcpy(p->idx[p->n], idx, proofsize * sizeof(uint32_t));
		p->proofsize = (int) proofsize;
		p->n++;
	}
}

/* the submitted bytes: compactSize + packed indices; returns the length */
static int eq_pack(const struct eq_variant *v, const uint32_t *idx, uint8_t *out)
{
	const int size = eq_solsize(v->wn, v->wk);
	const int pfx = eq_solprefix(size);
	std::vector<uint32_t> in(idx, idx + ((size_t)1 << v->wk));
	std::vector<unsigned char> min = eq_minimal_from_indices(in, v->wn / (v->wk + 1));
	if (pfx == 1) {
		out[0] = (uint8_t) size;
	} else {
		out[0] = 0xfd;
		out[1] = (uint8_t) size;
		out[2] = (uint8_t) (size >> 8);
	}
	memcpy(out + pfx, min.data(), size);
	return pfx + size;
}

/* packed solution -> indices (25-bit big-endian fields) */
static void eq_unpack(const struct eq_variant *v, const uint8_t *sol, uint32_t *idx)
{
	const int bits = v->wn / (v->wk + 1) + 1, n = 1 << v->wk;
	uint64_t acc = 0;
	int nb = 0, k = 0;
	for (int i = 0; k < n; i++) {
		acc = (acc << 8) | sol[i];
		nb += 8;
		if (nb >= bits) {
			nb -= bits;
			idx[k++] = (uint32_t) (acc >> nb) & ((1U << bits) - 1);
		}
	}
}

/* ---- known-answer vector, used by the init self-test and --test-vectors ---- */

struct eq_vec {
	uint8_t header[140];
	uint8_t sol[3 + 400];
	uint8_t hash[32];
	uint32_t idx[EQ_MAX_PROOF];
	int sollen;
};

static bool eq_parse_vector(const struct eq_variant *v, struct eq_vec *t)
{
	const int size = eq_solsize(v->wn, v->wk);
	const int pfx = eq_solprefix(size);
	if (strlen(v->tv_header) != 280 || strlen(v->tv_solution) != (size_t) size * 2 ||
	    !hex2bin(t->header, v->tv_header, 140) || !hex2bin(t->sol + pfx, v->tv_solution, size) ||
	    !hex2bin(t->hash, v->tv_hash, 32))
		return false;
	if (pfx == 1) {
		t->sol[0] = (uint8_t) size;
	} else {
		t->sol[0] = 0xfd;
		t->sol[1] = (uint8_t) size;
		t->sol[2] = (uint8_t) (size >> 8);
	}
	t->sollen = pfx + size;
	eq_unpack(v, t->sol + pfx, t->idx);
	return true;
}

/* CPU: the verifier accepts the vector, the packing gives its bytes back, sha256d matches */
static bool eq_cpu_check(const struct eq_variant *v, const struct eq_vec *t)
{
	uint8_t full[140 + 3 + 400], packed[3 + 400];
	uint32_t _ALIGN(64) hash[8];
	char personal[16];
	snprintf(personal, sizeof(personal), "%s", v->personal);
	if (v->verify((const char *) t->header, personal, t->idx) != 0)
		return false;
	if (eq_pack(v, t->idx, packed) != t->sollen || memcmp(packed, t->sol, t->sollen))
		return false;
	memcpy(full, t->header, 140);
	memcpy(full + 140, t->sol, t->sollen);
	equi_hash(full, hash, 140 + t->sollen);
	return memcmp(hash, t->hash, 32) == 0;
}

/* GPU: one solve of the vector header must emit its solution, every proof must verify.
 * 1 passed, 0 failed, -1 a CUDA error (logged) */
static int eq_gpu_check(int thr_id, const struct eq_variant *v, const struct eq_solver *sv,
	void *ctx, void *pipe, const struct eq_vec *t)
{
	static struct eq_proofs proofs[MAX_GPUS];
	struct eq_proofs *p = &proofs[thr_id];
	char personal[16];
	unsigned drops[4] = { 0, 0, 0, 0 };
	snprintf(personal, sizeof(personal), "%s", v->personal);
	p->n = 0;
	sv->pipe_reset(pipe);
	if (sv->launch(ctx, pipe, (const char *) t->header, personal) ||
	    sv->collect(ctx, pipe, eq_emit, p, drops) < 0) {
		const cudaError_t err = cudaGetLastError();
		gpulog(LOG_ERR, thr_id, "equihash%d self-test: %s", v->wn,
			err != cudaSuccess ? cudaGetErrorString(err) : "solver error");
		return -1;
	}
	bool found = false;
	for (int s = 0; s < p->n; s++) {
		if (v->verify((const char *) t->header, personal, p->idx[s]) != 0) {
			gpulog(LOG_ERR, thr_id, "equihash%d: GPU proof %d does not verify", v->wn, s);
			return 0;
		}
		if (!memcmp(p->idx[s], t->idx, ((size_t)1 << v->wk) * sizeof(uint32_t)))
			found = true;
	}
	if (!found)
		gpulog(LOG_ERR, thr_id, "equihash%d: GPU did not find the vector solution (%d proofs)",
			v->wn, p->n);
	return found ? 1 : 0;
}

/* ---- scan ---- */

static bool init[MAX_GPUS] = { 0 };
static void *eq_ctx[MAX_GPUS], *eq_pipe[MAX_GPUS];
static const struct eq_solver *eq_sv[MAX_GPUS];   /* the solver eq_ctx belongs to */
/* queued solves (oldest first) with their headers and personalizations; kept over a
 * return from scanhash if the next header differs at most in the nonce words and the
 * personalization is the same */
static uint32_t eq_pend_hdr[MAX_GPUS][2][35];
static char eq_pend_pers[MAX_GPUS][2][16];
static int eq_pend_n[MAX_GPUS];
static struct eq_proofs eq_found[MAX_GPUS];
static uint32_t eq_bad[MAX_GPUS], eq_lost[MAX_GPUS];

/* --debug: solves, solutions and proofs failing the CPU check per 5 s; about 2 solutions
 * per nonce is expected, fewer means lost pairs (Sol/s alone cannot show that) */
static struct { uint32_t solves, sols, bad; time_t since; } eq_dbg[MAX_GPUS];

static void eq_debug_count(int thr_id, const uint32_t *hdr, const char *pers,
	const struct eq_proofs *p, int nsol)
{
	const time_t now = time(NULL);
	eq_dbg[thr_id].solves++;
	eq_dbg[thr_id].sols += (uint32_t) nsol;
	for (int s = 0; s < p->n; s++)
		if (eqv->verify((const char *) hdr, pers, p->idx[s]) != 0)
			eq_dbg[thr_id].bad++;
	if (!eq_dbg[thr_id].since)
		eq_dbg[thr_id].since = now;
	if (now - eq_dbg[thr_id].since < 5)
		return;
	const double el = (double) (now - eq_dbg[thr_id].since);
	gpulog(LOG_DEBUG, thr_id, "equihash%d: %u solves, %.2f Sol/s, %.2f sol/nonce, %u failed"
		" the CPU check", eqv->wn, eq_dbg[thr_id].solves, eq_dbg[thr_id].sols / el,
		(double) eq_dbg[thr_id].sols / eq_dbg[thr_id].solves, eq_dbg[thr_id].bad);
	memset(&eq_dbg[thr_id], 0, sizeof(eq_dbg[thr_id]));
	eq_dbg[thr_id].since = now;
}

static bool eq_launch(int thr_id, const uint32_t *hdr)
{
	const int n = eq_pend_n[thr_id];
	const int rc = eq_sv[thr_id]->launch(eq_ctx[thr_id], eq_pipe[thr_id], (const char *) hdr, eq_personal);
	if (rc) {
		gpulog(LOG_WARNING, thr_id, "equihash solver launch failed (%d)", rc);
		return false;
	}
	memcpy(eq_pend_hdr[thr_id][n], hdr, 140);
	memcpy(eq_pend_pers[thr_id][n], eq_personal, 16);
	eq_pend_n[thr_id] = n + 1;
	return true;
}

static void eq_reset(int thr_id)
{
	eq_sv[thr_id]->pipe_reset(eq_pipe[thr_id]);
	eq_pend_n[thr_id] = 0;
}

/* return to the miner loop about once a second: max_nonce cannot bound the scan, the
 * rate is counted in solutions */
static bool eq_slice_done(const struct timeval *start)
{
	struct timeval now, d;
	gettimeofday(&now, NULL);
	timeval_subtract(&d, &now, (struct timeval *) start);
	return d.tv_sec >= 1;
}

extern "C" int scanhash_equihash(int thr_id, struct work *work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t _ALIGN(64) endiandata[35];
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t nonce_increment = (rand() & 0xFF) | 1; /* odd: never a zero step */
	struct timeval tv_start;
	uint32_t sols = 0;

	if (!eqv) {
		applog(LOG_ERR, "equihash 200/9 is not available: use -a equihash144 or -a equihash192");
		proper_exit(EXIT_CODE_SW_INIT_ERROR);
		return -1;
	}
	if (opt_benchmark)
		ptarget[7] = 0xfffff;

	if (!init[thr_id]) {
		const int dev_id = device_map[thr_id];
		cudaSetDevice(dev_id);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
		}
		/* the first solver whose arena the GPU can allocate */
		double need = 0.0;
		eq_ctx[thr_id] = eq_pipe[thr_id] = NULL;
		for (int i = 0; i < 2 && !eq_pipe[thr_id] && eqv->solver[i]; i++) {
			const struct eq_solver *sv = eqv->solver[i];
			need = (double) sv->arena() / (1 << 30);
			eq_ctx[thr_id] = sv->init();
			eq_pipe[thr_id] = eq_ctx[thr_id] ? sv->pipe_new() : NULL;
			if (eq_ctx[thr_id] && !eq_pipe[thr_id]) {
				sv->free(eq_ctx[thr_id]);
				eq_ctx[thr_id] = NULL;
			}
			cudaGetLastError();
			eq_sv[thr_id] = sv;
		}
		if (!eq_pipe[thr_id]) {
			gpulog(LOG_ERR, thr_id, "equihash%d/%d needs %.2f GiB of VRAM, the allocation failed",
				eqv->wn, eqv->wk, need);
			proper_exit(EXIT_CODE_CUDA_ERROR);
			return -1;
		}
		gpulog(LOG_INFO, thr_id, "equihash%d/%d: %s solver, %.2f GiB arena", eqv->wn, eqv->wk,
			eq_sv[thr_id]->name, need);
		gpus_intensity[thr_id] = 8192;
		api_set_throughput(thr_id, gpus_intensity[thr_id]);

		/* self-test: the vector header must give its solution */
		struct eq_vec t;
		const int rc = eq_parse_vector(eqv, &t) ? eq_gpu_check(thr_id, eqv, eq_sv[thr_id], eq_ctx[thr_id],
			eq_pipe[thr_id], &t) : 0;
		if (rc < 0)
			selftest_cuda_fault();
		selftest_gate(thr_id, "equihash", rc == 1);
		eq_pend_n[thr_id] = 0;
		init[thr_id] = true;
	}

	gettimeofday(&tv_start, NULL);
	memcpy(endiandata, pdata, 140);
	work->valid_nonces = 0;

	/* queued solves carry over when only the nonce words the miner loop steps between
	 * calls differ (data[30], data[31] with the GPU id in its top byte) */
	if (eq_pend_n[thr_id]) {
		const uint32_t *q = eq_pend_hdr[thr_id][0];
		if (memcmp(q, endiandata, NONCE_OFT * 4) ||
		    memcmp(q + NONCE_OFT + 2, endiandata + NONCE_OFT + 2, 140 - (NONCE_OFT + 2) * 4) ||
		    strncmp(eq_pend_pers[thr_id][0], eq_personal, 8))
			eq_reset(thr_id);
	}
	if (!eq_pend_n[thr_id] && !eq_launch(thr_id, endiandata))
		goto out;

	do {
		/* queue the next nonce first, so the GPU works while this one is checked */
		if (eq_pend_n[thr_id] == 1 && !work_restart[thr_id].restart) {
			uint32_t next[35];
			memcpy(next, eq_pend_hdr[thr_id][0], 140);
			next[NONCE_OFT] += nonce_increment;
			eq_launch(thr_id, next);
		}
		char pers[16];
		memcpy(endiandata, eq_pend_hdr[thr_id][0], 140);
		memcpy(pers, eq_pend_pers[thr_id][0], 16);
		memcpy(eq_pend_hdr[thr_id][0], eq_pend_hdr[thr_id][1], 140);
		memcpy(eq_pend_pers[thr_id][0], eq_pend_pers[thr_id][1], 16);
		eq_pend_n[thr_id]--;

		struct eq_proofs *p = &eq_found[thr_id];
		unsigned drops[4] = { 0, 0, 0, 0 };
		p->n = 0;
		const int nsol = eq_sv[thr_id]->collect(eq_ctx[thr_id], eq_pipe[thr_id], eq_emit, p, drops);
		if (nsol < 0) {
			gpulog(LOG_ERR, thr_id, "equihash solver failed (%d)", nsol);
			eq_reset(thr_id);
			break;
		}
		sols += (uint32_t) nsol;
		*hashes_done = sols;
		if (opt_debug)
			eq_debug_count(thr_id, endiandata, pers, p, nsol);

		/* the scatter tail is expected; staging or capacity losses mean wrong sizing */
		if (drops[1] || drops[3]) {
			eq_lost[thr_id] += drops[1] + drops[3];
			gpulog(LOG_WARNING, thr_id, "equihash: %u staging and %u capacity losses (%u in"
				" all), solutions are lost", drops[1], drops[3], eq_lost[thr_id]);
		}

		for (int s = 0; s < p->n; s++) {
			uint8_t _ALIGN(64) full[140 + 3 + 400];
			uint32_t _ALIGN(64) vhash[8];
			memcpy(full, endiandata, 140);
			const int len = eq_pack(eqv, p->idx[s], full + 140);
			equi_hash(full, vhash, 140 + len);
			if (vhash[7] > ptarget[7] || !fulltest(vhash, ptarget))
				continue;
			if (eqv->verify((const char *) endiandata, pers, p->idx[s]) != 0) {
				eq_bad[thr_id]++;
				gpu_increment_reject(thr_id);
				gpulog(LOG_WARNING, thr_id, "equihash solution does not verify on the CPU (%u so far)",
					eq_bad[thr_id]);
				continue;
			}
			/* one share per return: work->extra holds one solution */
			work->valid_nonces = 1;
			memcpy(work->data, endiandata, 140);
			equi_store_work_solution(work, vhash, full + 140);
			work->nonces[0] = endiandata[NONCE_OFT];
			goto out;
		}
		if (eq_slice_done(&tv_start))
			break;
	} while (!work_restart[thr_id].restart);

out:
	*hashes_done = sols;
	/* resume at the queued nonce, so the next call collects that solve */
	pdata[NONCE_OFT] = eq_pend_n[thr_id] ? eq_pend_hdr[thr_id][0][NONCE_OFT]
		: endiandata[NONCE_OFT] + nonce_increment;
	return work->valid_nonces;
}

extern "C" void free_equihash(int thr_id)
{
	if (!init[thr_id])
		return;
	eq_sv[thr_id]->pipe_reset(eq_pipe[thr_id]);
	eq_sv[thr_id]->pipe_free(eq_pipe[thr_id]);
	eq_sv[thr_id]->free(eq_ctx[thr_id]);
	eq_pipe[thr_id] = eq_ctx[thr_id] = NULL;
	eq_pend_n[thr_id] = 0;
	init[thr_id] = false;
}

/* ---- --test-vectors ---- */

void equihash_test_vectors(int *passed, int *failed)
{
	for (int i = 0; i < 2; i++) {
		const struct eq_variant *v = &eq_variants[i];
		struct eq_vec t;
		const bool cpu_ok = eq_parse_vector(v, &t) && eq_cpu_check(v, &t);
		applog(cpu_ok ? LOG_INFO : LOG_ERR, "CPU: equihash%d vector 1 %s", v->wn,
			cpu_ok ? "OK" : "FAILED");
		cpu_ok ? (*passed)++ : (*failed)++;
		if (!cpu_ok)
			continue;

		/* every solver of the variant on every GPU that holds its arena */
		for (int thr_id = 0; thr_id < opt_n_threads; thr_id++)
		for (int k = 0; k < 2 && v->solver[k]; k++) {
			const struct eq_solver *sv = v->solver[k];
			size_t mfree = 0, mtotal = 0;
			cudaSetDevice(device_map[thr_id]);
			cudaMemGetInfo(&mfree, &mtotal);
			const unsigned long long need = sv->arena();
			void *ctx = need < mfree ? sv->init() : NULL;
			void *pipe = ctx ? sv->pipe_new() : NULL;
			if (!ctx || !pipe) {
				gpulog(LOG_WARNING, thr_id, "equihash%d vector 1 (%s) skipped: needs %.2f GiB,"
					" %.2f GiB free", v->wn, sv->name, (double) need / (1 << 30),
					(double) mfree / (1 << 30));
				if (ctx) sv->free(ctx);
				cudaGetLastError();
				continue;
			}
			const bool gpu_ok = eq_gpu_check(thr_id, v, sv, ctx, pipe, &t) == 1;
			gpulog(gpu_ok ? LOG_INFO : LOG_ERR, thr_id, "equihash%d vector 1 (%s) %s", v->wn,
				sv->name, gpu_ok ? "OK" : "FAILED");
			gpu_ok ? (*passed)++ : (*failed)++;
			sv->pipe_reset(pipe);
			sv->pipe_free(pipe);
			sv->free(ctx);
		}
	}
}
