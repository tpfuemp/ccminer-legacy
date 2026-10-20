/**
 * Made to benchmark and test algo switch
 *
 * 2015 - tpruvot@github
 */

#include <unistd.h>

#include "miner.h"
#include "algos.h"
#include "algo_registry.h"
#include <cuda_runtime.h>

#ifdef __APPLE__
#include "compat/pthreads/pthread_barrier.hpp"
#endif

int bench_algo = -1;

static double algo_hashrates[MAX_GPUS][ALGO_COUNT] = { 0 };
static uint32_t algo_throughput[MAX_GPUS][ALGO_COUNT] = { 0 };
static int algo_mem_used[MAX_GPUS][ALGO_COUNT] = { 0 };
static int device_mem_free[MAX_GPUS] = { 0 };

static pthread_barrier_t miner_barr;
static pthread_barrier_t algo_barr;
static pthread_mutex_t bench_lock = PTHREAD_MUTEX_INITIALIZER;

extern double thr_hashrates[MAX_GPUS];

// algos the benchmark does not run, even when compiled in
static bool bench_skip_algo(int algo, int dev_id)
{
	switch (algo) {
	// duplicated algos
	case ALGO_C11: // same as x11
	case ALGO_DMD_GR: // same as groestl
	case ALGO_HEAVY: // dead
	case ALGO_MJOLLNIR: // same as heavy
	case ALGO_KECCAKC: // same as keccak
	case ALGO_WHIRLCOIN: // same as whirlpool
	case ALGO_WHIRLPOOLX: // disabled
	// todo: algo switch from RPC 2.0
	case ALGO_CRYPTOLIGHT:
	case ALGO_CRYPTONIGHT:
	case ALGO_WILDKECCAK:
	case ALGO_ANIME: // to fix
	case ALGO_QUARK: // to fix
	// and unwanted ones...
	case ALGO_SCRYPT:
	case ALGO_SCRYPT_JANE:
		return true;
	case ALGO_LBRY:
		return CUDART_VERSION < 7000;
	case ALGO_GROESTL:
	case ALGO_MYR_GR:
	case ALGO_NEOSCRYPT:
		// incompatible SM 2.1 kernels...
		return device_sm[dev_id] && device_sm[dev_id] < 300;
	}
	return false;
}

// next benchmarked algo after 'algo', ALGO_COUNT when done
static int bench_next_algo(int algo, int dev_id)
{
	do {
		algo++;
	} while (algo < ALGO_COUNT && (!algo_find(algo) || bench_skip_algo(algo, dev_id)));
	return algo;
}

void bench_init(int threads)
{
	bench_algo = opt_algo = (enum sha_algos) bench_next_algo(-1, device_map[0]); /* first */
	applog(LOG_BLUE, "Starting benchmark mode with %s", algo_names[opt_algo]);
	pthread_barrier_init(&miner_barr, NULL, threads);
	pthread_barrier_init(&algo_barr, NULL, threads);
	// required for usage of first algo.
	for (int n=0; n < opt_n_threads; n++) {
		device_mem_free[n] = cuda_available_memory(n);
	}
}

void bench_free()
{
	pthread_barrier_destroy(&miner_barr);
	pthread_barrier_destroy(&algo_barr);
}

// benchmark all algos (called once per mining thread)
bool bench_algo_switch_next(int thr_id)
{
	int algo = (int) opt_algo;
	int prev_algo = algo;
	int dev_id = device_map[thr_id % MAX_GPUS];
	int mfree, mused;
	// doesnt seems enough to prevent device slow down
	// after some algo switchs
	bool need_reset = (gpu_threads == 1);

	algo = bench_next_algo(algo, dev_id);

	// free current algo memory and track mem usage
	mused = cuda_available_memory(thr_id);
	algo_free_all(thr_id);
	CUDA_LOG_ERROR();

	// device can take some time to free
	mfree = cuda_available_memory(thr_id);
	if (device_mem_free[thr_id] > mfree) {
		sleep(1);
		mfree = cuda_available_memory(thr_id);
	}

	// we need to wait completion on all cards before the switch
	if (opt_n_threads > 1) {
		pthread_barrier_wait(&miner_barr);
	}

	char rate[32] = { 0 };
	double hashrate = stats_get_speed(thr_id, thr_hashrates[thr_id]);
	format_hashrate(hashrate, rate);
	gpulog(LOG_NOTICE, thr_id, "%s hashrate = %s", algo_names[prev_algo], rate);

	// ensure memory leak is still real after the barrier
	if (device_mem_free[thr_id] > mfree) {
		mfree = cuda_available_memory(thr_id);
	}

	// check if there is memory leak
	if (device_mem_free[thr_id] - mfree > 1) {
		gpulog(LOG_WARNING, thr_id, "possible %d MB memory leak in %s! %d MB free",
			(device_mem_free[thr_id] - mfree), algo_names[prev_algo], mfree);
		cuda_reset_device(thr_id, NULL); // force to free the leak
		need_reset = false;
		mfree = cuda_available_memory(thr_id);
	}
	// store used memory per algo
	algo_mem_used[thr_id][opt_algo] = device_mem_free[thr_id] - mused;
	device_mem_free[thr_id] = mfree;

	// store to dump a table per gpu later
	algo_hashrates[thr_id][prev_algo] = hashrate;

	// wait the other threads to display logs correctly
	if (opt_n_threads > 1) {
		pthread_barrier_wait(&algo_barr);
	}

	if (algo == ALGO_AUTO)
		return false; // all algos done

	// mutex primary used for the stats purge
	pthread_mutex_lock(&bench_lock);
	stats_purge_all();

	opt_algo = (enum sha_algos) algo;
	global_hashrate = 0;
	thr_hashrates[thr_id] = 0; // reset for minmax64
	pthread_mutex_unlock(&bench_lock);

	if (need_reset)
		cuda_reset_device(thr_id, NULL);

	if (thr_id == 0)
		applog(LOG_BLUE, "Benchmark algo %s...", algo_names[algo]);

	return true;
}

void bench_set_throughput(int thr_id, uint32_t throughput)
{
	algo_throughput[thr_id][opt_algo] = throughput;
}

void bench_display_results()
{
	for (int n=0; n < opt_n_threads; n++)
	{
		int dev_id = device_map[n];
		applog(LOG_BLUE, "Benchmark results for GPU #%d - %s:", dev_id, device_name[dev_id]);
		for (int i=0; i < ALGO_COUNT-1; i++) {
			double rate = algo_hashrates[n][i];
			if (rate == 0.0) continue;
			applog(LOG_INFO, "%12s : %12.1f kH/s, %5d MB, %8u thr.", algo_names[i],
				rate / 1024., algo_mem_used[n][i], algo_throughput[n][i]);
		}
	}
}
