/**
 * Verthash (Vertcoin): SHA-3 and 4096 random 32-byte reads per nonce from a fixed
 * 1.28 GB data file (verthash.dat), kept in VRAM. The nonce is hashed little-endian.
 */
#include <stdio.h>
#include <string.h>

#include "miner.h"
#include "cuda_helper.h"
#include "verthash.h"

#define VH_MAX_CAND (VH_MAX_RESULTS - 1)

/* ---- data file, shared by every GPU thread and the CPU check ---- */

static pthread_mutex_t s_dat_lock = PTHREAD_MUTEX_INITIALIZER;
static uint8_t *s_dat;
static size_t s_dat_size;
static int s_dat_state; /* 0 not tried, 1 ready, -1 failed */

bool verthash_data_ready(void)
{
	pthread_mutex_lock(&s_dat_lock);
	if (!s_dat_state) {
		const char *path = opt_verthash_data ? opt_verthash_data : "verthash.dat";
		s_dat_state = -1;
		applog(LOG_INFO, "verthash: loading %s", path);
		if (verthash_data_load(path, &s_dat, &s_dat_size) != 0) {
			applog(LOG_ERR, "verthash: cannot read %s (--verthash-data <file>, or create it with"
				" --generate-verthash-dat)", path);
		} else if (!verthash_data_verify(s_dat, s_dat_size)) {
			applog(LOG_ERR, "verthash: %s is not the Vertcoin data file (%zu bytes, SHA-256 differs)",
				path, s_dat_size);
			free(s_dat);
			s_dat = NULL;
		} else {
			applog(LOG_INFO, "verthash: %s verified, %zu bytes", path, s_dat_size);
			s_dat_state = 1;
		}
	}
	const bool ready = s_dat_state == 1;
	pthread_mutex_unlock(&s_dat_lock);
	return ready;
}

const uint8_t *verthash_data(size_t *size)
{
	*size = s_dat_size;
	return s_dat;
}

extern "C" void verthash_hash(void *output, const void *input)
{
	if (!verthash_data_ready()) {
		memset(output, 0xff, 32);
		return;
	}
	verthash_hash_blob(s_dat, s_dat_size, input, output);
}

/* ---- GPU ---- */

static uint64_t *d_memory[MAX_GPUS], *d_memory_odd[MAX_GPUS];
static uint64_t *d_iohashes[MAX_GPUS], *d_kstates[MAX_GPUS];
static uint32_t *d_res[MAX_GPUS];

void verthash_setBlock(int thr_id, const uint32_t *endiandata)
{
	verthash_cuda_setBlock(endiandata, d_kstates[thr_id]);
}

uint32_t verthash_scan(int thr_id, uint32_t threads, uint32_t startNonce, uint32_t target7,
	uint32_t *out, uint32_t max_out, uint32_t *total)
{
	uint32_t h_res[VH_MAX_RESULTS];
	cudaMemset(d_res[thr_id], 0, sizeof(uint32_t));
	verthash_cuda_scan(threads, startNonce, target7, d_kstates[thr_id], d_memory[thr_id],
		d_memory_odd[thr_id], d_iohashes[thr_id], d_res[thr_id]);
	cudaMemcpy(h_res, d_res[thr_id], sizeof(h_res), cudaMemcpyDeviceToHost);

	*total = h_res[0];
	const uint32_t all = min(h_res[0], (uint32_t) VH_MAX_CAND);
	uint32_t *c = &h_res[1];
	// ascending, so the two lowest are reported and the resume skips nothing
	for (uint32_t i = 1; i < all; i++)
		for (uint32_t j = i; j > 0 && c[j - 1] - startNonce > c[j] - startNonce; j--) {
			const uint32_t s = c[j]; c[j] = c[j - 1]; c[j - 1] = s;
		}
	const uint32_t n = min(all, max_out);
	memcpy(out, c, n * sizeof(uint32_t));
	return n;
}

void verthash_read_hashes(int thr_id, uint32_t count, uint32_t *out)
{
	cudaMemcpy(out, d_iohashes[thr_id], (size_t) count * 32, cudaMemcpyDeviceToHost);
}

static bool init[MAX_GPUS] = { 0 };

extern "C" int scanhash_verthash(int thr_id, struct work *work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t _ALIGN(64) endiandata[20];
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];

	if (opt_benchmark)
		ptarget[7] = 0x00ff;

	if (!verthash_data_ready())
		proper_exit(EXIT_CODE_SW_INIT_ERROR);

	const int dev_id = device_map[thr_id];
	// Kepler: smaller batches keep one launch chain well below the 2 s WDDM TDR
	const int intensity = (device_sm[dev_id] < 500) ? 14 : 18;
	const uint32_t throughput = cuda_default_throughput(thr_id, 1U << intensity);

	if (!init[thr_id])
	{
		cudaSetDevice(dev_id);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			// reduce cpu usage
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
			CUDA_LOG_ERROR();
		}
		gpulog(LOG_INFO, thr_id, "Intensity set to %g, %u cuda threads", throughput2intensity(throughput), throughput);

		size_t size;
		const uint8_t *dat = verthash_data(&size);
		CUDA_CALL_OR_RET_X(cudaMalloc(&d_memory[thr_id], size), -1);
		cudaMemcpy(d_memory[thr_id], dat, size, cudaMemcpyHostToDevice);

		const size_t padded = (throughput + VH_PAD - 1) / VH_PAD * VH_PAD;
		CUDA_CALL_OR_RET_X(cudaMalloc(&d_iohashes[thr_id], padded * 32), -1);
		CUDA_CALL_OR_RET_X(cudaMalloc(&d_kstates[thr_id], 8 * 25 * sizeof(uint64_t)), -1);
		CUDA_CALL_OR_RET_X(cudaMalloc(&d_res[thr_id], VH_MAX_RESULTS * sizeof(uint32_t)), -1);

		// a second copy shifted by 16 bytes aligns every read, if it fits with 256 MB to spare
		size_t free_mem = 0, total_mem = 0;
		cudaMemGetInfo(&free_mem, &total_mem);
		if (free_mem > size + (256U << 20) && cudaMalloc(&d_memory_odd[thr_id], size) == cudaSuccess) {
			cudaMemcpy(d_memory_odd[thr_id], dat + 16, size - 16, cudaMemcpyHostToDevice);
			gpulog(LOG_INFO, thr_id, "verthash: aligned second copy of the data file in VRAM");
		} else {
			cudaGetLastError();
			d_memory_odd[thr_id] = NULL;
		}
		verthash_cuda_set_mdiv(verthash_data_mdiv(size));
		CUDA_LOG_ERROR();

		verthash_device_selftest(thr_id);
		init[thr_id] = true;
	}

	for (int k = 0; k < 19; k++)
		be32enc(&endiandata[k], pdata[k]);
	verthash_setBlock(thr_id, endiandata);

	do {
		// the last launch covers only the rest of the range
		const uint32_t batch = (uint32_t) min((uint64_t) throughput, (uint64_t) max_nonce - pdata[19]);
		if (!batch)
			break;

		uint32_t cand[VH_MAX_CAND], total = 0;
		const uint32_t count = verthash_scan(thr_id, batch, pdata[19], ptarget[7], cand, VH_MAX_CAND, &total);
		*hashes_done = pdata[19] - first_nonce + batch;

		if (total > VH_MAX_CAND)
			gpulog(LOG_WARNING, thr_id, "candidates flood: %u", total);

		// CPU-check every candidate, keep the two lowest valid ones
		uint32_t valid[2];
		int nvalid = 0;
		for (uint32_t n = 0; n < count && nvalid < 2; n++) {
			uint32_t _ALIGN(64) vhash[8];
			endiandata[19] = cand[n];
			verthash_hash(vhash, endiandata);
			if (!fulltest(vhash, ptarget)) {
				// the GPU screens on the top word only: a miss at equality is normal
				if (vhash[7] > ptarget[7]) {
					gpu_increment_reject(thr_id);
					if (!opt_quiet)
						gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU!", cand[n]);
				}
				continue;
			}
			valid[nvalid++] = cand[n];
		}

		if (nvalid) {
			uint32_t _ALIGN(64) vhash[8];
			work->valid_nonces = nvalid;
			for (int i = 0; i < nvalid; i++) {
				work->nonces[i] = valid[i];
				endiandata[19] = valid[i];
				verthash_hash(vhash, endiandata);
				if (i == 0)
					work_set_target_ratio(work, vhash);
				else
					bn_set_target_ratio(work, vhash, 1);
			}
			// resume after the last reported nonce
			pdata[19] = valid[nvalid - 1] + 1;
			return nvalid;
		}

		if ((uint64_t) batch + pdata[19] >= max_nonce) {
			pdata[19] = max_nonce;
			break;
		}

		pdata[19] += batch;

	} while (!work_restart[thr_id].restart);

	*hashes_done = pdata[19] - first_nonce;
	return 0;
}

// cleanup
extern "C" void free_verthash(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaDeviceSynchronize();
	cudaFree(d_memory[thr_id]);
	cudaFree(d_memory_odd[thr_id]);
	cudaFree(d_iohashes[thr_id]);
	cudaFree(d_kstates[thr_id]);
	cudaFree(d_res[thr_id]);
	d_memory_odd[thr_id] = NULL;
	init[thr_id] = false;
	cudaDeviceSynchronize();
}
