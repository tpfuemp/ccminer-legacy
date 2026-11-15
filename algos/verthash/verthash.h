#ifndef VERTHASH_H
#define VERTHASH_H

#include <stddef.h>
#include <stdint.h>

/* candidate slots: [0] counts, [1..VH_MAX_RESULTS-1] hold nonces */
#define VH_MAX_RESULTS 16

/* known-answer vector: Bitcoin genesis header with time byte 34 and nonce 19509
 * (little-endian), the lowest hash of nonces 0..65535 (cpuminer-opt agrees) */
#define VERTHASH_TV_HEADER "01000000" "0000000000000000000000000000000000000000000000000000000000000000" \
	"3ba3edfd7a7b12b27ac72c3e67768f617fc81bc3888a51323a9fb8aa4b1e5e4a" "34ab5f49" "ffff001d" "354c0000"
#define VERTHASH_TV_HASH "70ee9c6d2b354a96b531495b6be02c966ab29de34654663d260c2145a6a10000"

#ifdef __cplusplus
extern "C" {
#endif

/* data file (verthash_data.cpp): load, SHA-256 against the canonical file, index modulus,
 * generator. The core calls load / verify / generate for --generate-verthash-dat. */
int verthash_data_load(const char *path, uint8_t **out_buf, size_t *out_size);
int verthash_data_verify(const uint8_t *buf, size_t size);
uint32_t verthash_data_mdiv(size_t size);
int verthash_generate_data_file(const char *path);

/* hash of the 80-byte header (nonce bytes 76..79 little-endian) over blob */
void verthash_hash_blob(const uint8_t *blob, size_t blob_size, const void *input, void *output);

#ifdef __cplusplus
}
#endif

/* the data file shared by all GPUs and the CPU check, loaded and verified on first use
 * (--verthash-data, default verthash.dat); false if missing or not the canonical file */
bool verthash_data_ready(void);
const uint8_t *verthash_data(size_t *size);

/* CPU hash of the 80-byte header over the loaded data file (all ff without it) */
extern "C" void verthash_hash(void *output, const void *input);

/* per GPU (verthash.cu): job header (endiandata words 0..18), one launch over
 * [startNonce, startNonce + threads) screened on hash word 7 <= target7, up to max_out
 * candidates ascending into out (returns how many, *total gets all found), and the
 * hashes of the first count nonces of the last launch */
void verthash_setBlock(int thr_id, const uint32_t *endiandata);
uint32_t verthash_scan(int thr_id, uint32_t threads, uint32_t startNonce, uint32_t target7,
	uint32_t *out, uint32_t max_out, uint32_t *total);
void verthash_read_hashes(int thr_id, uint32_t count, uint32_t *out);

/* call after the buffers are allocated, before mining */
bool verthash_device_selftest(int thr_id);

/* kernels (cuda_verthash.cu). d_memory_odd: copy shifted by 16 bytes or NULL. The batch is
 * padded to VH_PAD nonces, extra nonces are not reported. */
#define VH_PAD 256
void verthash_cuda_set_mdiv(uint32_t mdiv);
void verthash_cuda_setBlock(const uint32_t *endiandata, uint64_t *d_kstates);
void verthash_cuda_scan(uint32_t threads, uint32_t startNonce, uint32_t target7,
	const uint64_t *d_kstates, const uint64_t *d_memory, const uint64_t *d_memory_odd,
	uint64_t *d_iohashes, uint32_t *d_res);

#endif
