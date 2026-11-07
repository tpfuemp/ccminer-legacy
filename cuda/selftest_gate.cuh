// SPDX-License-Identifier: GPL-3.0-or-later
/*
 * Fail-closed gate for init-time device self-tests: a wrong hash exits, a
 * test that could not run (CUDA resource error, flagged with
 * selftest_cuda_fault()) only warns. End the self-test with
 * `return selftest_gate(thr_id, "name", passed);`.
 */

#ifndef CUDA_SELFTEST_GATE_CUH
#define CUDA_SELFTEST_GATE_CUH

#include <miner.h> // gpulog, proper_exit, EXIT_CODE_SW_INIT_ERROR

/* one flag per translation unit, cleared by selftest_gate() */
static bool s_selftest_cuda_fault = false;

/* Record "could not run" and return the false the caller wants to propagate. */
static inline bool selftest_cuda_fault(void)
{
	s_selftest_cuda_fault = true;
	return false;
}

/* A CUDA error in a self-test: out of memory is "could not run", anything else (launch
 * failure, illegal address, watchdog) fails the test. Logs it, returns false. */
static inline bool selftest_cuda_error(int thr_id, const char *name, cudaError_t err)
{
	if (thr_id < 0)
		applog(LOG_ERR, "%s self-test: %s", name, cudaGetErrorString(err));
	else
		gpulog(LOG_ERR, thr_id, "%s self-test: %s", name, cudaGetErrorString(err));
	if (err == cudaErrorMemoryAllocation)
		s_selftest_cuda_fault = true;
	return false;
}

/* returns `passed`; exits on a genuine mismatch */
static inline bool selftest_gate(int thr_id, const char *name, bool passed)
{
	const bool could_not_run = s_selftest_cuda_fault;
	s_selftest_cuda_fault = false;

	if (!passed) {
		if (could_not_run) {
			/* the caller has already logged its FAILED line by now */
			gpulog(LOG_WARNING, thr_id, "%s self-test could not run (CUDA resource failure);"
				" the failure above is not evidence of a wrong hash, continuing", name);
		} else {
			gpulog(LOG_ERR, thr_id, "%s: refusing to start after a failed device self-test", name);
			proper_exit(EXIT_CODE_SW_INIT_ERROR);
		}
	}
	return passed;
}

#endif // CUDA_SELFTEST_GATE_CUH
