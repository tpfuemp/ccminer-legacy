#!/bin/sh
# Builds the four Linux test variants: cuda8, cuda10, cuda8-jit52, cuda8-jit35
# (see "Test builds" in README.md). The two PTX-only builds run in parallel.
# INCREMENTAL=1 is passed on to build.sh. Logs go to build-linux/logs/.
set -e
cd "$(dirname "$0")"
mkdir -p build-linux/logs
LOG=build-linux/logs

run() {
	name=$1; shift
	if "$@" > "$LOG/$name.log" 2>&1; then
		echo "$name: ok ($(grep -m 1 -E '^(incremental|full rebuild)' "$LOG/$name.log" || echo full build))"
	else
		echo "$name: FAILED, see $LOG/$name.log"
		return 1
	fi
}

run cuda8 ./build.sh
run cuda10 env CUDA=10.1 NO_AUTOGEN=1 ./build.sh

# PTX only: light enough to share the CPUs
run cuda8-jit52 env TAG=-jit52 NO_AUTOGEN=1 JOBS=4 ./build.sh --with-cuda-arch=ptx52 &
p52=$!
run cuda8-jit35 env TAG=-jit35 NO_AUTOGEN=1 JOBS=4 ./build.sh --with-cuda-arch=ptx35 &
p35=$!
rc=0
wait $p52 || rc=1
wait $p35 || rc=1
exit $rc
