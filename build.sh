#!/bin/sh
# Out-of-tree Linux build into build-linux/cuda<major><TAG>/, arguments go to
# configure. CUDA=10.1 selects /usr/local/cuda-10.1 (default 8.0), e.g.
#   CUDA=10.1 TAG=-jit52 ./build.sh --with-cuda-arch=ptx52
# configure runs via a relative path so no build path ends up in __FILE__.
# INCREMENTAL=1: only make, unless headers/build files or configure arguments
# changed (see "Test builds" in README.md). NO_AUTOGEN=1 skips autogen.sh, JOBS
# sets make -j.
set -e
cd "$(dirname "$0")"

CUDA=${CUDA:-8.0}
CUDA_PATH=/usr/local/cuda-$CUDA
BUILD_DIR=build-linux/cuda${CUDA%%.*}${TAG}
JOBS=${JOBS:-$(nproc)}
STAMP=$BUILD_DIR/.build-stamp
CONFIG="CUDA=$CUDA $*"

# prints why a full rebuild is needed, nothing when make alone is enough
full_rebuild_reason() {
	[ -f "$BUILD_DIR/Makefile" ] && [ -f "$STAMP" ] || { echo "no previous build"; return; }
	[ "$(cat "$STAMP")" = "$CONFIG" ] || { echo "configure arguments changed"; return; }
	changed=$(find core cuda algos sph compat \
		\( -name '*.h' -o -name '*.cuh' -o -name '*.hpp' \) -newer "$STAMP" -print | head -n 1)
	[ -z "$changed" ] || { echo "header changed: $changed"; return; }
	changed=$(find configure.ac Makefile.am autogen.sh build.sh algos compat -maxdepth 2 \
		\( -name configure.ac -o -name Makefile.am -o -name autogen.sh -o -name build.sh \
		-o -name algo.mk \) -newer "$STAMP" -print | head -n 1)
	[ -z "$changed" ] || echo "build file changed: $changed"
}

if [ "$INCREMENTAL" = 1 ]; then
	reason=$(full_rebuild_reason)
	if [ -z "$reason" ]; then
		echo "incremental build in $BUILD_DIR"
		cd "$BUILD_DIR"
		make -j "$JOBS"
		echo "$CONFIG" > .build-stamp
		exit 0
	fi
	echo "full rebuild of $BUILD_DIR: $reason"
fi

if [ "$NO_AUTOGEN" != 1 ]; then
	# An in-tree configure blocks out-of-tree builds
	[ -f Makefile ] && make distclean >/dev/null 2>&1 || true
	rm -f config.status Makefile

	./autogen.sh
fi

rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

CFLAGS="-O2 -mtune=generic" CXXFLAGS="-O2 -mtune=generic" \
	../../configure --with-cuda="$CUDA_PATH" "$@"

make -j "$JOBS"
echo "$CONFIG" > .build-stamp
