#!/bin/sh
# Out-of-tree Linux build into build-linux/cuda<major><TAG>/, arguments go to
# configure. CUDA=10.1 selects /usr/local/cuda-10.1 (default 8.0), e.g.
#   CUDA=10.1 TAG=-jit52 ./build.sh --with-cuda-arch=ptx52
# configure runs via a relative path so no build path ends up in __FILE__.
set -e
cd "$(dirname "$0")"

CUDA=${CUDA:-8.0}
CUDA_PATH=/usr/local/cuda-$CUDA
BUILD_DIR=build-linux/cuda${CUDA%%.*}${TAG}

# An in-tree configure blocks out-of-tree builds
[ -f Makefile ] && make distclean >/dev/null 2>&1 || true
rm -f config.status Makefile

./autogen.sh

rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

CFLAGS="-O2 -mtune=generic" CXXFLAGS="-O2 -mtune=generic" \
	../../configure --with-cuda="$CUDA_PATH" "$@"

make -j "$(nproc)"
