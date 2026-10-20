#!/bin/sh
# Out-of-tree Linux build into build-linux/cuda<major>/. Arguments are passed
# to configure, e.g. ./build.sh --with-cuda-arch="ptx52".
# CUDA selects the toolkit in /usr/local/cuda-$CUDA (default 8.0), e.g.
# CUDA=10.1 ./build.sh builds build-linux/cuda10/ccminer. TAG is appended to
# the directory name, e.g. TAG=-jit52 ./build.sh --with-cuda-arch=ptx52 builds
# build-linux/cuda8-jit52/ccminer.
#
# configure is run through a relative path so sources reach the compilers as
# relative paths and no build directory ends up in the binary (__FILE__).
# No -march=native: the host code is only the CPU reference, and a portable
# baseline keeps the binary running on older CPUs.
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
