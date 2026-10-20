#!/bin/sh
# Install the Linux build toolchain on Ubuntu 16.04: GCC 5.4, autotools, the
# curl/OpenSSL/zlib headers and the CUDA 8.0 and 10.1 toolkits (both accept
# GCC 5; the 10.1 installer needs libxml2). /usr/local/cuda points to 8.0.
# Used by docker/Dockerfile; also works as root on a plain Ubuntu 16.04 VM or
# WSL distro. Toolkits already installed are skipped, runfiles already in /tmp
# are not downloaded again.
set -e

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
	build-essential autoconf automake perl wget ca-certificates \
	libcurl4-openssl-dev libssl-dev zlib1g-dev \
	libxml2
rm -rf /var/lib/apt/lists/*

# install_cuda <version> <runfile> <url> <sha256>
install_cuda() {
	[ -x "/usr/local/cuda-$1/bin/nvcc" ] && return 0
	[ -f "$2" ] || wget -q -O "$2" "$3"
	echo "$4  $2" | sha256sum -c -
	sh "$2" --silent --toolkit --toolkitpath="/usr/local/cuda-$1"
	rm -f "$2"
}

cd /tmp
install_cuda 8.0 cuda_8.0.61_375.26_linux.run \
	https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda_8.0.61_375.26_linux-run \
	9ceca9c2397f841024e03410bfd6eabfd72b384256fbed1c1e4834b5b0ce9dc4
install_cuda 10.1 cuda_10.1.105_418.39_linux.run \
	https://developer.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda_10.1.105_418.39_linux.run \
	33ac60685a3e29538db5094259ea85c15906cbd0f74368733f4111eab6187c8f
ln -sfn /usr/local/cuda-8.0 /usr/local/cuda
