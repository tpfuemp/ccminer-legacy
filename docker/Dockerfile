# Linux build environment: Ubuntu 16.04 (GCC 5.4) + CUDA 8.0 and 10.1 toolkits.
# NVIDIA no longer publishes the nvidia/cuda:8.0 images, so the toolkits are
# installed from the runfiles.
#
#   docker build -t ccminer-legacy-build docker/
#   docker run --rm -u "$(id -u):$(id -g)" -v "$PWD:/src" -w /src \
#       ccminer-legacy-build ./build.sh
#   docker run --rm -u "$(id -u):$(id -g)" -v "$PWD:/src" -w /src \
#       -e CUDA=10.1 ccminer-legacy-build ./build.sh
#   docker run --rm --gpus all -v "$PWD:/src" -w /src \
#       ccminer-legacy-build build-linux/cuda8/ccminer -a sha256d --benchmark
#
# The binary links the Ubuntu 16.04 curl/OpenSSL 1.0 libraries, so run it in
# this image (NVIDIA Container Toolkit provides the host driver).
FROM ubuntu:16.04

COPY provision.sh /tmp/provision.sh
RUN sh /tmp/provision.sh && rm /tmp/provision.sh

ENV PATH=/usr/local/cuda/bin:$PATH \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility
