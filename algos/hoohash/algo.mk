# hoohash (included from Makefile.am, keep in sync with ccminer.props).
# ref/ is PePe-core's reference code unchanged; blake3.h maps its hasher API.
ccminer_SOURCES += algos/hoohash/cuda_hoohash.cu algos/hoohash/register.cpp \
		   algos/hoohash/ref/hoohash.c algos/hoohash/ref/hoohash.h \
		   algos/hoohash/blake3.h algos/hoohash/hoohash_device.cuh
# the reference rounds every product and sum: no fused multiply-add
algos/hoohash/cuda_hoohash.$(OBJEXT): nvcc_FLAGS += --fmad=false
