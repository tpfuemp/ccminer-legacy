# heavyhash (included from Makefile.am, keep in sync with ccminer.props).
ccminer_SOURCES += algos/heavyhash/cuda_heavyhash.cu algos/heavyhash/heavyhash_cpu.c \
		   algos/heavyhash/heavyhash_cpu.h algos/heavyhash/register.cpp
