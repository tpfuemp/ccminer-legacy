# whirlpool (included from Makefile.am, keep in sync with ccminer.props).
ccminer_SOURCES += algos/whirlpool/whirlpool.cu algos/whirlpool/cuda_whirlpool.cu \
		   algos/whirlpool/cuda_whirlpool_selftest.cu algos/whirlpool/register.cpp \
		   algos/whirlpool/whirlpool.h algos/whirlpool/cuda_whirlpool_tables.cuh
