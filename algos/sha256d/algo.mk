# sha256d (included from Makefile.am, keep in sync with ccminer.props).
ccminer_SOURCES += algos/sha256d/sha256d.cu algos/sha256d/cuda_sha256d.cu \
		   algos/sha256d/cuda_sha256_selftest.cu algos/sha256d/register.cpp
