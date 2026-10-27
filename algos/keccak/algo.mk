# keccak, keccakc (included from Makefile.am, keep in sync with ccminer.props).
ccminer_SOURCES += algos/keccak/keccak256.cu algos/keccak/cuda_keccak256.cu \
		   algos/keccak/cuda_keccak_selftest.cu algos/keccak/register.cpp
