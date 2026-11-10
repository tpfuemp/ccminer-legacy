# lbry (included from Makefile.am, keep in sync with ccminer.props).
ccminer_SOURCES += algos/lbry/lbry.cu algos/lbry/cuda_lbry_merged.cu \
		   algos/lbry/cuda_sha256_lbry.cu algos/lbry/cuda_sha512_lbry.cu \
		   algos/lbry/cuda_lbry_selftest.cu algos/lbry/register.cpp algos/lbry/lbry.h
