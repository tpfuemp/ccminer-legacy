# blake, blakecoin (included from Makefile.am, keep in sync with ccminer.props).
ccminer_SOURCES += algos/blake/blake256.cu algos/blake/cuda_blake_selftest.cu \
		   algos/blake/register.cpp
