# tribus (included from Makefile.am, keep in sync with ccminer.props).
ccminer_SOURCES += algos/tribus/tribus.cu algos/tribus/cuda_tribus_echo512.cu \
		   algos/tribus/cuda_tribus_selftest.cu algos/tribus/register.cpp \
		   algos/tribus/tribus.h algos/tribus/cuda_echo512_aes.cuh
