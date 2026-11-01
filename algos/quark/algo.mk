# quark (included from Makefile.am, keep in sync with ccminer.props).
ccminer_SOURCES += algos/quark/quark.cu algos/quark/cuda_quark_split.cu \
		   algos/quark/cuda_quark_selftest.cu algos/quark/register.cpp \
		   algos/quark/quark.h
