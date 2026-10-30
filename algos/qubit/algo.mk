# qubit (included from Makefile.am, keep in sync with ccminer.props).
ccminer_SOURCES += algos/qubit/qubit.cu algos/qubit/cuda_qubit_luffa512.cu \
		   algos/qubit/cuda_qubit_selftest.cu algos/qubit/register.cpp \
		   algos/qubit/qubit.h
