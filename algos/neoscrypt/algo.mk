# neoscrypt (included from Makefile.am, keep in sync with ccminer.props).
ccminer_SOURCES += algos/neoscrypt/neoscrypt.cu algos/neoscrypt/cuda_neoscrypt.cu \
		   algos/neoscrypt/cuda_neoscrypt_selftest.cu algos/neoscrypt/neoscrypt-cpu.c \
		   algos/neoscrypt/register.cpp algos/neoscrypt/neoscrypt.h \
		   algos/neoscrypt/neoscrypt-cpu.h
