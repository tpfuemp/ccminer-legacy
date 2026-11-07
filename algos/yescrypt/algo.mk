# yescrypt family (included from Makefile.am, keep in sync with ccminer.props).
ccminer_SOURCES += algos/yescrypt/yescrypt.cu algos/yescrypt/cuda_yescrypt.cu \
		   algos/yescrypt/cuda_yescrypt_selftest.cu algos/yescrypt/register.cpp \
		   algos/yescrypt/yescrypt-common.c algos/yescrypt/yescrypt-opt.c \
		   algos/yescrypt/sha256_Y.c algos/yescrypt/sha256_Y.h algos/yescrypt/yescrypt.h \
		   algos/yescrypt/yescrypt-platform.h algos/yescrypt/cuda_yescrypt.h \
		   algos/yescrypt/yescrypt_algo.h
