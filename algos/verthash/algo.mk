# verthash (included from Makefile.am, keep in sync with ccminer.props).
ALGO_DEFINES += -DWITH_VERTHASH
ccminer_SOURCES += algos/verthash/verthash.cu algos/verthash/cuda_verthash.cu \
		   algos/verthash/cuda_verthash_selftest.cu algos/verthash/verthash_data.cpp \
		   algos/verthash/register.cpp algos/verthash/verthash.h
