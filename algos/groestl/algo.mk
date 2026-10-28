# groestl, dmd-gr, myr-gr (included from Makefile.am, keep in sync with ccminer.props).
ccminer_SOURCES += algos/groestl/groestlcoin.cpp algos/groestl/cuda_groestlcoin.cu \
		   algos/groestl/myriadgroestl.cpp algos/groestl/cuda_myriadgroestl.cu \
		   algos/groestl/cuda_groestl_selftest.cu algos/groestl/register.cpp \
		   algos/groestl/cuda_groestlcoin.h algos/groestl/groestl_functions_quad.h \
		   algos/groestl/groestl_transf_quad.h
