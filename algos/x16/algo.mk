# x16 family blocks (included from Makefile.am, keep in sync with ccminer.props).
ccminer_SOURCES += algos/x16/cuda_x16_simd512_80.cu algos/x16/cuda_x16_echo512.cu \
		   algos/x16/cuda_x16_echo512_64.cu algos/x16/cuda_x16_fugue512.cu \
		   algos/x16/cuda_x16_shabal512.cu algos/x16/cuda_shabal512.cu \
		   algos/x16/cuda_x16_whirlpool.cu algos/x16/cuda_sha512.cu algos/x16/cuda_luffa512.cu \
		   algos/x16/cuda_tiger192.cu algos/x16/x16.cu algos/x16/cuda_x16_selftest.cu \
		   algos/x16/register.cpp algos/x16/cuda_x16.h algos/x16/x16.h \
		   algos/x16/cuda_x16_whirlpool_tables.cuh
