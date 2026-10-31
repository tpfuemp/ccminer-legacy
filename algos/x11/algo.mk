# x11 and the X11 family building blocks (included from Makefile.am, keep in sync with
# ccminer.props).
ccminer_SOURCES += algos/x11/x11.cu algos/x11/cuda_x11_selftest.cu algos/x11/register.cpp \
		   algos/x11/x11.h \
		   algos/x11/cuda_blake512.cu algos/x11/cuda_bmw512_64.cu \
		   algos/x11/cuda_groestl512.cu algos/x11/cuda_skein512.cu \
		   algos/x11/cuda_jh512.cu algos/x11/cuda_keccak512.cu \
		   algos/x11/cuda_luffa_cubehash512.cu algos/x11/cuda_cubehash512.cu \
		   algos/x11/cuda_shavite512.cu algos/x11/cuda_simd512.cu algos/x11/cuda_echo512.cu \
		   algos/x11/cuda_x11.h algos/x11/cuda_x11_aes.cuh algos/x11/cuda_simd512_func.cuh \
		   algos/x11/cuda_blake512_sp.cuh
