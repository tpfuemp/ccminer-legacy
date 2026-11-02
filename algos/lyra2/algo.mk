# Lyra2 family building blocks (included from Makefile.am, keep in sync with
# ccminer.props).
ccminer_SOURCES += algos/lyra2/Lyra2.c algos/lyra2/Sponge.c \
		   algos/lyra2/Lyra2.h algos/lyra2/Sponge.h algos/lyra2/cuda_lyra2.h \
		   algos/lyra2/cuda_lyra2.cu algos/lyra2/cuda_blake256_keccak256.cu \
		   algos/lyra2/cuda_cubehash256.cu algos/lyra2/cuda_skein256.cu \
		   algos/lyra2/cuda_groestl256.cu \
		   algos/lyra2/cuda_lyra2_sm2.cuh algos/lyra2/cuda_lyra2_sm5.cuh \
		   algos/lyra2/cuda_lyra2_vectors.h
