# equihash 144/5 + 192/7 (included from Makefile.am, keep in sync with ccminer.props).
ALGO_DEFINES += -DWITH_EQUIHASH
ccminer_SOURCES += algos/equihash/equihash.cu algos/equihash/equi-stratum.cpp \
		   algos/equihash/cuda_equi24b_144.cu algos/equihash/cuda_equi24b_192.cu \
		   algos/equihash/cuda_equi24l_144.cu \
		   algos/equihash/equi_verify_144.cpp algos/equihash/equi_verify_192.cpp \
		   algos/equihash/blake2/blake2bx.cpp algos/equihash/register.cpp \
		   algos/equihash/equihash.h algos/equihash/cuda_equi24b.h algos/equihash/cuda_equi24b.cuh algos/equihash/equi24_warp.cuh \
		   algos/equihash/equi24b_params.h algos/equihash/blake2b_tromp.cuh \
		   algos/equihash/equi_verify.h algos/equihash/equi_tromp.h algos/equihash/equi_pack.h \
		   algos/equihash/equi_vectors.h algos/equihash/blake2/blake2.h \
		   algos/equihash/blake2/blake2-config.h algos/equihash/blake2/blake2-impl.h \
		   algos/equihash/blake2/blake2-round.h algos/equihash/blake2/blake2b-load-sse2.h \
		   algos/equihash/blake2/blake2b-load-sse41.h algos/equihash/blake2/blake2b-round.h
