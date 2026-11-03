# bc3hash (included from Makefile.am, keep in sync with ccminer.props). The ref/
# sources are compiled as part of bc3hash_oracle.cpp.
ccminer_SOURCES += algos/bc3hash/bc3hash.cu algos/bc3hash/bc3hash_oracle.cpp \
		   algos/bc3hash/register.cpp algos/bc3hash/cuda/bc3_constants.cuh \
		   algos/bc3hash/cuda/bc3_ec_device.cuh \
		   algos/bc3hash/cuda/bc3_exp_device.cuh \
		   algos/bc3hash/cuda/bc3_hashes_device.cuh \
		   algos/bc3hash/cuda/bc3_mix_device.cuh \
		   algos/bc3hash/cuda/bc3_modinv_device.cuh \
		   algos/bc3hash/cuda/bc3_secp_device.cuh \
		   algos/bc3hash/cuda/bc3_uint_device.cuh algos/bc3hash/ref/bc3_shim.h \
		   algos/bc3hash/ref/constants.hpp algos/bc3hash/ref/mix_internal.hpp \
		   algos/bc3hash/ref/standard_hashes.hpp algos/bc3hash/ref/uint.hpp \
		   algos/bc3hash/ref/include/bc3/bc3.hpp cuda/secp256k1_field_device.cuh \
		   cuda/secp256k1_group_device.cuh cuda/secp256k1_modinv32_device.cuh
EXTRA_ccminer_SOURCES += algos/bc3hash/ref/mix.cpp algos/bc3hash/ref/hash.cpp \
		   algos/bc3hash/ref/standard_hashes.cpp
