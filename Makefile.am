# Linux build. Keep the source lists in sync with ccminer.props: core and
# shared CUDA headers are listed here, each ported algo adds its files, per-file
# nvcc rules and WITH_* defines in algos/<name>/algo.mk. check-sync.sh compares
# both builds.

if WANT_JANSSON
JANSSON_INCLUDES = -I$(top_srcdir)/compat/jansson
endif

EXTRA_DIST = autogen.sh build.sh check-sync.sh README.md LICENSE.txt CREDITS.txt \
	     ccminer.sln ccminer.props ccminer-cuda8.vcxproj ccminer-cuda10.vcxproj \
	     docker/Dockerfile docker/provision.sh

SUBDIRS = compat

bin_PROGRAMS = ccminer

ccminer_SOURCES = \
	core/algo_registry.cpp core/algo_registry.h core/algos.h \
	core/api.cpp core/api_control.cpp core/api_control.h \
	core/api_http.c core/api_http.h core/api_metrics.c core/api_metrics.h \
	core/api_model.cpp core/api_model.h core/api_routes.cpp core/api_routes.h \
	core/bench.cpp core/bignum.cpp core/bignum.hpp core/ccminer.cpp \
	core/compat.h core/crc32.c core/cuda.cpp core/elist.h core/hashlog.cpp \
	core/miner.h core/nvapi.cpp core/nvml.cpp core/nvml.h core/nvsettings.cpp \
	core/pools.cpp core/serialize.hpp core/sha256.c core/stats.cpp \
	core/sysinfos.cpp core/test_vectors.cpp core/uint256.h core/util.cpp \
	cuda/cuda_compat.h cuda/cuda_helper.h cuda/selftest_gate.cuh cuda/sha256_device.cuh \
	sph/blake.c sph/keccak.c sph/skein.c sph/sph_blake.h sph/sph_keccak.h \
	sph/sph_skein.h sph/sph_types.h

# WITH_* switches of the ported algos, added by their algo.mk
ALGO_DEFINES =

include $(srcdir)/algos/common/algo.mk
include $(srcdir)/algos/blake/algo.mk
include $(srcdir)/algos/sha256d/algo.mk
include $(srcdir)/algos/keccak/algo.mk
include $(srcdir)/algos/skein/algo.mk

COMMON_INCLUDES = -I$(top_builddir) -I$(top_srcdir) -I$(top_srcdir)/core \
		  -I$(top_srcdir)/cuda $(JANSSON_INCLUDES) @CUDA_INCLUDES@
COMMON_DEFINES = -DUSE_WRAPNVML $(ALGO_DEFINES)

ccminer_CPPFLAGS = $(COMMON_INCLUDES) $(COMMON_DEFINES) @LIBCURL_CPPFLAGS@ \
		   $(PTHREAD_FLAGS) -fno-strict-aliasing
ccminer_CXXFLAGS = -std=gnu++11
ccminer_LDFLAGS  = $(PTHREAD_FLAGS) @CUDA_LDFLAGS@
ccminer_LDADD    = @LIBCURL@ @JANSSON_LIBS@ @PTHREAD_LIBS@ @CUDA_LIBS@ @LIBS@

# GPU archs come from configure --with-cuda-arch; "make NVCC_ARCH=..." overrides.
NVCC_ARCH = @NVCC_ARCH@
NVCC_CCBIN = @NVCC_CCBIN@
nvcc_FLAGS = $(NVCC_ARCH) $(NVCC_CCBIN) -std=c++11 $(COMMON_INCLUDES) $(COMMON_DEFINES) \
	     @CUDA_CFLAGS@ -Xcompiler -Wall

# Relative source paths keep build paths out of __FILE__ (no -ffile-prefix-map
# in GCC 5); out-of-tree builds need the object dir created here.
.cu.o:
	@$(MKDIR_P) $(@D)
	$(NVCC) $(nvcc_FLAGS) -o $@ -c $<
