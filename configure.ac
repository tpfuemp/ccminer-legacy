AC_INIT([ccminer-legacy], [2026.10], [], [ccminer-legacy], [https://github.com/tpfuemp/ccminer-legacy])

AC_PREREQ([2.69])
AC_CONFIG_SRCDIR([core/ccminer.cpp])
AM_INIT_AUTOMAKE([foreign subdir-objects])
AC_CONFIG_HEADERS([ccminer-config.h])

dnl Make sure anyone changing configure.ac/Makefile.am has a clue
AM_MAINTAINER_MODE

dnl No -g by default: DWARF records the absolute build directory, and shipped
dnl binaries must not contain build paths.
: ${CFLAGS="-O2"}
: ${CXXFLAGS="-O2"}

dnl Checks for programs
AC_PROG_CC_C99
AM_PROG_CC_C_O
AC_PROG_CXX
AC_PROG_RANLIB
AM_PROG_AR

dnl Checks for header files
AC_HEADER_STDC
AC_CHECK_HEADERS([sys/endian.h sys/param.h syslog.h])
# sys/sysctl.h requires sys/types.h on FreeBSD
AC_CHECK_HEADERS([sys/sysctl.h], [], [],
[#include <sys/types.h>
#ifdef HAVE_SYS_PARAM_H
#include <sys/param.h>
#endif
])

AC_CHECK_DECLS([be32dec, le32dec, be32enc, le32enc], [], [],
[AC_INCLUDES_DEFAULT
#ifdef HAVE_SYS_ENDIAN_H
#include <sys/endian.h>
#endif
])

AC_FUNC_ALLOCA
AC_CHECK_FUNCS([getopt_long])

PTHREAD_FLAGS="-pthread"

AC_CHECK_LIB(jansson, json_loads, request_jansson=false, request_jansson=true)
AC_CHECK_LIB([pthread], [pthread_create], PTHREAD_LIBS="-lpthread",
  [AC_MSG_ERROR([pthread library required])])

AC_CHECK_LIB([z],[gzopen], [], [])
AC_CHECK_LIB([ssl],[SSL_free], [], [AC_MSG_ERROR([OpenSSL library required])])
AC_CHECK_LIB([crypto],[EVP_DigestFinal_ex], [], [AC_MSG_ERROR([OpenSSL library required])])

AM_CONDITIONAL([WANT_JANSSON], [test x$request_jansson = xtrue])

if test x$request_jansson = xtrue
then
	JANSSON_LIBS="compat/jansson/libjansson.a"
else
	JANSSON_LIBS=-ljansson
fi

LIBCURL_CHECK_CONFIG(, 7.15.2, ,
  [AC_MSG_ERROR([Missing required libcurl >= 7.15.2])])

AC_SUBST(JANSSON_LIBS)
AC_SUBST(PTHREAD_FLAGS)
AC_SUBST(PTHREAD_LIBS)

dnl CUDA toolkit: 8.0 (widest driver range, GCC <= 5) or 10.1 (last release
dnl with sm_30 that runs on the last Kepler drivers, GCC <= 8).
AC_ARG_WITH([cuda],
  [AS_HELP_STRING([--with-cuda=PATH], [CUDA toolkit prefix @<:@default=/usr/local/cuda@:>@])],
  [], [with_cuda=/usr/local/cuda])

AC_ARG_VAR([CUDA_CFLAGS], [extra nvcc flags @<:@default=-O3@:>@])
: ${CUDA_CFLAGS="-O3"}

CUDA_INCLUDES="-I$with_cuda/include"
CUDA_LDFLAGS="-L$with_cuda/lib64"
CUDA_LIBS="-lcudart_static -ldl -lrt"
NVCC="$with_cuda/bin/nvcc"
AS_IF([test -x "$NVCC"], [], [AC_MSG_ERROR([nvcc not found in $with_cuda/bin, use --with-cuda=PATH])])

NVCC_VERSION=`"$NVCC" --version | sed -n 's/.*release \([[0-9]]*\.[[0-9]]*\).*/\1/p'`
AC_MSG_NOTICE([nvcc release $NVCC_VERSION])

dnl GPU code to embed, same defaults as CudaArchs in the vcxproj files.
dnl "N" embeds sm_N SASS plus compute_N PTX, "ptxN" compute_N PTX only.
dnl CUDA 8 builds Kepler as PTX only, the driver JIT generates faster code.
AS_CASE([$NVCC_VERSION],
  [8.*], [cuda_arch_default="ptx30 ptx35 ptx37 50 52 61"],
  [cuda_arch_default="30 35 37 50 52 61"])
AC_ARG_WITH([cuda-arch],
  [AS_HELP_STRING([--with-cuda-arch=LIST], [GPU archs to build @<:@default: "ptx30 ptx35 ptx37 50 52 61" with CUDA 8, "30 35 37 50 52 61" otherwise@:>@])],
  [], [with_cuda_arch="$cuda_arch_default"])

NVCC_ARCH=""
for arch in $with_cuda_arch; do
  case $arch in
    ptx*) cc=${arch#ptx}; NVCC_ARCH="$NVCC_ARCH -gencode=arch=compute_$cc,code=compute_$cc" ;;
    *)    NVCC_ARCH="$NVCC_ARCH -gencode=arch=compute_$arch,code=sm_$arch -gencode=arch=compute_$arch,code=compute_$arch" ;;
  esac
done
AC_MSG_NOTICE([CUDA archs: $with_cuda_arch])

AC_SUBST(CUDA_INCLUDES)
AC_SUBST(CUDA_LDFLAGS)
AC_SUBST(CUDA_LIBS)
AC_SUBST(NVCC)
AC_SUBST(NVCC_ARCH)

AC_CONFIG_FILES([
	Makefile
	compat/Makefile
	compat/jansson/Makefile
	])

AC_OUTPUT
