/*
 * Unity build of the vendored libsecp256k1 (compat/secp256k1, MIT, see its
 * COPYING), used by the host reference code of the bc3hash algo.
 *
 * Portable config: 32-bit field and scalar limbs, no GMP, built-in inverses,
 * no precomputed ecmult_gen table (built at context creation).
 */
#define USE_NUM_NONE 1
#define USE_FIELD_INV_BUILTIN 1
#define USE_SCALAR_INV_BUILTIN 1
#define USE_FIELD_10X26 1
#define USE_SCALAR_8X32 1

/* config complete above; HAVE_CONFIG_H would pull a missing libsecp256k1-config.h */
#undef HAVE_CONFIG_H

/* the library's narrowing conversions and unused static helpers (gcc: no pop) */
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable: 4244 4267)
#elif defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wunused-function"
#endif

/* needs compat and compat/secp256k1 on the include path */
#include "secp256k1/src/secp256k1.c"

#if defined(_MSC_VER)
#pragma warning(pop)
#endif
