# rinhash (included from Makefile.am, keep in sync with ccminer.props).
# argon2/ is the Argon2 reference (PHC release 20190702) unchanged; its
# blake2/blake2.h includes <argon2.h>, hence the include path below.
ccminer_SOURCES += algos/rinhash/cuda_rinhash.cu algos/rinhash/rinhash_cpu.cpp \
		   algos/rinhash/register.cpp algos/rinhash/argon2_device.cuh \
		   algos/rinhash/argon2/argon2.c algos/rinhash/argon2/core.c \
		   algos/rinhash/argon2/encoding.c algos/rinhash/argon2/ref.c \
		   algos/rinhash/argon2/thread.c algos/rinhash/argon2/blake2/blake2b.c \
		   algos/rinhash/argon2/argon2.h algos/rinhash/argon2/core.h \
		   algos/rinhash/argon2/encoding.h algos/rinhash/argon2/thread.h \
		   algos/rinhash/argon2/blake2/blake2.h algos/rinhash/argon2/blake2/blake2-impl.h \
		   algos/rinhash/argon2/blake2/blamka-round-ref.h
algos/rinhash/argon2/ccminer-core.$(OBJEXT) algos/rinhash/argon2/ccminer-ref.$(OBJEXT) \
algos/rinhash/argon2/blake2/ccminer-blake2b.$(OBJEXT): CPPFLAGS += -I$(srcdir)/algos/rinhash/argon2
