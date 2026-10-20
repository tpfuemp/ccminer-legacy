#!/bin/sh
# Generate configure and Makefile.in (needs autoconf, automake and libcurl.m4).
set -e
cd "$(dirname "$0")"
aclocal
autoheader
automake --add-missing --copy --foreign
autoconf
