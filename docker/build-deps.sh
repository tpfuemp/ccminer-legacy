#!/bin/sh
# Build static zlib, OpenSSL and libcurl (HTTP/HTTPS only) into PREFIX
# (default /opt/ccminer-deps) for `configure --with-static-deps=PREFIX`. Same
# versions as the Windows libraries in compat/ (see compat/BUILD-DEPS.txt).
# Tarballs already in the current directory are not downloaded again.
set -e

PREFIX=${1:-/opt/ccminer-deps}
JOBS=$(nproc)

# fetch <file> <url> <sha256>
fetch() {
	[ -f "$1" ] || wget -q -O "$1" "$2"
	echo "$3  $1" | sha256sum -c -
}

fetch zlib-1.3.1.tar.gz https://zlib.net/fossils/zlib-1.3.1.tar.gz \
	9a93b2b7dfdac77ceba5a558a580e74667dd6fede4585b91eefb60f03b72df23
fetch openssl-1.1.1w.tar.gz https://www.openssl.org/source/old/1.1.1/openssl-1.1.1w.tar.gz \
	cf3098950cb4d853ad95c0841f1f9c6d3dc102dccfcacd521d93925208b76ac8
fetch curl-8.11.1.tar.gz https://curl.se/download/curl-8.11.1.tar.gz \
	a889ac9dbba3644271bd9d1302b5c22a088893719b72be3487bc3d401e5c4e80

rm -rf zlib-1.3.1 openssl-1.1.1w curl-8.11.1

tar xzf zlib-1.3.1.tar.gz
(cd zlib-1.3.1 && CFLAGS="-O2 -fPIC" ./configure --static --prefix="$PREFIX" &&
	make -j "$JOBS" && make install)

# openssldir: the Debian/Ubuntu default, where the system CA store is
tar xzf openssl-1.1.1w.tar.gz
(cd openssl-1.1.1w && ./config no-shared no-tests no-engine no-dso \
	--prefix="$PREFIX" --openssldir=/usr/lib/ssl &&
	make -j "$JOBS" build_libs && make install_dev)

tar xzf curl-8.11.1.tar.gz
(cd curl-8.11.1 && PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig" ./configure \
	--prefix="$PREFIX" --disable-shared --enable-static \
	--with-openssl="$PREFIX" --with-zlib="$PREFIX" \
	--with-ca-bundle=/etc/ssl/certs/ca-certificates.crt --with-ca-fallback \
	--disable-ldap --disable-ldaps --disable-rtsp --disable-dict --disable-telnet \
	--disable-tftp --disable-pop3 --disable-imap --disable-smb --disable-smtp \
	--disable-gopher --disable-mqtt --disable-manual --disable-docs \
	--without-libpsl --without-libidn2 --without-nghttp2 --without-brotli \
	--without-zstd --without-librtmp --without-libssh2 --without-gssapi &&
	make -j "$JOBS" -C lib && make -C lib install && make -C include install &&
	make install-binSCRIPTS install-pkgconfigDATA)

rm -rf zlib-1.3.1 openssl-1.1.1w curl-8.11.1
