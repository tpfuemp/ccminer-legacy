#!/bin/sh
# Check that the Windows (ccminer.props) and Linux (Makefile.am + algo.mk)
# builds list the same sources and per-file nvcc register limits.
#
# Per-file register limit, same value on both sides:
#   ccminer.props:  <CudaCompile Include="algos\x\foo.cu">
#                     <MaxRegCount>64</MaxRegCount>
#                   </CudaCompile>
#   algo.mk:        algos/x/foo.$(OBJEXT): nvcc_FLAGS += --maxrregcount=64
#
# Exit status 0 when in sync, 1 otherwise.
set -e
cd "$(dirname "$0")"
LC_ALL=C; export LC_ALL
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
fail=0

# report <message> <file with one entry per line>
report() {
	[ -s "$2" ] || return 0
	echo "$1"; sed 's/^/  /' "$2"; fail=1
}

# sources listed in an automake variable: join continuation lines, keep the
# assignments to $2, print one path per line
am_sources() {
	sed -e ':a' -e '/\\$/N; s/\\\n//; ta' "$1" |
	sed -n "s/^[ 	]*$2[ 	]*+*=//p" | tr ' \t' '\n\n' | sed '/^$/d'
}

# --- sources ---------------------------------------------------------------
sed -n 's/.*<\(ClCompile\|CudaCompile\|ClInclude\|None\) Include="\([^"]*\)".*/\2/p' \
	ccminer.props | tr '\\' '/' | grep -v '^compat/' | sort -u > "$tmp/props"
{
	am_sources Makefile.am ccminer_SOURCES
	for mk in algos/*/algo.mk; do am_sources "$mk" ccminer_SOURCES; done
} | sort -u > "$tmp/make"
find core cuda algos -type f \( -name '*.c' -o -name '*.cpp' -o -name '*.cu' \
	-o -name '*.h' -o -name '*.hpp' -o -name '*.cuh' \) | sort -u > "$tmp/disk"

comm -23 "$tmp/props" "$tmp/make" > "$tmp/x"
report "in ccminer.props, missing in Makefile.am/algo.mk:" "$tmp/x"
comm -13 "$tmp/props" "$tmp/make" > "$tmp/x"
report "in Makefile.am/algo.mk, missing in ccminer.props:" "$tmp/x"
sort -u "$tmp/props" "$tmp/make" > "$tmp/listed"
comm -23 "$tmp/disk" "$tmp/listed" > "$tmp/x"
report "on disk, listed in neither build:" "$tmp/x"
comm -13 "$tmp/disk" "$tmp/listed" > "$tmp/x"
report "listed, missing on disk:" "$tmp/x"

# bundled jansson (Windows builds it in the main project)
sed -n 's/.*<ClCompile Include="compat\\jansson\\\([^"]*\.c\)".*/\1/p' \
	ccminer.props | sort -u > "$tmp/props"
am_sources compat/jansson/Makefile.am libjansson_a_SOURCES |
	grep '\.c$' | sort -u > "$tmp/make"
comm -3 "$tmp/props" "$tmp/make" > "$tmp/x"
report "jansson sources differ (props left, compat/jansson/Makefile.am right):" "$tmp/x"

# every algo.mk is included, the vcxproj files carry no own items
for mk in algos/*/algo.mk; do
	grep -q "^include \$(srcdir)/$mk\$" Makefile.am || echo "$mk"
done > "$tmp/x"
report "algo.mk not included from Makefile.am:" "$tmp/x"
grep -l 'Include="[^"]*\.\(c\|cpp\|cu\|h\|cuh\|hpp\)"' *.vcxproj > "$tmp/x" || true
report "sources belong in ccminer.props, not in:" "$tmp/x"

# --- register limits -------------------------------------------------------
awk '
	/<CudaCompile Include="/ {
		f = $0; sub(/.*Include="/, "", f); sub(/".*/, "", f); gsub(/\\/, "/", f)
	}
	/<MaxRegCount>/ && f != "" {
		n = $0; sub(/.*<MaxRegCount>/, "", n); sub(/<.*/, "", n); print f, n
	}
	/<\/CudaCompile>|\/>/ { f = "" }
' ccminer.props | sort > "$tmp/props"
cat algos/*/algo.mk | sed -n \
	's/^\([^ :]*\)\.\$(OBJEXT)[ 	]*:.*--maxrregcount[= ]*\([0-9]*\).*/\1.cu \2/p' |
	sort > "$tmp/make"
comm -3 "$tmp/props" "$tmp/make" > "$tmp/x"
report "maxrregcount differs (props left, algo.mk right):" "$tmp/x"

[ $fail = 0 ] && echo "build lists in sync"
exit $fail
