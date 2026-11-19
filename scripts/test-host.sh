#!/bin/sh
# test-host.sh: go test every package on this host and fail only on a failure
# that is not in scripts/known-failures.txt.
#
# Packages that import gocv (OpenCV) are skipped, and listed, when
# `pkg-config --exists opencv4` fails. Every other package runs. A listed
# package that passes is reported so it can leave the list.
#
# Usage: scripts/test-host.sh [go test flags], e.g. scripts/test-host.sh -race
set -u
cd "$(dirname "$0")/.." || exit 2
list=scripts/known-failures.txt
mod=$(go list -m) || exit 2
tmp=$(mktemp -d) || exit 2
trap 'rm -rf "$tmp"' EXIT

# short drops the module path: x/devices/nmea, as in the list.
short() { sed "s#^$mod/##"; }

go list -e ./... | sort -u >"$tmp/all"
if pkg-config --exists opencv4 2>/dev/null; then
	: >"$tmp/skip"
else
	# A package is skipped when it or one of its test packages depends on gocv.
	go list -e -test -f '{{.ImportPath}} {{join .Deps " "}}' ./... |
		awk '/ gocv\.io\/x\/gocv( |$)/ { p = $1; sub(/(_test|\.test)$/, "", p); print p }' |
		sort -u | comm -12 - "$tmp/all" >"$tmp/skip"
fi
comm -23 "$tmp/all" "$tmp/skip" >"$tmp/run"

if [ -s "$tmp/skip" ]; then
	echo "test-host: skipped, OpenCV 4 not found by pkg-config:"
	short <"$tmp/skip" | sed 's/^/  /'
fi

# shellcheck disable=SC2046 # one argument per package
go test -count=1 "$@" $(cat "$tmp/run") >"$tmp/log" 2>&1
cat "$tmp/log"

awk '$1 == "FAIL" && NF >= 2 { print $2 }' "$tmp/log" | short | sort -u >"$tmp/failed"
short <"$tmp/run" | sort -u >"$tmp/ran"
grep -v '^[[:space:]]*\(#\|$\)' "$list" | awk '{ print $1 }' | sort -u >"$tmp/known"
grep -v '^[[:space:]]*\(#\|$\)' "$list" | awk '$2 == "may-fail" { print $1 }' | sort -u >"$tmp/mayfail"

comm -23 "$tmp/failed" "$tmp/known" >"$tmp/new"
comm -13 "$tmp/failed" "$tmp/known" | comm -23 - "$tmp/mayfail" | comm -12 - "$tmp/ran" >"$tmp/fixed"

echo
echo "test-host: $(wc -l <"$tmp/ran" | tr -d ' ') packages run, $(wc -l <"$tmp/failed" | tr -d ' ') failed, $(comm -12 "$tmp/failed" "$tmp/known" | wc -l | tr -d ' ') of them listed in $list"
if [ -s "$tmp/fixed" ]; then
	echo "test-host: listed but passing now; remove from $list:"
	sed 's/^/  /' "$tmp/fixed"
fi
if [ -s "$tmp/new" ]; then
	echo "test-host: NEW failures, not in $list:"
	sed 's/^/  /' "$tmp/new"
	exit 1
fi
echo "test-host: no new failures"
