//go:build !tinygo

package flip

// div is n/d truncated, for 1 ≤ d < divMax: the host's divider (see div.go).
func div(n, d int32) int32 { return n / d }
