//go:build 386 || arm || mips || mipsle

package internal

// ClampableToInt types that need clamping when converting to int (32-bit platform).
// On 32-bit platforms, int64 also needs clamping when converting to int.
type ClampableToInt interface {
	~float64 | ~int64 | ~float32
}
