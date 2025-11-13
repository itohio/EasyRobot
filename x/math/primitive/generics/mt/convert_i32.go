//go:build 386 || arm || mips || mipsle

package mt

import (
	. "github.com/itohio/EasyRobot/pkg/core/math/primitive/generics/helpers"

	"math"
)

// clampableToInt is now in helpers package as ClampableToInt

// elemConvertToInt handles conversions to int with clamping.
// Handles: float32/float64/int64 -> int (needs clamping)
func elemConvertToInt[U Numeric](dst []int, src []U) {
	n := len(dst)
	if len(src) < n {
		n = len(src)
	}
	if n > 0 {
		switch s := any(src).(type) {
		case []float32:
			clampToInt(dst, s, n)
		case []float64:
			clampToInt(dst, s, n)
		case []int64:
			clampToInt(dst, s, n)
		default:
			// Up-conversion: direct conversion
			for i := 0; i < n; i++ {
				dst[i] = int(src[i])
			}
		}
	}
}

// clampToInt implements the hot path inner loop for clamping to int range.
// Generic over source types that need clamping (float32, float64, int64).
func clampToInt[U ClampableToInt](dst []int, src []U, n int) {
	if n == 0 {
		return
	}
	_ = dst[n-1]
	_ = src[n-1]
	for i := 0; i < n; i++ {
		val := src[i]
		// Direct assignment to avoid type conversion issues and improve performance
		if val > U(math.MaxInt) {
			dst[i] = math.MaxInt
		} else if val < U(math.MinInt) {
			dst[i] = math.MinInt
		} else {
			dst[i] = int(val)
		}
	}
}

// elemConvertToIntStrided handles conversions to int with clamping.
func elemConvertToIntStrided[U Numeric](dst []int, src []U, shape []int, srcStrides, dstStrides []int) {
	switch s := any(src).(type) {
	case []float32:
		clampToIntStrided(dst, s, shape, srcStrides, dstStrides)
	case []float64:
		clampToIntStrided(dst, s, shape, srcStrides, dstStrides)
	case []int64:
		clampToIntStrided(dst, s, shape, srcStrides, dstStrides)
	}
}

// clampToIntStrided implements the hot path for strided copying with clamping to int.
// Generic over source types that need clamping (float32, float64, int64).
func clampToIntStrided[U ClampableToInt](dst []int, src []U, shape []int, srcStrides, dstStrides []int) {
	ndims := len(shape)
	if ndims == 0 {
		return
	}

	// Use stack-allocated arrays and AdvanceOffsets pattern
	var indicesStatic [MAX_DIMS]int
	var offsetsStatic [2]int
	indices := indicesStatic[:ndims]
	offsets := offsetsStatic[:2]
	for {
		// Hot path: use AdvanceOffsets pattern
		val := src[offsets[1]]
		if val > U(math.MaxInt) {
			dst[offsets[0]] = math.MaxInt
		} else if val < U(math.MinInt) {
			dst[offsets[0]] = math.MinInt
		} else {
			dst[offsets[0]] = int(val)
		}
		if !AdvanceOffsets(shape, indices, offsets, dstStrides, srcStrides) {
			break
		}
	}
}

// clampToIntValue now uses helpers.ClampToIntValue
