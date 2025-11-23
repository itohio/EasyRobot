package st

import . "github.com/itohio/EasyRobot/x/types/internal"

// CastFromComplex copies the real part of each value from src (complex64) to dst (any NumericReal type).
// NOTE: No range checks or clamping are performed; values outside the range of T may overflow or lose precision.
func CastFromComplex[T NumberReal](dst []T, src []complex64) {
	n := min(len(src), len(dst))
	for i := range n {
		dst[i] = T(real(src[i]))
	}
}

// CastToComplex copies values from src (any NumericReal type) to dst (complex64), assigning to the real part
// and filling the imaginary part with zero.
// NOTE: No range checks or clamping are performed; values outside the range of float32 may overflow or lose precision.
func CastToComplex[T NumberReal](dst []complex64, src []T) {
	n := min(len(src), len(dst))
	for i := range n {
		dst[i] = complex(float32(src[i]), 0)
	}
}

// Cast copies values from src to dst, converting each value to type T.
// NOTE: No range checks or clamping are performed; values outside the range of T may overflow or lose precision.
func Cast[T NumberReal, U NumberReal](dst []T, src []U) {
	n := min(len(src), len(dst))
	for i := range n {
		dst[i] = T(src[i])
	}
}
