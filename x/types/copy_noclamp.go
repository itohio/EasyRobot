package types

import (
	"github.com/itohio/EasyRobot/x/types/st"
)

// CastFromComplex copies the real part of each value from src (complex64) to dst (any NumericReal type).
// NOTE: No range checks or clamping are performed; values outside the range of T may overflow or lose precision.
func CastFromComplex[T NumberReal](dst []T, src []complex64) {
	st.CastFromComplex(dst, src)
}

// CastToComplex copies values from src (any NumericReal type) to dst (complex64), assigning to the real part
// and filling the imaginary part with zero.
// NOTE: No range checks or clamping are performed; values outside the range of float32 may overflow or lose precision.
func CastToComplex[T NumberReal](dst []complex64, src []T) {
	st.CastToComplex(dst, src)
}

// Cast copies values from src to dst, converting each value to type T.
// NOTE: No range checks or clamping are performed; values outside the range of T may overflow or lose precision.
func Cast[T NumberReal, U NumberReal](dst []T, src []U) {
	st.Cast(dst, src)
}
