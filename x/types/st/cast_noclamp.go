package st

import . "github.com/itohio/EasyRobot/x/types/internal"

// CastFromComplex copies values from src (complex64) to dst (any NumericReal type).
// If dst is at least twice the size of src, copies real parts to the first half and imaginary parts to the second half.
// Otherwise, copies only the real parts.
// NOTE: No range checks or clamping are performed; values outside the range of T may overflow or lose precision.
func CastFromComplex[T NumberReal](dst []T, src []complex64) {
	srcLen := len(src)
	dstLen := len(dst)

	if dstLen >= 2*srcLen {
		// Copy real parts to first half, imaginary parts to second half
		for i := range srcLen {
			dst[i] = T(real(src[i]))
			dst[i+srcLen] = T(imag(src[i]))
		}
	} else {
		// Copy only real parts
		n := min(srcLen, dstLen)
		for i := range n {
			dst[i] = T(real(src[i]))
		}
	}
}

// CastToComplex copies values from src (any NumericReal type) to dst (complex64).
// If src is at least twice the size of dst, copies first half to real parts and second half to imaginary parts.
// Otherwise, copies only to real parts (imaginary parts set to zero).
// NOTE: No range checks or clamping are performed; values outside the range of float32 may overflow or lose precision.
func CastToComplex[T NumberReal](dst []complex64, src []T) {
	srcLen := len(src)
	dstLen := len(dst)

	if srcLen >= 2*dstLen {
		// Copy first half to real parts, second half to imaginary parts
		for i := range dstLen {
			dst[i] = complex(float32(src[i]), float32(src[i+dstLen]))
		}
	} else {
		// Copy only to real parts
		n := min(srcLen, dstLen)
		for i := range n {
			dst[i] = complex(float32(src[i]), 0)
		}
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
