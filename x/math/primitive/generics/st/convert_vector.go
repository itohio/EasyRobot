package st

import (
	. "github.com/itohio/EasyRobot/x/math/primitive/generics/helpers"
)

// ElemVecConvertStrided converts src into dst for a vector with stride support.
// Optimized for 1D vector operations.
// n is the vector length, strideDst and strideSrc are the strides.
func ElemVecConvertStrided[T, U Numeric](dst []T, src []U, n int, strideDst, strideSrc int) {
	if n == 0 {
		return
	}

	if strideDst == 1 && strideSrc == 1 {
		// Fast path: contiguous vectors - use elemConvertNumeric directly
		_ = elemConvertNumeric(dst[:n], src[:n])
		return
	}

	// Strided path: convert each element individually
	// Handle complex64 cases explicitly to avoid type conversion issues
	switch d := any(dst).(type) {
	case []complex64:
		switch s := any(src).(type) {
		case []complex64:
			// Same type - copy
			dIdx := 0
			sIdx := 0
			for i := 0; i < n; i++ {
				d[dIdx] = s[sIdx]
				dIdx += strideDst
				sIdx += strideSrc
			}
			return
		case []float64:
			dIdx := 0
			sIdx := 0
			for i := 0; i < n; i++ {
				FromTToComplex(d[dIdx:dIdx+1], s[sIdx:sIdx+1])
				dIdx += strideDst
				sIdx += strideSrc
			}
			return
		case []float32:
			dIdx := 0
			sIdx := 0
			for i := 0; i < n; i++ {
				FromTToComplex(d[dIdx:dIdx+1], s[sIdx:sIdx+1])
				dIdx += strideDst
				sIdx += strideSrc
			}
			return
		case []int64:
			dIdx := 0
			sIdx := 0
			for i := 0; i < n; i++ {
				FromTToComplex(d[dIdx:dIdx+1], s[sIdx:sIdx+1])
				dIdx += strideDst
				sIdx += strideSrc
			}
			return
		case []int:
			dIdx := 0
			sIdx := 0
			for i := 0; i < n; i++ {
				FromTToComplex(d[dIdx:dIdx+1], s[sIdx:sIdx+1])
				dIdx += strideDst
				sIdx += strideSrc
			}
			return
		case []int32:
			dIdx := 0
			sIdx := 0
			for i := 0; i < n; i++ {
				FromTToComplex(d[dIdx:dIdx+1], s[sIdx:sIdx+1])
				dIdx += strideDst
				sIdx += strideSrc
			}
			return
		case []int16:
			dIdx := 0
			sIdx := 0
			for i := 0; i < n; i++ {
				FromTToComplex(d[dIdx:dIdx+1], s[sIdx:sIdx+1])
				dIdx += strideDst
				sIdx += strideSrc
			}
			return
		case []int8:
			dIdx := 0
			sIdx := 0
			for i := 0; i < n; i++ {
				FromTToComplex(d[dIdx:dIdx+1], s[sIdx:sIdx+1])
				dIdx += strideDst
				sIdx += strideSrc
			}
			return
		}
	case []float64:
		switch s := any(src).(type) {
		case []complex64:
			dIdx := 0
			sIdx := 0
			for i := 0; i < n; i++ {
				FromComplexToT(d[dIdx:dIdx+1], s[sIdx:sIdx+1])
				dIdx += strideDst
				sIdx += strideSrc
			}
			return
		}
	case []float32:
		switch s := any(src).(type) {
		case []complex64:
			dIdx := 0
			sIdx := 0
			for i := 0; i < n; i++ {
				FromComplexToT(d[dIdx:dIdx+1], s[sIdx:sIdx+1])
				dIdx += strideDst
				sIdx += strideSrc
			}
			return
		}
	case []int64:
		switch s := any(src).(type) {
		case []complex64:
			dIdx := 0
			sIdx := 0
			for i := 0; i < n; i++ {
				FromComplexToT(d[dIdx:dIdx+1], s[sIdx:sIdx+1])
				dIdx += strideDst
				sIdx += strideSrc
			}
			return
		}
	case []int:
		switch s := any(src).(type) {
		case []complex64:
			dIdx := 0
			sIdx := 0
			for i := 0; i < n; i++ {
				FromComplexToT(d[dIdx:dIdx+1], s[sIdx:sIdx+1])
				dIdx += strideDst
				sIdx += strideSrc
			}
			return
		}
	case []int32:
		switch s := any(src).(type) {
		case []complex64:
			dIdx := 0
			sIdx := 0
			for i := 0; i < n; i++ {
				FromComplexToT(d[dIdx:dIdx+1], s[sIdx:sIdx+1])
				dIdx += strideDst
				sIdx += strideSrc
			}
			return
		}
	case []int16:
		switch s := any(src).(type) {
		case []complex64:
			dIdx := 0
			sIdx := 0
			for i := 0; i < n; i++ {
				FromComplexToT(d[dIdx:dIdx+1], s[sIdx:sIdx+1])
				dIdx += strideDst
				sIdx += strideSrc
			}
			return
		}
	case []int8:
		switch s := any(src).(type) {
		case []complex64:
			dIdx := 0
			sIdx := 0
			for i := 0; i < n; i++ {
				FromComplexToT(d[dIdx:dIdx+1], s[sIdx:sIdx+1])
				dIdx += strideDst
				sIdx += strideSrc
			}
			return
		}
	}

	// NumericReal to NumericReal - direct conversion
	// Handle each destination type explicitly to avoid generic constraint issues
	switch d := any(dst).(type) {
	case []float64:
		switch s := any(src).(type) {
		case []float64:
			dIdx, sIdx := 0, 0
			for i := 0; i < n; i++ {
				d[dIdx] = float64(s[sIdx])
				dIdx += strideDst
				sIdx += strideSrc
			}
			return
		case []float32:
			dIdx, sIdx := 0, 0
			for i := 0; i < n; i++ {
				d[dIdx] = float64(s[sIdx])
				dIdx += strideDst
				sIdx += strideSrc
			}
			return
		case []int64:
			dIdx, sIdx := 0, 0
			for i := 0; i < n; i++ {
				d[dIdx] = float64(s[sIdx])
				dIdx += strideDst
				sIdx += strideSrc
			}
			return
		case []int:
			dIdx, sIdx := 0, 0
			for i := 0; i < n; i++ {
				d[dIdx] = float64(s[sIdx])
				dIdx += strideDst
				sIdx += strideSrc
			}
			return
		case []int32:
			dIdx, sIdx := 0, 0
			for i := 0; i < n; i++ {
				d[dIdx] = float64(s[sIdx])
				dIdx += strideDst
				sIdx += strideSrc
			}
			return
		case []int16:
			dIdx, sIdx := 0, 0
			for i := 0; i < n; i++ {
				d[dIdx] = float64(s[sIdx])
				dIdx += strideDst
				sIdx += strideSrc
			}
			return
		case []int8:
			dIdx, sIdx := 0, 0
			for i := 0; i < n; i++ {
				d[dIdx] = float64(s[sIdx])
				dIdx += strideDst
				sIdx += strideSrc
			}
			return
		}
	case []float32:
		switch s := any(src).(type) {
		case []float64:
			dIdx, sIdx := 0, 0
			for i := 0; i < n; i++ {
				d[dIdx] = float32(s[sIdx])
				dIdx += strideDst
				sIdx += strideSrc
			}
			return
		case []float32:
			dIdx, sIdx := 0, 0
			for i := 0; i < n; i++ {
				d[dIdx] = float32(s[sIdx])
				dIdx += strideDst
				sIdx += strideSrc
			}
			return
		case []int64:
			dIdx, sIdx := 0, 0
			for i := 0; i < n; i++ {
				d[dIdx] = float32(s[sIdx])
				dIdx += strideDst
				sIdx += strideSrc
			}
			return
		case []int:
			dIdx, sIdx := 0, 0
			for i := 0; i < n; i++ {
				d[dIdx] = float32(s[sIdx])
				dIdx += strideDst
				sIdx += strideSrc
			}
			return
		case []int32:
			dIdx, sIdx := 0, 0
			for i := 0; i < n; i++ {
				d[dIdx] = float32(s[sIdx])
				dIdx += strideDst
				sIdx += strideSrc
			}
			return
		case []int16:
			dIdx, sIdx := 0, 0
			for i := 0; i < n; i++ {
				d[dIdx] = float32(s[sIdx])
				dIdx += strideDst
				sIdx += strideSrc
			}
			return
		case []int8:
			dIdx, sIdx := 0, 0
			for i := 0; i < n; i++ {
				d[dIdx] = float32(s[sIdx])
				dIdx += strideDst
				sIdx += strideSrc
			}
			return
		}
	case []int64:
		switch s := any(src).(type) {
		case []float64:
			dIdx, sIdx := 0, 0
			for i := 0; i < n; i++ {
				d[dIdx] = int64(s[sIdx])
				dIdx += strideDst
				sIdx += strideSrc
			}
			return
		case []float32:
			dIdx, sIdx := 0, 0
			for i := 0; i < n; i++ {
				d[dIdx] = int64(s[sIdx])
				dIdx += strideDst
				sIdx += strideSrc
			}
			return
		case []int64:
			dIdx, sIdx := 0, 0
			for i := 0; i < n; i++ {
				d[dIdx] = int64(s[sIdx])
				dIdx += strideDst
				sIdx += strideSrc
			}
			return
		case []int:
			dIdx, sIdx := 0, 0
			for i := 0; i < n; i++ {
				d[dIdx] = int64(s[sIdx])
				dIdx += strideDst
				sIdx += strideSrc
			}
			return
		case []int32:
			dIdx, sIdx := 0, 0
			for i := 0; i < n; i++ {
				d[dIdx] = int64(s[sIdx])
				dIdx += strideDst
				sIdx += strideSrc
			}
			return
		case []int16:
			dIdx, sIdx := 0, 0
			for i := 0; i < n; i++ {
				d[dIdx] = int64(s[sIdx])
				dIdx += strideDst
				sIdx += strideSrc
			}
			return
		case []int8:
			dIdx, sIdx := 0, 0
			for i := 0; i < n; i++ {
				d[dIdx] = int64(s[sIdx])
				dIdx += strideDst
				sIdx += strideSrc
			}
			return
		}
	case []int:
		switch s := any(src).(type) {
		case []float64:
			dIdx, sIdx := 0, 0
			for i := 0; i < n; i++ {
				d[dIdx] = int(s[sIdx])
				dIdx += strideDst
				sIdx += strideSrc
			}
			return
		case []float32:
			dIdx, sIdx := 0, 0
			for i := 0; i < n; i++ {
				d[dIdx] = int(s[sIdx])
				dIdx += strideDst
				sIdx += strideSrc
			}
			return
		case []int64:
			dIdx, sIdx := 0, 0
			for i := 0; i < n; i++ {
				d[dIdx] = int(s[sIdx])
				dIdx += strideDst
				sIdx += strideSrc
			}
			return
		case []int:
			dIdx, sIdx := 0, 0
			for i := 0; i < n; i++ {
				d[dIdx] = int(s[sIdx])
				dIdx += strideDst
				sIdx += strideSrc
			}
			return
		case []int32:
			dIdx, sIdx := 0, 0
			for i := 0; i < n; i++ {
				d[dIdx] = int(s[sIdx])
				dIdx += strideDst
				sIdx += strideSrc
			}
			return
		case []int16:
			dIdx, sIdx := 0, 0
			for i := 0; i < n; i++ {
				d[dIdx] = int(s[sIdx])
				dIdx += strideDst
				sIdx += strideSrc
			}
			return
		case []int8:
			dIdx, sIdx := 0, 0
			for i := 0; i < n; i++ {
				d[dIdx] = int(s[sIdx])
				dIdx += strideDst
				sIdx += strideSrc
			}
			return
		}
	case []int32:
		switch s := any(src).(type) {
		case []float64:
			dIdx, sIdx := 0, 0
			for i := 0; i < n; i++ {
				d[dIdx] = int32(s[sIdx])
				dIdx += strideDst
				sIdx += strideSrc
			}
			return
		case []float32:
			dIdx, sIdx := 0, 0
			for i := 0; i < n; i++ {
				d[dIdx] = int32(s[sIdx])
				dIdx += strideDst
				sIdx += strideSrc
			}
			return
		case []int64:
			dIdx, sIdx := 0, 0
			for i := 0; i < n; i++ {
				d[dIdx] = int32(s[sIdx])
				dIdx += strideDst
				sIdx += strideSrc
			}
			return
		case []int:
			dIdx, sIdx := 0, 0
			for i := 0; i < n; i++ {
				d[dIdx] = int32(s[sIdx])
				dIdx += strideDst
				sIdx += strideSrc
			}
			return
		case []int32:
			dIdx, sIdx := 0, 0
			for i := 0; i < n; i++ {
				d[dIdx] = int32(s[sIdx])
				dIdx += strideDst
				sIdx += strideSrc
			}
			return
		case []int16:
			dIdx, sIdx := 0, 0
			for i := 0; i < n; i++ {
				d[dIdx] = int32(s[sIdx])
				dIdx += strideDst
				sIdx += strideSrc
			}
			return
		case []int8:
			dIdx, sIdx := 0, 0
			for i := 0; i < n; i++ {
				d[dIdx] = int32(s[sIdx])
				dIdx += strideDst
				sIdx += strideSrc
			}
			return
		}
	case []int16:
		switch s := any(src).(type) {
		case []float64:
			dIdx, sIdx := 0, 0
			for i := 0; i < n; i++ {
				d[dIdx] = int16(s[sIdx])
				dIdx += strideDst
				sIdx += strideSrc
			}
			return
		case []float32:
			dIdx, sIdx := 0, 0
			for i := 0; i < n; i++ {
				d[dIdx] = int16(s[sIdx])
				dIdx += strideDst
				sIdx += strideSrc
			}
			return
		case []int64:
			dIdx, sIdx := 0, 0
			for i := 0; i < n; i++ {
				d[dIdx] = int16(s[sIdx])
				dIdx += strideDst
				sIdx += strideSrc
			}
			return
		case []int:
			dIdx, sIdx := 0, 0
			for i := 0; i < n; i++ {
				d[dIdx] = int16(s[sIdx])
				dIdx += strideDst
				sIdx += strideSrc
			}
			return
		case []int32:
			dIdx, sIdx := 0, 0
			for i := 0; i < n; i++ {
				d[dIdx] = int16(s[sIdx])
				dIdx += strideDst
				sIdx += strideSrc
			}
			return
		case []int16:
			dIdx, sIdx := 0, 0
			for i := 0; i < n; i++ {
				d[dIdx] = int16(s[sIdx])
				dIdx += strideDst
				sIdx += strideSrc
			}
			return
		case []int8:
			dIdx, sIdx := 0, 0
			for i := 0; i < n; i++ {
				d[dIdx] = int16(s[sIdx])
				dIdx += strideDst
				sIdx += strideSrc
			}
			return
		}
	case []int8:
		switch s := any(src).(type) {
		case []float64:
			dIdx, sIdx := 0, 0
			for i := 0; i < n; i++ {
				d[dIdx] = int8(s[sIdx])
				dIdx += strideDst
				sIdx += strideSrc
			}
			return
		case []float32:
			dIdx, sIdx := 0, 0
			for i := 0; i < n; i++ {
				d[dIdx] = int8(s[sIdx])
				dIdx += strideDst
				sIdx += strideSrc
			}
			return
		case []int64:
			dIdx, sIdx := 0, 0
			for i := 0; i < n; i++ {
				d[dIdx] = int8(s[sIdx])
				dIdx += strideDst
				sIdx += strideSrc
			}
			return
		case []int:
			dIdx, sIdx := 0, 0
			for i := 0; i < n; i++ {
				d[dIdx] = int8(s[sIdx])
				dIdx += strideDst
				sIdx += strideSrc
			}
			return
		case []int32:
			dIdx, sIdx := 0, 0
			for i := 0; i < n; i++ {
				d[dIdx] = int8(s[sIdx])
				dIdx += strideDst
				sIdx += strideSrc
			}
			return
		case []int16:
			dIdx, sIdx := 0, 0
			for i := 0; i < n; i++ {
				d[dIdx] = int8(s[sIdx])
				dIdx += strideDst
				sIdx += strideSrc
			}
			return
		case []int8:
			dIdx, sIdx := 0, 0
			for i := 0; i < n; i++ {
				d[dIdx] = int8(s[sIdx])
				dIdx += strideDst
				sIdx += strideSrc
			}
			return
		}
	}
}

// ElemMatConvertStrided converts src into dst for a matrix with leading dimension support.
// Optimized for 2D matrix operations.
// rows and cols are the matrix dimensions, ldDst and ldSrc are leading dimensions (typically cols for row-major).
func ElemMatConvertStrided[T, U Numeric](dst []T, src []U, rows, cols int, ldDst, ldSrc int) {
	if rows == 0 || cols == 0 {
		return
	}

	if ldDst == cols && ldSrc == cols {
		// Fast path: contiguous matrices - use elemConvertNumeric directly
		size := rows * cols
		_ = elemConvertNumeric(dst[:size], src[:size])
		return
	}

	// Strided path: iterate row by row, convert each element
	for i := 0; i < rows; i++ {
		dstRow := dst[i*ldDst:]
		srcRow := src[i*ldSrc:]
		// Convert row using elemConvertNumeric
		_ = elemConvertNumeric(dstRow[:cols], srcRow[:cols])
	}
}
