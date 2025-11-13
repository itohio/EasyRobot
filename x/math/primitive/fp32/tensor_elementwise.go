package fp32

import (
	"math"

	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/pkg/core/math/primitive/generics"
	helpers "github.com/itohio/EasyRobot/pkg/core/math/primitive/generics/helpers"
)

// ElemAdd writes element-wise sum of a and b into dst for the provided shape/strides.
func ElemAdd(dst, a, b []float32, shape []int, stridesDst, stridesA, stridesB []int) {
	generics.ElemApplyBinaryStrided[float32](dst, a, b, shape, stridesDst, stridesA, stridesB, func(av, bv float32) float32 {
		return av + bv
	})
}

// ElemSub writes element-wise difference of a and b into dst (dst = a - b).
func ElemSub(dst, a, b []float32, shape []int, stridesDst, stridesA, stridesB []int) {
	generics.ElemApplyBinaryStrided[float32](dst, a, b, shape, stridesDst, stridesA, stridesB, func(av, bv float32) float32 {
		return av - bv
	})
}

// ElemMul writes element-wise product of a and b into dst.
func ElemMul(dst, a, b []float32, shape []int, stridesDst, stridesA, stridesB []int) {
	generics.ElemApplyBinaryStrided[float32](dst, a, b, shape, stridesDst, stridesA, stridesB, func(av, bv float32) float32 {
		return av * bv
	})
}

// ElemDiv writes element-wise division of a by b into dst, skipping positions where b == 0.
// When the divisor is zero, the destination retains its previous value to match existing tensor semantics.
func ElemDiv(dst, a, b []float32, shape []int, stridesDst, stridesA, stridesB []int) {
	size := SizeFromShape(shape)
	if len(shape) == 0 || size == 0 {
		return
	}

	stridesDst = EnsureStrides(stridesDst, shape)
	stridesA = EnsureStrides(stridesA, shape)
	stridesB = EnsureStrides(stridesB, shape)

	if IsContiguous(stridesDst, shape) && IsContiguous(stridesA, shape) && IsContiguous(stridesB, shape) {
		for i := 0; i < size; i++ {
			bv := b[i]
			if bv != 0 {
				dst[i] = a[i] / bv
			}
		}
		return
	}

	strideSet := [][]int{stridesDst, stridesA, stridesB}
	process := func(indices []int, offsets []int) {
		for {
			dIdx := offsets[0]
			aIdx := offsets[1]
			bIdx := offsets[2]
			bv := b[bIdx]
			if bv != 0 {
				dst[dIdx] = a[aIdx] / bv
			}
			if !advanceOffsets(shape, indices, offsets, strideSet) {
				break
			}
		}
	}

	if len(shape) <= helpers.MAX_DIMS {
		var indicesArr [helpers.MAX_DIMS]int
		var offsetsArr [3]int
		process(indicesArr[:len(shape)], offsetsArr[:len(strideSet)])
		return
	}

	var offsetsArr [3]int
	process(make([]int, len(shape)), offsetsArr[:len(strideSet)])
}

// ElemScaleInPlace multiplies dst by the given scalar (in-place) for the provided shape/strides.
func ElemScaleInPlace(dst []float32, scalar float32, shape []int, stridesDst []int) {
	if scalar == 1.0 {
		return
	}

	stridesDst = EnsureStrides(stridesDst, shape)
	size := SizeFromShape(shape)
	if size == 0 {
		return
	}
	if IsContiguous(stridesDst, shape) {
		Scal(dst, 1, size, scalar)
		return
	}

	strideSet := [][]int{stridesDst}
	process := func(indices []int, offsets []int) {
		for {
			dIdx := offsets[0]
			dst[dIdx] *= scalar
			if !advanceOffsets(shape, indices, offsets, strideSet) {
				break
			}
		}
	}

	if len(shape) <= helpers.MAX_DIMS {
		var indicesArr [helpers.MAX_DIMS]int
		var offsetsArr [1]int
		process(indicesArr[:len(shape)], offsetsArr[:len(strideSet)])
		return
	}

	var offsetsArr [1]int
	process(make([]int, len(shape)), offsetsArr[:len(strideSet)])
}

// ElemCopy copies src into dst respecting the supplied shape/strides.
//
// Deprecated: Use generics.ElemCopyStrided[float32] directly instead.
// This function is a thin wrapper that just passes all parameters to generics.
func ElemCopy(dst, src []float32, shape []int, stridesDst, stridesSrc []int) {
	generics.ElemCopyStrided[float32](dst, src, shape, stridesDst, stridesSrc)
}

// ElemWhere writes elements from a where condition is true, otherwise from b.
// condition, a, b must have compatible shapes/strides.
//
// Deprecated: Use generics.ElemWhere[float32] directly instead.
// This function is a thin wrapper that just passes all parameters to generics.
func ElemWhere(dst, condition, a, b []float32, shape []int, stridesDst, stridesCond, stridesA, stridesB []int) {
	generics.ElemWhere[float32](dst, condition, a, b, shape, stridesDst, stridesCond, stridesA, stridesB)
}

// ElemGreaterThan writes 1.0 where a > b, 0.0 otherwise.
//
// Deprecated: Use generics.ElemGreaterThanStrided[float32] directly instead.
// This function is a thin wrapper that just passes all parameters to generics.
func ElemGreaterThan(dst, a, b []float32, shape []int, stridesDst, stridesA, stridesB []int) {
	generics.ElemGreaterThanStrided[float32](dst, a, b, shape, stridesDst, stridesA, stridesB)
}

// ElemEqual writes 1.0 where a == b, 0.0 otherwise.
//
// Deprecated: Use generics.ElemEqualStrided[float32] directly instead.
// This function is a thin wrapper that just passes all parameters to generics.
func ElemEqual(dst, a, b []float32, shape []int, stridesDst, stridesA, stridesB []int) {
	generics.ElemEqualStrided[float32](dst, a, b, shape, stridesDst, stridesA, stridesB)
}

// ElemLess writes 1.0 where a < b, 0.0 otherwise.
//
// Deprecated: Use generics.ElemLessStrided[float32] directly instead.
// This function is a thin wrapper that just passes all parameters to generics.
func ElemLess(dst, a, b []float32, shape []int, stridesDst, stridesA, stridesB []int) {
	generics.ElemLessStrided[float32](dst, a, b, shape, stridesDst, stridesA, stridesB)
}

// ElemSquare writes element-wise square of src into dst: dst[i] = src[i]^2
func ElemSquare(dst, src []float32, shape []int, stridesDst, stridesSrc []int) {
	generics.ElemApplyUnaryStrided[float32](dst, src, shape, stridesDst, stridesSrc, func(v float32) float32 {
		return v * v
	})
}

// ElemSqrt writes element-wise square root of src into dst: dst[i] = sqrt(src[i])
func ElemSqrt(dst, src []float32, shape []int, stridesDst, stridesSrc []int) {
	generics.ElemApplyUnaryStrided[float32](dst, src, shape, stridesDst, stridesSrc, func(v float32) float32 {
		if v < 0 {
			return 0.0 // Handle negative values gracefully
		}
		return math32.Sqrt(v)
	})
}

// ElemExp writes element-wise exponential of src into dst: dst[i] = exp(src[i])
func ElemExp(dst, src []float32, shape []int, stridesDst, stridesSrc []int) {
	generics.ElemApplyUnaryStrided[float32](dst, src, shape, stridesDst, stridesSrc, func(v float32) float32 {
		if v > 88.0 { // Prevent overflow
			return float32(math.Inf(1))
		}
		if v < -88.0 {
			return 0.0
		}
		return math32.Exp(v)
	})
}

// ElemLog writes element-wise natural logarithm of src into dst: dst[i] = log(src[i])
func ElemLog(dst, src []float32, shape []int, stridesDst, stridesSrc []int) {
	generics.ElemApplyUnaryStrided[float32](dst, src, shape, stridesDst, stridesSrc, func(v float32) float32 {
		if v <= 0 {
			return float32(math.Inf(-1)) // log(0) or negative = -Inf
		}
		return math32.Log(v)
	})
}

// ElemPow writes element-wise power of src^power into dst: dst[i] = src[i]^power
func ElemPow(dst, src []float32, power float32, shape []int, stridesDst, stridesSrc []int) {
	generics.ElemApplyUnaryScalarStrided[float32](dst, src, power, shape, stridesDst, stridesSrc, func(v, p float32) float32 {
		if v < 0 && p != float32(int32(p)) {
			// Negative base with non-integer exponent results in NaN
			return float32(math.NaN())
		}
		if v == 0 && p < 0 {
			return float32(math.Inf(1)) // 0^(-n) = Inf
		}
		return math32.Pow(v, p)
	})
}

// ElemAbs writes element-wise absolute value of src into dst: dst[i] = abs(src[i])
func ElemAbs(dst, src []float32, shape []int, stridesDst, stridesSrc []int) {
	generics.ElemApplyUnaryStrided[float32](dst, src, shape, stridesDst, stridesSrc, func(v float32) float32 {
		return math32.Abs(v)
	})
}

// ElemSign writes element-wise sign of src into dst: dst[i] = sign(src[i]) (-1, 0, or 1)
//
// Deprecated: Use generics.ElemSignStrided[float32] directly instead.
// This function is a thin wrapper that just passes all parameters to generics.
func ElemSign(dst, src []float32, shape []int, stridesDst, stridesSrc []int) {
	generics.ElemSignStrided[float32](dst, src, shape, stridesDst, stridesSrc)
}

// ElemCos writes element-wise cosine of src into dst: dst[i] = cos(src[i])
func ElemCos(dst, src []float32, shape []int, stridesDst, stridesSrc []int) {
	generics.ElemApplyUnaryStrided[float32](dst, src, shape, stridesDst, stridesSrc, func(v float32) float32 {
		return math32.Cos(v)
	})
}

// ElemSin writes element-wise sine of src into dst: dst[i] = sin(src[i])
func ElemSin(dst, src []float32, shape []int, stridesDst, stridesSrc []int) {
	generics.ElemApplyUnaryStrided[float32](dst, src, shape, stridesDst, stridesSrc, func(v float32) float32 {
		return math32.Sin(v)
	})
}

// ElemTanh writes element-wise hyperbolic tangent of src into dst: dst[i] = tanh(src[i])
func ElemTanh(dst, src []float32, shape []int, stridesDst, stridesSrc []int) {
	generics.ElemApplyUnaryStrided[float32](dst, src, shape, stridesDst, stridesSrc, func(v float32) float32 {
		return math32.Tanh(v)
	})
}

// ElemNegative writes element-wise negation of src into dst: dst[i] = -src[i]
//
// Deprecated: Use generics.ElemNegativeStrided[float32] directly instead.
// This function is a thin wrapper that just passes all parameters to generics.
func ElemNegative(dst, src []float32, shape []int, stridesDst, stridesSrc []int) {
	generics.ElemNegativeStrided[float32](dst, src, shape, stridesDst, stridesSrc)
}

// ElemFill writes constant value to dst: dst[i] = value
//
// Deprecated: Use generics.ElemFillStrided[float32] directly instead.
// This function is a thin wrapper that just passes all parameters to generics.
func ElemFill(dst []float32, value float32, shape []int, stridesDst []int) {
	generics.ElemFillStrided[float32](dst, value, shape, stridesDst)
}

// ElemAddScalar writes src + scalar to dst: dst[i] = src[i] + scalar
func ElemAddScalar(dst, src []float32, scalar float32, shape []int, stridesDst, stridesSrc []int) {
	generics.ElemApplyUnaryScalarStrided[float32](dst, src, scalar, shape, stridesDst, stridesSrc, func(v, s float32) float32 {
		return v + s
	})
}

// ElemSubScalar writes src - scalar to dst: dst[i] = src[i] - scalar
func ElemSubScalar(dst, src []float32, scalar float32, shape []int, stridesDst, stridesSrc []int) {
	generics.ElemApplyUnaryScalarStrided[float32](dst, src, scalar, shape, stridesDst, stridesSrc, func(v, s float32) float32 {
		return v - s
	})
}

// ElemScale multiplies each element of src by scalar and writes to dst: dst[i] = src[i] * scalar
func ElemScale(dst, src []float32, scalar float32, shape []int, stridesDst, stridesSrc []int) {
	generics.ElemApplyUnaryScalarStrided[float32](dst, src, scalar, shape, stridesDst, stridesSrc, func(v, s float32) float32 {
		return v * s
	})
}

// ElemDivScalar writes src / scalar to dst: dst[i] = src[i] / scalar
// When scalar is zero, the destination retains its previous value to match existing tensor semantics.
func ElemDivScalar(dst, src []float32, scalar float32, shape []int, stridesDst, stridesSrc []int) {
	size := SizeFromShape(shape)
	if len(shape) == 0 || size == 0 {
		return
	}

	if scalar == 0 {
		return // Skip division by zero
	}

	stridesDst = EnsureStrides(stridesDst, shape)
	stridesSrc = EnsureStrides(stridesSrc, shape)

	if IsContiguous(stridesDst, shape) && IsContiguous(stridesSrc, shape) {
		for i := 0; i < size; i++ {
			dst[i] = src[i] / scalar
		}
		return
	}

	strideSet := [][]int{stridesDst, stridesSrc}
	process := func(indices []int, offsets []int) {
		for {
			dIdx := offsets[0]
			sIdx := offsets[1]
			dst[dIdx] = src[sIdx] / scalar
			if !advanceOffsets(shape, indices, offsets, strideSet) {
				break
			}
		}
	}

	if len(shape) <= helpers.MAX_DIMS {
		var indicesArr [helpers.MAX_DIMS]int
		var offsetsArr [2]int
		process(indicesArr[:len(shape)], offsetsArr[:len(strideSet)])
		return
	}

	var offsetsArr [2]int
	process(make([]int, len(shape)), offsetsArr[:len(strideSet)])
}

// ElemNotEqual writes 1.0 where a != b, 0.0 otherwise
//
// Deprecated: Use generics.ElemNotEqualStrided[float32] directly instead.
// This function is a thin wrapper that just passes all parameters to generics.
func ElemNotEqual(dst, a, b []float32, shape []int, stridesDst, stridesA, stridesB []int) {
	generics.ElemNotEqualStrided[float32](dst, a, b, shape, stridesDst, stridesA, stridesB)
}

// ElemLessEqual writes 1.0 where a <= b, 0.0 otherwise
//
// Deprecated: Use generics.ElemLessEqualStrided[float32] directly instead.
// This function is a thin wrapper that just passes all parameters to generics.
func ElemLessEqual(dst, a, b []float32, shape []int, stridesDst, stridesA, stridesB []int) {
	generics.ElemLessEqualStrided[float32](dst, a, b, shape, stridesDst, stridesA, stridesB)
}

// ElemGreaterEqual writes 1.0 where a >= b, 0.0 otherwise
//
// Deprecated: Use generics.ElemGreaterEqualStrided[float32] directly instead.
// This function is a thin wrapper that just passes all parameters to generics.
func ElemGreaterEqual(dst, a, b []float32, shape []int, stridesDst, stridesA, stridesB []int) {
	generics.ElemGreaterEqualStrided[float32](dst, a, b, shape, stridesDst, stridesA, stridesB)
}

// ElemAddScaledMul computes dst = (1 + scalar) * other
func ElemAddScaledMul(dst, other []float32, scalar float32, shape []int, stridesDst, stridesOther []int) {
	generics.ElemApplyUnaryScalarStrided[float32](dst, other, scalar, shape, stridesDst, stridesOther, func(v, s float32) float32 {
		return (1.0 + s) * v
	})
}

// ElemAddScaledSquareMul computes dst = (1 + scalar * other^2) * other
func ElemAddScaledSquareMul(dst, other []float32, scalar float32, shape []int, stridesDst, stridesOther []int) {
	generics.ElemApplyUnaryScalarStrided[float32](dst, other, scalar, shape, stridesDst, stridesOther, func(v, s float32) float32 {
		return (1.0 + s*v*v) * v
	})
}
