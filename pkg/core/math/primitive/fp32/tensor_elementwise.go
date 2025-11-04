package fp32

import (
	"math"

	"github.com/chewxy/math32"
)

// ElemAdd writes element-wise sum of a and b into dst for the provided shape/strides.
func ElemAdd(dst, a, b []float32, shape []int, stridesDst, stridesA, stridesB []int) {
	applyElemBinary(dst, a, b, shape, stridesDst, stridesA, stridesB, func(av, bv float32) float32 {
		return av + bv
	})
}

// ElemSub writes element-wise difference of a and b into dst (dst = a - b).
func ElemSub(dst, a, b []float32, shape []int, stridesDst, stridesA, stridesB []int) {
	applyElemBinary(dst, a, b, shape, stridesDst, stridesA, stridesB, func(av, bv float32) float32 {
		return av - bv
	})
}

// ElemMul writes element-wise product of a and b into dst.
func ElemMul(dst, a, b []float32, shape []int, stridesDst, stridesA, stridesB []int) {
	applyElemBinary(dst, a, b, shape, stridesDst, stridesA, stridesB, func(av, bv float32) float32 {
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

	indices := make([]int, len(shape))
	offsets := make([]int, 3)
	strideSet := [][]int{stridesDst, stridesA, stridesB}
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

// ElemScale multiplies dst by the given scalar (in-place) for the provided shape/strides.
func ElemScale(dst []float32, scalar float32, shape []int, stridesDst []int) {
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

	indices := make([]int, len(shape))
	offsets := make([]int, 1)
	strideSet := [][]int{stridesDst}
	for {
		dIdx := offsets[0]
		dst[dIdx] *= scalar
		if !advanceOffsets(shape, indices, offsets, strideSet) {
			break
		}
	}
}

// ElemCopy copies src into dst respecting the supplied shape/strides.
func ElemCopy(dst, src []float32, shape []int, stridesDst, stridesSrc []int) {
	stridesDst = EnsureStrides(stridesDst, shape)
	stridesSrc = EnsureStrides(stridesSrc, shape)
	size := SizeFromShape(shape)
	if size == 0 {
		return
	}

	if IsContiguous(stridesDst, shape) && IsContiguous(stridesSrc, shape) {
		Copy(dst, src, 1, 1, size)
		return
	}

	indices := make([]int, len(shape))
	offsets := make([]int, 2)
	strideSet := [][]int{stridesDst, stridesSrc}
	for {
		dIdx := offsets[0]
		sIdx := offsets[1]
		dst[dIdx] = src[sIdx]
		if !advanceOffsets(shape, indices, offsets, strideSet) {
			break
		}
	}
}

// ElemWhere writes elements from a where condition is true, otherwise from b.
// condition, a, b must have compatible shapes/strides.
func ElemWhere(dst, condition, a, b []float32, shape []int, stridesDst, stridesCond, stridesA, stridesB []int) {
	applyElemTernary(dst, condition, a, b, shape, stridesDst, stridesCond, stridesA, stridesB, func(cv, av, bv float32) float32 {
		if cv > 0 { // condition > 0 means true
			return av
		}
		return bv
	})
}

// ElemGreaterThan writes 1.0 where a > b, 0.0 otherwise.
func ElemGreaterThan(dst, a, b []float32, shape []int, stridesDst, stridesA, stridesB []int) {
	applyElemBinary(dst, a, b, shape, stridesDst, stridesA, stridesB, func(av, bv float32) float32 {
		if av > bv {
			return 1.0
		}
		return 0.0
	})
}

// ElemEqual writes 1.0 where a == b, 0.0 otherwise.
func ElemEqual(dst, a, b []float32, shape []int, stridesDst, stridesA, stridesB []int) {
	applyElemBinary(dst, a, b, shape, stridesDst, stridesA, stridesB, func(av, bv float32) float32 {
		if av == bv {
			return 1.0
		}
		return 0.0
	})
}

// ElemLess writes 1.0 where a < b, 0.0 otherwise.
func ElemLess(dst, a, b []float32, shape []int, stridesDst, stridesA, stridesB []int) {
	applyElemBinary(dst, a, b, shape, stridesDst, stridesA, stridesB, func(av, bv float32) float32 {
		if av < bv {
			return 1.0
		}
		return 0.0
	})
}

// ElemSquare writes element-wise square of src into dst: dst[i] = src[i]^2
func ElemSquare(dst, src []float32, shape []int, stridesDst, stridesSrc []int) {
	applyElemUnary(dst, src, shape, stridesDst, stridesSrc, func(v float32) float32 {
		return v * v
	})
}

// ElemSqrt writes element-wise square root of src into dst: dst[i] = sqrt(src[i])
func ElemSqrt(dst, src []float32, shape []int, stridesDst, stridesSrc []int) {
	applyElemUnary(dst, src, shape, stridesDst, stridesSrc, func(v float32) float32 {
		if v < 0 {
			return 0.0 // Handle negative values gracefully
		}
		return math32.Sqrt(v)
	})
}

// ElemExp writes element-wise exponential of src into dst: dst[i] = exp(src[i])
func ElemExp(dst, src []float32, shape []int, stridesDst, stridesSrc []int) {
	applyElemUnary(dst, src, shape, stridesDst, stridesSrc, func(v float32) float32 {
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
	applyElemUnary(dst, src, shape, stridesDst, stridesSrc, func(v float32) float32 {
		if v <= 0 {
			return float32(math.Inf(-1)) // log(0) or negative = -Inf
		}
		return math32.Log(v)
	})
}

// ElemPow writes element-wise power of src^power into dst: dst[i] = src[i]^power
func ElemPow(dst, src []float32, power float32, shape []int, stridesDst, stridesSrc []int) {
	applyElemUnaryScalar(dst, src, power, shape, stridesDst, stridesSrc, func(v, p float32) float32 {
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
	applyElemUnary(dst, src, shape, stridesDst, stridesSrc, func(v float32) float32 {
		return math32.Abs(v)
	})
}

// ElemSign writes element-wise sign of src into dst: dst[i] = sign(src[i]) (-1, 0, or 1)
func ElemSign(dst, src []float32, shape []int, stridesDst, stridesSrc []int) {
	applyElemUnary(dst, src, shape, stridesDst, stridesSrc, func(v float32) float32 {
		if v > 0 {
			return 1.0
		}
		if v < 0 {
			return -1.0
		}
		return 0.0
	})
}

// ElemCos writes element-wise cosine of src into dst: dst[i] = cos(src[i])
func ElemCos(dst, src []float32, shape []int, stridesDst, stridesSrc []int) {
	applyElemUnary(dst, src, shape, stridesDst, stridesSrc, func(v float32) float32 {
		return math32.Cos(v)
	})
}

// ElemSin writes element-wise sine of src into dst: dst[i] = sin(src[i])
func ElemSin(dst, src []float32, shape []int, stridesDst, stridesSrc []int) {
	applyElemUnary(dst, src, shape, stridesDst, stridesSrc, func(v float32) float32 {
		return math32.Sin(v)
	})
}

// ElemTanh writes element-wise hyperbolic tangent of src into dst: dst[i] = tanh(src[i])
func ElemTanh(dst, src []float32, shape []int, stridesDst, stridesSrc []int) {
	applyElemUnary(dst, src, shape, stridesDst, stridesSrc, func(v float32) float32 {
		return math32.Tanh(v)
	})
}

// ElemNegative writes element-wise negation of src into dst: dst[i] = -src[i]
func ElemNegative(dst, src []float32, shape []int, stridesDst, stridesSrc []int) {
	applyElemUnary(dst, src, shape, stridesDst, stridesSrc, func(v float32) float32 {
		return -v
	})
}

func applyElemBinary(dst, a, b []float32, shape []int, stridesDst, stridesA, stridesB []int, op func(float32, float32) float32) {
	size := SizeFromShape(shape)
	if len(shape) == 0 || size == 0 {
		return
	}

	stridesDst = EnsureStrides(stridesDst, shape)
	stridesA = EnsureStrides(stridesA, shape)
	stridesB = EnsureStrides(stridesB, shape)

	if IsContiguous(stridesDst, shape) && IsContiguous(stridesA, shape) && IsContiguous(stridesB, shape) {
		for i := 0; i < size; i++ {
			dst[i] = op(a[i], b[i])
		}
		return
	}

	indices := make([]int, len(shape))
	offsets := make([]int, 3)
	strideSet := [][]int{stridesDst, stridesA, stridesB}
	for {
		dIdx := offsets[0]
		aIdx := offsets[1]
		bIdx := offsets[2]
		dst[dIdx] = op(a[aIdx], b[bIdx])
		if !advanceOffsets(shape, indices, offsets, strideSet) {
			break
		}
	}
}

func applyElemTernary(dst, condition, a, b []float32, shape []int, stridesDst, stridesCond, stridesA, stridesB []int, op func(float32, float32, float32) float32) {
	size := SizeFromShape(shape)
	if len(shape) == 0 || size == 0 {
		return
	}

	stridesDst = EnsureStrides(stridesDst, shape)
	stridesCond = EnsureStrides(stridesCond, shape)
	stridesA = EnsureStrides(stridesA, shape)
	stridesB = EnsureStrides(stridesB, shape)

	if IsContiguous(stridesDst, shape) && IsContiguous(stridesCond, shape) && IsContiguous(stridesA, shape) && IsContiguous(stridesB, shape) {
		for i := 0; i < size; i++ {
			dst[i] = op(condition[i], a[i], b[i])
		}
		return
	}

	indices := make([]int, len(shape))
	offsets := make([]int, 4)
	strideSet := [][]int{stridesDst, stridesCond, stridesA, stridesB}
	for {
		dIdx := offsets[0]
		cIdx := offsets[1]
		aIdx := offsets[2]
		bIdx := offsets[3]
		dst[dIdx] = op(condition[cIdx], a[aIdx], b[bIdx])
		if !advanceOffsets(shape, indices, offsets, strideSet) {
			break
		}
	}
}

// applyElemUnary applies a unary operation to each element: dst[i] = op(src[i])
func applyElemUnary(dst, src []float32, shape []int, stridesDst, stridesSrc []int, op func(float32) float32) {
	size := SizeFromShape(shape)
	if len(shape) == 0 || size == 0 {
		return
	}

	stridesDst = EnsureStrides(stridesDst, shape)
	stridesSrc = EnsureStrides(stridesSrc, shape)

	if IsContiguous(stridesDst, shape) && IsContiguous(stridesSrc, shape) {
		for i := 0; i < size; i++ {
			dst[i] = op(src[i])
		}
		return
	}

	indices := make([]int, len(shape))
	offsets := make([]int, 2)
	strideSet := [][]int{stridesDst, stridesSrc}
	for {
		dIdx := offsets[0]
		sIdx := offsets[1]
		dst[dIdx] = op(src[sIdx])
		if !advanceOffsets(shape, indices, offsets, strideSet) {
			break
		}
	}
}

// applyElemUnaryScalar applies a unary operation with a scalar parameter: dst[i] = op(src[i], scalar)
func applyElemUnaryScalar(dst, src []float32, scalar float32, shape []int, stridesDst, stridesSrc []int, op func(float32, float32) float32) {
	size := SizeFromShape(shape)
	if len(shape) == 0 || size == 0 {
		return
	}

	stridesDst = EnsureStrides(stridesDst, shape)
	stridesSrc = EnsureStrides(stridesSrc, shape)

	if IsContiguous(stridesDst, shape) && IsContiguous(stridesSrc, shape) {
		for i := 0; i < size; i++ {
			dst[i] = op(src[i], scalar)
		}
		return
	}

	indices := make([]int, len(shape))
	offsets := make([]int, 2)
	strideSet := [][]int{stridesDst, stridesSrc}
	for {
		dIdx := offsets[0]
		sIdx := offsets[1]
		dst[dIdx] = op(src[sIdx], scalar)
		if !advanceOffsets(shape, indices, offsets, strideSet) {
			break
		}
	}
}
