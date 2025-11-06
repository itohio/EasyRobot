package generics

import (
	"testing"

	. "github.com/itohio/EasyRobot/pkg/core/math/primitive/generics/helpers"
)

var (
	applyBenchSrcA = make([]float32, 10000)
	applyBenchSrcB = make([]float32, 10000)
	applyBenchDst  = make([]float32, 10000)
)

func init() {
	for i := range applyBenchSrcA {
		applyBenchSrcA[i] = float32(i)
		applyBenchSrcB[i] = float32(i * 2)
	}
}

// BenchmarkElemApplyUnary_Generic benchmarks generic unary apply (contiguous)
func BenchmarkElemApplyUnary_Generic(b *testing.B) {
	op := func(x float32) float32 { return x * 2 }
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ElemApplyUnary(applyBenchDst, applyBenchSrcA, 10000, op)
	}
}

// BenchmarkElemApplyUnary_NonGeneric benchmarks non-generic unary apply
func BenchmarkElemApplyUnary_NonGeneric(b *testing.B) {
	op := func(x float32) float32 { return x * 2 }
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		applyUnaryNonGeneric(applyBenchDst, applyBenchSrcA, 10000, op)
	}
}

// BenchmarkElemApplyUnary_DirectLoop benchmarks direct loop without closure
func BenchmarkElemApplyUnary_DirectLoop(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for j := 0; j < 10000; j++ {
			applyBenchDst[j] = applyBenchSrcA[j] * 2
		}
	}
}

// applyUnaryNonGeneric is a non-generic float32-specific unary apply
func applyUnaryNonGeneric(dst, src []float32, n int, op func(float32) float32) {
	if n == 0 {
		return
	}
	for i := 0; i < n; i++ {
		dst[i] = op(src[i])
	}
}

// applyUnaryStridedNonGeneric is a non-generic float32-specific strided unary apply
func applyUnaryStridedNonGeneric(dst, src []float32, shape []int, stridesDst, stridesSrc []int, op func(float32) float32) {
	size := SizeFromShape(shape)
	if size == 0 {
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
		if !AdvanceOffsets(shape, indices, offsets, strideSet) {
			break
		}
	}
}

// BenchmarkElemApplyBinary_Generic benchmarks generic binary apply (contiguous)
func BenchmarkElemApplyBinary_Generic(b *testing.B) {
	op := func(x, y float32) float32 { return x + y }
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ElemApplyBinary(applyBenchDst, applyBenchSrcA, applyBenchSrcB, 10000, op)
	}
}

// BenchmarkElemApplyBinary_NonGeneric benchmarks non-generic binary apply
func BenchmarkElemApplyBinary_NonGeneric(b *testing.B) {
	op := func(x, y float32) float32 { return x + y }
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		applyBinaryNonGeneric(applyBenchDst, applyBenchSrcA, applyBenchSrcB, 10000, op)
	}
}

// BenchmarkElemApplyBinary_DirectLoop benchmarks direct loop without closure
func BenchmarkElemApplyBinary_DirectLoop(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for j := 0; j < 10000; j++ {
			applyBenchDst[j] = applyBenchSrcA[j] + applyBenchSrcB[j]
		}
	}
}

// BenchmarkElemApplyBinaryStrided_Generic benchmarks generic binary apply with strides
func BenchmarkElemApplyBinaryStrided_Generic(b *testing.B) {
	shape := []int{100, 100}
	strides := []int{100, 1}
	op := func(x, y float32) float32 { return x + y }
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ElemApplyBinaryStrided(applyBenchDst, applyBenchSrcA, applyBenchSrcB, shape, strides, strides, strides, op)
	}
}

// BenchmarkElemApplyBinaryStrided_NonGeneric benchmarks non-generic strided binary apply
func BenchmarkElemApplyBinaryStrided_NonGeneric(b *testing.B) {
	shape := []int{100, 100}
	strides := []int{100, 1}
	op := func(x, y float32) float32 { return x + y }
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		applyBinaryStridedNonGeneric(applyBenchDst, applyBenchSrcA, applyBenchSrcB, shape, strides, strides, strides, op)
	}
}

// BenchmarkElemApplyBinaryStrided_DirectLoop benchmarks direct strided binary loop
func BenchmarkElemApplyBinaryStrided_DirectLoop(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for row := 0; row < 100; row++ {
			for col := 0; col < 100; col++ {
				idx := row*100 + col
				applyBenchDst[idx] = applyBenchSrcA[idx] + applyBenchSrcB[idx]
			}
		}
	}
}

// applyBinaryNonGeneric is a non-generic float32-specific binary apply
func applyBinaryNonGeneric(dst, a, b []float32, n int, op func(float32, float32) float32) {
	if n == 0 {
		return
	}
	for i := 0; i < n; i++ {
		dst[i] = op(a[i], b[i])
	}
}

// applyBinaryStridedNonGeneric is a non-generic float32-specific strided binary apply
func applyBinaryStridedNonGeneric(dst, a, b []float32, shape []int, stridesDst, stridesA, stridesB []int, op func(float32, float32) float32) {
	size := SizeFromShape(shape)
	if size == 0 {
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
		if !AdvanceOffsets(shape, indices, offsets, strideSet) {
			break
		}
	}
}

// BenchmarkElemApplyUnaryScalar_Generic benchmarks generic unary scalar apply (contiguous)
func BenchmarkElemApplyUnaryScalar_Generic(b *testing.B) {
	scalar := float32(5.0)
	op := func(x, s float32) float32 { return x * s }
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ElemApplyUnaryScalar(applyBenchDst, applyBenchSrcA, scalar, 10000, op)
	}
}

// BenchmarkElemApplyUnaryScalar_NonGeneric benchmarks non-generic unary scalar apply
func BenchmarkElemApplyUnaryScalar_NonGeneric(b *testing.B) {
	scalar := float32(5.0)
	op := func(x, s float32) float32 { return x * s }
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		applyUnaryScalarNonGeneric(applyBenchDst, applyBenchSrcA, scalar, 10000, op)
	}
}

// BenchmarkElemApplyUnaryScalar_DirectLoop benchmarks direct loop without closure
func BenchmarkElemApplyUnaryScalar_DirectLoop(b *testing.B) {
	scalar := float32(5.0)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for j := 0; j < 10000; j++ {
			applyBenchDst[j] = applyBenchSrcA[j] * scalar
		}
	}
}

// BenchmarkElemApplyUnaryScalarStrided_Generic benchmarks generic unary scalar apply with strides
func BenchmarkElemApplyUnaryScalarStrided_Generic(b *testing.B) {
	shape := []int{100, 100}
	strides := []int{100, 1}
	scalar := float32(5.0)
	op := func(x, s float32) float32 { return x * s }
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ElemApplyUnaryScalarStrided(applyBenchDst, applyBenchSrcA, scalar, shape, strides, strides, op)
	}
}

// BenchmarkElemApplyUnaryScalarStrided_NonGeneric benchmarks non-generic strided unary scalar apply
func BenchmarkElemApplyUnaryScalarStrided_NonGeneric(b *testing.B) {
	shape := []int{100, 100}
	strides := []int{100, 1}
	scalar := float32(5.0)
	op := func(x, s float32) float32 { return x * s }
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		applyUnaryScalarStridedNonGeneric(applyBenchDst, applyBenchSrcA, scalar, shape, strides, strides, op)
	}
}

// BenchmarkElemApplyUnaryScalarStrided_DirectLoop benchmarks direct strided scalar loop
func BenchmarkElemApplyUnaryScalarStrided_DirectLoop(b *testing.B) {
	scalar := float32(5.0)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for row := 0; row < 100; row++ {
			for col := 0; col < 100; col++ {
				idx := row*100 + col
				applyBenchDst[idx] = applyBenchSrcA[idx] * scalar
			}
		}
	}
}

// applyUnaryScalarNonGeneric is a non-generic float32-specific unary scalar apply
func applyUnaryScalarNonGeneric(dst, src []float32, scalar float32, n int, op func(float32, float32) float32) {
	if n == 0 {
		return
	}
	for i := 0; i < n; i++ {
		dst[i] = op(src[i], scalar)
	}
}

// applyUnaryScalarStridedNonGeneric is a non-generic float32-specific strided unary scalar apply
func applyUnaryScalarStridedNonGeneric(dst, src []float32, scalar float32, shape []int, stridesDst, stridesSrc []int, op func(float32, float32) float32) {
	size := SizeFromShape(shape)
	if size == 0 {
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
		if !AdvanceOffsets(shape, indices, offsets, strideSet) {
			break
		}
	}
}

// BenchmarkElemVecApply_Generic benchmarks generic vector apply
func BenchmarkElemVecApply_Generic(b *testing.B) {
	op := func(x float32) float32 { return x * 2 }
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ElemVecApply(applyBenchDst, applyBenchSrcA, 10000, 1, 1, op)
	}
}

// BenchmarkElemVecApply_NonGeneric benchmarks non-generic vector apply
func BenchmarkElemVecApply_NonGeneric(b *testing.B) {
	op := func(x float32) float32 { return x * 2 }
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		vecApplyNonGeneric(applyBenchDst, applyBenchSrcA, 10000, 1, 1, op)
	}
}

// BenchmarkElemVecApply_DirectLoop benchmarks direct vector loop
func BenchmarkElemVecApply_DirectLoop(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for j := 0; j < 10000; j++ {
			applyBenchDst[j] = applyBenchSrcA[j] * 2
		}
	}
}

// vecApplyNonGeneric is a non-generic float32-specific vector apply
func vecApplyNonGeneric(dst, src []float32, n int, strideDst, strideSrc int, op func(float32) float32) {
	if n == 0 {
		return
	}
	if strideDst == 1 && strideSrc == 1 {
		for i := 0; i < n; i++ {
			dst[i] = op(src[i])
		}
		return
	}
	dIdx := 0
	sIdx := 0
	for i := 0; i < n; i++ {
		dst[dIdx] = op(src[sIdx])
		dIdx += strideDst
		sIdx += strideSrc
	}
}

// BenchmarkElemMatApply_Generic benchmarks generic matrix apply
func BenchmarkElemMatApply_Generic(b *testing.B) {
	op := func(x float32) float32 { return x * 2 }
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ElemMatApply(applyBenchDst, applyBenchSrcA, 100, 100, 100, 100, op)
	}
}

// BenchmarkElemMatApply_NonGeneric benchmarks non-generic matrix apply
func BenchmarkElemMatApply_NonGeneric(b *testing.B) {
	op := func(x float32) float32 { return x * 2 }
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		matApplyNonGeneric(applyBenchDst, applyBenchSrcA, 100, 100, 100, 100, op)
	}
}

// BenchmarkElemMatApply_DirectLoop benchmarks direct matrix loop
func BenchmarkElemMatApply_DirectLoop(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for row := 0; row < 100; row++ {
			for col := 0; col < 100; col++ {
				idx := row*100 + col
				applyBenchDst[idx] = applyBenchSrcA[idx] * 2
			}
		}
	}
}

// matApplyNonGeneric is a non-generic float32-specific matrix apply
func matApplyNonGeneric(dst, src []float32, rows, cols int, ldDst, ldSrc int, op func(float32) float32) {
	if rows == 0 || cols == 0 {
		return
	}
	if ldDst == cols && ldSrc == cols {
		size := rows * cols
		for i := 0; i < size; i++ {
			dst[i] = op(src[i])
		}
		return
	}
	for i := 0; i < rows; i++ {
		dstRow := dst[i*ldDst:]
		srcRow := src[i*ldSrc:]
		for j := 0; j < cols; j++ {
			dstRow[j] = op(srcRow[j])
		}
	}
}
