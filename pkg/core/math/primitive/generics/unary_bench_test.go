package generics

import (
	"testing"

	. "github.com/itohio/EasyRobot/pkg/core/math/primitive/generics/helpers"
)

var (
	unaryBenchSrc = make([]float32, 10000)
	unaryBenchDst = make([]float32, 10000)
)

func init() {
	for i := range unaryBenchSrc {
		unaryBenchSrc[i] = float32(i - 5000) // Mix of positive and negative
	}
}

// BenchmarkElemSign_Generic benchmarks generic sign (contiguous)
func BenchmarkElemSign_Generic(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ElemSign(unaryBenchDst, unaryBenchSrc, 10000)
	}
}

// BenchmarkElemSign_NonGeneric benchmarks non-generic sign
func BenchmarkElemSign_NonGeneric(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		signNonGeneric(unaryBenchDst, unaryBenchSrc, 10000)
	}
}

// BenchmarkElemSign_DirectLoop benchmarks direct loop
func BenchmarkElemSign_DirectLoop(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for j := 0; j < 10000; j++ {
			v := unaryBenchSrc[j]
			if v > 0 {
				unaryBenchDst[j] = 1
			} else if v < 0 {
				unaryBenchDst[j] = -1
			} else {
				unaryBenchDst[j] = 0
			}
		}
	}
}

// BenchmarkElemSignStrided_Generic benchmarks generic sign with strides
func BenchmarkElemSignStrided_Generic(b *testing.B) {
	shape := []int{100, 100}
	strides := []int{100, 1}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ElemSignStrided(unaryBenchDst, unaryBenchSrc, shape, strides, strides)
	}
}

// BenchmarkElemSignStrided_NonGeneric benchmarks non-generic strided sign
func BenchmarkElemSignStrided_NonGeneric(b *testing.B) {
	shape := []int{100, 100}
	strides := []int{100, 1}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		signStridedNonGeneric(unaryBenchDst, unaryBenchSrc, shape, strides, strides)
	}
}

// BenchmarkElemSignStrided_DirectLoop benchmarks direct strided loop
func BenchmarkElemSignStrided_DirectLoop(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for row := 0; row < 100; row++ {
			for col := 0; col < 100; col++ {
				idx := row*100 + col
				v := unaryBenchSrc[idx]
				if v > 0 {
					unaryBenchDst[idx] = 1
				} else if v < 0 {
					unaryBenchDst[idx] = -1
				} else {
					unaryBenchDst[idx] = 0
				}
			}
		}
	}
}

// BenchmarkElemNegative_Generic benchmarks generic negative (contiguous)
func BenchmarkElemNegative_Generic(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ElemNegative(unaryBenchDst, unaryBenchSrc, 10000)
	}
}

// BenchmarkElemNegative_NonGeneric benchmarks non-generic negative
func BenchmarkElemNegative_NonGeneric(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		negativeNonGeneric(unaryBenchDst, unaryBenchSrc, 10000)
	}
}

// BenchmarkElemNegative_DirectLoop benchmarks direct loop
func BenchmarkElemNegative_DirectLoop(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for j := 0; j < 10000; j++ {
			unaryBenchDst[j] = -unaryBenchSrc[j]
		}
	}
}

// BenchmarkElemNegativeStrided_Generic benchmarks generic negative with strides
func BenchmarkElemNegativeStrided_Generic(b *testing.B) {
	shape := []int{100, 100}
	strides := []int{100, 1}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ElemNegativeStrided(unaryBenchDst, unaryBenchSrc, shape, strides, strides)
	}
}

// BenchmarkElemNegativeStrided_NonGeneric benchmarks non-generic strided negative
func BenchmarkElemNegativeStrided_NonGeneric(b *testing.B) {
	shape := []int{100, 100}
	strides := []int{100, 1}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		negativeStridedNonGeneric(unaryBenchDst, unaryBenchSrc, shape, strides, strides)
	}
}

// BenchmarkElemNegativeStrided_DirectLoop benchmarks direct strided loop
func BenchmarkElemNegativeStrided_DirectLoop(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for row := 0; row < 100; row++ {
			for col := 0; col < 100; col++ {
				idx := row*100 + col
				unaryBenchDst[idx] = -unaryBenchSrc[idx]
			}
		}
	}
}

// Non-generic helper functions
func signNonGeneric(dst, src []float32, n int) {
	if n == 0 {
		return
	}
	for i := 0; i < n; i++ {
		v := src[i]
		if v > 0 {
			dst[i] = 1
		} else if v < 0 {
			dst[i] = -1
		} else {
			dst[i] = 0
		}
	}
}

func signStridedNonGeneric(dst, src []float32, shape []int, stridesDst, stridesSrc []int) {
	size := SizeFromShape(shape)
	if size == 0 {
		return
	}
	// Use stack-allocated arrays for stride computation
	var dstStridesStatic [MAX_DIMS]int
	var srcStridesStatic [MAX_DIMS]int
	stridesDst = EnsureStrides(dstStridesStatic[:len(shape)], stridesDst, shape)
	stridesSrc = EnsureStrides(srcStridesStatic[:len(shape)], stridesSrc, shape)
	if IsContiguous(stridesDst, shape) && IsContiguous(stridesSrc, shape) {
		for i := 0; i < size; i++ {
			v := src[i]
			if v > 0 {
				dst[i] = 1
			} else if v < 0 {
				dst[i] = -1
			} else {
				dst[i] = 0
			}
		}
		return
	}
	rank := len(shape)
	var indicesStatic [MAX_DIMS]int
	var offsetsStatic [2]int
	indices := indicesStatic[:rank]
	offsets := offsetsStatic[:2]
	for {
		dIdx := offsets[0]
		sIdx := offsets[1]
		v := src[sIdx]
		if v > 0 {
			dst[dIdx] = 1
		} else if v < 0 {
			dst[dIdx] = -1
		} else {
			dst[dIdx] = 0
		}
		if !AdvanceOffsets(shape, indices, offsets, stridesDst, stridesSrc) {
			break
		}
	}
}

func negativeNonGeneric(dst, src []float32, n int) {
	if n == 0 {
		return
	}
	for i := 0; i < n; i++ {
		dst[i] = -src[i]
	}
}

func negativeStridedNonGeneric(dst, src []float32, shape []int, stridesDst, stridesSrc []int) {
	size := SizeFromShape(shape)
	if size == 0 {
		return
	}
	// Use stack-allocated arrays for stride computation
	var dstStridesStatic [MAX_DIMS]int
	var srcStridesStatic [MAX_DIMS]int
	stridesDst = EnsureStrides(dstStridesStatic[:len(shape)], stridesDst, shape)
	stridesSrc = EnsureStrides(srcStridesStatic[:len(shape)], stridesSrc, shape)
	if IsContiguous(stridesDst, shape) && IsContiguous(stridesSrc, shape) {
		for i := 0; i < size; i++ {
			dst[i] = -src[i]
		}
		return
	}
	rank := len(shape)
	var indicesStatic [MAX_DIMS]int
	var offsetsStatic [2]int
	indices := indicesStatic[:rank]
	offsets := offsetsStatic[:2]
	for {
		dIdx := offsets[0]
		sIdx := offsets[1]
		dst[dIdx] = -src[sIdx]
		if !AdvanceOffsets(shape, indices, offsets, stridesDst, stridesSrc) {
			break
		}
	}
}
