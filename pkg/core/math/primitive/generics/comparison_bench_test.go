package generics

import (
	"testing"
)

var (
	compBenchSrcA = make([]float32, 10000)
	compBenchSrcB = make([]float32, 10000)
	compBenchDst  = make([]float32, 10000)
)

func init() {
	for i := range compBenchSrcA {
		compBenchSrcA[i] = float32(i)
		compBenchSrcB[i] = float32(i * 2)
	}
}

// BenchmarkElemGreaterThan_Generic benchmarks generic greater than (contiguous)
func BenchmarkElemGreaterThan_Generic(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ElemGreaterThan(compBenchDst, compBenchSrcA, compBenchSrcB, 10000)
	}
}

// BenchmarkElemGreaterThan_NonGeneric benchmarks non-generic greater than
func BenchmarkElemGreaterThan_NonGeneric(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		greaterThanNonGeneric(compBenchDst, compBenchSrcA, compBenchSrcB, 10000)
	}
}

// BenchmarkElemGreaterThan_DirectLoop benchmarks direct loop without function calls
func BenchmarkElemGreaterThan_DirectLoop(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for j := 0; j < 10000; j++ {
			if compBenchSrcA[j] > compBenchSrcB[j] {
				compBenchDst[j] = 1
			} else {
				compBenchDst[j] = 0
			}
		}
	}
}

// BenchmarkElemGreaterThanStrided_Generic benchmarks generic greater than with strides
func BenchmarkElemGreaterThanStrided_Generic(b *testing.B) {
	shape := []int{100, 100}
	strides := []int{100, 1}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ElemGreaterThanStrided(compBenchDst, compBenchSrcA, compBenchSrcB, shape, strides, strides, strides)
	}
}

// BenchmarkElemGreaterThanStrided_NonGeneric benchmarks non-generic strided greater than
func BenchmarkElemGreaterThanStrided_NonGeneric(b *testing.B) {
	shape := []int{100, 100}
	strides := []int{100, 1}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		greaterThanStridedNonGeneric(compBenchDst, compBenchSrcA, compBenchSrcB, shape, strides, strides, strides)
	}
}

// BenchmarkElemGreaterThanStrided_DirectLoop benchmarks direct strided loop
func BenchmarkElemGreaterThanStrided_DirectLoop(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for row := 0; row < 100; row++ {
			for col := 0; col < 100; col++ {
				idx := row*100 + col
				if compBenchSrcA[idx] > compBenchSrcB[idx] {
					compBenchDst[idx] = 1
				} else {
					compBenchDst[idx] = 0
				}
			}
		}
	}
}

// BenchmarkElemEqual_Generic benchmarks generic equal (contiguous)
func BenchmarkElemEqual_Generic(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ElemEqual(compBenchDst, compBenchSrcA, compBenchSrcB, 10000)
	}
}

// BenchmarkElemEqual_NonGeneric benchmarks non-generic equal
func BenchmarkElemEqual_NonGeneric(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		equalNonGeneric(compBenchDst, compBenchSrcA, compBenchSrcB, 10000)
	}
}

// BenchmarkElemEqual_DirectLoop benchmarks direct loop
func BenchmarkElemEqual_DirectLoop(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for j := 0; j < 10000; j++ {
			if compBenchSrcA[j] == compBenchSrcB[j] {
				compBenchDst[j] = 1
			} else {
				compBenchDst[j] = 0
			}
		}
	}
}

// BenchmarkElemLess_Generic benchmarks generic less than (contiguous)
func BenchmarkElemLess_Generic(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ElemLess(compBenchDst, compBenchSrcA, compBenchSrcB, 10000)
	}
}

// BenchmarkElemLess_NonGeneric benchmarks non-generic less than
func BenchmarkElemLess_NonGeneric(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		lessNonGeneric(compBenchDst, compBenchSrcA, compBenchSrcB, 10000)
	}
}

// BenchmarkElemLess_DirectLoop benchmarks direct loop
func BenchmarkElemLess_DirectLoop(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for j := 0; j < 10000; j++ {
			if compBenchSrcA[j] < compBenchSrcB[j] {
				compBenchDst[j] = 1
			} else {
				compBenchDst[j] = 0
			}
		}
	}
}

// Non-generic helper functions
func greaterThanNonGeneric(dst, a, b []float32, n int) {
	if n == 0 {
		return
	}
	for i := 0; i < n; i++ {
		if a[i] > b[i] {
			dst[i] = 1
		} else {
			dst[i] = 0
		}
	}
}

func greaterThanStridedNonGeneric(dst, a, b []float32, shape []int, stridesDst, stridesA, stridesB []int) {
	size := SizeFromShape(shape)
	if size == 0 {
		return
	}
	stridesDst = EnsureStrides(stridesDst, shape)
	stridesA = EnsureStrides(stridesA, shape)
	stridesB = EnsureStrides(stridesB, shape)
	if IsContiguous(stridesDst, shape) && IsContiguous(stridesA, shape) && IsContiguous(stridesB, shape) {
		for i := 0; i < size; i++ {
			if a[i] > b[i] {
				dst[i] = 1
			} else {
				dst[i] = 0
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
		if a[aIdx] > b[bIdx] {
			dst[dIdx] = 1
		} else {
			dst[dIdx] = 0
		}
		if !AdvanceOffsets(shape, indices, offsets, strideSet) {
			break
		}
	}
}

func equalNonGeneric(dst, a, b []float32, n int) {
	if n == 0 {
		return
	}
	for i := 0; i < n; i++ {
		if a[i] == b[i] {
			dst[i] = 1
		} else {
			dst[i] = 0
		}
	}
}

func lessNonGeneric(dst, a, b []float32, n int) {
	if n == 0 {
		return
	}
	for i := 0; i < n; i++ {
		if a[i] < b[i] {
			dst[i] = 1
		} else {
			dst[i] = 0
		}
	}
}

