package generics

import (
	"testing"

	. "github.com/itohio/EasyRobot/pkg/core/math/primitive/generics/helpers"
)

var (
	copyBenchSrc = make([]float32, 10000)
	copyBenchDst = make([]float32, 10000)
)

func init() {
	for i := range copyBenchSrc {
		copyBenchSrc[i] = float32(i)
	}
}

// BenchmarkElemCopy_Generic benchmarks the generic copy implementation
func BenchmarkElemCopy_Generic(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ElemCopy(copyBenchDst, copyBenchSrc, 10000)
	}
}

// BenchmarkElemCopy_NonGeneric benchmarks a non-generic float32-specific loop
func BenchmarkElemCopy_NonGeneric(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		copyNonGeneric(copyBenchDst, copyBenchSrc, 10000)
	}
}

// BenchmarkElemCopy_DirectLoop benchmarks a direct loop without function calls
func BenchmarkElemCopy_DirectLoop(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for j := 0; j < 10000; j++ {
			copyBenchDst[j] = copyBenchSrc[j]
		}
	}
}

// copyNonGeneric is a non-generic float32-specific implementation for comparison
func copyNonGeneric(dst, src []float32, n int) {
	if n == 0 {
		return
	}
	copy(dst[:n], src[:n])
}

// BenchmarkElemCopyStrided_Generic benchmarks generic copy with strides
func BenchmarkElemCopyStrided_Generic(b *testing.B) {
	shape := []int{100, 100}
	strides := []int{100, 1}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ElemCopyStrided(copyBenchDst, copyBenchSrc, shape, strides, strides)
	}
}

// BenchmarkElemCopyStrided_NonGeneric benchmarks non-generic strided copy
func BenchmarkElemCopyStrided_NonGeneric(b *testing.B) {
	shape := []int{100, 100}
	strides := []int{100, 1}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		copyStridedNonGeneric(copyBenchDst, copyBenchSrc, shape, strides, strides)
	}
}

// BenchmarkElemCopyStrided_DirectLoop benchmarks direct strided loop
func BenchmarkElemCopyStrided_DirectLoop(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for row := 0; row < 100; row++ {
			for col := 0; col < 100; col++ {
				dstIdx := row*100 + col
				srcIdx := row*100 + col
				copyBenchDst[dstIdx] = copyBenchSrc[srcIdx]
			}
		}
	}
}

// copyStridedNonGeneric is a non-generic strided copy for comparison
func copyStridedNonGeneric(dst, src []float32, shape []int, stridesDst, stridesSrc []int) {
	rank := len(shape)
	var indicesStatic [MAX_DIMS]int
	var offsetsStatic [2]int
	indices := indicesStatic[:rank]
	offsets := offsetsStatic[:2]
	for {
		dIdx := offsets[0]
		sIdx := offsets[1]
		dst[dIdx] = src[sIdx]
		if !AdvanceOffsets(shape, indices, offsets, stridesDst, stridesSrc) {
			break
		}
	}
}

// BenchmarkElemSwap_Generic benchmarks generic swap (contiguous)
func BenchmarkElemSwap_Generic(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ElemSwap(copyBenchDst, copyBenchSrc, 10000)
	}
}

// BenchmarkElemSwap_NonGeneric benchmarks non-generic swap
func BenchmarkElemSwap_NonGeneric(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		swapNonGeneric(copyBenchDst, copyBenchSrc, 10000)
	}
}

// BenchmarkElemSwapStrided_Generic benchmarks generic swap with strides
func BenchmarkElemSwapStrided_Generic(b *testing.B) {
	shape := []int{100, 100}
	strides := []int{100, 1}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ElemSwapStrided(copyBenchDst, copyBenchSrc, shape, strides, strides)
	}
}

// BenchmarkElemSwapStrided_NonGeneric benchmarks non-generic strided swap
func BenchmarkElemSwapStrided_NonGeneric(b *testing.B) {
	shape := []int{100, 100}
	strides := []int{100, 1}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		swapStridedNonGeneric(copyBenchDst, copyBenchSrc, shape, strides, strides)
	}
}

// BenchmarkElemSwapStrided_DirectLoop benchmarks direct strided swap loop
func BenchmarkElemSwapStrided_DirectLoop(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for row := 0; row < 100; row++ {
			for col := 0; col < 100; col++ {
				idx := row*100 + col
				copyBenchDst[idx], copyBenchSrc[idx] = copyBenchSrc[idx], copyBenchDst[idx]
			}
		}
	}
}

// swapStridedNonGeneric is a non-generic strided swap for comparison
func swapStridedNonGeneric(dst, src []float32, shape []int, stridesDst, stridesSrc []int) {
	rank := len(shape)
	var indicesStatic [MAX_DIMS]int
	var offsetsStatic [2]int
	indices := indicesStatic[:rank]
	offsets := offsetsStatic[:2]
	for {
		dIdx := offsets[0]
		sIdx := offsets[1]
		dst[dIdx], src[sIdx] = src[sIdx], dst[dIdx]
		if !AdvanceOffsets(shape, indices, offsets, stridesDst, stridesSrc) {
			break
		}
	}
}

// BenchmarkElemSwap_DirectLoop benchmarks direct swap loop
func BenchmarkElemSwap_DirectLoop(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for j := 0; j < 10000; j++ {
			copyBenchDst[j], copyBenchSrc[j] = copyBenchSrc[j], copyBenchDst[j]
		}
	}
}

// swapNonGeneric is a non-generic float32-specific swap
func swapNonGeneric(dst, src []float32, n int) {
	if n == 0 {
		return
	}
	for i := 0; i < n; i++ {
		dst[i], src[i] = src[i], dst[i]
	}
}
