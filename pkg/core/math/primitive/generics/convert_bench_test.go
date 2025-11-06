package generics

import (
	"math"
	"testing"
)

var (
	convertBenchSrcF32 = make([]float32, 10000)
	convertBenchSrcF64 = make([]float64, 10000)
	convertBenchSrcI32 = make([]int32, 10000)
	convertBenchDstF32 = make([]float32, 10000)
	convertBenchDstF64 = make([]float64, 10000)
	convertBenchDstI32 = make([]int32, 10000)
	convertBenchDstI8  = make([]int8, 10000)
)

func init() {
	for i := range convertBenchSrcF32 {
		convertBenchSrcF32[i] = float32(i)
		convertBenchSrcF64[i] = float64(i)
		convertBenchSrcI32[i] = int32(i)
	}
}

// BenchmarkElemConvert_Generic benchmarks generic convert (contiguous)
func BenchmarkElemConvert_Generic(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ElemConvert(convertBenchDstF64, convertBenchSrcF32, 10000)
	}
}

// BenchmarkElemConvert_NonGeneric benchmarks non-generic convert
func BenchmarkElemConvert_NonGeneric(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		convertNonGeneric(convertBenchDstF64, convertBenchSrcF32, 10000)
	}
}

// BenchmarkElemConvert_DirectLoop benchmarks direct loop without function calls
func BenchmarkElemConvert_DirectLoop(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for j := 0; j < 10000; j++ {
			convertBenchDstF64[j] = float64(convertBenchSrcF32[j])
		}
	}
}

// convertNonGeneric is a non-generic float32->float64 conversion for comparison
func convertNonGeneric(dst []float64, src []float32, n int) {
	if n == 0 {
		return
	}
	if len(src) < n {
		n = len(src)
	}
	if len(dst) < n {
		n = len(dst)
	}
	for i := 0; i < n; i++ {
		dst[i] = float64(src[i])
	}
}

// BenchmarkElemConvert_Clamping benchmarks convert with clamping (float64 to int8)
func BenchmarkElemConvert_Clamping(b *testing.B) {
	src := make([]float64, 10000)
	for i := range src {
		src[i] = float64(i * 10) // Values that need clamping
	}
	dst := make([]int8, 10000)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ElemConvert(dst, src, 10000)
	}
}

// BenchmarkElemConvert_Clamping_NonGeneric benchmarks non-generic clamping convert
func BenchmarkElemConvert_Clamping_NonGeneric(b *testing.B) {
	src := make([]float64, 10000)
	for i := range src {
		src[i] = float64(i * 10)
	}
	dst := make([]int8, 10000)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		convertClampingNonGeneric(dst, src, 10000)
	}
}

// BenchmarkElemConvert_Clamping_DirectLoop benchmarks direct clamping loop
func BenchmarkElemConvert_Clamping_DirectLoop(b *testing.B) {
	src := make([]float64, 10000)
	for i := range src {
		src[i] = float64(i * 10)
	}
	dst := make([]int8, 10000)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for j := 0; j < 10000; j++ {
			val := src[j]
			if val > math.MaxInt8 {
				dst[j] = math.MaxInt8
			} else if val < math.MinInt8 {
				dst[j] = math.MinInt8
			} else {
				dst[j] = int8(val)
			}
		}
	}
}

// convertClampingNonGeneric is a non-generic float64->int8 conversion with clamping
func convertClampingNonGeneric(dst []int8, src []float64, n int) {
	if n == 0 {
		return
	}
	if len(src) < n {
		n = len(src)
	}
	if len(dst) < n {
		n = len(dst)
	}
	for i := 0; i < n; i++ {
		val := src[i]
		if val > math.MaxInt8 {
			dst[i] = math.MaxInt8
		} else if val < math.MinInt8 {
			dst[i] = math.MinInt8
		} else {
			dst[i] = int8(val)
		}
	}
}

// BenchmarkElemConvertStrided_Generic benchmarks generic convert with strides
func BenchmarkElemConvertStrided_Generic(b *testing.B) {
	shape := []int{100, 100}
	strides := []int{100, 1}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ElemConvertStrided(convertBenchDstF64, convertBenchSrcF32, shape, strides, strides)
	}
}

// BenchmarkElemConvertStrided_NonGeneric benchmarks non-generic strided convert
func BenchmarkElemConvertStrided_NonGeneric(b *testing.B) {
	shape := []int{100, 100}
	strides := []int{100, 1}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		convertStridedNonGeneric(convertBenchDstF64, convertBenchSrcF32, shape, strides, strides)
	}
}

// BenchmarkElemConvertStrided_DirectLoop benchmarks direct strided convert loop
func BenchmarkElemConvertStrided_DirectLoop(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for row := 0; row < 100; row++ {
			for col := 0; col < 100; col++ {
				idx := row*100 + col
				convertBenchDstF64[idx] = float64(convertBenchSrcF32[idx])
			}
		}
	}
}

// convertStridedNonGeneric is a non-generic strided convert for comparison
func convertStridedNonGeneric(dst []float64, src []float32, shape []int, stridesDst, stridesSrc []int) {
	ndims := len(shape)
	if ndims == 0 {
		return
	}
	indices := make([]int, ndims)
	dim := 0
	for {
		if dim == ndims {
			sIdx := computeStrideOffset(indices, stridesSrc)
			dIdx := computeStrideOffset(indices, stridesDst)
			dst[dIdx] = float64(src[sIdx])
			dim--
			if dim < 0 {
				break
			}
			indices[dim]++
			continue
		}
		if indices[dim] >= shape[dim] {
			indices[dim] = 0
			dim--
			if dim < 0 {
				break
			}
			indices[dim]++
			continue
		}
		dim++
	}
}

// BenchmarkValueConvert_Generic benchmarks generic value convert
func BenchmarkValueConvert_Generic(b *testing.B) {
	val := float32(42.5)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = ValueConvert[float32, float64](val)
	}
}

// BenchmarkValueConvert_NonGeneric benchmarks non-generic value convert
func BenchmarkValueConvert_NonGeneric(b *testing.B) {
	val := float32(42.5)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = float64(val)
	}
}

// BenchmarkValueConvert_Clamping benchmarks value convert with clamping
func BenchmarkValueConvert_Clamping(b *testing.B) {
	val := float64(1000)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = ValueConvert[float64, int8](val)
	}
}

// BenchmarkValueConvert_Clamping_NonGeneric benchmarks non-generic clamping value convert
func BenchmarkValueConvert_Clamping_NonGeneric(b *testing.B) {
	val := float64(1000)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var result int8
		if val > math.MaxInt8 {
			result = math.MaxInt8
		} else if val < math.MinInt8 {
			result = math.MinInt8
		} else {
			result = int8(val)
		}
		_ = result
	}
}

