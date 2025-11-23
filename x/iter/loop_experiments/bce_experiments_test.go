package loop_experiments

import (
	"testing"
)

const (
	rows = 1000
	cols = 1000
	size = rows * cols
)

// 200MB array: 200 * 1024 * 1024 / 4 (float32) = 52,428,800 elements
const hugeArraySize = 52_428_800

var (
	hugeDst = make([]float32, hugeArraySize)
	hugeSrc = make([]float32, hugeArraySize)
)

func init() {
	// Initialize huge arrays with test data
	for i := range hugeSrc {
		hugeSrc[i] = float32(i) * 0.001
	}
}

// ========== CONTIGUOUS BENCHMARKS ==========

func Benchmark_BaselineNestedLoops(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		offset := GetCacheOffset(i, hugeArraySize)
		BaselineNestedLoops(hugeDst[offset:offset+size], hugeSrc[offset:offset+size], rows, cols)
	}
}

func Benchmark_BCE_Hint_AccessLast(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		offset := GetCacheOffset(i, hugeArraySize)
		BCE_Hint_AccessLast(hugeDst[offset:offset+size], hugeSrc[offset:offset+size], rows, cols)
	}
}

func Benchmark_BCE_Reslice_ExactSize(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		offset := GetCacheOffset(i, hugeArraySize)
		BCE_Reslice_ExactSize(hugeDst[offset:offset+size], hugeSrc[offset:offset+size], rows, cols)
	}
}

func Benchmark_BCE_RowSlices(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		offset := GetCacheOffset(i, hugeArraySize)
		BCE_RowSlices(hugeDst[offset:offset+size], hugeSrc[offset:offset+size], rows, cols)
	}
}

func Benchmark_BCE_RowSlices_Reslice(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		offset := GetCacheOffset(i, hugeArraySize)
		BCE_RowSlices_Reslice(hugeDst[offset:offset+size], hugeSrc[offset:offset+size], rows, cols)
	}
}

func Benchmark_BCE_RangeRows(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		offset := GetCacheOffset(i, hugeArraySize)
		BCE_RangeRows(hugeDst[offset:offset+size], hugeSrc[offset:offset+size], rows, cols)
	}
}

func Benchmark_BCE_RangeBoth(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		offset := GetCacheOffset(i, hugeArraySize)
		BCE_RangeBoth(hugeDst[offset:offset+size], hugeSrc[offset:offset+size], rows, cols)
	}
}

func Benchmark_BCE_Reslice_RangeBoth(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		offset := GetCacheOffset(i, hugeArraySize)
		BCE_Reslice_RangeBoth(hugeDst[offset:offset+size], hugeSrc[offset:offset+size], rows, cols)
	}
}

func Benchmark_BCE_RowSlices_RangeCols(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		offset := GetCacheOffset(i, hugeArraySize)
		BCE_RowSlices_RangeCols(hugeDst[offset:offset+size], hugeSrc[offset:offset+size], rows, cols)
	}
}

func Benchmark_BCE_RowSlices_Reslice_Range(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		offset := GetCacheOffset(i, hugeArraySize)
		BCE_RowSlices_Reslice_Range(hugeDst[offset:offset+size], hugeSrc[offset:offset+size], rows, cols)
	}
}

func Benchmark_BCE_Flatten_Reslice(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		offset := GetCacheOffset(i, hugeArraySize)
		BCE_Flatten_Reslice(hugeDst[offset:offset+size], hugeSrc[offset:offset+size], rows, cols)
	}
}

func Benchmark_BCE_AccessLastPerRow(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		offset := GetCacheOffset(i, hugeArraySize)
		BCE_AccessLastPerRow(hugeDst[offset:offset+size], hugeSrc[offset:offset+size], rows, cols)
	}
}

func Benchmark_BCE_PrecomputeOffsets(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		offset := GetCacheOffset(i, hugeArraySize)
		BCE_PrecomputeOffsets(hugeDst[offset:offset+size], hugeSrc[offset:offset+size], rows, cols)
	}
}

func Benchmark_BCE_PrecomputeOffsets_Range(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		offset := GetCacheOffset(i, hugeArraySize)
		BCE_PrecomputeOffsets_Range(hugeDst[offset:offset+size], hugeSrc[offset:offset+size], rows, cols)
	}
}

func Benchmark_BCE_RowSlices_AccessLast(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		offset := GetCacheOffset(i, hugeArraySize)
		BCE_RowSlices_AccessLast(hugeDst[offset:offset+size], hugeSrc[offset:offset+size], rows, cols)
	}
}

// ========== STRIDED BENCHMARKS ==========

const ldDst = 1000
const ldSrc = 1000

func Benchmark_StridedBaseline(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		offset := GetCacheOffset(i, hugeArraySize)
		StridedBaseline(hugeDst[offset:offset+size], hugeSrc[offset:offset+size], rows, cols, ldDst, ldSrc)
	}
}

func Benchmark_Strided_RowSlices(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		offset := GetCacheOffset(i, hugeArraySize)
		Strided_RowSlices(hugeDst[offset:offset+size], hugeSrc[offset:offset+size], rows, cols, ldDst, ldSrc)
	}
}

func Benchmark_Strided_RowSlices_Reslice(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		offset := GetCacheOffset(i, hugeArraySize)
		Strided_RowSlices_Reslice(hugeDst[offset:offset+size], hugeSrc[offset:offset+size], rows, cols, ldDst, ldSrc)
	}
}

func Benchmark_Strided_RowSlices_Range(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		offset := GetCacheOffset(i, hugeArraySize)
		Strided_RowSlices_Range(hugeDst[offset:offset+size], hugeSrc[offset:offset+size], rows, cols, ldDst, ldSrc)
	}
}

func Benchmark_Strided_RowSlices_Reslice_Range(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		offset := GetCacheOffset(i, hugeArraySize)
		Strided_RowSlices_Reslice_Range(hugeDst[offset:offset+size], hugeSrc[offset:offset+size], rows, cols, ldDst, ldSrc)
	}
}

func Benchmark_Strided_RowSlices_Reslice_Range_Unrolled(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		offset := GetCacheOffset(i, hugeArraySize)
		Strided_RowSlices_Reslice_Range_Unrolled(hugeDst[offset:offset+size], hugeSrc[offset:offset+size], rows, cols, ldDst, ldSrc)
	}
}

func Benchmark_Strided_PrecomputeOffsets(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		offset := GetCacheOffset(i, hugeArraySize)
		Strided_PrecomputeOffsets(hugeDst[offset:offset+size], hugeSrc[offset:offset+size], rows, cols, ldDst, ldSrc)
	}
}

func Benchmark_Strided_PrecomputeOffsets_Range(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		offset := GetCacheOffset(i, hugeArraySize)
		Strided_PrecomputeOffsets_Range(hugeDst[offset:offset+size], hugeSrc[offset:offset+size], rows, cols, ldDst, ldSrc)
	}
}

// ========== OVERHEAD MEASUREMENT BENCHMARKS ==========

// Benchmark_Contiguous_FlatLoop: Single flat loop over contiguous array
func Benchmark_Contiguous_FlatLoop(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		offset := GetCacheOffset(i, hugeArraySize)
		dst := hugeDst[offset : offset+size]
		src := hugeSrc[offset : offset+size]
		for j := range size {
			dst[j] = op(src[j])
		}
	}
}

// Benchmark_Contiguous_FlatLoop_Reslice: Single flat loop with reslice
func Benchmark_Contiguous_FlatLoop_Reslice(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		offset := GetCacheOffset(i, hugeArraySize)
		dst := hugeDst[offset : offset+size]
		src := hugeSrc[offset : offset+size]
		dst = dst[:size]
		src = src[:size]
		for j := range size {
			dst[j] = op(src[j])
		}
	}
}

// Benchmark_OperationOnly: Just the operation, no looping overhead
// This measures the pure operation cost to understand loop overhead
func Benchmark_OperationOnly(b *testing.B) {
	x := float32(1.5)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// Just the operation, no array access
		_ = op(x)
	}
}

// Benchmark_OperationOnly_Inline: Operation inlined directly
func Benchmark_OperationOnly_Inline(b *testing.B) {
	x := float32(1.5)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// Inline: x*x + 1.0
		_ = x*x + 1.0
	}
}

// Benchmark_ArrayAccessOnly: Just array access, no operation
// Measures array access overhead
func Benchmark_ArrayAccessOnly(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		offset := GetCacheOffset(i, hugeArraySize)
		src := hugeSrc[offset : offset+size]
		var sum float32
		for j := range size {
			sum += src[j] // Just read, no operation
		}
		_ = sum
	}
}

// Benchmark_ArrayAccessWriteOnly: Just array write, no operation
// Measures array write overhead
func Benchmark_ArrayAccessWriteOnly(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		offset := GetCacheOffset(i, hugeArraySize)
		dst := hugeDst[offset : offset+size]
		src := hugeSrc[offset : offset+size]
		for j := range size {
			dst[j] = src[j] // Just copy, no operation
		}
	}
}

// Assembly-based benchmarks (calling op function - not inlined)
func Benchmark_Contiguous_Assembly(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		offset := GetCacheOffset(i, hugeArraySize)
		dst := hugeDst[offset : offset+size]
		src := hugeSrc[offset : offset+size]
		BCE_Assembly(dst, src, rows, cols)
	}
}

func Benchmark_Contiguous_Assembly_Unrolled(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		offset := GetCacheOffset(i, hugeArraySize)
		dst := hugeDst[offset : offset+size]
		src := hugeSrc[offset : offset+size]
		BCE_Assembly_Unrolled(dst, src, rows, cols)
	}
}

// Assembly-based benchmarks (with inline op function - compiler may inline)
func Benchmark_Contiguous_Assembly_Inline(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		offset := GetCacheOffset(i, hugeArraySize)
		dst := hugeDst[offset : offset+size]
		src := hugeSrc[offset : offset+size]
		BCE_Assembly_Inline(dst, src, rows, cols)
	}
}

func Benchmark_Contiguous_Assembly_Unrolled_Inline(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		offset := GetCacheOffset(i, hugeArraySize)
		dst := hugeDst[offset : offset+size]
		src := hugeSrc[offset : offset+size]
		BCE_Assembly_Unrolled_Inline(dst, src, rows, cols)
	}
}

// Platform-specific assembly benchmarks (direct implementation, no function calls)
func Benchmark_Contiguous_Assembly_Direct(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		offset := GetCacheOffset(i, hugeArraySize)
		dst := hugeDst[offset : offset+size]
		src := hugeSrc[offset : offset+size]
		BCE_Assembly_Direct(dst, src, rows, cols)
	}
}

func Benchmark_Contiguous_Assembly_Unrolled_Direct(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		offset := GetCacheOffset(i, hugeArraySize)
		dst := hugeDst[offset : offset+size]
		src := hugeSrc[offset : offset+size]
		BCE_Assembly_Unrolled_Direct(dst, src, rows, cols)
	}
}
