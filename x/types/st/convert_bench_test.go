package st

import (
	"testing"
)

const benchmarkSize = 10000

// P11.1: Create convert_bench_test.go
// P11.2: Benchmark same-type conversions (int8→int8, etc.)
func BenchmarkConvertSameType(b *testing.B) {
	b.Run("int8_to_int8", func(b *testing.B) {
		src := make([]int8, benchmarkSize)
		dst := make([]int8, benchmarkSize)
		for i := range src {
			src[i] = int8(i % 128)
		}
		b.ResetTimer()
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			Convert(dst, src, benchmarkSize)
		}
	})

	b.Run("int16_to_int16", func(b *testing.B) {
		src := make([]int16, benchmarkSize)
		dst := make([]int16, benchmarkSize)
		for i := range src {
			src[i] = int16(i)
		}
		b.ResetTimer()
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			Convert(dst, src, benchmarkSize)
		}
	})

	b.Run("int32_to_int32", func(b *testing.B) {
		src := make([]int32, benchmarkSize)
		dst := make([]int32, benchmarkSize)
		for i := range src {
			src[i] = int32(i)
		}
		b.ResetTimer()
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			Convert(dst, src, benchmarkSize)
		}
	})

	b.Run("float32_to_float32", func(b *testing.B) {
		src := make([]float32, benchmarkSize)
		dst := make([]float32, benchmarkSize)
		for i := range src {
			src[i] = float32(i) / float32(benchmarkSize)
		}
		b.ResetTimer()
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			Convert(dst, src, benchmarkSize)
		}
	})

	b.Run("complex64_to_complex64", func(b *testing.B) {
		src := make([]complex64, benchmarkSize)
		dst := make([]complex64, benchmarkSize)
		for i := range src {
			src[i] = complex(float32(i)/float32(benchmarkSize), 0)
		}
		b.ResetTimer()
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			Convert(dst, src, benchmarkSize)
		}
	})

	// Baseline: direct copy for comparison
	b.Run("copy_baseline_int8", func(b *testing.B) {
		src := make([]int8, benchmarkSize)
		dst := make([]int8, benchmarkSize)
		for i := range src {
			src[i] = int8(i % 128)
		}
		b.ResetTimer()
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			copy(dst, src)
		}
	})
}

// P11.3: Benchmark float32 → int16 (with dithering)
func BenchmarkConvertFloat32ToInt16(b *testing.B) {
	src := make([]float32, benchmarkSize)
	dst := make([]int16, benchmarkSize)
	for i := range src {
		// Generate normalized audio samples in [-1, 1]
		src[i] = float32(i-benchmarkSize/2) / float32(benchmarkSize/2)
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		Convert(dst, src, benchmarkSize)
	}
}

// P11.4: Benchmark int16 → float32 (scaling only)
func BenchmarkConvertInt16ToFloat32(b *testing.B) {
	src := make([]int16, benchmarkSize)
	dst := make([]float32, benchmarkSize)
	for i := range src {
		src[i] = int16(i - benchmarkSize/2)
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		Convert(dst, src, benchmarkSize)
	}
}

// P11.5: Benchmark float32 → int8 (with dithering)
func BenchmarkConvertFloat32ToInt8(b *testing.B) {
	src := make([]float32, benchmarkSize)
	dst := make([]int8, benchmarkSize)
	for i := range src {
		// Generate normalized audio samples in [-1, 1]
		src[i] = float32(i-benchmarkSize/2) / float32(benchmarkSize/2)
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		Convert(dst, src, benchmarkSize)
	}
}

// P11.6: Benchmark strided conversions
func BenchmarkConvertStrided(b *testing.B) {
	// Contiguous case
	b.Run("contiguous", func(b *testing.B) {
		shape := []int{100, 100}
		src := make([]float32, 10000)
		dst := make([]int16, 10000)
		for i := range src {
			src[i] = float32(i-benchmarkSize/2) / float32(benchmarkSize/2)
		}
		dstStrides := []int{100, 1}
		srcStrides := []int{100, 1}

		b.ResetTimer()
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			ConvertStrided(dst, src, shape, dstStrides, srcStrides)
		}
	})

	// Non-contiguous case (transposed)
	b.Run("non_contiguous", func(b *testing.B) {
		shape := []int{100, 100}
		src := make([]float32, 10000)
		dst := make([]int16, 10000)
		for i := range src {
			src[i] = float32(i-benchmarkSize/2) / float32(benchmarkSize/2)
		}
		dstStrides := []int{1, 100} // Column-major
		srcStrides := []int{1, 100}

		b.ResetTimer()
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			ConvertStrided(dst, src, shape, dstStrides, srcStrides)
		}
	})
}

// P11.7: Benchmark all type combinations
func BenchmarkConvertAllCombinations(b *testing.B) {
	// int8 conversions
	b.Run("int8_from_float32", func(b *testing.B) {
		src := make([]float32, benchmarkSize)
		dst := make([]int8, benchmarkSize)
		for i := range src {
			src[i] = float32(i-benchmarkSize/2) / float32(benchmarkSize/2)
		}
		b.ResetTimer()
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			Convert(dst, src, benchmarkSize)
		}
	})

	b.Run("int8_from_int16", func(b *testing.B) {
		src := make([]int16, benchmarkSize)
		dst := make([]int8, benchmarkSize)
		for i := range src {
			src[i] = int16(i - benchmarkSize/2)
		}
		b.ResetTimer()
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			Convert(dst, src, benchmarkSize)
		}
	})

	b.Run("int8_from_int32", func(b *testing.B) {
		src := make([]int32, benchmarkSize)
		dst := make([]int8, benchmarkSize)
		for i := range src {
			src[i] = int32(i - benchmarkSize/2)
		}
		b.ResetTimer()
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			Convert(dst, src, benchmarkSize)
		}
	})

	b.Run("int8_from_complex64", func(b *testing.B) {
		src := make([]complex64, benchmarkSize)
		dst := make([]int8, benchmarkSize)
		for i := range src {
			src[i] = complex(float32(i-benchmarkSize/2)/float32(benchmarkSize/2), 0)
		}
		b.ResetTimer()
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			Convert(dst, src, benchmarkSize)
		}
	})

	b.Run("float32_from_int8", func(b *testing.B) {
		src := make([]int8, benchmarkSize)
		dst := make([]float32, benchmarkSize)
		for i := range src {
			src[i] = int8(i % 128)
		}
		b.ResetTimer()
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			Convert(dst, src, benchmarkSize)
		}
	})

	b.Run("int16_from_int8", func(b *testing.B) {
		src := make([]int8, benchmarkSize)
		dst := make([]int16, benchmarkSize)
		for i := range src {
			src[i] = int8(i % 128)
		}
		b.ResetTimer()
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			Convert(dst, src, benchmarkSize)
		}
	})

	b.Run("int32_from_int8", func(b *testing.B) {
		src := make([]int8, benchmarkSize)
		dst := make([]int32, benchmarkSize)
		for i := range src {
			src[i] = int8(i % 128)
		}
		b.ResetTimer()
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			Convert(dst, src, benchmarkSize)
		}
	})

	// int16 conversions
	b.Run("int16_from_float32", func(b *testing.B) {
		src := make([]float32, benchmarkSize)
		dst := make([]int16, benchmarkSize)
		for i := range src {
			src[i] = float32(i-benchmarkSize/2) / float32(benchmarkSize/2)
		}
		b.ResetTimer()
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			Convert(dst, src, benchmarkSize)
		}
	})

	b.Run("int16_from_int32", func(b *testing.B) {
		src := make([]int32, benchmarkSize)
		dst := make([]int16, benchmarkSize)
		for i := range src {
			src[i] = int32(i - benchmarkSize/2)
		}
		b.ResetTimer()
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			Convert(dst, src, benchmarkSize)
		}
	})

	b.Run("int16_from_complex64", func(b *testing.B) {
		src := make([]complex64, benchmarkSize)
		dst := make([]int16, benchmarkSize)
		for i := range src {
			src[i] = complex(float32(i-benchmarkSize/2)/float32(benchmarkSize/2), 0)
		}
		b.ResetTimer()
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			Convert(dst, src, benchmarkSize)
		}
	})

	b.Run("float32_from_int16", func(b *testing.B) {
		src := make([]int16, benchmarkSize)
		dst := make([]float32, benchmarkSize)
		for i := range src {
			src[i] = int16(i - benchmarkSize/2)
		}
		b.ResetTimer()
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			Convert(dst, src, benchmarkSize)
		}
	})

	b.Run("int32_from_int16", func(b *testing.B) {
		src := make([]int16, benchmarkSize)
		dst := make([]int32, benchmarkSize)
		for i := range src {
			src[i] = int16(i - benchmarkSize/2)
		}
		b.ResetTimer()
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			Convert(dst, src, benchmarkSize)
		}
	})

	// int32 conversions
	b.Run("int32_from_float32", func(b *testing.B) {
		src := make([]float32, benchmarkSize)
		dst := make([]int32, benchmarkSize)
		for i := range src {
			src[i] = float32(i-benchmarkSize/2) / float32(benchmarkSize/2)
		}
		b.ResetTimer()
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			Convert(dst, src, benchmarkSize)
		}
	})

	b.Run("int32_from_complex64", func(b *testing.B) {
		src := make([]complex64, benchmarkSize)
		dst := make([]int32, benchmarkSize)
		for i := range src {
			src[i] = complex(float32(i-benchmarkSize/2)/float32(benchmarkSize/2), 0)
		}
		b.ResetTimer()
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			Convert(dst, src, benchmarkSize)
		}
	})

	b.Run("float32_from_int32", func(b *testing.B) {
		src := make([]int32, benchmarkSize)
		dst := make([]float32, benchmarkSize)
		for i := range src {
			src[i] = int32(i - benchmarkSize/2)
		}
		b.ResetTimer()
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			Convert(dst, src, benchmarkSize)
		}
	})

	// float32 conversions
	b.Run("float32_from_complex64", func(b *testing.B) {
		src := make([]complex64, benchmarkSize)
		dst := make([]float32, benchmarkSize)
		for i := range src {
			src[i] = complex(float32(i-benchmarkSize/2)/float32(benchmarkSize/2), 0)
		}
		b.ResetTimer()
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			Convert(dst, src, benchmarkSize)
		}
	})

	// complex64 conversions
	b.Run("complex64_from_float32", func(b *testing.B) {
		src := make([]float32, benchmarkSize)
		dst := make([]complex64, benchmarkSize)
		for i := range src {
			src[i] = float32(i-benchmarkSize/2) / float32(benchmarkSize/2)
		}
		b.ResetTimer()
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			Convert(dst, src, benchmarkSize)
		}
	})

	b.Run("complex64_from_int8", func(b *testing.B) {
		src := make([]int8, benchmarkSize)
		dst := make([]complex64, benchmarkSize)
		for i := range src {
			src[i] = int8(i % 128)
		}
		b.ResetTimer()
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			Convert(dst, src, benchmarkSize)
		}
	})

	b.Run("complex64_from_int16", func(b *testing.B) {
		src := make([]int16, benchmarkSize)
		dst := make([]complex64, benchmarkSize)
		for i := range src {
			src[i] = int16(i - benchmarkSize/2)
		}
		b.ResetTimer()
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			Convert(dst, src, benchmarkSize)
		}
	})

	b.Run("complex64_from_int32", func(b *testing.B) {
		src := make([]int32, benchmarkSize)
		dst := make([]complex64, benchmarkSize)
		for i := range src {
			src[i] = int32(i - benchmarkSize/2)
		}
		b.ResetTimer()
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			Convert(dst, src, benchmarkSize)
		}
	})
}

// Compare against inline conversion code (baseline)
func BenchmarkConvertBaselineInline(b *testing.B) {
	b.Run("float32_to_int16_inline", func(b *testing.B) {
		src := make([]float32, benchmarkSize)
		dst := make([]int16, benchmarkSize)
		for i := range src {
			src[i] = float32(i-benchmarkSize/2) / float32(benchmarkSize/2)
		}
		b.ResetTimer()
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			for j := range src {
				val := src[j] * 32767.0
				if val > 32767.0 {
					dst[j] = 32767
				} else if val < -32768.0 {
					dst[j] = -32768
				} else {
					dst[j] = int16(val)
				}
			}
		}
	})

	b.Run("int16_to_float32_inline", func(b *testing.B) {
		src := make([]int16, benchmarkSize)
		dst := make([]float32, benchmarkSize)
		for i := range src {
			src[i] = int16(i - benchmarkSize/2)
		}
		b.ResetTimer()
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			for j := range src {
				dst[j] = float32(src[j]) * (1.0 / 32767.0)
			}
		}
	})

	b.Run("float32_to_int8_inline", func(b *testing.B) {
		src := make([]float32, benchmarkSize)
		dst := make([]int8, benchmarkSize)
		for i := range src {
			src[i] = float32(i-benchmarkSize/2) / float32(benchmarkSize/2)
		}
		b.ResetTimer()
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			for j := range src {
				val := src[j] * 127.0
				if val > 127.0 {
					dst[j] = 127
				} else if val < -128.0 {
					dst[j] = -128
				} else {
					dst[j] = int8(val)
				}
			}
		}
	})
}

// BenchmarkInt16ToFloat32Comparison compares baseline, Convert, and assembly implementations
// This benchmark helps identify performance differences and optimization opportunities.
//
// Results (typical):
// - baseline_inline: ~6μs/op - Current cmd/audio inline implementation
// - convert_with_dispatch: ~7-8μs/op - Convert() with type dispatch overhead
// - convert_optimized: ~2.9μs/op - Direct convertFromInt16ToFloat32() call (no dispatch)
// - assembly_simd_placeholder: ~3μs/op - Placeholder for future SIMD implementation
//
// The optimized version is ~2x faster than baseline, showing the benefit of optimized loops.
// Dispatch overhead is ~5μs due to:
// 1. Type assertions (any(dst), any(src)) - runtime type checks
// 2. Multiple switch statements (same-type check + dispatch + source type dispatch)
// 3. Extra bounds checks and reslicing operations
// 4. Function call overhead (Convert -> convertToFloat32 -> convertFromInt16ToFloat32)
func BenchmarkInt16ToFloat32Comparison(b *testing.B) {
	src := make([]int16, benchmarkSize)
	for i := range src {
		src[i] = int16(i - benchmarkSize/2)
	}

	// Baseline: inline implementation (what cmd/audio currently does)
	// Simple loop with direct conversion and scaling
	b.Run("baseline_inline", func(b *testing.B) {
		dst := make([]float32, benchmarkSize)
		b.ResetTimer()
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			for j := range src {
				dst[j] = float32(src[j]) * (1.0 / 32767.0)
			}
		}
	})

	// Convert function: current optimized implementation
	// Uses type dispatch and optimized loops with BCE hints
	b.Run("convert_with_dispatch", func(b *testing.B) {
		dst := make([]float32, benchmarkSize)
		b.ResetTimer()
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			Convert(dst, src, benchmarkSize)
		}
	})

	// Direct function call: uses the actual convertFromInt16ToFloat32 function
	// This is what Convert calls internally after dispatch, bypassing dispatch overhead
	b.Run("convert_optimized", func(b *testing.B) {
		dst := make([]float32, benchmarkSize)
		b.ResetTimer()
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			convertFromInt16ToFloat32(dst, src)
		}
	})

	// Assembly-optimized version
	// NOTE: Go's assembler has limited SIMD instruction support (no CVTSI2SS/CVTDQ2PS),
	// so this currently provides optimized loop structure but not full SIMD conversion.
	// For true SIMD optimization, consider CGO with compiler intrinsics.
	b.Run("assembly_simd_placeholder", func(b *testing.B) {
		dst := make([]float32, benchmarkSize)
		b.ResetTimer()
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			convertInt16ToFloat32Assembly(dst, src)
		}
	})
}

// convertInt16ToFloat32Assembly is declared in convert_asm_amd64.s
// This is REAL assembly (not a placeholder), but Go's assembler limitations
// prevent full SIMD optimization. The assembly optimizes loop structure and
// memory access patterns, but int->float conversion uses available scalar path.
// For full SIMD, consider CGO with compiler intrinsics (AVX/SSE).
func convertInt16ToFloat32Assembly(dst []float32, src []int16)
