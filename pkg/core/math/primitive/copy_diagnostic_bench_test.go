package primitive

import (
	"testing"
)

// Benchmark to diagnose float64->float32 conversion performance
func BenchmarkFloatConversionDiagnostic(b *testing.B) {
	const size = 1000
	src := make([]float64, size)
	dst64 := make([]float64, size)
	dst32 := make([]float32, size)
	
	for i := range src {
		src[i] = float64(i)
	}

	b.Run("float64_to_float64_same_type", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			CopyWithConversion(dst64, src)
		}
	})

	b.Run("float64_to_float32_conversion", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			CopyWithConversion(dst32, src)
		}
	})

	// Test direct copy for comparison
	b.Run("float64_to_float64_direct_copy", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			copy(dst64, src)
		}
	})

	// Test manual conversion loop
	b.Run("float64_to_float32_manual_loop", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			for j := 0; j < size; j++ {
				dst32[j] = float32(src[j])
			}
		}
	})

	// Test manual conversion loop with boundary check elimination
	b.Run("float64_to_float32_manual_loop_with_bce", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if size > 0 {
				_ = dst32[size-1]
				_ = src[size-1]
			}
			for j := 0; j < size; j++ {
				dst32[j] = float32(src[j])
			}
		}
	})
}

