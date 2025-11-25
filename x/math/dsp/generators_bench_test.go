package dsp

import (
	"testing"
)

const (
	benchSampleRate = 96000.0
	benchSamples    = 96000 // 1 second at 96kHz
)

func BenchmarkSineGenerator_Generate(b *testing.B) {
	gen := NewSineGenerator(benchSampleRate, 20000.0, 1.0, 0.0)
	dst := make([]float32, benchSamples)
	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		gen.Generate(dst)
	}
}

func BenchmarkSineGenerator_GenerateWithOp(b *testing.B) {
	gen := NewSineGenerator(benchSampleRate, 20000.0, 1.0, 0.0)
	dst := make([]float32, benchSamples)
	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		gen.GenerateWithOp(dst, false, func(idx int, value float32) {
			dst[idx] = value
		})
	}
}

func BenchmarkCosineGenerator_Generate(b *testing.B) {
	gen := NewCosineGenerator(benchSampleRate, 20000.0, 1.0, 0.0)
	dst := make([]float32, benchSamples)
	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		gen.Generate(dst)
	}
}

func BenchmarkCosineGenerator_GenerateWithOp(b *testing.B) {
	gen := NewCosineGenerator(benchSampleRate, 20000.0, 1.0, 0.0)
	dst := make([]float32, benchSamples)
	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		gen.GenerateWithOp(dst, false, func(idx int, value float32) {
			dst[idx] = value
		})
	}
}

func BenchmarkCosineGenerator_GenerateWithOp_Multiply(b *testing.B) {
	gen := NewCosineGenerator(benchSampleRate, 20000.0, 1.0, 0.0)
	baseband := make([]float32, benchSamples)
	// Initialize baseband with some values
	for i := range baseband {
		baseband[i] = 0.5
	}
	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		gen.GenerateWithOp(baseband, false, func(idx int, carrierValue float32) {
			baseband[idx] *= carrierValue
		})
	}
}

func BenchmarkSincGenerator_Generate(b *testing.B) {
	gen := NewSincGenerator(float32(benchSamples)/2, 1.0, 1.0)
	dst := make([]float32, benchSamples)
	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		gen.Generate(dst)
	}
}

func BenchmarkSincGenerator_GenerateWithOp(b *testing.B) {
	gen := NewSincGenerator(float32(benchSamples)/2, 1.0, 1.0)
	dst := make([]float32, benchSamples)
	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		gen.GenerateWithOp(dst, func(idx int, value float32) {
			dst[idx] = value
		})
	}
}

func BenchmarkGaussianGenerator_Generate(b *testing.B) {
	gen := NewGaussianGenerator(float32(benchSamples)/2, 1000.0, 1.0)
	dst := make([]float32, benchSamples)
	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		gen.Generate(dst)
	}
}

func BenchmarkGaussianGenerator_GenerateWithOp(b *testing.B) {
	gen := NewGaussianGenerator(float32(benchSamples)/2, 1000.0, 1.0)
	dst := make([]float32, benchSamples)
	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		gen.GenerateWithOp(dst, func(idx int, value float32) {
			dst[idx] = value
		})
	}
}

func BenchmarkSquareGenerator_Generate(b *testing.B) {
	gen := NewSquareGenerator(benchSampleRate, 20000.0, 1.0)
	dst := make([]float32, benchSamples)
	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		gen.Generate(dst)
	}
}

func BenchmarkSquareGenerator_GenerateWithOp(b *testing.B) {
	gen := NewSquareGenerator(benchSampleRate, 20000.0, 1.0)
	dst := make([]float32, benchSamples)
	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		gen.GenerateWithOp(dst, func(idx int, value float32) {
			dst[idx] = value
		})
	}
}

func BenchmarkTriangleGenerator_Generate(b *testing.B) {
	gen := NewTriangleGenerator(benchSampleRate, 20000.0, 1.0)
	dst := make([]float32, benchSamples)
	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		gen.Generate(dst)
	}
}

func BenchmarkTriangleGenerator_GenerateWithOp(b *testing.B) {
	gen := NewTriangleGenerator(benchSampleRate, 20000.0, 1.0)
	dst := make([]float32, benchSamples)
	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		gen.GenerateWithOp(dst, func(idx int, value float32) {
			dst[idx] = value
		})
	}
}

func BenchmarkNoiseWhiteGenerator_Generate(b *testing.B) {
	gen := NewNoiseWhiteGenerator(1.0)
	dst := make([]float32, benchSamples)
	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		gen.Generate(dst)
	}
}

func BenchmarkNoiseWhiteGenerator_GenerateWithOp(b *testing.B) {
	gen := NewNoiseWhiteGenerator(1.0)
	dst := make([]float32, benchSamples)
	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		gen.GenerateWithOp(dst, func(idx int, value float32) {
			dst[idx] = value
		})
	}
}

// Benchmark throughput in samples per second
func BenchmarkThroughput_SineGenerator(b *testing.B) {
	gen := NewSineGenerator(benchSampleRate, 20000.0, 1.0, 0.0)
	dst := make([]float32, benchSamples)
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		gen.Generate(dst)
	}
	b.SetBytes(int64(benchSamples * 4)) // 4 bytes per float32
}

func BenchmarkThroughput_CosineGenerator(b *testing.B) {
	gen := NewCosineGenerator(benchSampleRate, 20000.0, 1.0, 0.0)
	dst := make([]float32, benchSamples)
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		gen.Generate(dst)
	}
	b.SetBytes(int64(benchSamples * 4)) // 4 bytes per float32
}

func BenchmarkThroughput_SincGenerator(b *testing.B) {
	gen := NewSincGenerator(float32(benchSamples)/2, 1.0, 1.0)
	dst := make([]float32, benchSamples)
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		gen.Generate(dst)
	}
	b.SetBytes(int64(benchSamples * 4)) // 4 bytes per float32
}

func BenchmarkThroughput_GaussianGenerator(b *testing.B) {
	gen := NewGaussianGenerator(float32(benchSamples)/2, 1000.0, 1.0)
	dst := make([]float32, benchSamples)
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		gen.Generate(dst)
	}
	b.SetBytes(int64(benchSamples * 4)) // 4 bytes per float32
}

// Benchmark small buffer sizes (typical for chip-level processing)
func BenchmarkSmallBuffer_CosineGenerator(b *testing.B) {
	gen := NewCosineGenerator(benchSampleRate, 20000.0, 1.0, 0.0)
	dst := make([]float32, 8) // Small buffer for chip samples
	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		gen.Generate(dst)
	}
}

func BenchmarkSmallBuffer_CosineGenerator_GenerateWithOp(b *testing.B) {
	gen := NewCosineGenerator(benchSampleRate, 20000.0, 1.0, 0.0)
	baseband := make([]float32, 8) // Small buffer for chip samples
	for i := range baseband {
		baseband[i] = 0.5
	}
	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		gen.GenerateWithOp(baseband, false, func(idx int, carrierValue float32) {
			baseband[idx] *= carrierValue
		})
	}
}

// Benchmark state continuity (multiple calls)
func BenchmarkStateContinuity_CosineGenerator(b *testing.B) {
	gen := NewCosineGenerator(benchSampleRate, 20000.0, 1.0, 0.0)
	dst := make([]float32, 1000) // Smaller chunks
	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		gen.Generate(dst)
		if i%100 == 0 {
			// Simulate periodic reset (not actually resetting, just showing state is maintained)
		}
	}
}

