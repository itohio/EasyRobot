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
		gen.GenerateWithOp(dst, false, func(d *float32, value float32) {
			*d = value
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
		gen.GenerateWithOp(dst, false, func(d *float32, value float32) {
			*d = value
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
		gen.GenerateWithOp(baseband, false, func(d *float32, carrierValue float32) {
			*d *= carrierValue
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
		gen.GenerateWithOp(dst, false, func(d *float32, value float32) {
			*d = value
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
		gen.GenerateWithOp(dst, false, func(d *float32, value float32) {
			*d = value
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
		gen.GenerateWithOp(dst, false, func(d *float32, value float32) {
			*d = value
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
		gen.GenerateWithOp(dst, false, func(d *float32, value float32) {
			*d = value
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
		gen.GenerateWithOp(dst, false, func(d *float32, value float32) {
			*d = value
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

func BenchmarkPinkChirpGenerator_Generate(b *testing.B) {
	gen := NewPinkChirpGenerator(benchSampleRate, 1000.0, 20000.0, 1.0)
	dst := make([]float32, benchSamples)
	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		gen.Generate(dst)
	}
}

func BenchmarkPinkChirpGenerator_GenerateWithOp(b *testing.B) {
	gen := NewPinkChirpGenerator(benchSampleRate, 1000.0, 20000.0, 1.0)
	dst := make([]float32, benchSamples)
	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		gen.GenerateWithOp(dst, false, func(d *float32, value float32) {
			*d = value
		})
	}
}

func BenchmarkPinkLogarithmicChirpGenerator_Generate(b *testing.B) {
	gen := NewPinkLogarithmicChirpGenerator(benchSampleRate, 1000.0, 20000.0, 1.0)
	dst := make([]float32, benchSamples)
	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		gen.Generate(dst)
	}
}

func BenchmarkPinkLogarithmicChirpGenerator_GenerateWithOp(b *testing.B) {
	gen := NewPinkLogarithmicChirpGenerator(benchSampleRate, 1000.0, 20000.0, 1.0)
	dst := make([]float32, benchSamples)
	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		gen.GenerateWithOp(dst, false, func(d *float32, value float32) {
			*d = value
		})
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
		gen.GenerateWithOp(baseband, false, func(d *float32, carrierValue float32) {
			*d *= carrierValue
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

// Benchmark to show samples per second more clearly
func BenchmarkSamplesPerSecond_SquareGenerator(b *testing.B) {
	gen := NewSquareGenerator(benchSampleRate, 20000.0, 1.0)
	dst := make([]float32, benchSamples)
	b.ResetTimer()

	totalSamples := int64(0)
	for i := 0; i < b.N; i++ {
		gen.Generate(dst)
		totalSamples += int64(benchSamples)
	}
	b.SetBytes(totalSamples * 4) // 4 bytes per float32
}

func BenchmarkSamplesPerSecond_NoiseWhiteGenerator(b *testing.B) {
	gen := NewNoiseWhiteGenerator(1.0)
	dst := make([]float32, benchSamples)
	b.ResetTimer()

	totalSamples := int64(0)
	for i := 0; i < b.N; i++ {
		gen.Generate(dst)
		totalSamples += int64(benchSamples)
	}
	b.SetBytes(totalSamples * 4) // 4 bytes per float32
}
