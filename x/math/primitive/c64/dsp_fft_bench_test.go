package c64

import (
	"fmt"
	"testing"
)

func BenchmarkFFT(b *testing.B) {
	sizes := []int{64, 256, 1024, 4096}
	for _, size := range sizes {
		b.Run(fmt.Sprintf("size_%d", size), func(b *testing.B) {
			data := make([]complex64, size)
			for i := range size {
				data[i] = complex64(complex(float32(i), float32(i)))
			}
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				FFT(data)
			}
		})
	}
}

func BenchmarkIFFT(b *testing.B) {
	sizes := []int{64, 256, 1024, 4096}
	for _, size := range sizes {
		b.Run(fmt.Sprintf("size_%d", size), func(b *testing.B) {
			data := make([]complex64, size)
			for i := range size {
				data[i] = complex64(complex(float32(i), float32(i)))
			}
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				IFFT(data)
			}
		})
	}
}

func BenchmarkFFTWithLUT(b *testing.B) {
	sizes := []int{64, 256, 1024, 4096}
	for _, size := range sizes {
		b.Run(fmt.Sprintf("size_%d", size), func(b *testing.B) {
			lut := NewFFTLUT(size)
			totalFactors := len(lut)
			data := make([]complex64, size)
			for i := range size {
				data[i] = complex64(complex(float32(i), float32(i)))
			}
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				FFTWithLUT(data, lut, totalFactors)
			}
		})
	}
}

func BenchmarkFFT_vs_FFTWithLUT(b *testing.B) {
	size := 1024
	lut := NewFFTLUT(size)
	totalFactors := len(lut)

	b.Run("FFT", func(b *testing.B) {
		data := make([]complex64, size)
		for i := range size {
			data[i] = complex64(complex(float32(i), float32(i)))
		}
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			FFT(data)
		}
	})

	b.Run("FFTWithLUT", func(b *testing.B) {
		data := make([]complex64, size)
		for i := range size {
			data[i] = complex64(complex(float32(i), float32(i)))
		}
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			FFTWithLUT(data, lut, totalFactors)
		}
	})
}

func BenchmarkComputeFFTLUT(b *testing.B) {
	sizes := []int{64, 256, 1024, 4096}
	for _, size := range sizes {
		b.Run(fmt.Sprintf("size_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_ = ComputeFFTLUT(size)
			}
		})
	}
}

func BenchmarkFillFFTLUT(b *testing.B) {
	sizes := []int{64, 256, 1024, 4096}
	for _, size := range sizes {
		b.Run(fmt.Sprintf("size_%d", size), func(b *testing.B) {
			lutSize := ComputeFFTLUT(size)
			lut := make([]complex64, lutSize)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				FillFFTLUT(lut, size)
			}
		})
	}
}

func BenchmarkNewFFTLUT(b *testing.B) {
	sizes := []int{64, 256, 1024, 4096}
	for _, size := range sizes {
		b.Run(fmt.Sprintf("size_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_ = NewFFTLUT(size)
			}
		})
	}
}
