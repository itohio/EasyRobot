package fp32

import (
	"fmt"
	"testing"
)

func BenchmarkCrossCorrelate1D(b *testing.B) {
	sizes := []int{64, 256, 1024, 4096}
	for _, size := range sizes {
		b.Run(fmt.Sprintf("size_%d", size), func(b *testing.B) {
			signal1 := make([]float32, size)
			signal2 := make([]float32, size/2)
			dst := make([]float32, size+size/2-1)
			for i := range size {
				signal1[i] = float32(i)
			}
			for i := range size / 2 {
				signal2[i] = float32(i)
			}
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				CrossCorrelate1D(dst, signal1, signal2)
			}
		})
	}
}

func BenchmarkCrossCorrelate1DByConvolution(b *testing.B) {
	size := 1024
	signal1 := make([]float32, size)
	signal2 := make([]float32, size/2)
	dst := make([]float32, size+size/2-1)
	for i := range size {
		signal1[i] = float32(i)
	}
	for i := range size / 2 {
		signal2[i] = float32(i)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		CrossCorrelate1DByConvolution(dst, signal1, signal2)
	}
}

func BenchmarkCrossCorrelate2D(b *testing.B) {
	rows, cols := 64, 64
	signal1 := make([]float32, rows*cols)
	signal2 := make([]float32, rows/2*cols/2)
	for i := range len(signal1) {
		signal1[i] = float32(i)
	}
	for i := range len(signal2) {
		signal2[i] = float32(i)
	}

	dstRows := rows + rows/2 - 1
	dstCols := cols + cols/2 - 1
	dst := make([]float32, dstRows*dstCols)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		CrossCorrelate2D(dst, dstRows, dstCols, signal1, rows, cols, signal2, rows/2, cols/2)
	}
}

func BenchmarkFindPeakCorrelation(b *testing.B) {
	sizes := []int{64, 256, 1024, 4096}
	for _, size := range sizes {
		b.Run(fmt.Sprintf("size_%d", size), func(b *testing.B) {
			corr := make([]float32, size)
			for i := range size {
				corr[i] = float32(i)
			}
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, _ = FindPeakCorrelation(corr)
			}
		})
	}
}

func BenchmarkFindPeakCorrelation2D(b *testing.B) {
	rows, cols := 64, 64
	corr := make([]float32, rows*cols)
	for i := range len(corr) {
		corr[i] = float32(i)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _, _ = FindPeakCorrelation2D(corr, rows, cols)
	}
}

