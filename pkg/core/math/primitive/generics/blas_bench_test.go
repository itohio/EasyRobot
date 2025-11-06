package generics

import (
	"testing"
)

var (
	blasBenchX = make([]float32, 10000)
	blasBenchY = make([]float32, 10000)
)

func init() {
	for i := range blasBenchX {
		blasBenchX[i] = float32(i)
		blasBenchY[i] = float32(i * 2)
	}
}

// BenchmarkCopy_Generic benchmarks generic copy (contiguous)
func BenchmarkCopy_Generic(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Copy(blasBenchY, blasBenchX, 10000)
	}
}

// BenchmarkCopy_NonGeneric benchmarks non-generic copy
func BenchmarkCopy_NonGeneric(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		blasCopyNonGeneric(blasBenchY, blasBenchX, 10000)
	}
}

// BenchmarkCopy_DirectLoop benchmarks direct copy builtin
func BenchmarkCopy_DirectLoop(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		copy(blasBenchY[:10000], blasBenchX[:10000])
	}
}

// BenchmarkCopyStrided_Generic benchmarks generic copy with strides
func BenchmarkCopyStrided_Generic(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		CopyStrided(blasBenchY, blasBenchX, 1, 1, 10000)
	}
}

// BenchmarkCopyStrided_NonGeneric benchmarks non-generic strided copy
func BenchmarkCopyStrided_NonGeneric(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		blasCopyStridedNonGeneric(blasBenchY, blasBenchX, 1, 1, 10000)
	}
}

// BenchmarkSwap_Generic benchmarks generic swap (contiguous)
func BenchmarkSwap_Generic(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Swap(blasBenchX, blasBenchY, 10000)
	}
}

// BenchmarkSwap_NonGeneric benchmarks non-generic swap
func BenchmarkSwap_NonGeneric(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		blasSwapNonGeneric(blasBenchX, blasBenchY, 10000)
	}
}

// BenchmarkSwap_DirectLoop benchmarks direct loop swap
func BenchmarkSwap_DirectLoop(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for j := 0; j < 10000; j++ {
			blasBenchX[j], blasBenchY[j] = blasBenchY[j], blasBenchX[j]
		}
	}
}

// BenchmarkSwapStrided_Generic benchmarks generic swap with strides
func BenchmarkSwapStrided_Generic(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		SwapStrided(blasBenchX, blasBenchY, 1, 1, 10000)
	}
}

// BenchmarkSwapStrided_NonGeneric benchmarks non-generic strided swap
func BenchmarkSwapStrided_NonGeneric(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		blasSwapStridedNonGeneric(blasBenchX, blasBenchY, 1, 1, 10000)
	}
}

// Non-generic helper functions
func blasCopyNonGeneric(y, x []float32, n int) {
	if n == 0 {
		return
	}
	copy(y[:n], x[:n])
}

func blasCopyStridedNonGeneric(y, x []float32, strideY, strideX, n int) {
	if n == 0 {
		return
	}
	if strideY == 1 && strideX == 1 {
		copy(y[:n], x[:n])
		return
	}
	py := 0
	px := 0
	for i := 0; i < n; i++ {
		y[py] = x[px]
		py += strideY
		px += strideX
	}
}

func blasSwapNonGeneric(x, y []float32, n int) {
	if n == 0 {
		return
	}
	for i := 0; i < n; i++ {
		x[i], y[i] = y[i], x[i]
	}
}

func blasSwapStridedNonGeneric(x, y []float32, strideX, strideY, n int) {
	if n == 0 {
		return
	}
	if strideX == 1 && strideY == 1 {
		for i := 0; i < n; i++ {
			x[i], y[i] = y[i], x[i]
		}
		return
	}
	px := 0
	py := 0
	for i := 0; i < n; i++ {
		x[px], y[py] = y[py], x[px]
		px += strideX
		py += strideY
	}
}
