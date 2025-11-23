package loop_experiments

import (
	"testing"
)

const (
	matSize = 100
)

var (
	matA = make([]float32, matSize*matSize)
	matB = make([]float32, matSize*matSize)
	matC = make([]float32, matSize*matSize)
)

func init() {
	// Initialize test matrices
	for i := range matA {
		matA[i] = float32(i) * 0.001
		matB[i] = float32(i) * 0.002
	}
}

func Benchmark_MatMul_Naive(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		MatMulNaive(matC, matA, matB, matSize, matSize, matSize)
	}
}

func Benchmark_MatMul_Optimized(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		MatMulOptimized(matC, matA, matB, matSize, matSize, matSize)
	}
}

func Benchmark_MatMul_OptimizedTranspose(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		MatMulOptimizedTranspose(matC, matA, matB, matSize, matSize, matSize)
	}
}

func Benchmark_MatMul_Flattened(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		MatMulFlattened(matC, matA, matB, matSize, matSize, matSize)
	}
}

// Assembly version (amd64 only) - BASELINE
func Benchmark_MatMul_Assembly(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		asmMatMul(&matC[0], &matA[0], &matB[0], matSize, matSize, matSize)
	}
}

// Experimental implementations
func Benchmark_MatMul_Exp1(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		MatMulExp1(matC, matA, matB, matSize, matSize, matSize)
	}
}

func Benchmark_MatMul_Exp2(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		MatMulExp2(matC, matA, matB, matSize, matSize, matSize)
	}
}

func Benchmark_MatMul_Exp3(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		MatMulExp3(matC, matA, matB, matSize, matSize, matSize)
	}
}

func Benchmark_MatMul_Exp4(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		MatMulExp4(matC, matA, matB, matSize, matSize, matSize)
	}
}

func Benchmark_MatMul_Exp5(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		MatMulExp5(matC, matA, matB, matSize, matSize, matSize)
	}
}

func Benchmark_MatMul_Exp6(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		MatMulExp6(matC, matA, matB, matSize, matSize, matSize)
	}
}

func Benchmark_MatMul_Exp7(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		MatMulExp7(matC, matA, matB, matSize, matSize, matSize)
	}
}

func Benchmark_MatMul_Exp8(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		MatMulExp8(matC, matA, matB, matSize, matSize, matSize)
	}
}

func Benchmark_MatMul_Exp9(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		MatMulExp9(matC, matA, matB, matSize, matSize, matSize)
	}
}

func Benchmark_MatMul_Exp10(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		MatMulExp10(matC, matA, matB, matSize, matSize, matSize)
	}
}

func Benchmark_MatMul_Exp11(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		MatMulExp11(matC, matA, matB, matSize, matSize, matSize)
	}
}

func Benchmark_MatMul_Exp12(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		MatMulExp12(matC, matA, matB, matSize, matSize, matSize)
	}
}
