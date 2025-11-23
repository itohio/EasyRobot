package shape

import (
	"testing"
)

var (
	benchShapeMax = []int{2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 1}
)

func BenchmarkComputeStrides_1D(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ComputeStrides(nil, benchShape1D)
	}
}

func BenchmarkComputeStrides_2D(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ComputeStrides(nil, benchShape2D)
	}
}

func BenchmarkComputeStrides_3D(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ComputeStrides(nil, benchShape3D)
	}
}

func BenchmarkComputeStrides_4D(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ComputeStrides(nil, benchShape4D)
	}
}

func BenchmarkComputeStrides_MAX_DIMS(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ComputeStrides(nil, benchShapeMax)
	}
}

func BenchmarkComputeStrides_WithDst_2D(b *testing.B) {
	var dst [MAX_DIMS]int
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ComputeStrides(dst[:len(benchShape2D)], benchShape2D)
	}
}

func BenchmarkEnsureStrides_Valid(b *testing.B) {
	strides := []int{100, 1}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		EnsureStrides(nil, strides, benchShape2D)
	}
}

func BenchmarkEnsureStrides_Invalid(b *testing.B) {
	strides := []int{1} // Wrong rank
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		EnsureStrides(nil, strides, benchShape2D)
	}
}

func BenchmarkEnsureStrides_WithDst(b *testing.B) {
	var dst [MAX_DIMS]int
	strides := []int{1} // Wrong rank, will compute
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		EnsureStrides(dst[:len(benchShape2D)], strides, benchShape2D)
	}
}

func BenchmarkIsContiguous_True_2D(b *testing.B) {
	strides := []int{100, 1}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		IsContiguous(strides, benchShape2D)
	}
}

func BenchmarkIsContiguous_False_2D(b *testing.B) {
	strides := []int{200, 1} // Non-contiguous
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		IsContiguous(strides, benchShape2D)
	}
}

func BenchmarkIsContiguous_True_3D(b *testing.B) {
	strides := []int{400, 20, 1}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		IsContiguous(strides, benchShape3D)
	}
}

func BenchmarkIsContiguous_MAX_DIMS(b *testing.B) {
	strides := ComputeStrides(nil, benchShapeMax)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		IsContiguous(strides, benchShapeMax)
	}
}
