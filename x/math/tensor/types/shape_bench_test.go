package types

import (
	"testing"
)

var (
	benchShape1D    = NewShape(1000)
	benchShape2D    = NewShape(100, 100)
	benchShape3D    = NewShape(50, 50, 50)
	benchShape4D    = NewShape(10, 10, 10, 10)
	benchShape5D    = NewShape(5, 5, 5, 5, 5)
	benchShape16D   = NewShape(2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2)
	benchShapeSmall = NewShape(10)
)

// BenchmarkShape_Rank benchmarks Rank operation
func BenchmarkShape_Rank_1D(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = benchShape1D.Rank()
	}
}

func BenchmarkShape_Rank_5D(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = benchShape5D.Rank()
	}
}

// BenchmarkShape_Size benchmarks Size operation
func BenchmarkShape_Size_1D(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = benchShape1D.Size()
	}
}

func BenchmarkShape_Size_2D(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = benchShape2D.Size()
	}
}

func BenchmarkShape_Size_3D(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = benchShape3D.Size()
	}
}

func BenchmarkShape_Size_5D(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = benchShape5D.Size()
	}
}

func BenchmarkShape_Size_16D(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = benchShape16D.Size()
	}
}

// BenchmarkShape_Strides_Nil benchmarks Strides with nil destination (allocates)
func BenchmarkShape_Strides_Nil_1D(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = benchShape1D.Strides(nil)
	}
}

func BenchmarkShape_Strides_Nil_2D(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = benchShape2D.Strides(nil)
	}
}

func BenchmarkShape_Strides_Nil_3D(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = benchShape3D.Strides(nil)
	}
}

func BenchmarkShape_Strides_Nil_4D(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = benchShape4D.Strides(nil)
	}
}

func BenchmarkShape_Strides_Nil_5D(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = benchShape5D.Strides(nil)
	}
}

func BenchmarkShape_Strides_Nil_16D(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = benchShape16D.Strides(nil)
	}
}

// BenchmarkShape_Strides_Preallocated benchmarks Strides with preallocated destination
func BenchmarkShape_Strides_Preallocated_1D(b *testing.B) {
	dst := make([]int, benchShape1D.Rank())
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = benchShape1D.Strides(dst)
	}
}

func BenchmarkShape_Strides_Preallocated_2D(b *testing.B) {
	dst := make([]int, benchShape2D.Rank())
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = benchShape2D.Strides(dst)
	}
}

func BenchmarkShape_Strides_Preallocated_3D(b *testing.B) {
	dst := make([]int, benchShape3D.Rank())
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = benchShape3D.Strides(dst)
	}
}

func BenchmarkShape_Strides_Preallocated_4D(b *testing.B) {
	dst := make([]int, benchShape4D.Rank())
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = benchShape4D.Strides(dst)
	}
}

func BenchmarkShape_Strides_Preallocated_5D(b *testing.B) {
	dst := make([]int, benchShape5D.Rank())
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = benchShape5D.Strides(dst)
	}
}

func BenchmarkShape_Strides_Preallocated_16D(b *testing.B) {
	dst := make([]int, benchShape16D.Rank())
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = benchShape16D.Strides(dst)
	}
}

// BenchmarkShape_Strides_StackAllocated benchmarks Strides with stack-allocated destination
func BenchmarkShape_Strides_StackAllocated_1D(b *testing.B) {
	var dst [1]int
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = benchShape1D.Strides(dst[:])
	}
}

func BenchmarkShape_Strides_StackAllocated_2D(b *testing.B) {
	var dst [2]int
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = benchShape2D.Strides(dst[:])
	}
}

func BenchmarkShape_Strides_StackAllocated_3D(b *testing.B) {
	var dst [3]int
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = benchShape3D.Strides(dst[:])
	}
}

func BenchmarkShape_Strides_StackAllocated_4D(b *testing.B) {
	var dst [4]int
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = benchShape4D.Strides(dst[:])
	}
}

func BenchmarkShape_Strides_StackAllocated_5D(b *testing.B) {
	var dst [5]int
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = benchShape5D.Strides(dst[:])
	}
}

// BenchmarkShape_Equal benchmarks Equal operation
func BenchmarkShape_Equal_True(b *testing.B) {
	other := NewShape(100, 100)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = benchShape2D.Equal(other)
	}
}

func BenchmarkShape_Equal_False(b *testing.B) {
	other := NewShape(50, 50)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = benchShape2D.Equal(other)
	}
}

func BenchmarkShape_Equal_DifferentRank(b *testing.B) {
	other := NewShape(100)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = benchShape2D.Equal(other)
	}
}

// BenchmarkShape_Clone benchmarks Clone operation
func BenchmarkShape_Clone_1D(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = benchShape1D.Clone()
	}
}

func BenchmarkShape_Clone_2D(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = benchShape2D.Clone()
	}
}

func BenchmarkShape_Clone_5D(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = benchShape5D.Clone()
	}
}

func BenchmarkShape_Clone_16D(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = benchShape16D.Clone()
	}
}

// BenchmarkShape_ToSlice benchmarks ToSlice operation
func BenchmarkShape_ToSlice_1D(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = benchShape1D.ToSlice()
	}
}

func BenchmarkShape_ToSlice_2D(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = benchShape2D.ToSlice()
	}
}

func BenchmarkShape_ToSlice_5D(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = benchShape5D.ToSlice()
	}
}

func BenchmarkShape_ToSlice_16D(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = benchShape16D.ToSlice()
	}
}

// BenchmarkShape_IsContiguous benchmarks IsContiguous operation
func BenchmarkShape_IsContiguous_True_2D(b *testing.B) {
	strides := benchShape2D.Strides(nil)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = benchShape2D.IsContiguous(strides)
	}
}

func BenchmarkShape_IsContiguous_False_2D(b *testing.B) {
	strides := []int{50, 1} // Non-contiguous strides
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = benchShape2D.IsContiguous(strides)
	}
}

func BenchmarkShape_IsContiguous_True_3D(b *testing.B) {
	strides := benchShape3D.Strides(nil)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = benchShape3D.IsContiguous(strides)
	}
}

func BenchmarkShape_IsContiguous_True_5D(b *testing.B) {
	strides := benchShape5D.Strides(nil)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = benchShape5D.IsContiguous(strides)
	}
}

// BenchmarkShape_ValidateAxes benchmarks ValidateAxes operation
func BenchmarkShape_ValidateAxes_Single(b *testing.B) {
	axes := []int{1}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		axesCopy := make([]int, len(axes))
		copy(axesCopy, axes)
		_ = benchShape3D.ValidateAxes(axesCopy)
	}
}

func BenchmarkShape_ValidateAxes_Multiple(b *testing.B) {
	axes := []int{0, 2}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		axesCopy := make([]int, len(axes))
		copy(axesCopy, axes)
		_ = benchShape3D.ValidateAxes(axesCopy)
	}
}

func BenchmarkShape_ValidateAxes_All(b *testing.B) {
	axes := []int{0, 1, 2}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		axesCopy := make([]int, len(axes))
		copy(axesCopy, axes)
		_ = benchShape3D.ValidateAxes(axesCopy)
	}
}

func BenchmarkShape_ValidateAxes_Empty(b *testing.B) {
	axes := []int{}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		axesCopy := make([]int, len(axes))
		copy(axesCopy, axes)
		_ = benchShape3D.ValidateAxes(axesCopy)
	}
}

func BenchmarkShape_ValidateAxes_Many(b *testing.B) {
	axes := []int{0, 1, 2, 3, 4}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		axesCopy := make([]int, len(axes))
		copy(axesCopy, axes)
		_ = benchShape5D.ValidateAxes(axesCopy)
	}
}

// BenchmarkNewShape benchmarks NewShape constructor
func BenchmarkNewShape_1D(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = NewShape(1000)
	}
}

func BenchmarkNewShape_2D(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = NewShape(100, 100)
	}
}

func BenchmarkNewShape_5D(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = NewShape(5, 5, 5, 5, 5)
	}
}

func BenchmarkNewShape_16D(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = NewShape(2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2)
	}
}
