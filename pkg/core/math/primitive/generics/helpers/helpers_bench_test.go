package helpers

import (
	"testing"
)

var (
	benchShape1D  = []int{1000}
	benchShape2D  = []int{100, 100}
	benchShape3D  = []int{20, 20, 20}
	benchShape4D  = []int{10, 10, 10, 10}
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

func BenchmarkAdvanceOffsets_2D(b *testing.B) {
	shape := benchShape2D
	stridesDst := []int{100, 1}
	stridesSrc := []int{100, 1}
	var indices [MAX_DIMS]int
	var offsets [2]int
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// Reset for each iteration
		for j := range indices {
			indices[j] = 0
		}
		offsets[0] = 0
		offsets[1] = 0
		// Advance through a few elements
		for j := 0; j < 100 && AdvanceOffsets(shape, indices[:len(shape)], offsets[:], stridesDst, stridesSrc); j++ {
		}
	}
}

func BenchmarkAdvanceOffsets_3D(b *testing.B) {
	shape := benchShape3D
	stridesDst := []int{400, 20, 1}
	stridesSrc := []int{400, 20, 1}
	var indices [MAX_DIMS]int
	var offsets [2]int
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// Reset for each iteration
		for j := range indices {
			indices[j] = 0
		}
		offsets[0] = 0
		offsets[1] = 0
		// Advance through a few elements
		for j := 0; j < 100 && AdvanceOffsets(shape, indices[:len(shape)], offsets[:], stridesDst, stridesSrc); j++ {
		}
	}
}

func BenchmarkComputeStrideOffset_2D(b *testing.B) {
	indices := []int{50, 50}
	strides := []int{100, 1}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ComputeStrideOffset(indices, strides)
	}
}

func BenchmarkComputeStrideOffset_3D(b *testing.B) {
	indices := []int{10, 10, 10}
	strides := []int{400, 20, 1}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ComputeStrideOffset(indices, strides)
	}
}

func BenchmarkSizeFromShape_2D(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		SizeFromShape(benchShape2D)
	}
}

func BenchmarkSizeFromShape_3D(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		SizeFromShape(benchShape3D)
	}
}

func BenchmarkSizeFromShape_MAX_DIMS(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		SizeFromShape(benchShapeMax)
	}
}

func BenchmarkIterateOffsets_2D(b *testing.B) {
	stridesDst := []int{100, 1}
	stridesSrc := []int{100, 1}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		count := 0
		IterateOffsets(benchShape2D, stridesDst, stridesSrc, func(offsets []int) {
			count++
		})
	}
}

func BenchmarkIterateOffsets_3D(b *testing.B) {
	stridesDst := []int{400, 20, 1}
	stridesSrc := []int{400, 20, 1}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		count := 0
		IterateOffsets(benchShape3D, stridesDst, stridesSrc, func(offsets []int) {
			count++
		})
	}
}

func BenchmarkIterateOffsetsWithIndices_2D(b *testing.B) {
	stridesDst := []int{100, 1}
	stridesSrc := []int{100, 1}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		count := 0
		IterateOffsetsWithIndices(benchShape2D, stridesDst, stridesSrc, func(indices []int, offsets []int) {
			count++
		})
	}
}

// Benchmark comparison: old vs new AdvanceOffsets signature
func BenchmarkAdvanceOffsets_OldSignature(b *testing.B) {
	shape := benchShape2D
	stridesDst := []int{100, 1}
	stridesSrc := []int{100, 1}
	strideSet := [][]int{stridesDst, stridesSrc}
	var indices [MAX_DIMS]int
	var offsets [2]int
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// Reset for each iteration
		for j := range indices {
			indices[j] = 0
		}
		offsets[0] = 0
		offsets[1] = 0
		// Simulate old signature behavior
		for j := 0; j < 100; j++ {
			indices[1]++
			offsets[0] += strideSet[0][1]
			offsets[1] += strideSet[1][1]
			if indices[1] >= shape[1] {
				offsets[0] -= strideSet[0][1] * shape[1]
				offsets[1] -= strideSet[1][1] * shape[1]
				indices[1] = 0
				indices[0]++
				offsets[0] += strideSet[0][0]
				offsets[1] += strideSet[1][0]
				if indices[0] >= shape[0] {
					break
				}
			}
		}
	}
}

func BenchmarkAdvanceOffsets_NewSignature(b *testing.B) {
	shape := benchShape2D
	stridesDst := []int{100, 1}
	stridesSrc := []int{100, 1}
	var indices [MAX_DIMS]int
	var offsets [2]int
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// Reset for each iteration
		for j := range indices {
			indices[j] = 0
		}
		offsets[0] = 0
		offsets[1] = 0
		// Use new signature
		for j := 0; j < 100 && AdvanceOffsets(shape, indices[:len(shape)], offsets[:], stridesDst, stridesSrc); j++ {
		}
	}
}
