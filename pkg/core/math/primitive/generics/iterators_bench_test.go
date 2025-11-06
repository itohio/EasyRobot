package generics

import (
	"testing"
)

// BenchmarkElements_Generic benchmarks generic Elements iterator
func BenchmarkElements_Generic(b *testing.B) {
	shape := []int{100, 100}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		count := 0
		for indices := range Elements(shape) {
			_ = indices
			count++
		}
		_ = count
	}
}

// BenchmarkElements_NonGeneric benchmarks non-generic iterator
func BenchmarkElements_NonGeneric(b *testing.B) {
	shape := []int{100, 100}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		count := 0
		for indices := range elementsNonGeneric(shape) {
			_ = indices
			count++
		}
		_ = count
	}
}

// BenchmarkElements_DirectLoop benchmarks direct nested loop
func BenchmarkElements_DirectLoop(b *testing.B) {
	rows, cols := 100, 100
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		count := 0
		for row := 0; row < rows; row++ {
			for col := 0; col < cols; col++ {
				_ = []int{row, col}
				count++
			}
		}
		_ = count
	}
}

// BenchmarkElementsVec_Generic benchmarks generic ElementsVec iterator
func BenchmarkElementsVec_Generic(b *testing.B) {
	n := 10000
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		count := 0
		for idx := range ElementsVec(n) {
			_ = idx
			count++
		}
		_ = count
	}
}

// BenchmarkElementsVec_NonGeneric benchmarks non-generic vector iterator
func BenchmarkElementsVec_NonGeneric(b *testing.B) {
	n := 10000
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		count := 0
		for idx := range elementsVecNonGeneric(n) {
			_ = idx
			count++
		}
		_ = count
	}
}

// BenchmarkElementsVec_DirectLoop benchmarks direct loop
func BenchmarkElementsVec_DirectLoop(b *testing.B) {
	n := 10000
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		count := 0
		for idx := 0; idx < n; idx++ {
			_ = idx
			count++
		}
		_ = count
	}
}

// BenchmarkElementsVecStrided_Generic benchmarks generic ElementsVecStrided iterator
func BenchmarkElementsVecStrided_Generic(b *testing.B) {
	n := 10000
	stride := 1
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		count := 0
		for idx := range ElementsVecStrided(n, stride) {
			_ = idx
			count++
		}
		_ = count
	}
}

// BenchmarkElementsVecStrided_NonGeneric benchmarks non-generic strided vector iterator
func BenchmarkElementsVecStrided_NonGeneric(b *testing.B) {
	n := 10000
	stride := 1
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		count := 0
		for idx := range elementsVecStridedNonGeneric(n, stride) {
			_ = idx
			count++
		}
		_ = count
	}
}

// BenchmarkElementsMat_Generic benchmarks generic ElementsMat iterator
func BenchmarkElementsMat_Generic(b *testing.B) {
	rows, cols := 100, 100
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		count := 0
		for idx := range ElementsMat(rows, cols) {
			_ = idx
			count++
		}
		_ = count
	}
}

// BenchmarkElementsMat_NonGeneric benchmarks non-generic matrix iterator
func BenchmarkElementsMat_NonGeneric(b *testing.B) {
	rows, cols := 100, 100
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		count := 0
		for idx := range elementsMatNonGeneric(rows, cols) {
			_ = idx
			count++
		}
		_ = count
	}
}

// BenchmarkElementsMat_DirectLoop benchmarks direct nested loop
func BenchmarkElementsMat_DirectLoop(b *testing.B) {
	rows, cols := 100, 100
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		count := 0
		for row := 0; row < rows; row++ {
			for col := 0; col < cols; col++ {
				_ = [2]int{row, col}
				count++
			}
		}
		_ = count
	}
}

// BenchmarkElementsMatStrided_Generic benchmarks generic ElementsMatStrided iterator
func BenchmarkElementsMatStrided_Generic(b *testing.B) {
	rows, cols, ld := 100, 100, 100
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		count := 0
		for idx := range ElementsMatStrided(rows, cols, ld) {
			_ = idx
			count++
		}
		_ = count
	}
}

// BenchmarkElementsMatStrided_NonGeneric benchmarks non-generic strided matrix iterator
func BenchmarkElementsMatStrided_NonGeneric(b *testing.B) {
	rows, cols, ld := 100, 100, 100
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		count := 0
		for idx := range elementsMatStridedNonGeneric(rows, cols, ld) {
			_ = idx
			count++
		}
		_ = count
	}
}

// Non-generic helper functions
func elementsNonGeneric(shape []int) func(func([]int) bool) {
	if len(shape) == 0 {
		return func(yield func([]int) bool) {
			yield([]int{})
		}
	}
	size := 1
	for _, dim := range shape {
		size *= dim
	}
	if size == 0 {
		return func(yield func([]int) bool) {
		}
	}
	return func(yield func([]int) bool) {
		indices := make([]int, len(shape))
		for {
			indicesCopy := make([]int, len(indices))
			copy(indicesCopy, indices)
			if !yield(indicesCopy) {
				return
			}
			advanced := false
			for i := len(indices) - 1; i >= 0; i-- {
				indices[i]++
				if indices[i] < shape[i] {
					advanced = true
					break
				}
				indices[i] = 0
			}
			if !advanced {
				break
			}
		}
	}
}

func elementsVecNonGeneric(n int) func(func(int) bool) {
	return func(yield func(int) bool) {
		for i := 0; i < n; i++ {
			if !yield(i) {
				return
			}
		}
	}
}

func elementsVecStridedNonGeneric(n int, stride int) func(func(int) bool) {
	return func(yield func(int) bool) {
		idx := 0
		for i := 0; i < n; i++ {
			if !yield(idx) {
				return
			}
			idx += stride
		}
	}
}

func elementsMatNonGeneric(rows, cols int) func(func([2]int) bool) {
	return func(yield func([2]int) bool) {
		for i := 0; i < rows; i++ {
			for j := 0; j < cols; j++ {
				if !yield([2]int{i, j}) {
					return
				}
			}
		}
	}
}

func elementsMatStridedNonGeneric(rows, cols int, ld int) func(func([2]int) bool) {
	return func(yield func([2]int) bool) {
		for i := 0; i < rows; i++ {
			for j := 0; j < cols; j++ {
				if !yield([2]int{i, j}) {
					return
				}
			}
		}
	}
}

