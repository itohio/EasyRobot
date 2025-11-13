package generics

import (
	"math"
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

// BenchmarkElements_Naive benchmarks naive implementation (baseline)
func BenchmarkElements_Naive(b *testing.B) {
	shape := []int{100, 100}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		count := 0
		for indices := range elementsNaive(shape) {
			_ = indices
			count++
		}
		_ = count
	}
}

// BenchmarkElements_DirectLoop benchmarks direct nested loop (baseline)
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
// Uses complex math computation: pow(idx, 1.5) * sin(idx)
func BenchmarkElementsVec_Generic(b *testing.B) {
	n := 10000
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sum := 0.0
		for idx := range ElementsVec(n) {
			sum += math.Pow(float64(idx), 1.5) * math.Sin(float64(idx))
		}
		_ = sum
	}
}

// BenchmarkElementsVec_NonGeneric benchmarks non-generic vector iterator
func BenchmarkElementsVec_NonGeneric(b *testing.B) {
	n := 10000
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sum := 0.0
		for idx := range elementsVecNonGeneric(n) {
			sum += math.Pow(float64(idx), 1.5) * math.Sin(float64(idx))
		}
		_ = sum
	}
}

// BenchmarkElementsVec_DirectLoop benchmarks direct loop
func BenchmarkElementsVec_DirectLoop(b *testing.B) {
	n := 10000
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sum := 0.0
		for idx := 0; idx < n; idx++ {
			sum += math.Pow(float64(idx), 1.5) * math.Sin(float64(idx))
		}
		_ = sum
	}
}

// BenchmarkElementsVecStrided_Generic benchmarks generic ElementsVecStrided iterator
func BenchmarkElementsVecStrided_Generic(b *testing.B) {
	n := 10000
	stride := 1
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sum := 0.0
		for idx := range ElementsVecStrided(n, stride) {
			sum += math.Pow(float64(idx), 1.5) * math.Sin(float64(idx))
		}
		_ = sum
	}
}

// BenchmarkElementsVecStrided_NonGeneric benchmarks non-generic strided vector iterator
func BenchmarkElementsVecStrided_NonGeneric(b *testing.B) {
	n := 10000
	stride := 1
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sum := 0.0
		for idx := range elementsVecStridedNonGeneric(n, stride) {
			sum += math.Pow(float64(idx), 1.5) * math.Sin(float64(idx))
		}
		_ = sum
	}
}

// BenchmarkElementsMat_Generic benchmarks generic ElementsMat iterator
// Uses complex math computation: sqrt(row*row + col*col) * exp(-row/10)
func BenchmarkElementsMat_Generic(b *testing.B) {
	rows, cols := 100, 100
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sum := 0.0
		for idx := range ElementsMat(rows, cols) {
			row, col := float64(idx[0]), float64(idx[1])
			sum += math.Sqrt(row*row+col*col) * math.Exp(-row/10.0)
		}
		_ = sum
	}
}

// BenchmarkElementsMat_NonGeneric benchmarks non-generic matrix iterator
func BenchmarkElementsMat_NonGeneric(b *testing.B) {
	rows, cols := 100, 100
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sum := 0.0
		for idx := range elementsMatNonGeneric(rows, cols) {
			row, col := float64(idx[0]), float64(idx[1])
			sum += math.Sqrt(row*row+col*col) * math.Exp(-row/10.0)
		}
		_ = sum
	}
}

// BenchmarkElementsMat_DirectLoop benchmarks direct nested loop
func BenchmarkElementsMat_DirectLoop(b *testing.B) {
	rows, cols := 100, 100
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sum := 0.0
		for row := 0; row < rows; row++ {
			for col := 0; col < cols; col++ {
				r, c := float64(row), float64(col)
				sum += math.Sqrt(r*r+c*c) * math.Exp(-r/10.0)
			}
		}
		_ = sum
	}
}

// BenchmarkElementsMatStrided_Generic benchmarks generic ElementsMatStrided iterator
func BenchmarkElementsMatStrided_Generic(b *testing.B) {
	rows, cols, ld := 100, 100, 100
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sum := 0.0
		for idx := range ElementsMatStrided(rows, cols, ld) {
			row, col := float64(idx[0]), float64(idx[1])
			sum += math.Sqrt(row*row+col*col) * math.Exp(-row/10.0)
		}
		_ = sum
	}
}

// BenchmarkElementsMatStrided_NonGeneric benchmarks non-generic strided matrix iterator
func BenchmarkElementsMatStrided_NonGeneric(b *testing.B) {
	rows, cols, ld := 100, 100, 100
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sum := 0.0
		for idx := range elementsMatStridedNonGeneric(rows, cols, ld) {
			row, col := float64(idx[0]), float64(idx[1])
			sum += math.Sqrt(row*row+col*col) * math.Exp(-row/10.0)
		}
		_ = sum
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

// Naive implementation of Elements (baseline - allocates on each iteration)
func elementsNaive(shape []int) func(func([]int) bool) {
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
			// Allocate new slice on each iteration (naive approach)
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

// Naive implementation of ElementsWindow (baseline - direct nested loops)
func elementsWindowNaive(
	windowOffset, windowShape, parentShape []int,
) func(func([]int, bool) bool) {
	if len(windowShape) == 0 || len(parentShape) == 0 {
		return func(yield func([]int, bool) bool) {
		}
	}
	if len(windowOffset) != len(parentShape) || len(windowShape) != len(parentShape) {
		return func(yield func([]int, bool) bool) {
		}
	}
	return func(yield func([]int, bool) bool) {
		// Direct nested loops for 2D case (most common)
		if len(windowShape) == 2 {
			for kh := 0; kh < windowShape[0]; kh++ {
				for kw := 0; kw < windowShape[1]; kw++ {
					inH := windowOffset[0] + kh
					inW := windowOffset[1] + kw
					isValid := inH >= 0 && inH < parentShape[0] && inW >= 0 && inW < parentShape[1]
					if !yield([]int{inH, inW}, isValid) {
						return
					}
				}
			}
		} else {
			// Generic case using recursion-like approach
			indices := make([]int, len(windowShape))
			for {
				absIndices := make([]int, len(parentShape))
				isValid := true
				for i := range parentShape {
					absPos := windowOffset[i] + indices[i]
					absIndices[i] = absPos
					if absPos < 0 || absPos >= parentShape[i] {
						isValid = false
					}
				}
				if !yield(absIndices, isValid) {
					return
				}
				// Advance indices
				advanced := false
				for i := len(indices) - 1; i >= 0; i-- {
					indices[i]++
					if indices[i] < windowShape[i] {
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
}

// Naive implementation of ElementsWindows (baseline - direct nested loops)
func elementsWindowsNaive(
	outputShape, kernelShape, inputShape []int,
	stride, padding []int,
) func(func([]int, []int, bool) bool) {
	if len(outputShape) == 0 || len(kernelShape) == 0 || len(inputShape) == 0 {
		return func(yield func([]int, []int, bool) bool) {
		}
	}
	if len(outputShape) != len(inputShape) || len(kernelShape) != len(inputShape) {
		return func(yield func([]int, []int, bool) bool) {
		}
	}
	if len(stride) != len(inputShape) || len(padding) != len(inputShape) {
		return func(yield func([]int, []int, bool) bool) {
		}
	}
	return func(yield func([]int, []int, bool) bool) {
		// Direct nested loops for 2D case (most common for convolutions)
		if len(outputShape) == 2 {
			for outH := 0; outH < outputShape[0]; outH++ {
				for outW := 0; outW < outputShape[1]; outW++ {
					for kh := 0; kh < kernelShape[0]; kh++ {
						for kw := 0; kw < kernelShape[1]; kw++ {
							inH := outH*stride[0] + kh - padding[0]
							inW := outW*stride[1] + kw - padding[1]
							isValid := inH >= 0 && inH < inputShape[0] && inW >= 0 && inW < inputShape[1]
							if !yield([]int{outH, outW}, []int{inH, inW}, isValid) {
								return
							}
						}
					}
				}
			}
		} else {
			// Generic case - iterate over output positions, then kernel positions
			outIndices := make([]int, len(outputShape))
			for {
				// Iterate over kernel positions for this output position
				kernelIndices := make([]int, len(kernelShape))
				for {
					// Calculate input position
					inputIndices := make([]int, len(inputShape))
					isValid := true
					for i := range inputShape {
						inPos := outIndices[i]*stride[i] + kernelIndices[i] - padding[i]
						inputIndices[i] = inPos
						if inPos < 0 || inPos >= inputShape[i] {
							isValid = false
						}
					}
					outCopy := make([]int, len(outIndices))
					copy(outCopy, outIndices)
					if !yield(outCopy, inputIndices, isValid) {
						return
					}
					// Advance kernel indices
					advanced := false
					for i := len(kernelIndices) - 1; i >= 0; i-- {
						kernelIndices[i]++
						if kernelIndices[i] < kernelShape[i] {
							advanced = true
							break
						}
						kernelIndices[i] = 0
					}
					if !advanced {
						break
					}
				}
				// Advance output indices
				advanced := false
				for i := len(outIndices) - 1; i >= 0; i-- {
					outIndices[i]++
					if outIndices[i] < outputShape[i] {
						advanced = true
						break
					}
					outIndices[i] = 0
				}
				if !advanced {
					break
				}
			}
		}
	}
}

// BenchmarkElementsWindow_Generic benchmarks generic ElementsWindow iterator
func BenchmarkElementsWindow_Generic(b *testing.B) {
	windowOffset := []int{0, 0}
	windowShape := []int{3, 3}
	parentShape := []int{10, 10}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		count := 0
		for absIndices, isValid := range ElementsWindow(windowOffset, windowShape, parentShape) {
			_ = absIndices
			_ = isValid
			count++
		}
		_ = count
	}
}

// BenchmarkElementsWindow_Naive benchmarks naive implementation (baseline)
func BenchmarkElementsWindow_Naive(b *testing.B) {
	windowOffset := []int{0, 0}
	windowShape := []int{3, 3}
	parentShape := []int{10, 10}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		count := 0
		iter := elementsWindowNaive(windowOffset, windowShape, parentShape)
		iter(func(absIndices []int, isValid bool) bool {
			_ = absIndices
			_ = isValid
			count++
			return true
		})
		_ = count
	}
}

// BenchmarkElementsWindow_DirectLoop benchmarks direct nested loop (baseline)
func BenchmarkElementsWindow_DirectLoop(b *testing.B) {
	windowOffset := []int{0, 0}
	windowShape := []int{3, 3}
	parentShape := []int{10, 10}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		count := 0
		for kh := 0; kh < windowShape[0]; kh++ {
			for kw := 0; kw < windowShape[1]; kw++ {
				inH := windowOffset[0] + kh
				inW := windowOffset[1] + kw
				isValid := inH >= 0 && inH < parentShape[0] && inW >= 0 && inW < parentShape[1]
				_ = []int{inH, inW}
				_ = isValid
				count++
			}
		}
		_ = count
	}
}

// BenchmarkElementsWindows_Generic benchmarks generic ElementsWindows iterator
func BenchmarkElementsWindows_Generic(b *testing.B) {
	outputShape := []int{8, 8} // (10-3)/1+1 = 8
	kernelShape := []int{3, 3}
	inputShape := []int{10, 10}
	stride := []int{1, 1}
	padding := []int{0, 0}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		count := 0
		iter := ElementsWindows(outputShape, kernelShape, inputShape, stride, padding)
		iter(func(outIdx, inIdx []int, isValid bool) bool {
			_ = outIdx
			_ = inIdx
			_ = isValid
			count++
			return true
		})
		_ = count
	}
}

// BenchmarkElementsWindows_Naive benchmarks naive implementation (baseline)
func BenchmarkElementsWindows_Naive(b *testing.B) {
	outputShape := []int{8, 8} // (10-3)/1+1 = 8
	kernelShape := []int{3, 3}
	inputShape := []int{10, 10}
	stride := []int{1, 1}
	padding := []int{0, 0}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		count := 0
		iter := elementsWindowsNaive(outputShape, kernelShape, inputShape, stride, padding)
		iter(func(outIdx, inIdx []int, isValid bool) bool {
			_ = outIdx
			_ = inIdx
			_ = isValid
			count++
			return true
		})
		_ = count
	}
}

// BenchmarkElementsWindows_DirectLoop benchmarks direct nested loop (baseline)
func BenchmarkElementsWindows_DirectLoop(b *testing.B) {
	outputShape := []int{8, 8} // (10-3)/1+1 = 8
	kernelShape := []int{3, 3}
	inputShape := []int{10, 10}
	stride := []int{1, 1}
	padding := []int{0, 0}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		count := 0
		for outH := 0; outH < outputShape[0]; outH++ {
			for outW := 0; outW < outputShape[1]; outW++ {
				for kh := 0; kh < kernelShape[0]; kh++ {
					for kw := 0; kw < kernelShape[1]; kw++ {
						inH := outH*stride[0] + kh - padding[0]
						inW := outW*stride[1] + kw - padding[1]
						isValid := inH >= 0 && inH < inputShape[0] && inW >= 0 && inW < inputShape[1]
						_ = []int{outH, outW}
						_ = []int{inH, inW}
						_ = isValid
						count++
					}
				}
			}
		}
		_ = count
	}
}

// BenchmarkElementsWindows_Large benchmarks ElementsWindows with larger tensors
func BenchmarkElementsWindows_Large(b *testing.B) {
	outputShape := []int{28, 28} // Typical conv output
	kernelShape := []int{3, 3}
	inputShape := []int{30, 30}
	stride := []int{1, 1}
	padding := []int{0, 0}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		count := 0
		iter := ElementsWindows(outputShape, kernelShape, inputShape, stride, padding)
		iter(func(outIdx, inIdx []int, isValid bool) bool {
			_ = outIdx
			_ = inIdx
			_ = isValid
			count++
			return true
		})
		_ = count
	}
}

// BenchmarkElementsWindows_Large_Naive benchmarks naive implementation with larger tensors
func BenchmarkElementsWindows_Large_Naive(b *testing.B) {
	outputShape := []int{28, 28} // Typical conv output
	kernelShape := []int{3, 3}
	inputShape := []int{30, 30}
	stride := []int{1, 1}
	padding := []int{0, 0}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		count := 0
		iter := elementsWindowsNaive(outputShape, kernelShape, inputShape, stride, padding)
		iter(func(outIdx, inIdx []int, isValid bool) bool {
			_ = outIdx
			_ = inIdx
			_ = isValid
			count++
			return true
		})
		_ = count
	}
}

// BenchmarkElementsWindows_Large_DirectLoop benchmarks direct loop with larger tensors
func BenchmarkElementsWindows_Large_DirectLoop(b *testing.B) {
	outputShape := []int{28, 28} // Typical conv output
	kernelShape := []int{3, 3}
	inputShape := []int{30, 30}
	stride := []int{1, 1}
	padding := []int{0, 0}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		count := 0
		for outH := 0; outH < outputShape[0]; outH++ {
			for outW := 0; outW < outputShape[1]; outW++ {
				for kh := 0; kh < kernelShape[0]; kh++ {
					for kw := 0; kw < kernelShape[1]; kw++ {
						inH := outH*stride[0] + kh - padding[0]
						inW := outW*stride[1] + kw - padding[1]
						isValid := inH >= 0 && inH < inputShape[0] && inW >= 0 && inW < inputShape[1]
						_ = []int{outH, outW}
						_ = []int{inH, inW}
						_ = isValid
						count++
					}
				}
			}
		}
		_ = count
	}
}

// BenchmarkElementsWindows_WithPadding benchmarks ElementsWindows with padding
func BenchmarkElementsWindows_WithPadding(b *testing.B) {
	outputShape := []int{10, 10} // (8+2*1-3)/1+1 = 8, but let's use 10 for padding
	kernelShape := []int{3, 3}
	inputShape := []int{8, 8}
	stride := []int{1, 1}
	padding := []int{1, 1}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		count := 0
		iter := ElementsWindows(outputShape, kernelShape, inputShape, stride, padding)
		iter(func(outIdx, inIdx []int, isValid bool) bool {
			_ = outIdx
			_ = inIdx
			_ = isValid
			count++
			return true
		})
		_ = count
	}
}

// BenchmarkElementsWindows_WithPadding_Naive benchmarks naive with padding
func BenchmarkElementsWindows_WithPadding_Naive(b *testing.B) {
	outputShape := []int{10, 10} // (8+2*1-3)/1+1 = 8, but let's use 10 for padding
	kernelShape := []int{3, 3}
	inputShape := []int{8, 8}
	stride := []int{1, 1}
	padding := []int{1, 1}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		count := 0
		iter := elementsWindowsNaive(outputShape, kernelShape, inputShape, stride, padding)
		iter(func(outIdx, inIdx []int, isValid bool) bool {
			_ = outIdx
			_ = inIdx
			_ = isValid
			count++
			return true
		})
		_ = count
	}
}

// BenchmarkElementsWindows_Stride2 benchmarks ElementsWindows with stride 2
func BenchmarkElementsWindows_Stride2(b *testing.B) {
	outputShape := []int{4, 4} // (8-3)/2+1 = 3, but let's use 4
	kernelShape := []int{3, 3}
	inputShape := []int{8, 8}
	stride := []int{2, 2}
	padding := []int{0, 0}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		count := 0
		iter := ElementsWindows(outputShape, kernelShape, inputShape, stride, padding)
		iter(func(outIdx, inIdx []int, isValid bool) bool {
			_ = outIdx
			_ = inIdx
			_ = isValid
			count++
			return true
		})
		_ = count
	}
}

// BenchmarkElementsWindows_Stride2_Naive benchmarks naive with stride 2
func BenchmarkElementsWindows_Stride2_Naive(b *testing.B) {
	outputShape := []int{4, 4} // (8-3)/2+1 = 3, but let's use 4
	kernelShape := []int{3, 3}
	inputShape := []int{8, 8}
	stride := []int{2, 2}
	padding := []int{0, 0}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		count := 0
		iter := elementsWindowsNaive(outputShape, kernelShape, inputShape, stride, padding)
		iter(func(outIdx, inIdx []int, isValid bool) bool {
			_ = outIdx
			_ = inIdx
			_ = isValid
			count++
			return true
		})
		_ = count
	}
}

// BenchmarkElementsIndices_AllDims benchmarks ElementsIndices with all dimensions
func BenchmarkElementsIndices_AllDims(b *testing.B) {
	shape := []int{100, 100}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		count := 0
		for indices := range ElementsIndices(shape) {
			_ = indices
			count++
		}
		_ = count
	}
}

// BenchmarkElementsIndices_AllDims_CompareElements benchmarks comparison with Elements
func BenchmarkElementsIndices_AllDims_CompareElements(b *testing.B) {
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

// BenchmarkElementsIndices_SingleDim benchmarks ElementsIndices with single dimension
func BenchmarkElementsIndices_SingleDim(b *testing.B) {
	shape := []int{100, 100, 100}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		count := 0
		for indices := range ElementsIndices(shape, 0, 0) {
			_ = indices
			count++
		}
		_ = count
	}
}

// BenchmarkElementsIndices_TwoDims benchmarks ElementsIndices with two dimensions
func BenchmarkElementsIndices_TwoDims(b *testing.B) {
	shape := []int{100, 100, 100}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		count := 0
		for indices := range ElementsIndices(shape, 0, 1) {
			_ = indices
			count++
		}
		_ = count
	}
}

// BenchmarkElementsIndices_ThreeDims benchmarks ElementsIndices with three dimensions fixed
func BenchmarkElementsIndices_ThreeDims(b *testing.B) {
	shape := []int{50, 50, 50}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		count := 0
		for indices := range ElementsIndices(shape, 0, 1, 2, 2) {
			_ = indices
			count++
		}
		_ = count
	}
}

// BenchmarkElementsIndices_OutOfOrder benchmarks ElementsIndices with dimensions out of order
func BenchmarkElementsIndices_OutOfOrder(b *testing.B) {
	shape := []int{100, 100, 100}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		count := 0
		for indices := range ElementsIndices(shape, 2, 0) {
			_ = indices
			count++
		}
		_ = count
	}
}

// BenchmarkElementsIndices_LargeShape benchmarks ElementsIndices with larger shape
func BenchmarkElementsIndices_LargeShape(b *testing.B) {
	shape := []int{200, 200}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		count := 0
		for indices := range ElementsIndices(shape) {
			_ = indices
			count++
		}
		_ = count
	}
}

// BenchmarkElementsIndices_HighRank benchmarks ElementsIndices with high rank
func BenchmarkElementsIndices_HighRank(b *testing.B) {
	shape := []int{10, 10, 10, 10}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		count := 0
		for indices := range ElementsIndices(shape, 0, 2) {
			_ = indices
			count++
		}
		_ = count
	}
}

// Naive implementation of ElementsIndices (baseline - allocates on each iteration)
func elementsIndicesNaive(shape []int, dims ...int) func(func([]int) bool) {
	rank := len(shape)
	if rank == 0 {
		return func(yield func([]int) bool) {
			yield([]int{})
		}
	}

	// If no dims specified, iterate over all dimensions
	if len(dims) == 0 {
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
				// Allocate new slice on each iteration (naive approach)
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

	// Validate dimensions
	for _, dim := range dims {
		if dim < 0 || dim >= rank {
			return func(yield func([]int) bool) {
			}
		}
	}

	// Check for duplicates
	for i := 0; i < len(dims); i++ {
		for j := i + 1; j < len(dims); j++ {
			if dims[i] == dims[j] {
				return func(yield func([]int) bool) {
				}
			}
		}
	}

	// Build selected shape
	selectedShape := make([]int, len(dims))
	selectedSize := 1
	for i, dim := range dims {
		dimSize := shape[dim]
		if dimSize <= 0 {
			return func(yield func([]int) bool) {
			}
		}
		selectedShape[i] = dimSize
		selectedSize *= dimSize
	}

	if selectedSize == 0 {
		return func(yield func([]int) bool) {
		}
	}

	return func(yield func([]int) bool) {
		indices := make([]int, len(dims))
		for {
			// Allocate new slice on each iteration (naive approach)
			indicesCopy := make([]int, len(indices))
			copy(indicesCopy, indices)
			if !yield(indicesCopy) {
				return
			}
			advanced := false
			for i := len(indices) - 1; i >= 0; i-- {
				dim := dims[i]
				indices[i]++
				if indices[i] < shape[dim] {
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

// BenchmarkElementsIndices_Naive benchmarks naive implementation (baseline)
func BenchmarkElementsIndices_Naive(b *testing.B) {
	shape := []int{100, 100}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		count := 0
		iter := elementsIndicesNaive(shape)
		iter(func(indices []int) bool {
			_ = indices
			count++
			return true
		})
		_ = count
	}
}

// BenchmarkElementsIndices_SingleDim_Naive benchmarks naive with single dimension
func BenchmarkElementsIndices_SingleDim_Naive(b *testing.B) {
	shape := []int{100, 100, 100}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		count := 0
		iter := elementsIndicesNaive(shape, 0)
		iter(func(indices []int) bool {
			_ = indices
			count++
			return true
		})
		_ = count
	}
}

// BenchmarkElementsIndices_TwoDims_Naive benchmarks naive with two dimensions
func BenchmarkElementsIndices_TwoDims_Naive(b *testing.B) {
	shape := []int{100, 100, 100}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		count := 0
		iter := elementsIndicesNaive(shape, 0, 1)
		iter(func(indices []int) bool {
			_ = indices
			count++
			return true
		})
		_ = count
	}
}

// BenchmarkElementsIndicesStrided_AllDims benchmarks ElementsIndicesStrided with all dimensions
func BenchmarkElementsIndicesStrided_AllDims(b *testing.B) {
	shape := []int{100, 100}
	strides := []int{100, 1}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		count := 0
		for indices := range ElementsIndicesStrided(shape, strides) {
			_ = indices
			count++
		}
		_ = count
	}
}

// BenchmarkElementsIndicesStrided_AllDims_CompareElementsStrided benchmarks comparison with ElementsStrided
func BenchmarkElementsIndicesStrided_AllDims_CompareElementsStrided(b *testing.B) {
	shape := []int{100, 100}
	strides := []int{100, 1}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		count := 0
		for indices := range ElementsStrided(shape, strides) {
			_ = indices
			count++
		}
		_ = count
	}
}

// BenchmarkElementsIndicesStrided_SingleDim benchmarks ElementsIndicesStrided with single dimension
func BenchmarkElementsIndicesStrided_SingleDim(b *testing.B) {
	shape := []int{100, 100, 100}
	strides := []int{10000, 100, 1}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		count := 0
		for indices := range ElementsIndicesStrided(shape, strides, 0) {
			_ = indices
			count++
		}
		_ = count
	}
}

// BenchmarkElementsIndicesStrided_TwoDims benchmarks ElementsIndicesStrided with two dimensions
func BenchmarkElementsIndicesStrided_TwoDims(b *testing.B) {
	shape := []int{100, 100, 100}
	strides := []int{10000, 100, 1}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		count := 0
		for indices := range ElementsIndicesStrided(shape, strides, 0, 1) {
			_ = indices
			count++
		}
		_ = count
	}
}

// BenchmarkElementsIndicesStrided_NonContiguous benchmarks ElementsIndicesStrided with non-contiguous strides
func BenchmarkElementsIndicesStrided_NonContiguous(b *testing.B) {
	shape := []int{100, 100}
	strides := []int{200, 2} // Non-contiguous
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		count := 0
		for indices := range ElementsIndicesStrided(shape, strides) {
			_ = indices
			count++
		}
		_ = count
	}
}

// BenchmarkElementsIndicesStrided_LargeShape benchmarks ElementsIndicesStrided with larger shape
func BenchmarkElementsIndicesStrided_LargeShape(b *testing.B) {
	shape := []int{200, 200}
	strides := []int{400, 2} // Non-contiguous
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		count := 0
		for indices := range ElementsIndicesStrided(shape, strides) {
			_ = indices
			count++
		}
		_ = count
	}
}
