package primitive

import (
	"github.com/chewxy/math32"
)

// Axpy computes: y = alpha*x + y
// This is BLAS AXPY operation
func Axpy(y, x []float32, strideY, strideX, n int, alpha float32) {
	if n == 0 {
		return
	}

	py := 0
	px := 0

	for i := 0; i < n; i++ {
		y[py] = alpha*x[px] + y[py]
		py += strideY
		px += strideX
	}
}

// Dot computes: dot = x^T * y
// This is BLAS DOT operation
func Dot(x, y []float32, strideX, strideY, n int) float32 {
	if n == 0 {
		return 0
	}

	acc := float32(0.0)
	px := 0
	py := 0

	for i := 0; i < n; i++ {
		acc += x[px] * y[py]
		px += strideX
		py += strideY
	}

	return acc
}

// Nrm2 computes: norm = ||x||_2 (Euclidean norm)
// This is BLAS NRM2 operation
func Nrm2(x []float32, stride, n int) float32 {
	if n == 0 {
		return 0
	}

	acc := float32(0.0)
	px := 0

	for i := 0; i < n; i++ {
		val := x[px]
		acc += val * val
		px += stride
	}

	return math32.Sqrt(acc)
}

// Asum computes: sum = ||x||_1 (L1 norm)
// This is BLAS ASUM operation
func Asum(x []float32, stride, n int) float32 {
	if n == 0 {
		return 0
	}

	acc := float32(0.0)
	px := 0

	for i := 0; i < n; i++ {
		val := x[px]
		if val < 0 {
			acc -= val
		} else {
			acc += val
		}
		px += stride
	}

	return acc
}

// Scal computes: x = alpha*x
// This is BLAS SCAL operation
func Scal(x []float32, stride, n int, alpha float32) {
	if n == 0 || alpha == 1.0 {
		return
	}

	px := 0

	for i := 0; i < n; i++ {
		x[px] *= alpha
		px += stride
	}
}

// Copy computes: y = x
// This is BLAS COPY operation
func Copy(y, x []float32, strideY, strideX, n int) {
	if n == 0 {
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

// Swap swaps x and y: x ↔ y
// This is BLAS SWAP operation
func Swap(x, y []float32, strideX, strideY, n int) {
	if n == 0 {
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

// Iamax returns the index of element with maximum absolute value
// This is BLAS IAMAX operation
func Iamax(x []float32, stride, n int) int {
	if n == 0 {
		return -1
	}

	maxVal := float32(0)
	maxIdx := 0
	px := 0

	for i := 0; i < n; i++ {
		val := x[px]
		if val < 0 {
			val = -val
		}
		if val > maxVal {
			maxVal = val
			maxIdx = i
		}
		px += stride
	}

	return maxIdx
}
