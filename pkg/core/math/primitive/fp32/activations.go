package fp32

import (
	"math"
)

const float32ExpMax = 88.0 // max value for exp to avoid overflow

// ReLU applies the Rectified Linear Unit activation function: dst[i] = max(0, src[i])
// dst and src can be the same slice for in-place operation.
func ReLU(dst, src []float32, size int) {
	if size <= 0 {
		return
	}

	for i := 0; i < size; i++ {
		if src[i] > 0 {
			dst[i] = src[i]
		} else {
			dst[i] = 0
		}
	}
}

// ReLUGrad computes the ReLU gradient: dst[i] = gradOutput[i] * (input[i] > 0 ? 1 : 0)
// dst, gradOutput, and input can be the same slice for in-place operation.
func ReLUGrad(dst, gradOutput, input []float32, size int) {
	if size <= 0 {
		return
	}

	for i := 0; i < size; i++ {
		if input[i] > 0 {
			dst[i] = gradOutput[i]
		} else {
			dst[i] = 0
		}
	}
}

// Sigmoid applies the sigmoid activation function: dst[i] = 1 / (1 + exp(-src[i]))
// dst and src can be the same slice for in-place operation.
func Sigmoid(dst, src []float32, size int) {
	if size <= 0 {
		return
	}

	for i := 0; i < size; i++ {
		x := -src[i]
		if x > float32ExpMax {
			dst[i] = 0.0
		} else if x < -float32ExpMax {
			dst[i] = 1.0
		} else {
			dst[i] = 1.0 / (1.0 + float32(math.Exp(float64(x))))
		}
	}
}

// SigmoidGrad computes the sigmoid gradient: dst[i] = gradOutput[i] * output[i] * (1 - output[i])
// dst, gradOutput, and output can be the same slice for in-place operation.
func SigmoidGrad(dst, gradOutput, output []float32, size int) {
	if size <= 0 {
		return
	}

	for i := 0; i < size; i++ {
		dst[i] = gradOutput[i] * output[i] * (1 - output[i])
	}
}

// Tanh applies the hyperbolic tangent activation function: dst[i] = tanh(src[i])
// dst and src can be the same slice for in-place operation.
func Tanh(dst, src []float32, size int) {
	if size <= 0 {
		return
	}

	for i := 0; i < size; i++ {
		x := float64(src[i])
		dst[i] = float32(math.Tanh(x))
	}
}

// TanhGrad computes the tanh gradient: dst[i] = gradOutput[i] * (1 - output[i]^2)
// dst, gradOutput, and output can be the same slice for in-place operation.
func TanhGrad(dst, gradOutput, output []float32, size int) {
	if size <= 0 {
		return
	}

	for i := 0; i < size; i++ {
		dst[i] = gradOutput[i] * (1 - output[i]*output[i])
	}
}

// Softmax1D applies softmax to a 1D array in-place: dst[i] = exp(dst[i] - max) / sum(exp(dst[j] - max))
func Softmax1D(dst []float32, size int) {
	if size <= 0 {
		return
	}

	// Find max value for numerical stability
	maxVal := dst[0]
	for i := 1; i < size; i++ {
		if dst[i] > maxVal {
			maxVal = dst[i]
		}
	}

	// Compute exp(x - max) and sum
	var sum float32
	for i := 0; i < size; i++ {
		dst[i] = float32(math.Exp(float64(dst[i] - maxVal)))
		sum += dst[i]
	}

	// Normalize
	if sum > 0 {
		invSum := 1.0 / sum
		for i := 0; i < size; i++ {
			dst[i] *= invSum
		}
	}
}

// Softmax2DRows applies softmax along rows (dim=0) of a 2D array.
// For each column j: dst[i][j] = exp(dst[i][j] - max) / sum(exp(dst[k][j] - max))
func Softmax2DRows(dst []float32, rows, cols int) {
	if rows <= 0 || cols <= 0 {
		return
	}

	for j := 0; j < cols; j++ {
		// Find max value in this column
		maxVal := dst[j]
		for i := 1; i < rows; i++ {
			val := dst[i*cols+j]
			if val > maxVal {
				maxVal = val
			}
		}

		// Compute exp(x - max) and sum for this column
		var sum float32
		for i := 0; i < rows; i++ {
			idx := i*cols + j
			val := dst[idx] - maxVal
			dst[idx] = float32(math.Exp(float64(val)))
			sum += dst[idx]
		}

		// Normalize this column
		if sum > 0 {
			invSum := 1.0 / sum
			for i := 0; i < rows; i++ {
				idx := i*cols + j
				dst[idx] *= invSum
			}
		}
	}
}

// Softmax2DCols applies softmax along columns (dim=1) of a 2D array.
// For each row i: dst[i][j] = exp(dst[i][j] - max) / sum(exp(dst[i][k] - max))
func Softmax2DCols(dst []float32, rows, cols int) {
	if rows <= 0 || cols <= 0 {
		return
	}

	for i := 0; i < rows; i++ {
		rowStart := i * cols

		// Find max value in this row
		maxVal := dst[rowStart]
		for j := 1; j < cols; j++ {
			val := dst[rowStart+j]
			if val > maxVal {
				maxVal = val
			}
		}

		// Compute exp(x - max) and sum for this row
		var sum float32
		for j := 0; j < cols; j++ {
			idx := rowStart + j
			val := dst[idx] - maxVal
			dst[idx] = float32(math.Exp(float64(val)))
			sum += dst[idx]
		}

		// Normalize this row
		if sum > 0 {
			invSum := 1.0 / sum
			for j := 0; j < cols; j++ {
				idx := rowStart + j
				dst[idx] *= invSum
			}
		}
	}
}

// Softmax1DGrad computes softmax gradient for 1D case.
// dst[i] = output[i] * (gradOutput[i] - sum(gradOutput[j] * output[j]))
func Softmax1DGrad(dst, gradOutput, output []float32, size int) {
	if size <= 0 {
		return
	}

	// Compute sum(gradOutput[j] * output[j])
	var sum float32
	for i := 0; i < size; i++ {
		sum += gradOutput[i] * output[i]
	}

	// Apply gradient: output[i] * (gradOutput[i] - sum)
	for i := 0; i < size; i++ {
		dst[i] = output[i] * (gradOutput[i] - sum)
	}
}

// Softmax2DRowsGrad computes softmax gradient along rows (dim=0).
// For each column j: dst[i][j] = output[i][j] * (gradOutput[i][j] - sum(gradOutput[k][j] * output[k][j]))
func Softmax2DRowsGrad(dst, gradOutput, output []float32, rows, cols int) {
	if rows <= 0 || cols <= 0 {
		return
	}

	for j := 0; j < cols; j++ {
		// Compute sum for this column
		var sum float32
		for i := 0; i < rows; i++ {
			idx := i*cols + j
			sum += gradOutput[idx] * output[idx]
		}

		// Apply gradient for this column
		for i := 0; i < rows; i++ {
			idx := i*cols + j
			dst[idx] = output[idx] * (gradOutput[idx] - sum)
		}
	}
}

// Softmax2DColsGrad computes softmax gradient along columns (dim=1).
// For each row i: dst[i][j] = output[i][j] * (gradOutput[i][j] - sum(gradOutput[i][k] * output[i][k]))
func Softmax2DColsGrad(dst, gradOutput, output []float32, rows, cols int) {
	if rows <= 0 || cols <= 0 {
		return
	}

	for i := 0; i < rows; i++ {
		rowStart := i * cols

		// Compute sum for this row
		var sum float32
		for j := 0; j < cols; j++ {
			idx := rowStart + j
			sum += gradOutput[idx] * output[idx]
		}

		// Apply gradient for this row
		for j := 0; j < cols; j++ {
			idx := rowStart + j
			dst[idx] = output[idx] * (gradOutput[idx] - sum)
		}
	}
}

// ElemMask applies element-wise mask multiplication: dst[i] = src[i] * mask[i]
// This is used for both dropout forward and backward operations.
func ElemMask(dst, src, mask []float32, size int) {
	HadamardProduct(dst, src, mask, size, 1, 1)
}
