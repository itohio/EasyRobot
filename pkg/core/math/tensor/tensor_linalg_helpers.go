package tensor

import (
	"fmt"

	"github.com/itohio/EasyRobot/pkg/core/math/primitive/fp32"
)

// MatVecMulTransposed performs matrix-vector multiplication with matrix transposed: y = alpha*A^T*x + beta*y
// matrix: [M, N] tensor (will be transposed)
// vector: [M] tensor (input vector)
// result: [N] tensor (output vector)
// Uses fp32 primitive.Gemv_T internally.
func (t *Tensor) MatVecMulTransposed(matrix *Tensor, vector *Tensor, alpha, beta float32) *Tensor {
	if t == nil || matrix == nil || vector == nil {
		return nil
	}

	matrixShape := matrix.Shape()
	vectorShape := vector.Shape()

	// Validate shapes
	if len(matrixShape) != 2 {
		panic(fmt.Sprintf("tensor.MatVecMulTransposed: matrix must be 2D, got %v", matrixShape))
	}
	if len(vectorShape) != 1 {
		panic(fmt.Sprintf("tensor.MatVecMulTransposed: vector must be 1D, got %v", vectorShape))
	}
	if matrixShape[0] != vectorShape[0] {
		panic(fmt.Sprintf("tensor.MatVecMulTransposed: matrix rows (%d) must match vector length (%d)", matrixShape[0], vectorShape[0]))
	}

	M, N := matrixShape[0], matrixShape[1]

	// Ensure result tensor has correct shape
	var result *Tensor
	if len(t.Dim) == 1 && t.Dim[0] == N {
		// Reuse provided tensor if shape matches
		result = t
	} else {
		// Create new result tensor
		result = &Tensor{
			Dim:  []int{N},
			Data: make([]float32, N),
		}
	}

	// Leading dimension of matrix (row-major: number of columns)
	ldA := N

	// Use primitive.Gemv_T
	fp32.Gemv_T(
		result.Data, // y (output)
		matrix.Data, // A (matrix)
		vector.Data, // x (vector)
		ldA, M, N,   // leading dimension, rows, cols
		alpha, beta, // scaling factors
	)

	return result
}

// MatMulTransposed performs matrix multiplication with optional transposition.
// If transposeA is true: uses Gemm_TN (A^T @ B)
// If transposeB is true: uses Gemm_NT (A @ B^T)
// If both are false: uses Gemm_NN (A @ B)
// If both are true: uses Gemm_TT (A^T @ B^T)
// Returns result tensor (creates new one if dst is nil, otherwise uses dst).
func (t *Tensor) MatMulTransposed(other *Tensor, transposeA, transposeB bool, dst *Tensor) *Tensor {
	if t == nil || other == nil {
		return nil
	}

	tShape := t.Shape()
	otherShape := other.Shape()

	if len(tShape) < 2 || len(otherShape) < 2 {
		panic(fmt.Sprintf("tensor.MatMulTransposed: tensors must be at least 2D, got %v × %v", tShape, otherShape))
	}

	// Extract last 2 dimensions for matrix multiplication
	M := tShape[len(tShape)-2]
	K1 := tShape[len(tShape)-1]
	K2 := otherShape[len(otherShape)-2]
	N := otherShape[len(otherShape)-1]

	if transposeA {
		M, K1 = K1, M // Swap when transposing A
	}
	if transposeB {
		K2, N = N, K2 // Swap when transposing B
	}

	if K1 != K2 {
		panic(fmt.Sprintf("tensor.MatMulTransposed: dimension mismatch: K=%d vs K2=%d", K1, K2))
	}

	// Determine output shape (same batch dimensions as t)
	resultShape := make([]int, len(tShape))
	copy(resultShape, tShape)
	resultShape[len(resultShape)-2] = M
	resultShape[len(resultShape)-1] = N

	// Calculate leading dimensions
	ldA := tShape[len(tShape)-1]         // columns of A (or rows if transposed)
	ldB := otherShape[len(otherShape)-1] // columns of B (or rows if transposed)
	ldC := N                             // columns of result

	// Calculate output size
	outputSize := 1
	for _, d := range resultShape {
		outputSize *= d
	}

	// Prepare result tensor
	var result *Tensor
	if dst != nil {
		if len(dst.Dim) == 0 {
			dst.Dim = resultShape
			dst.Data = make([]float32, outputSize)
		}
		result = dst
	} else {
		result = &Tensor{
			Dim:  resultShape,
			Data: make([]float32, outputSize),
		}
	}

	// Handle 2D case
	if len(tShape) == 2 && len(otherShape) == 2 {
		switch {
		case !transposeA && !transposeB:
			// Gemm_NN: C = A @ B
			fp32.Gemm_NN(
				result.Data, t.Data, other.Data,
				ldC, ldA, ldB,
				M, N, K1,
				1.0, 0.0,
			)
		case !transposeA && transposeB:
			// Gemm_NT: C = A @ B^T
			fp32.Gemm_NT(
				result.Data, t.Data, other.Data,
				ldC, ldA, ldB,
				M, N, K1,
				1.0, 0.0,
			)
		case transposeA && !transposeB:
			// Gemm_TN: C = A^T @ B
			fp32.Gemm_TN(
				result.Data, t.Data, other.Data,
				ldC, ldA, ldB,
				M, N, K1,
				1.0, 0.0,
			)
		case transposeA && transposeB:
			// Gemm_TT: C = A^T @ B^T
			fp32.Gemm_TT(
				result.Data, t.Data, other.Data,
				ldC, ldA, ldB,
				M, N, K1,
				1.0, 0.0,
			)
		}
		return result
	}

	// Handle batched case (simplified - can be enhanced later)
	// For now, handle same batch size case
	if len(tShape) == len(otherShape) {
		batchSize := 1
		for i := 0; i < len(tShape)-2; i++ {
			if tShape[i] != otherShape[i] {
				panic(fmt.Sprintf("tensor.MatMulTransposed: batch dimension mismatch: %v vs %v", tShape, otherShape))
			}
			batchSize *= tShape[i]
		}

		tStride := M * K1
		otherStride := K2 * N
		resultStride := M * N

		for b := 0; b < batchSize; b++ {
			tOffset := b * tStride
			otherOffset := b * otherStride
			resultOffset := b * resultStride

			switch {
			case !transposeA && !transposeB:
				fp32.Gemm_NN(
					result.Data[resultOffset:],
					t.Data[tOffset:],
					other.Data[otherOffset:],
					ldC, ldA, ldB,
					M, N, K1,
					1.0, 0.0,
				)
			case !transposeA && transposeB:
				fp32.Gemm_NT(
					result.Data[resultOffset:],
					t.Data[tOffset:],
					other.Data[otherOffset:],
					ldC, ldA, ldB,
					M, N, K1,
					1.0, 0.0,
				)
			case transposeA && !transposeB:
				fp32.Gemm_TN(
					result.Data[resultOffset:],
					t.Data[tOffset:],
					other.Data[otherOffset:],
					ldC, ldA, ldB,
					M, N, K1,
					1.0, 0.0,
				)
			case transposeA && transposeB:
				fp32.Gemm_TT(
					result.Data[resultOffset:],
					t.Data[tOffset:],
					other.Data[otherOffset:],
					ldC, ldA, ldB,
					M, N, K1,
					1.0, 0.0,
				)
			}
		}
		return result
	}

	panic(fmt.Sprintf("tensor.MatMulTransposed: unsupported tensor shapes: %v × %v", tShape, otherShape))
}

// AddScaled adds a scaled tensor to this tensor in-place: t = t + alpha * other
// Uses fp32 primitive.Axpy internally.
// Returns the tensor itself for method chaining.
func (t *Tensor) AddScaled(other *Tensor, alpha float32) *Tensor {
	if t == nil || other == nil {
		return t
	}

	if !t.sameShape(other) {
		panic(fmt.Sprintf("tensor.AddScaled: shape mismatch: %v vs %v", t.Dim, other.Dim))
	}

	if !t.isContiguous() || !other.isContiguous() {
		// Handle strided case manually
		size := t.Size()
		for i := 0; i < size; i++ {
			t.Data[i] += alpha * other.Data[i]
		}
		return t
	}

	// Use primitive.Axpy for contiguous case
	size := t.Size()
	fp32.Axpy(t.Data, other.Data, 1, 1, size, alpha)
	return t
}
