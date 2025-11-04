package eager_tensor

import (
	"fmt"

	"github.com/itohio/EasyRobot/pkg/core/math/primitive/fp32"
	"github.com/itohio/EasyRobot/pkg/core/math/tensor/types"
)

// MatVecMulTransposed performs matrix-vector multiplication with matrix transposed: y = alpha*A^T*x + beta*y
// matrix: [M, N] tensor (will be transposed)
// vector: [M] tensor (input vector)
// result: [N] tensor (output vector)
// Uses fp32 primitive.Gemv_T internally.
// NOTE: Does not guarantee returning a
func (t Tensor) MatVecMulTransposed(matrix types.Tensor, vector types.Tensor, alpha, beta float32) types.Tensor {
	if t.shape == nil || matrix == nil || matrix.Shape() == nil || vector == nil || vector.Shape() == nil {
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
	var result Tensor
	tShape := t.Shape()
	if len(tShape) == 1 && tShape[0] == N {
		// Reuse provided tensor if shape matches
		result = t
	} else {
		// Create new result tensor
		result = New(t.DataType(), types.NewShape(N))
	}
	resultPtr := &result

	// Leading dimension of matrix (row-major: number of columns)
	ldA := N

	// Use primitive.Gemv_T
	matrixData := types.GetTensorData[[]float32](matrix)
	vectorData := types.GetTensorData[[]float32](vector)
	resultData := types.GetTensorData[[]float32](resultPtr)
	fp32.Gemv_T(
		resultData, // y (output)
		matrixData, // A (matrix)
		vectorData, // x (vector)
		ldA, M, N,  // leading dimension, rows, cols
		alpha, beta, // scaling factors
	)

	return resultPtr
}

// MatMulTransposed performs matrix multiplication with optional transposition.
// If transposeA is true: uses Gemm_TN (A^T @ B)
// If transposeB is true: uses Gemm_NT (A @ B^T)
// If both are false: uses Gemm_NN (A @ B)
// If both are true: uses Gemm_TT (A^T @ B^T)
// Returns result tensor (creates new one if dst is nil, otherwise uses dst).
func (t Tensor) MatMulTransposed(other types.Tensor, transposeA, transposeB bool, dst types.Tensor) types.Tensor {
	if t.shape == nil || other == nil || other.Shape() == nil {
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
	var result Tensor
	if dst != nil {
		dstShape := dst.Shape()
		// Check if dst has correct shape, if not create new one
		if len(dstShape) == len(resultShape) {
			match := true
			for i := range dstShape {
				if dstShape[i] != resultShape[i] {
					match = false
					break
				}
			}
			if match {
				// Can use dst, but need to cast - this is a limitation
				// For now, create new tensor to avoid casting
				result = New(t.DataType(), types.NewShape(resultShape...))
			} else {
				result = New(t.DataType(), types.NewShape(resultShape...))
			}
		} else {
			result = New(t.DataType(), types.NewShape(resultShape...))
		}
	} else {
		result = New(t.DataType(), types.NewShape(resultShape...))
	}

	// Handle 2D case
	tData := types.GetTensorData[[]float32](&t)
	otherData := types.GetTensorData[[]float32](other)
	resultData := types.GetTensorData[[]float32](&result)

	if len(tShape) == 2 && len(otherShape) == 2 {
		switch {
		case !transposeA && !transposeB:
			// Gemm_NN: C = A @ B
			fp32.Gemm_NN(
				resultData, tData, otherData,
				ldC, ldA, ldB,
				M, N, K1,
				1.0, 0.0,
			)
		case !transposeA && transposeB:
			// Gemm_NT: C = A @ B^T
			fp32.Gemm_NT(
				resultData, tData, otherData,
				ldC, ldA, ldB,
				M, N, K1,
				1.0, 0.0,
			)
		case transposeA && !transposeB:
			// Gemm_TN: C = A^T @ B
			fp32.Gemm_TN(
				resultData, tData, otherData,
				ldC, ldA, ldB,
				M, N, K1,
				1.0, 0.0,
			)
		case transposeA && transposeB:
			// Gemm_TT: C = A^T @ B^T
			fp32.Gemm_TT(
				resultData, tData, otherData,
				ldC, ldA, ldB,
				M, N, K1,
				1.0, 0.0,
			)
		}
		// If dst was provided and has correct shape, copy result to dst
		if dst != nil {
			copyTensorData(result, dst)
			return dst
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

		tData := types.GetTensorData[[]float32](&t)
		otherData := types.GetTensorData[[]float32](other)
		resultData := types.GetTensorData[[]float32](&result)

		for b := 0; b < batchSize; b++ {
			tOffset := b * tStride
			otherOffset := b * otherStride
			resultOffset := b * resultStride

			switch {
			case !transposeA && !transposeB:
				fp32.Gemm_NN(
					resultData[resultOffset:],
					tData[tOffset:],
					otherData[otherOffset:],
					ldC, ldA, ldB,
					M, N, K1,
					1.0, 0.0,
				)
			case !transposeA && transposeB:
				fp32.Gemm_NT(
					resultData[resultOffset:],
					tData[tOffset:],
					otherData[otherOffset:],
					ldC, ldA, ldB,
					M, N, K1,
					1.0, 0.0,
				)
			case transposeA && !transposeB:
				fp32.Gemm_TN(
					resultData[resultOffset:],
					tData[tOffset:],
					otherData[otherOffset:],
					ldC, ldA, ldB,
					M, N, K1,
					1.0, 0.0,
				)
			case transposeA && transposeB:
				fp32.Gemm_TT(
					resultData[resultOffset:],
					tData[tOffset:],
					otherData[otherOffset:],
					ldC, ldA, ldB,
					M, N, K1,
					1.0, 0.0,
				)
			}
		}
		// If dst was provided and has correct shape, copy result to dst
		if dst != nil {
			copyTensorData(&result, dst)
			return dst
		}
		return &result
	}

	panic(fmt.Sprintf("tensor.MatMulTransposed: unsupported tensor shapes: %v × %v", tShape, otherShape))
}

// AddScaled adds a scaled tensor to this tensor in-place: t = t + alpha * other
// Uses fp32 primitive.Axpy internally.
// Returns the tensor itself for method chaining.
func (t Tensor) AddScaled(other types.Tensor, alpha float32) types.Tensor {
	if t.shape == nil || other == nil || other.Shape() == nil {
		return t
	}

	tVal := t
	if !tVal.Shape().Equal(other.Shape()) {
		panic(fmt.Sprintf("tensor.AddScaled: shape mismatch: %v vs %v", t.Shape(), other.Shape()))
	}

	tData := types.GetTensorData[[]float32](&t)
	otherData := types.GetTensorData[[]float32](other)

	if !t.isContiguous() || !isTensorContiguous(other) {
		// Handle strided case manually
		size := t.Size()
		for i := 0; i < size; i++ {
			tData[i] += alpha * otherData[i]
		}
		return &t
	}

	// Use primitive.Axpy for contiguous case
	size := t.Size()
	fp32.Axpy(tData, otherData, 1, 1, size, alpha)
	return &t
}
