package eager_tensor

import (
	"fmt"

	"github.com/itohio/EasyRobot/pkg/core/math/primitive/fp32"
	"github.com/itohio/EasyRobot/pkg/core/math/primitive/generics"
	. "github.com/itohio/EasyRobot/pkg/core/math/primitive/generics/helpers"
	"github.com/itohio/EasyRobot/pkg/core/math/tensor/types"
)

// MatVecMulTransposed performs matrix-vector multiplication with matrix transposed: y = alpha*A^T*x + beta*y
// matrix: [M, N] tensor (will be transposed)
// vector: [M] tensor (input vector)
// result: [N] tensor (output vector)
// Uses fp32 primitive.Gemv_T internally.
// If dst is nil, creates a new tensor.
// If dst is provided, writes result to dst and returns dst.
func (t Tensor) MatVecMulTransposed(dst types.Tensor, matrix types.Tensor, vector types.Tensor, alpha, beta float64) types.Tensor {
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

	// Determine result tensor
	var result types.Tensor
	if IsNil(dst) {
		// Create new result tensor
		result = New(t.DataType(), types.NewShape(N))
	} else {
		// Validate dst shape
		dstShape := dst.Shape()
		if len(dstShape) != 1 || dstShape[0] != N {
			panic(fmt.Sprintf("tensor.MatVecMulTransposed: destination shape mismatch: expected [%d], got %v", N, dstShape))
		}
		result = dst
	}

	// Leading dimension of matrix (row-major: number of columns)
	ldA := N

	// Use primitive.Gemv_T
	// Convert float64 parameters to float32 for internal computation
	alpha32 := float32(alpha)
	beta32 := float32(beta)
	matrixData := types.GetTensorData[[]float32](matrix)
	vectorData := types.GetTensorData[[]float32](vector)
	resultData := types.GetTensorData[[]float32](result)
	fp32.Gemv_T(
		resultData, // y (output)
		matrixData, // A (matrix)
		vectorData, // x (vector)
		ldA, M, N,  // leading dimension, rows, cols
		alpha32, beta32, // scaling factors
	)

	return result
}

// MatMulTransposed performs matrix multiplication with optional transposition.
// If transposeA is true: uses Gemm_TN (A^T @ B)
// If transposeB is true: uses Gemm_NT (A @ B^T)
// If both are false: uses Gemm_NN (A @ B)
// If both are true: uses Gemm_TT (A^T @ B^T)
// If dst is nil, creates a new tensor.
// If dst is provided, writes result to dst and returns dst.
func (t Tensor) MatMulTransposed(dst types.Tensor, other types.Tensor, transposeA, transposeB bool) types.Tensor {
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

	// Prepare result tensor - try to use dst directly if it's an eager tensor and shape matches
	var result Tensor
	usedDst := false
	if dst != nil {
		dstShape := dst.Shape()
		// Check if dst has correct shape
		if len(dstShape) == len(resultShape) {
			match := true
			for i := range dstShape {
				if dstShape[i] != resultShape[i] {
					match = false
					break
				}
			}
			if match {
				// Try to use dst directly if it's an eager tensor
				if dstTensor, ok := dst.(Tensor); ok {
					result = dstTensor
					usedDst = true
				} else {
					// Not an eager tensor, need to create new and copy
					result = New(t.DataType(), types.NewShape(resultShape...))
				}
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
	tData := types.GetTensorData[[]float32](t)
	otherData := types.GetTensorData[[]float32](other)
	resultData := types.GetTensorData[[]float32](result)

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
		// If dst was provided and we used it directly, return it
		// Otherwise, if dst was provided but we created a new tensor, copy result to dst
		if dst != nil {
			if usedDst {
				return dst
			}
			// Need to copy result to dst
			resultData := types.GetTensorData[[]float32](result)
			dstData := types.GetTensorData[[]float32](dst)
			shapeSlice := result.Shape().ToSlice()
			// Use Strides(nil) for read-only operations - returns stored strides directly without copy
			dstStrides := dst.Strides(nil)
			resultStrides := result.Strides(nil)
			generics.ElemCopyStrided[float32](dstData, resultData, shapeSlice, dstStrides, resultStrides)
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

		// Use dst directly if it was provided and matches shape (for batched case)
		if !usedDst && dst != nil {
			if dstTensor, ok := dst.(Tensor); ok {
				dstShape := dstTensor.Shape()
				if len(dstShape) == len(resultShape) {
					match := true
					for i := range dstShape {
						if dstShape[i] != resultShape[i] {
							match = false
							break
						}
					}
					if match {
						result = dstTensor
						usedDst = true
					}
				}
			}
		}
		tData := types.GetTensorData[[]float32](t)
		otherData := types.GetTensorData[[]float32](other)
		resultData := types.GetTensorData[[]float32](result)

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
		// If dst was provided and we used it directly, return it
		// Otherwise, if dst was provided but we created a new tensor, copy result to dst
		if dst != nil {
			if usedDst {
				return dst
			}
			// Need to copy result to dst
			resultData := types.GetTensorData[[]float32](result)
			dstData := types.GetTensorData[[]float32](dst)
			shapeSlice := result.Shape().ToSlice()
			// Use Strides(nil) for read-only operations - returns stored strides directly without copy
			dstStrides := dst.Strides(nil)
			resultStrides := result.Strides(nil)
			generics.ElemCopyStrided[float32](dstData, resultData, shapeSlice, dstStrides, resultStrides)
			return dst
		}
		return &result
	}

	panic(fmt.Sprintf("tensor.MatMulTransposed: unsupported tensor shapes: %v × %v", tShape, otherShape))
}

// AddScaled adds a scaled tensor to this tensor in-place: t = t + alpha * other
// Uses fp32 primitive.Axpy internally.
// Returns the tensor itself for method chaining.
// Converts float64 alpha to float32 for internal computation.
// AddScaled computes dst = t + alpha * other (scaled addition).
// If dst is nil, operation is in-place (modifies t) and returns t.
// If dst is provided, writes result to dst and returns dst.
func (t Tensor) AddScaled(dst types.Tensor, other types.Tensor, alpha float64) types.Tensor {
	if t.shape == nil {
		return t
	}
	if IsNil(other) {
		return t
	}
	if !t.Shape().Equal(other.Shape()) {
		panic(fmt.Sprintf("tensor.AddScaled: shape mismatch: %v vs %v", t.Shape(), other.Shape()))
	}

	alpha32 := float32(alpha)

	// In-place operation (dst is nil)
	if IsNil(dst) {
		tData := types.GetTensorData[[]float32](t)
		otherData := types.GetTensorData[[]float32](other)

		if t.IsContiguous() && other.IsContiguous() {
			// Use primitive.Axpy for contiguous case
			size := t.Size()
			fp32.Axpy(tData, otherData, 1, 1, size, alpha32)
			return &t
		}

		// Handle strided case manually
		size := t.Size()
		for i := 0; i < size; i++ {
			tData[i] += alpha32 * otherData[i]
		}
		return &t
	}

	// Destination-based operation
	if !t.Shape().Equal(dst.Shape()) {
		panic(fmt.Sprintf("tensor.AddScaled: destination shape mismatch: %v vs %v", dst.Shape(), t.shape))
	}

	// Copy t to dst, then add scaled other
	tData := types.GetTensorData[[]float32](t)
	dstData := types.GetTensorData[[]float32](dst)
	shapeSlice := t.Shape().ToSlice()
	// Use Strides(nil) for read-only operations - returns stored strides directly without copy
	dstStrides := dst.Strides(nil)
	tStrides := t.Strides(nil)
	generics.ElemCopyStrided[float32](dstData, tData, shapeSlice, dstStrides, tStrides)
	otherData := types.GetTensorData[[]float32](other)
	otherStrides := other.Strides(nil)

	if IsContiguous(dstStrides, shapeSlice) && IsContiguous(otherStrides, shapeSlice) {
		size := t.Size()
		fp32.Axpy(dstData, otherData, 1, 1, size, alpha32)
		return dst
	}

	// Handle strided case
	size := t.Size()
	for i := 0; i < size; i++ {
		dstData[i] += alpha32 * otherData[i]
	}
	return dst
}
