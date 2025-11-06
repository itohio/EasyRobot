package eager_tensor

import (
	"fmt"

	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/pkg/core/math/primitive/fp32"
	"github.com/itohio/EasyRobot/pkg/core/math/primitive/generics"
	. "github.com/itohio/EasyRobot/pkg/core/math/primitive/generics/helpers"
	"github.com/itohio/EasyRobot/pkg/core/math/tensor/types"
)

// MatMul performs matrix multiplication (matches tf.matmul).
// For 2D tensors: [M, K] × [K, N] = [M, N]
// For batched tensors: [B, M, K] × [B, K, N] = [B, M, N] or [M, K] × [B, K, N] = [B, M, N]
// If dst is nil, creates a new tensor.
// If dst is provided, writes result to dst and returns dst.
func (t Tensor) MatMul(dst types.Tensor, other types.Tensor) types.Tensor {
	if t.shape == nil {
		return nil
	}
	if IsNil(other) {
		return nil
	}

	tShape := t.Shape()
	otherShape := other.Shape()

	var result types.Tensor
	// Handle 2D case: [M, K] × [K, N]
	if len(tShape) == 2 && len(otherShape) == 2 {
		result = t.matMul2D(other)
	} else if len(tShape) >= 2 && len(otherShape) >= 2 {
		// Handle batched case: [B, M, K] × [B, K, N] or [M, K] × [B, K, N]
		result = t.matMulBatched(other)
	} else {
		panic(fmt.Sprintf("tensor.MatMul: unsupported tensor shapes: %v × %v", tShape, otherShape))
	}

	if result == nil {
		return nil
	}

	if IsNil(dst) {
		return result
	}

	if !result.Shape().Equal(dst.Shape()) {
		panic(fmt.Sprintf("tensor.MatMul: destination shape mismatch: expected %v, got %v", result.Shape(), dst.Shape()))
	}

	// Copy result to dst using generics
	resultData := types.GetTensorData[[]float32](result)
	dstData := types.GetTensorData[[]float32](dst)
	shapeSlice := result.Shape().ToSlice()
	generics.ElemCopyStrided[float32](dstData, resultData, shapeSlice, dst.Shape().Strides(), result.Shape().Strides())
	return dst
}

// matMul2D performs matrix multiplication for 2D tensors: [M, K] × [K, N] = [M, N]
func (t Tensor) matMul2D(other types.Tensor) types.Tensor {
	tShape := t.Shape()
	otherShape := other.Shape()
	M := tShape[0]
	K := tShape[1]
	K2 := otherShape[0]
	N := otherShape[1]

	if K != K2 {
		panic(fmt.Sprintf("tensor.MatMul: dimension mismatch: %v × %v", tShape, otherShape))
	}

	ldA := K
	ldB := N
	ldC := N

	result := New(t.DataType(), types.NewShape(M, N))
	resultPtr := &result

	tData := types.GetTensorData[[]float32](t)
	otherData := types.GetTensorData[[]float32](other)
	resultData := types.GetTensorData[[]float32](resultPtr)

	fp32.Gemm_NN(
		resultData,
		tData,
		otherData,
		ldC, ldA, ldB,
		M, N, K,
		1.0, 0.0,
	)

	return resultPtr
}

// matMulBatched handles batched matrix multiplication
func (t Tensor) matMulBatched(other types.Tensor) types.Tensor {
	tShape := t.Shape()
	otherShape := other.Shape()

	// Extract matrix dimensions (last two dimensions)
	if len(tShape) < 2 || len(otherShape) < 2 {
		panic(fmt.Sprintf("tensor.MatMul: need at least 2D tensors, got %v × %v", tShape, otherShape))
	}

	M, K := tShape[len(tShape)-2], tShape[len(tShape)-1]
	K2, N := otherShape[len(otherShape)-2], otherShape[len(otherShape)-1]

	if K != K2 {
		panic(fmt.Sprintf("tensor.MatMul: dimension mismatch: K=%d vs K2=%d", K, K2))
	}

	// Check if we can use batched operations
	tBatchSize := 1
	otherBatchSize := 1
	for i := 0; i < len(tShape)-2; i++ {
		tBatchSize *= tShape[i]
	}
	for i := 0; i < len(otherShape)-2; i++ {
		otherBatchSize *= otherShape[i]
	}

	// For now, handle simple cases:
	// 1. [B, M, K] × [B, K, N] - both batched with same batch size
	// 2. [M, K] × [B, K, N] - broadcast first tensor
	// 3. [B, M, K] × [K, N] - broadcast second tensor

	if tBatchSize == otherBatchSize {
		// Both have same batch size
		return t.matMulSameBatch(other, tBatchSize, M, N, K)
	} else if tBatchSize == 1 && otherBatchSize > 1 {
		// Broadcast first tensor
		return t.matMulBroadcastFirst(other, otherBatchSize, M, N, K)
	} else if tBatchSize > 1 && otherBatchSize == 1 {
		// Broadcast second tensor
		return t.matMulBroadcastSecond(other, tBatchSize, M, N, K)
	}

	panic(fmt.Sprintf("tensor.MatMul: unsupported batch configuration: %v × %v", tShape, otherShape))
}

// matMulSameBatch handles [B, M, K] × [B, K, N]
func (t Tensor) matMulSameBatch(other types.Tensor, batchSize, M, N, K int) types.Tensor {
	ldA := K
	ldB := N
	ldC := N

	tStride := M * K
	otherStride := K * N
	resultStride := M * N

	resultShape := append([]int(nil), t.shape...)
	resultShape[len(resultShape)-2] = M
	resultShape[len(resultShape)-1] = N

	result := New(t.DataType(), types.NewShape(resultShape...))
	resultPtr := &result

	tData := types.GetTensorData[[]float32](t)
	otherData := types.GetTensorData[[]float32](other)
	resultData := types.GetTensorData[[]float32](resultPtr)

	otherStrides := other.Shape().Strides()
	tShapeSlice := []int{M, K}
	if t.isContiguous() && IsContiguous(otherStrides, tShapeSlice) {
		fp32.GemmStrided(
			resultData,
			tData,
			otherData,
			ldC, ldA, ldB,
			M, N, K,
			1.0, 0.0,
			batchSize,
			tStride,
			otherStride,
			resultStride,
		)
	} else {
		fp32.GemmBatched(
			resultData,
			tData,
			otherData,
			ldC, ldA, ldB,
			M, N, K,
			1.0, 0.0,
			batchSize,
			tStride,
			otherStride,
			resultStride,
		)
	}

	return resultPtr
}

// matMulBroadcastFirst handles [M, K] × [B, K, N] (broadcast first tensor)
func (t Tensor) matMulBroadcastFirst(other types.Tensor, batchSize, M, N, K int) types.Tensor {
	resultShape := []int{batchSize, M, N}
	result := New(t.DataType(), types.NewShape(resultShape...))
	resultPtr := &result

	sliceSize := K * N
	dstSize := M * N

	tData := types.GetTensorData[[]float32](t)
	otherData := types.GetTensorData[[]float32](other)
	resultData := types.GetTensorData[[]float32](resultPtr)

	for b := 0; b < batchSize; b++ {
		sliceOffset := b * sliceSize
		dstOffset := b * dstSize
		fp32.Gemm_NN(
			resultData[dstOffset:],
			tData,
			otherData[sliceOffset:],
			N, K, N,
			M, N, K,
			1.0, 0.0,
		)
	}

	return resultPtr
}

// matMulBroadcastSecond handles [B, M, K] × [K, N] (broadcast second tensor)
func (t Tensor) matMulBroadcastSecond(other types.Tensor, batchSize, M, N, K int) types.Tensor {
	resultShape := append([]int(nil), t.shape...)
	resultShape[len(resultShape)-1] = N

	result := New(t.DataType(), types.NewShape(resultShape...))
	resultPtr := &result

	ldA := K
	ldB := N
	ldC := N
	tStride := M * K
	resultStride := M * N

	tData := types.GetTensorData[[]float32](t)
	otherData := types.GetTensorData[[]float32](other)
	resultData := types.GetTensorData[[]float32](resultPtr)

	for b := 0; b < batchSize; b++ {
		tOffset := b * tStride
		resultOffset := b * resultStride

		fp32.Gemm_NN(
			resultData[resultOffset:],
			tData[tOffset:],
			otherData,
			ldC, ldA, ldB,
			M, N, K,
			1.0, 0.0,
		)
	}

	return resultPtr
}

// Transpose transposes tensor dimensions. Currently supports 2D transpose.
// Transpose transposes dimensions (matches tf.transpose).
// For 2D: [M, N] → [N, M] (swaps last two dimensions if no dims provided)
// For 4D+: uses Permute to rearrange dimensions
// If dst is nil, creates a new tensor.
// If dst is provided, writes result to dst and returns dst.
func (t Tensor) Transpose(dst types.Tensor, dims []int) types.Tensor {
	if t.shape == nil {
		return nil
	}

	shape := t.Shape()
	rank := shape.Rank()

	// Handle default dims for 2D
	if len(dims) == 0 {
		if rank == 2 {
			// Default: swap last two dimensions
			dims = []int{1, 0}
		} else if rank >= 2 {
			// Default: swap last two dimensions
			dims = make([]int, rank)
			for i := 0; i < rank-2; i++ {
				dims[i] = i
			}
			dims[rank-2] = rank - 1
			dims[rank-1] = rank - 2
		} else {
			panic(fmt.Sprintf("tensor.Transpose: need at least 2 dimensions, got %d", rank))
		}
	}

	var result types.Tensor
	if rank == 2 && len(dims) == 2 && dims[0] == 1 && dims[1] == 0 {
		// Simple 2D transpose: [M, N] -> [N, M]
		result = t.transpose2D()
	} else {
		// For higher dimensions, use Permute
		result = t.Permute(dims)
	}

	if result == nil {
		return nil
	}

	if IsNil(dst) {
		return result
	}

	if !result.Shape().Equal(dst.Shape()) {
		panic(fmt.Sprintf("tensor.Transpose: destination shape mismatch: expected %v, got %v", result.Shape(), dst.Shape()))
	}

	// Copy result to dst using generics
	resultData := types.GetTensorData[[]float32](result)
	dstData := types.GetTensorData[[]float32](dst)
	shapeSlice := result.Shape().ToSlice()
	generics.ElemCopyStrided[float32](dstData, resultData, shapeSlice, dst.Shape().Strides(), result.Shape().Strides())
	return dst
}

// Permute permutes dimensions according to the provided permutation
// dims: permutation of [0, 1, 2, ..., rank-1]
// Example: Permute([]int{1, 0, 2, 3}) swaps dimensions 0 and 1 in a 4D tensor
func (t Tensor) Permute(dims []int) types.Tensor {
	if t.shape == nil {
		return nil
	}

	shape := t.Shape()
	rank := shape.Rank()

	if len(dims) != rank {
		panic(fmt.Sprintf("tensor.Permute: permutation length %d must match tensor rank %d", len(dims), rank))
	}

	// Validate permutation
	used := make(map[int]bool)
	for i, d := range dims {
		if d < 0 || d >= rank {
			panic(fmt.Sprintf("tensor.Permute: invalid dimension %d at position %d (rank %d)", d, i, rank))
		}
		if used[d] {
			panic(fmt.Sprintf("tensor.Permute: duplicate dimension %d in permutation", d))
		}
		used[d] = true
	}

	// Compute permuted shape
	newShape := make(types.Shape, rank)
	for i, d := range dims {
		newShape[i] = shape[d]
	}

	// Compute source and destination strides
	srcStrides := shape.Strides()
	dstStrides := types.NewShape(newShape...).Strides()

	// Create result tensor
	result := New(t.DataType(), types.NewShape(newShape...))
	resultPtr := &result

	// Compute permuted source strides (map through permutation)
	permutedSrcStrides := make([]int, rank)
	for i, d := range dims {
		permutedSrcStrides[i] = srcStrides[d]
	}

	// Use fp32.ElemCopy with permuted strides
	resultData := types.GetTensorData[[]float32](resultPtr)
	tData := types.GetTensorData[[]float32](t)
	fp32.ElemCopy(
		resultData,
		tData,
		newShape.ToSlice(),
		dstStrides,
		permutedSrcStrides,
	)

	return resultPtr
}

// transpose2D transposes a 2D tensor: [M, N] -> [N, M]
func (t Tensor) transpose2D() types.Tensor {
	if t.shape.Rank() != 2 {
		panic("tensor.transpose2D: tensor must be 2D")
	}

	M, N := t.shape[0], t.shape[1]
	result := New(t.DataType(), types.NewShape(N, M))
	resultPtr := &result

	// Transpose: result[j][i] = t[i][j]
	resultData := types.GetTensorData[[]float32](resultPtr)
	tData := types.GetTensorData[[]float32](t)
	for i := 0; i < M; i++ {
		for j := 0; j < N; j++ {
			resultData[j*M+i] = tData[i*N+j]
		}
	}

	return resultPtr
}

// Dot computes dot product (vector) or Frobenius inner product (matrix).
// For vectors: dot product of two 1D tensors.
// For matrices: Frobenius inner product (sum of element-wise products).
// Uses fp32 primitive.Dot for vector case.
// Returns float64 result converted from internal float32 computation.
func (t Tensor) Dot(other types.Tensor) float64 {
	if t.shape == nil || other == nil || other.Shape() == nil {
		return 0
	}

	tShape := t.Shape()
	otherShape := other.Shape()

	// Vector dot product: both are 1D and same size
	if len(tShape) == 1 && len(otherShape) == 1 {
		if tShape[0] != otherShape[0] {
			panic(fmt.Sprintf("tensor.Dot: vector size mismatch: %d vs %d", tShape[0], otherShape[0]))
		}

		tData := types.GetTensorData[[]float32](t)
		otherData := types.GetTensorData[[]float32](other)
		otherStrides := other.Shape().Strides()
		tShapeSlice := []int{tShape[0]}
		if t.isContiguous() && IsContiguous(otherStrides, tShapeSlice) {
			return float64(fp32.Dot(tData, otherData, 1, 1, tShape[0]))
		}

		// Strided case
		return t.dotStrided(other, tShape[0])
	}

	// Matrix Frobenius inner product: sum of all element-wise products
	if !t.Shape().Equal(other.Shape()) {
		panic(fmt.Sprintf("tensor.Dot: shape mismatch for Frobenius product: %v vs %v", tShape, otherShape))
	}

	// Flatten and compute dot product
	return t.dotFrobenius(other)
}

// Tensordot is an alias for Dot (matches TensorFlow naming: tf.tensordot).
func (t Tensor) Tensordot(other types.Tensor) float64 {
	return t.Dot(other)
}

// dotStrided computes dot product for strided vectors
func (t Tensor) dotStrided(other types.Tensor, n int) float64 {
	tStrides := t.shape.Strides()
	otherStrides := other.Shape().Strides()

	var sum float32
	tData := types.GetTensorData[[]float32](t)
	otherData := types.GetTensorData[[]float32](other)
	for i := 0; i < n; i++ {
		tIdx := i * tStrides[0]
		otherIdx := i * otherStrides[0]
		sum += tData[tIdx] * otherData[otherIdx]
	}
	return float64(sum)
}

// dotFrobenius computes Frobenius inner product (sum of element-wise products)
func (t Tensor) dotFrobenius(other types.Tensor) float64 {
	var sum float32
	size := t.Size()
	tData := types.GetTensorData[[]float32](t)
	otherData := types.GetTensorData[[]float32](other)
	for i := 0; i < size; i++ {
		sum += tData[i] * otherData[i]
	}
	return float64(sum)
}

// Norm computes vector or matrix norm.
// ord: 0 = L1 norm (|x|_1), 1 = L2 norm (|x|_2), 2 = Frobenius norm for matrices
// Uses fp32 primitive.Nrm2 for L2 norm, fp32 primitive.Asum for L1 norm.
// Returns float64 result converted from internal float32 computation.
func (t Tensor) Norm(ord int) float64 {
	if t.shape == nil {
		return 0
	}

	switch ord {
	case 0:
		// L1 norm
		if t.isContiguous() {
			tData := types.GetTensorData[[]float32](t)
			return float64(fp32.Asum(tData, 1, t.Size()))
		}
		return t.norm1Strided()

	case 1:
		// L2 norm (Euclidean norm)
		if t.isContiguous() {
			tData := types.GetTensorData[[]float32](t)
			return float64(fp32.Nrm2(tData, 1, t.Size()))
		}
		return t.norm2Strided()

	case 2:
		// Frobenius norm for matrices (same as L2 norm on flattened matrix)
		if t.isContiguous() {
			tData := types.GetTensorData[[]float32](t)
			return float64(fp32.Nrm2(tData, 1, t.Size()))
		}
		return t.norm2Strided()

	default:
		panic(fmt.Sprintf("tensor.Norm: unsupported order %d (use 0=L1, 1=L2, 2=Frobenius)", ord))
	}
}

// norm1Strided computes L1 norm for strided tensor
func (t Tensor) norm1Strided() float64 {
	var sum float64
	strides := t.shape.Strides()
	indices := make([]int, t.shape.Rank())
	t.norm1StridedRecursive(&sum, indices, strides, 0)
	return sum
}

func (t Tensor) norm1StridedRecursive(sum *float64, indices []int, strides []int, dim int) {
	if dim == t.shape.Rank() {
		idx := t.elementIndex(indices, strides)
		tData := types.GetTensorData[[]float32](t)
		val := tData[idx]
		if val < 0 {
			*sum -= float64(val)
		} else {
			*sum += float64(val)
		}
		return
	}

	for i := 0; i < t.shape[dim]; i++ {
		indices[dim] = i
		t.norm1StridedRecursive(sum, indices, strides, dim+1)
	}
}

// norm2Strided computes L2 norm for strided tensor
func (t Tensor) norm2Strided() float64 {
	var sumSq float64
	strides := t.shape.Strides()
	indices := make([]int, t.shape.Rank())
	t.norm2StridedRecursive(&sumSq, indices, strides, 0)
	// Note: Need sqrt for L2 norm, but primitive.Nrm2 does that
	// For now, compute manually
	if sumSq == 0 {
		return 0
	}
	// Use simple sqrt approximation or call Nrm2 on flattened
	// For now, let's use the sum of squares and take sqrt
	// Better: use primitive.Nrm2 on a flattened view if possible
	return t.norm2StridedCompute(float32(sumSq))
}

func (t Tensor) norm2StridedCompute(sumSq float32) float64 {
	// Compute sqrt using primitive approach - for now use approximation
	// Actually, we can flatten and use Nrm2 if size is reasonable
	size := t.Size()
	if size > 0 {
		// For large tensors, compute properly
		// Use iterative sqrt approximation
		return t.sqrtApprox(sumSq)
	}
	return 0
}

func (t Tensor) sqrtApprox(x float32) float64 {
	// Use math32.Sqrt for square root computation
	if x <= 0 {
		return 0
	}
	return float64(math32.Sqrt(x))
}

func (t Tensor) norm2StridedRecursive(sum *float64, indices []int, strides []int, dim int) {
	if dim == t.shape.Rank() {
		idx := t.elementIndex(indices, strides)
		tData := types.GetTensorData[[]float32](t)
		val := tData[idx]
		*sum += float64(val * val)
		return
	}

	for i := 0; i < t.shape[dim]; i++ {
		indices[dim] = i
		t.norm2StridedRecursive(sum, indices, strides, dim+1)
	}
}

// L2Normalize performs L2 normalization along the specified dimension (matches tf.nn.l2_normalize).
// For vectors (1D): normalizes the entire vector
// For matrices (2D): normalizes along specified dimension (0=rows, 1=columns)
// If dst is nil, creates a new tensor.
// If dst is provided, writes result to dst and returns dst.
func (t Tensor) L2Normalize(dst types.Tensor, dim int) types.Tensor {
	if t.shape == nil {
		return nil
	}

	shape := t.Shape()
	var result types.Tensor
	if shape.Rank() == 1 {
		// Vector normalization
		result = t.normalizeVector()
	} else if shape.Rank() == 2 {
		// Matrix normalization along dimension
		result = t.normalizeMatrixDim(dim)
	} else {
		panic(fmt.Sprintf("tensor.L2Normalize: unsupported tensor shape %v (use 1D or 2D)", shape))
	}

	if result == nil {
		return nil
	}

	if IsNil(dst) {
		return result
	}

	if !result.Shape().Equal(dst.Shape()) {
		panic(fmt.Sprintf("tensor.L2Normalize: destination shape mismatch: expected %v, got %v", result.Shape(), dst.Shape()))
	}

	// Copy result to dst using generics
	resultData := types.GetTensorData[[]float32](result)
	dstData := types.GetTensorData[[]float32](dst)
	shapeSlice := result.Shape().ToSlice()
	generics.ElemCopyStrided[float32](dstData, resultData, shapeSlice, dst.Shape().Strides(), result.Shape().Strides())
	return dst
}

// Normalize is an alias for L2Normalize (matches TensorFlow naming).
func (t Tensor) Normalize(dst types.Tensor, dim int) types.Tensor {
	return t.L2Normalize(dst, dim)
}

// normalizeVector normalizes a 1D vector
func (t Tensor) normalizeVector() types.Tensor {
	result := t.Clone()

	resultStrides := result.Shape().Strides()
	resultShape := result.Shape().ToSlice()
	if !IsContiguous(resultStrides, resultShape) {
		// Handle strided case
		norm := result.Norm(1) // L2 norm
		if norm > 0 {
			scale := 1.0 / norm
			result.ScalarMul(result, scale)
		}
		return result
	}

	// Use fp32 operations
	resultData := types.GetTensorData[[]float32](result)
	norm := fp32.Nrm2(resultData, 1, result.Size())
	if norm > 0 {
		fp32.Scal(resultData, 1, result.Size(), 1.0/norm)
	}

	return result
}

// normalizeMatrixDim normalizes a matrix along specified dimension
func (t Tensor) normalizeMatrixDim(dim int) types.Tensor {
	shape := t.Shape()
	if dim < 0 || dim >= shape.Rank() {
		panic(fmt.Sprintf("tensor.Normalize: dimension %d out of range for shape %v", dim, shape))
	}

	result := t.Clone()

	M, N := shape[0], shape[1]
	resultData := types.GetTensorData[[]float32](result)

	if dim == 0 {
		// Normalize along rows (each row becomes unit vector)
		for i := 0; i < M; i++ {
			rowData := resultData[i*N : (i+1)*N]
			norm := fp32.Nrm2(rowData, 1, N)
			if norm > 0 {
				fp32.Scal(rowData, 1, N, 1.0/norm)
			}
		}
	} else if dim == 1 {
		// Normalize along columns (each column becomes unit vector)
		for j := 0; j < N; j++ {
			// Extract column (stride access)
			colData := make([]float32, M)
			for i := 0; i < M; i++ {
				colData[i] = resultData[i*N+j]
			}
			norm := fp32.Nrm2(colData, 1, M)
			if norm > 0 {
				scale := 1.0 / norm
				for i := 0; i < M; i++ {
					resultData[i*N+j] *= scale
				}
			}
		}
	}

	return result
}
