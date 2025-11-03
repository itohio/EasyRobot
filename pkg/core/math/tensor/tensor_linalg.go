package tensor

import (
	"fmt"

	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/pkg/core/math/primitive/fp32"
)

// MatMul performs matrix multiplication with another tensor.
// For 2D tensors: [M, K] × [K, N] = [M, N]
// For batched tensors: [B, M, K] × [B, K, N] = [B, M, N] or [M, K] × [B, K, N] = [B, M, N]
// Uses fp32 primitive.Gemm_NN by default. Automatically handles leading dimensions.
func (t *Tensor) MatMul(other *Tensor) *Tensor {
	if t == nil || other == nil {
		return nil
	}

	tShape := t.Shape()
	otherShape := other.Shape()

	// Handle 2D case: [M, K] × [K, N]
	if len(tShape) == 2 && len(otherShape) == 2 {
		return t.matMul2D(other)
	}

	// Handle batched case: [B, M, K] × [B, K, N] or [M, K] × [B, K, N]
	if len(tShape) >= 2 && len(otherShape) >= 2 {
		return t.matMulBatched(other)
	}

	panic(fmt.Sprintf("tensor.MatMul: unsupported tensor shapes: %v × %v", tShape, otherShape))
}

// MatMulTo performs matrix multiplication and stores result in dst (or creates new tensor if dst is nil).
func (t *Tensor) MatMulTo(other *Tensor, dst *Tensor) *Tensor {
	if t == nil || other == nil {
		return nil
	}

	result := t.MatMul(other)
	if result == nil {
		return nil
	}

	if dst == nil {
		return result
	}

	if !result.sameShape(dst) {
		panic(fmt.Sprintf("tensor.MatMulTo: destination shape mismatch: %v vs %v", dst.Shape(), result.Shape()))
	}

	// Copy result to dst
	result.copyTo(dst)
	return dst
}

// matMul2D performs matrix multiplication for 2D tensors: [M, K] × [K, N] = [M, N]
func (t *Tensor) matMul2D(other *Tensor) *Tensor {
	M := t.Dim[0]
	K := t.Dim[1]
	K2 := other.Dim[0]
	N := other.Dim[1]

	if K != K2 {
		panic(fmt.Sprintf("tensor.MatMul: dimension mismatch: %v × %v", t.Dim, other.Dim))
	}

	// Leading dimensions for row-major matrices
	ldA := K // A is [M, K], columns = K
	ldB := N // B is [K, N], columns = N
	ldC := N // C is [M, N], columns = N

	// Create output tensor
	result := &Tensor{
		Dim:  []int{M, N},
		Data: make([]float32, M*N),
	}

	// Use fp32.Gemm_NN
	fp32.Gemm_NN(
		result.Data,   // C
		t.Data,        // A
		other.Data,    // B
		ldC, ldA, ldB, // leading dimensions
		M, N, K, // dimensions
		1.0, 0.0, // alpha, beta
	)

	return result
}

// matMulBatched handles batched matrix multiplication
func (t *Tensor) matMulBatched(other *Tensor) *Tensor {
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
func (t *Tensor) matMulSameBatch(other *Tensor, batchSize, M, N, K int) *Tensor {
	// Leading dimensions
	ldA := K
	ldB := N
	ldC := N

	// Compute strides
	tStride := M * K
	otherStride := K * N
	resultStride := M * N

	// Create output tensor
	resultShape := make([]int, len(t.Dim))
	copy(resultShape, t.Dim)
	resultShape[len(resultShape)-2] = M
	resultShape[len(resultShape)-1] = N

	result := &Tensor{
		Dim:  resultShape,
		Data: make([]float32, batchSize*M*N),
	}

	// Check if contiguous (can use GemmStrided)
	if t.isContiguous() && other.isContiguous() {
		fp32.GemmStrided(
			result.Data,   // C
			t.Data,        // A
			other.Data,    // B
			ldC, ldA, ldB, // leading dimensions
			M, N, K, // dimensions
			1.0, 0.0, // alpha, beta
			batchSize,    // batch count
			tStride,      // strideA
			otherStride,  // strideB
			resultStride, // strideC
		)
	} else {
		// Use GemmBatched (handles strided access)
		fp32.GemmBatched(
			result.Data,
			t.Data,
			other.Data,
			ldC, ldA, ldB,
			M, N, K,
			1.0, 0.0,
			batchSize,
			tStride,
			otherStride,
			resultStride,
		)
	}

	return result
}

// matMulBroadcastFirst handles [M, K] × [B, K, N] (broadcast first tensor)
func (t *Tensor) matMulBroadcastFirst(other *Tensor, batchSize, M, N, K int) *Tensor {
	// Replicate t for each batch element
	resultShape := make([]int, len(other.Dim))
	copy(resultShape, other.Dim)
	resultShape[len(resultShape)-2] = M

	result := &Tensor{
		Dim:  resultShape,
		Data: make([]float32, batchSize*M*N),
	}

	ldA := K
	ldB := N
	ldC := N
	otherStride := K * N
	resultStride := M * N

	// For each batch, multiply t with other[batch]
	for b := 0; b < batchSize; b++ {
		otherOffset := b * otherStride
		resultOffset := b * resultStride

		fp32.Gemm_NN(
			result.Data[resultOffset:],
			t.Data,
			other.Data[otherOffset:],
			ldC, ldA, ldB,
			M, N, K,
			1.0, 0.0,
		)
	}

	return result
}

// matMulBroadcastSecond handles [B, M, K] × [K, N] (broadcast second tensor)
func (t *Tensor) matMulBroadcastSecond(other *Tensor, batchSize, M, N, K int) *Tensor {
	resultShape := make([]int, len(t.Dim))
	copy(resultShape, t.Dim)
	resultShape[len(resultShape)-1] = N

	result := &Tensor{
		Dim:  resultShape,
		Data: make([]float32, batchSize*M*N),
	}

	ldA := K
	ldB := N
	ldC := N
	tStride := M * K
	resultStride := M * N

	// For each batch, multiply t[batch] with other
	for b := 0; b < batchSize; b++ {
		tOffset := b * tStride
		resultOffset := b * resultStride

		fp32.Gemm_NN(
			result.Data[resultOffset:],
			t.Data[tOffset:],
			other.Data,
			ldC, ldA, ldB,
			M, N, K,
			1.0, 0.0,
		)
	}

	return result
}

// Transpose transposes tensor dimensions. Currently supports 2D transpose.
// Future: support arbitrary dimension permutation.
func (t *Tensor) Transpose(dims ...int) *Tensor {
	if t == nil {
		return nil
	}

	shape := t.Shape()
	if len(shape) == 2 {
		// Simple 2D transpose: [M, N] -> [N, M]
		return t.transpose2D()
	}

	// For now, only support 2D transpose
	if len(dims) == 0 {
		if len(shape) == 2 {
			return t.transpose2D()
		}
		panic(fmt.Sprintf("tensor.Transpose: need dims for %dD tensor", len(shape)))
	}

	// Future: implement general transpose with dimension permutation
	panic("tensor.Transpose: general dimension transpose not yet implemented")
}

// transpose2D transposes a 2D tensor: [M, N] -> [N, M]
func (t *Tensor) transpose2D() *Tensor {
	if len(t.Dim) != 2 {
		panic("tensor.transpose2D: tensor must be 2D")
	}

	M, N := t.Dim[0], t.Dim[1]
	result := &Tensor{
		Dim:  []int{N, M},
		Data: make([]float32, M*N),
	}

	// Transpose: result[j][i] = t[i][j]
	for i := 0; i < M; i++ {
		for j := 0; j < N; j++ {
			result.Data[j*M+i] = t.Data[i*N+j]
		}
	}

	return result
}

// TransposeTo transposes tensor and stores result in dst (or creates new tensor if dst is nil).
func (t *Tensor) TransposeTo(dst *Tensor, dims ...int) *Tensor {
	if t == nil {
		return nil
	}

	result := t.Transpose(dims...)
	if result == nil {
		return nil
	}

	if dst == nil {
		return result
	}

	if !result.sameShape(dst) {
		panic(fmt.Sprintf("tensor.TransposeTo: destination shape mismatch: %v vs %v", dst.Shape(), result.Shape()))
	}

	// Copy result to dst
	result.copyTo(dst)
	return dst
}

// Dot computes the dot product of two tensors.
// For vectors: dot product of two 1D tensors
// For matrices: Frobenius inner product (sum of element-wise products)
// Uses fp32 primitive.Dot for vector case.
func (t *Tensor) Dot(other *Tensor) float32 {
	if t == nil || other == nil {
		return 0
	}

	tShape := t.Shape()
	otherShape := other.Shape()

	// Vector dot product: both are 1D and same size
	if len(tShape) == 1 && len(otherShape) == 1 {
		if tShape[0] != otherShape[0] {
			panic(fmt.Sprintf("tensor.Dot: vector size mismatch: %d vs %d", tShape[0], otherShape[0]))
		}

		if t.isContiguous() && other.isContiguous() {
			return fp32.Dot(t.Data, other.Data, 1, 1, tShape[0])
		}

		// Strided case
		return t.dotStrided(other, tShape[0])
	}

	// Matrix Frobenius inner product: sum of all element-wise products
	if !t.sameShape(other) {
		panic(fmt.Sprintf("tensor.Dot: shape mismatch for Frobenius product: %v vs %v", tShape, otherShape))
	}

	// Flatten and compute dot product
	return t.dotFrobenius(other)
}

// dotStrided computes dot product for strided vectors
func (t *Tensor) dotStrided(other *Tensor, n int) float32 {
	tStrides := Shape(t.Dim).Strides()
	otherStrides := Shape(other.Dim).Strides()

	var sum float32
	for i := 0; i < n; i++ {
		tIdx := i * tStrides[0]
		otherIdx := i * otherStrides[0]
		sum += t.Data[tIdx] * other.Data[otherIdx]
	}
	return sum
}

// dotFrobenius computes Frobenius inner product (sum of element-wise products)
func (t *Tensor) dotFrobenius(other *Tensor) float32 {
	var sum float32
	size := t.Size()
	for i := 0; i < size; i++ {
		sum += t.Data[i] * other.Data[i]
	}
	return sum
}

// Norm computes vector or matrix norm.
// ord: 0 = L1 norm (|x|_1), 1 = L2 norm (|x|_2), 2 = Frobenius norm for matrices
// Uses fp32 primitive.Nrm2 for L2 norm, fp32 primitive.Asum for L1 norm.
func (t *Tensor) Norm(ord int) float32 {
	if t == nil {
		return 0
	}

	switch ord {
	case 0:
		// L1 norm
		if t.isContiguous() {
			return fp32.Asum(t.Data, 1, t.Size())
		}
		return t.norm1Strided()

	case 1:
		// L2 norm (Euclidean norm)
		if t.isContiguous() {
			return fp32.Nrm2(t.Data, 1, t.Size())
		}
		return t.norm2Strided()

	case 2:
		// Frobenius norm for matrices (same as L2 norm on flattened matrix)
		if t.isContiguous() {
			return fp32.Nrm2(t.Data, 1, t.Size())
		}
		return t.norm2Strided()

	default:
		panic(fmt.Sprintf("tensor.Norm: unsupported order %d (use 0=L1, 1=L2, 2=Frobenius)", ord))
	}
}

// norm1Strided computes L1 norm for strided tensor
func (t *Tensor) norm1Strided() float32 {
	var sum float32
	strides := Shape(t.Dim).Strides()
	indices := make([]int, len(t.Dim))
	t.norm1StridedRecursive(&sum, indices, strides, 0)
	return sum
}

func (t *Tensor) norm1StridedRecursive(sum *float32, indices []int, strides []int, dim int) {
	if dim == len(t.Dim) {
		idx := t.elementIndex(indices, strides)
		val := t.Data[idx]
		if val < 0 {
			*sum -= val
		} else {
			*sum += val
		}
		return
	}

	for i := 0; i < t.Dim[dim]; i++ {
		indices[dim] = i
		t.norm1StridedRecursive(sum, indices, strides, dim+1)
	}
}

// norm2Strided computes L2 norm for strided tensor
func (t *Tensor) norm2Strided() float32 {
	var sumSq float32
	strides := Shape(t.Dim).Strides()
	indices := make([]int, len(t.Dim))
	t.norm2StridedRecursive(&sumSq, indices, strides, 0)
	// Note: Need sqrt for L2 norm, but primitive.Nrm2 does that
	// For now, compute manually
	if sumSq == 0 {
		return 0
	}
	// Use simple sqrt approximation or call Nrm2 on flattened
	// For now, let's use the sum of squares and take sqrt
	// Better: use primitive.Nrm2 on a flattened view if possible
	return t.norm2StridedCompute(sumSq)
}

func (t *Tensor) norm2StridedCompute(sumSq float32) float32 {
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

func (t *Tensor) sqrtApprox(x float32) float32 {
	// Use math32.Sqrt for square root computation
	if x <= 0 {
		return 0
	}
	return math32.Sqrt(x)
}

func (t *Tensor) norm2StridedRecursive(sum *float32, indices []int, strides []int, dim int) {
	if dim == len(t.Dim) {
		idx := t.elementIndex(indices, strides)
		val := t.Data[idx]
		*sum += val * val
		return
	}

	for i := 0; i < t.Dim[dim]; i++ {
		indices[dim] = i
		t.norm2StridedRecursive(sum, indices, strides, dim+1)
	}
}

// Normalize normalizes the tensor along a dimension (L2 normalization).
// For vectors (1D): normalizes the entire vector
// For matrices (2D): normalizes along specified dimension (0=rows, 1=columns)
// Uses fp32 primitive.Nrm2 + fp32 primitive.Scal for efficient computation.
func (t *Tensor) Normalize(dim int) *Tensor {
	if t == nil {
		return nil
	}

	shape := t.Shape()
	if len(shape) == 1 {
		// Vector normalization
		return t.normalizeVector()
	}

	if len(shape) == 2 {
		// Matrix normalization along dimension
		return t.normalizeMatrixDim(dim)
	}

	panic(fmt.Sprintf("tensor.Normalize: unsupported tensor shape %v (use 1D or 2D)", shape))
}

// normalizeVector normalizes a 1D vector
func (t *Tensor) normalizeVector() *Tensor {
	result := t.Clone()

	if !result.isContiguous() {
		// Handle strided case
		norm := result.Norm(1) // L2 norm
		if norm > 0 {
			scale := 1.0 / norm
			result.Scale(scale)
		}
		return result
	}

	// Use fp32 operations
	norm := fp32.Nrm2(result.Data, 1, result.Size())
	if norm > 0 {
		fp32.Scal(result.Data, 1, result.Size(), 1.0/norm)
	}

	return result
}

// normalizeMatrixDim normalizes a matrix along specified dimension
func (t *Tensor) normalizeMatrixDim(dim int) *Tensor {
	if dim < 0 || dim >= len(t.Dim) {
		panic(fmt.Sprintf("tensor.Normalize: dimension %d out of range for shape %v", dim, t.Dim))
	}

	result := t.Clone()

	M, N := t.Dim[0], t.Dim[1]

	if dim == 0 {
		// Normalize along rows (each row becomes unit vector)
		for i := 0; i < M; i++ {
			rowData := result.Data[i*N : (i+1)*N]
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
				colData[i] = result.Data[i*N+j]
			}
			norm := fp32.Nrm2(colData, 1, M)
			if norm > 0 {
				scale := 1.0 / norm
				for i := 0; i < M; i++ {
					result.Data[i*N+j] *= scale
				}
			}
		}
	}

	return result
}
