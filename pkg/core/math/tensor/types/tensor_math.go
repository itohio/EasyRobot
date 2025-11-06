package types

// TensorMath defines common tensor math operations including reductions and linear algebra.
// This interface contains operations for mathematical computations on tensors.
type TensorMath interface {
	// Reduction Operations
	// Reduction operations reduce dimensions and return new tensors with reduced dimensions.

	// Sum sums along specified dimensions. If no dimensions specified, sums all elements.
	// Returns a new tensor with reduced dimensions. Panics if dimensions are out of range.
	Sum(dims ...int) Tensor

	// Mean computes mean along specified dimensions. If no dimensions specified, means all elements.
	// Returns a new tensor with reduced dimensions. Panics if dimensions are out of range.
	Mean(dims ...int) Tensor

	// Max computes maximum along specified dimensions. If no dimensions specified, finds global maximum.
	// Returns a new tensor with reduced dimensions. Panics if dimensions are out of range.
	Max(dims ...int) Tensor

	// Min computes minimum along specified dimensions. If no dimensions specified, finds global minimum.
	// Returns a new tensor with reduced dimensions. Panics if dimensions are out of range.
	Min(dims ...int) Tensor

	// ArgMax returns the index of the maximum element along the specified dimension.
	// Returns a new tensor with reduced dimension. Panics if dimension is out of range.
	ArgMax(dim int) Tensor

	// Linear Algebra Operations
	// All linear algebra operations use optimized BLAS/LAPACK operations when possible.

	// MatMul performs matrix multiplication.
	// For 2D: [M, K] × [K, N] = [M, N]
	// For batched: [B, M, K] × [B, K, N] = [B, M, N]
	// Supports broadcasting: [M, K] × [B, K, N] or [B, M, K] × [K, N]
	// Panics if dimensions are incompatible. Returns a new tensor.
	MatMul(other Tensor) Tensor

	// MatMulTo performs matrix multiplication and stores result in dst.
	// If dst is nil, creates a new tensor. If dst is provided, uses it (must match output shape).
	// Returns the destination tensor.
	MatMulTo(other Tensor, dst Tensor) Tensor

	// MatMulTransposed performs matrix multiplication with optional transposition.
	// Computes: dst = (transposeA ? t^T : t) × (transposeB ? other^T : other)
	// If dst is nil, creates a new tensor. Returns the destination tensor.
	MatMulTransposed(other Tensor, transposeA, transposeB bool, dst Tensor) Tensor

	// MatVecMulTransposed performs matrix-vector multiplication with scaling: result = alpha * matrix^T × vector + beta * result.
	// Returns a new tensor.
	MatVecMulTransposed(matrix, vector Tensor, alpha, beta float64) Tensor

	// Dot computes dot product (vector) or Frobenius inner product (matrix).
	// For vectors: dot product of two 1D tensors.
	// For matrices: Frobenius inner product (sum of element-wise products).
	// Panics if shapes are incompatible.
	Dot(other Tensor) float64

	// Norm computes vector/matrix norm.
	// ord: 0 = L1 norm, 1 = L2 norm, 2 = Frobenius norm (same as L2 for matrices).
	// Panics if ord is invalid.
	Norm(ord int) float64

	// Normalize performs L2 normalization along the specified dimension.
	// For 1D: normalizes entire vector. For 2D: normalizes along rows (dim=0) or columns (dim=1).
	// Returns a new tensor. Panics if dimension is out of range.
	Normalize(dim int) Tensor

	// AddScaled computes t = t + alpha * other (scaled addition).
	// Panics if shapes don't match. Returns self for method chaining.
	AddScaled(other Tensor, alpha float64) Tensor

	// Gradient Routing and Utility Operations
	// ScatterAdd adds values to destination tensor at positions specified by indices.
	// dst: destination tensor (modified in-place, should be zero-initialized)
	// index: indices tensor [batch, channels, outHeight, outWidth] (as int16, linear indices into dst)
	// value: values to add [batch, channels, outHeight, outWidth]
	// For each position in index, adds the corresponding value from value to dst[index[i]].
	// This is a general scatter operation useful for gradient routing in backpropagation.
	// Returns the destination tensor.
	ScatterAdd(dst, index, value Tensor) Tensor
}

