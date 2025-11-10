package types

// Math defines common tensor math operations including reductions and linear algebra.
// This interface contains operations for mathematical computations on tensors.
type Math interface {
	// Reduction Operations
	// Reduction operations reduce dimensions and return new tensors with reduced dimensions.

	// Sum sums along specified dimensions. If no dimensions specified, sums all elements (matches tf.reduce_sum).
	// If dst is nil, creates a new tensor.
	// If dst is provided, writes result to dst and returns dst.
	// Panics if dimensions are out of range.
	Sum(dst Tensor, dims []int) Tensor

	// ReduceSum is an alias for Sum (matches TensorFlow naming: tf.reduce_sum).
	ReduceSum(dst Tensor, dims []int) Tensor

	// Mean computes mean along specified dimensions. If no dimensions specified, means all elements (matches tf.reduce_mean).
	// If dst is nil, creates a new tensor.
	// If dst is provided, writes result to dst and returns dst.
	// Panics if dimensions are out of range.
	Mean(dst Tensor, dims []int) Tensor

	// ReduceMean is an alias for Mean (matches TensorFlow naming: tf.reduce_mean).
	ReduceMean(dst Tensor, dims []int) Tensor

	// Max computes maximum along specified dimensions. If no dimensions specified, finds global maximum (matches tf.reduce_max).
	// If dst is nil, creates a new tensor.
	// If dst is provided, writes result to dst and returns dst.
	// Panics if dimensions are out of range.
	Max(dst Tensor, dims []int) Tensor

	// ReduceMax is an alias for Max (matches TensorFlow naming: tf.reduce_max).
	ReduceMax(dst Tensor, dims []int) Tensor

	// Min computes minimum along specified dimensions. If no dimensions specified, finds global minimum (matches tf.reduce_min).
	// If dst is nil, creates a new tensor.
	// If dst is provided, writes result to dst and returns dst.
	// Panics if dimensions are out of range.
	Min(dst Tensor, dims []int) Tensor

	// ReduceMin is an alias for Min (matches TensorFlow naming: tf.reduce_min).
	ReduceMin(dst Tensor, dims []int) Tensor

	// ArgMax returns the index of the maximum element along the specified dimension (matches tf.argmax).
	// If dst is nil, creates a new tensor.
	// If dst is provided, writes result to dst and returns dst.
	// Panics if dimension is out of range.
	ArgMax(dst Tensor, dim int) Tensor

	// ArgMin returns the index of the minimum element along the specified dimension (matches tf.argmin).
	// If dst is nil, creates a new tensor.
	// If dst is provided, writes result to dst and returns dst.
	// Panics if dimension is out of range.
	ArgMin(dst Tensor, dim int) Tensor

	// Linear Algebra Operations
	// All linear algebra operations use optimized BLAS/LAPACK operations when possible.

	// MatMul performs matrix multiplication (matches tf.matmul).
	// For 2D: [M, K] × [K, N] = [M, N]
	// For batched: [B, M, K] × [B, K, N] = [B, M, N]
	// Supports broadcasting: [M, K] × [B, K, N] or [B, M, K] × [K, N]
	// If dst is nil, creates a new tensor.
	// If dst is provided, writes result to dst and returns dst.
	// Panics if dimensions are incompatible.
	MatMul(dst Tensor, other Tensor) Tensor

	// MatMulTransposed performs matrix multiplication with optional transposition.
	// Computes: dst = (transposeA ? t^T : t) × (transposeB ? other^T : other)
	// If dst is nil, creates a new tensor. Returns the destination tensor.
	MatMulTransposed(dst Tensor, other Tensor, transposeA, transposeB bool) Tensor

	// MatVecMulTransposed performs matrix-vector multiplication with scaling: dst = alpha * matrix^T × vector + beta * dst.
	// If dst is nil, creates a new tensor.
	// If dst is provided, writes result to dst and returns dst.
	MatVecMulTransposed(dst Tensor, matrix, vector Tensor, alpha, beta float64) Tensor

	// Dot computes dot product (vector) or Frobenius inner product (matrix).
	// For vectors: dot product of two 1D tensors.
	// For matrices: Frobenius inner product (sum of element-wise products).
	// Panics if shapes are incompatible.
	Dot(other Tensor) float64

	// Tensordot is an alias for Dot (matches TensorFlow naming: tf.tensordot).
	Tensordot(other Tensor) float64

	// Norm computes vector/matrix norm.
	// ord: 0 = L1 norm, 1 = L2 norm, 2 = Frobenius norm (same as L2 for matrices).
	// Panics if ord is invalid.
	Norm(ord int) float64

	// L2Normalize performs L2 normalization along the specified dimension (matches tf.nn.l2_normalize).
	// For 1D: normalizes entire vector. For 2D: normalizes along rows (dim=0) or columns (dim=1).
	// If dst is nil, creates a new tensor.
	// If dst is provided, writes result to dst and returns dst.
	// Panics if dimension is out of range.
	L2Normalize(dst Tensor, dim int) Tensor

	// Normalize is an alias for L2Normalize (matches TensorFlow naming).
	Normalize(dst Tensor, dim int) Tensor

	// AddScaled computes dst = t + alpha * other (scaled addition).
	// If dst is nil, operation is in-place (modifies t) and returns t.
	// If dst is provided, writes result to dst and returns dst.
	// Panics if shapes don't match.
	AddScaled(dst Tensor, other Tensor, alpha float64) Tensor

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
