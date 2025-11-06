package types

// TensorManipulation defines operations for copying, cloning, reshaping, and manipulating tensor structure.
// This interface contains operations that modify tensor shape or create new tensor views.
type TensorManipulation interface {
	// Clone creates a deep copy of the tensor and returns it as a Tensor interface.
	// The returned tensor is independent of the original.
	Clone() Tensor

	// Copy copies data from src tensor into this tensor.
	// Both tensors must have the same shape.
	// Supports data type conversion between different tensor data types.
	// Uses optimized primitive copy functions for efficient copying.
	// Returns self for method chaining. Panics if shapes don't match.
	Copy(src Tensor) Tensor

	// Reshape returns a new tensor with the same data but different shape (zero-copy when possible).
	// The total number of elements must remain the same.
	// Returns Tensor interface. Panics if newShape is incompatible with current size.
	Reshape(newShape Shape) Tensor

	// Slice extracts a contiguous slice along the specified dimension.
	// Returns a new tensor view (zero-copy when possible) with the sliced data.
	// Parameters:
	//   - dim: dimension along which to slice (0-based)
	//   - start: starting index along the dimension (inclusive)
	//   - length: number of elements to extract along the dimension
	// The result tensor has the same rank as the input, but dimension 'dim' is reduced to 'length'.
	// Panics if dim is out of range, or if start+length exceeds the dimension size.
	// Example: For tensor [2, 4, 3], Slice(1, 1, 2) returns [2, 2, 3] (slicing dimension 1 from index 1 to 3).
	Slice(dim int, start int, length int) Tensor

	// Transpose transposes dimensions (matches tf.transpose).
	// For 2D: [M, N] → [N, M] (swaps last two dimensions if no dims provided)
	// For 4D+: uses Permute to rearrange dimensions
	// If dst is nil, creates a new tensor.
	// If dst is provided, writes result to dst and returns dst.
	// Panics if dimensions are invalid.
	Transpose(dst Tensor, dims []int) Tensor

	// Permute permutes dimensions according to the provided permutation.
	// dims: permutation of [0, 1, 2, ..., rank-1]
	// Example: Permute([]int{1, 0, 2, 3}) swaps dimensions 0 and 1 in a 4D tensor.
	// Returns a new tensor. Panics if permutation is invalid.
	Permute(dims []int) Tensor

	// BroadcastTo broadcasts the tensor to the target shape.
	// Returns a new tensor with the target shape or an error if broadcasting is not possible.
	// Currently creates a clone if shapes match exactly.
	BroadcastTo(shape Shape) (Tensor, error)

	// Fill fills the tensor with a constant value.
	// If dst is nil, operation is in-place (modifies t) and returns t.
	// If dst is provided, writes result to dst and returns dst.
	// Uses optimized primitive for efficient computation.
	Fill(dst Tensor, value float64) Tensor

	// Pad adds padding to tensor with constant value (matches tf.pad).
	// padding: [padBeforeDim0, padAfterDim0, padBeforeDim1, padAfterDim1, ...]
	// Each dimension has two padding values: before and after.
	// value: constant value to pad with
	// If dst is nil, creates a new tensor.
	// If dst is provided, writes result to dst and returns dst.
	// Panics if padding values are invalid.
	Pad(dst Tensor, padding []int, value float64) Tensor

	// Unpad removes padding from tensor.
	// padding: [padBeforeDim0, padAfterDim0, padBeforeDim1, padAfterDim1, ...]
	// Each dimension has two padding values: before and after.
	// Returns a new tensor with padding removed.
	// Panics if padding values are invalid or result shape would be invalid.
	Unpad(padding []int) Tensor
}
