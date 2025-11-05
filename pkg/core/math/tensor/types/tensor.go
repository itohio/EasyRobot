package types

// Element interface represents a single tensor element with Get and Set methods.
type Element interface {
	Get() float64
	Set(value float64)
}

// Tensor defines the complete interface for tensor operations.
// This interface is the authoritative source of truth for all Tensor capabilities.
// All concrete implementations must satisfy this interface.
// Concrete implementations must be using Value receivers instead of pointers.
type Tensor interface {
	// Core Properties and Access

	// ID returns a unique identifier for the tensor.
	ID() uintptr

	// New creates a new tensor with the given shape and data type.
	// DataType returns the tensor's data type (e.g., DTFP32, DTINT8).
	DataType() DataType

	// Data returns the underlying data storage as any.
	// For FP32 tensors, returns []float32. For INT8 tensors, returns []int8.
	// Use DataType() to determine the actual type before type assertion.
	// Direct data access bypasses tensor abstractions; prefer Elements() for iteration when possible.
	Data() any

	// Shape returns a copy of the tensor's shape (dimensions).
	Shape() Shape

	// Rank returns the number of dimensions in the tensor.
	Rank() int

	// Size returns the total number of elements in the tensor.
	Size() int

	// Empty returns true if the tensor is empty (no shape or data).
	Empty() bool

	// At returns the element at the given multi-dimensional indices.
	// When only one index is provided and tensor rank > 1, uses linear indexing (direct data access).
	// Otherwise, indices must match the tensor's dimensions for multi-dimensional access.
	// Panics if indices are out of bounds or incorrect number of indices provided.
	At(indices ...int) float64

	// SetAt sets the element at the given multi-dimensional indices to the specified value.
	// When only one index is provided and tensor rank > 1, uses linear indexing (direct data access).
	// Otherwise, indices must match the tensor's dimensions for multi-dimensional access.
	// Panics if indices are out of bounds or incorrect number of indices provided.
	SetAt(value float64, indices ...int)

	// Elements creates an iterator over tensor elements (Go 1.22+ range-over-function).
	// Returns a function that can be used in range loops.
	// fixedAxisValuePairs are pairs of axis index and fixed value: axis1, value1, axis2, value2, ...
	// If no pairs provided, iterates over all elements. Iterates in row-major order.
	// Returns Element objects with Get() and Set() methods for element access.
	Elements(fixedAxisValuePairs ...int) func(func(Element) bool)

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

	// Element-wise Operations (In-Place)
	// All in-place operations modify the tensor and return self for method chaining.
	// Panics if shapes don't match (for binary operations) or on invalid operations.

	// Add adds another tensor element-wise in-place: t = t + other.
	// Panics if shapes don't match. Returns self for method chaining.
	Add(other Tensor) Tensor

	// Sub subtracts another tensor element-wise in-place: t = t - other.
	// Panics if shapes don't match. Returns self for method chaining.
	Sub(other Tensor) Tensor

	// Mul multiplies another tensor element-wise in-place: t = t * other.
	// Panics if shapes don't match. Returns self for method chaining.
	Mul(other Tensor) Tensor

	// Div divides by another tensor element-wise in-place: t = t / other.
	// Panics if shapes don't match or on division by zero. Returns self for method chaining.
	Div(other Tensor) Tensor

	// Scale multiplies the tensor by a scalar in-place: t = scalar * t.
	// Returns self for method chaining.
	Scale(scalar float64) Tensor

	// Square computes element-wise square in-place: t[i] = t[i]^2.
	// Returns self for method chaining.
	Square(dst Tensor) Tensor

	// Sqrt computes element-wise square root in-place: t[i] = sqrt(t[i]).
	// Panics on negative values. Returns self for method chaining.
	Sqrt(dst Tensor) Tensor

	// Exp computes element-wise exponential in-place: t[i] = exp(t[i]).
	// Returns self for method chaining.
	Exp(dst Tensor) Tensor

	// Log computes element-wise natural logarithm in-place: t[i] = log(t[i]).
	// Panics on non-positive values. Returns self for method chaining.
	Log(dst Tensor) Tensor

	// Pow computes element-wise power in-place: t[i] = t[i]^power.
	// Returns self for method chaining.
	Pow(dst Tensor, power float64) Tensor

	// Abs computes element-wise absolute value in-place: t[i] = |t[i]|.
	// Returns self for method chaining.
	Abs(dst Tensor) Tensor

	// Sign computes element-wise sign in-place: t[i] = sign(t[i]) (-1, 0, or 1).
	// Returns self for method chaining.
	Sign(dst Tensor) Tensor

	// Cos computes element-wise cosine in-place: t[i] = cos(t[i]).
	// Returns self for method chaining.
	Cos(dst Tensor) Tensor

	// Sin computes element-wise sine in-place: t[i] = sin(t[i]).
	// Returns self for method chaining.
	Sin(dst Tensor) Tensor

	// Negative computes element-wise negation in-place: t[i] = -t[i].
	// Returns self for method chaining.
	Negative(dst Tensor) Tensor

	// Element-wise Operations (Non-Mutating)
	// These operations create new tensors instead of modifying in-place.

	// AddTo computes result = t + other and stores it in dst.
	// If dst is nil, creates a new tensor. If dst is provided, uses it (must match shape).
	// Panics if shapes don't match. Returns the destination tensor.
	AddTo(other Tensor, dst Tensor) Tensor

	// MulTo computes result = t * other (element-wise) and stores it in dst.
	// If dst is nil, creates a new tensor. If dst is provided, uses it (must match shape).
	// Panics if shapes don't match. Returns the destination tensor.
	MulTo(other Tensor, dst Tensor) Tensor

	// Comparison Operations
	// Comparison operations return new tensors with 1.0 where condition is true, 0.0 otherwise (matching TensorFlow behavior).

	// Equal returns a tensor with 1.0 where t == other, 0.0 otherwise.
	// Panics if shapes don't match. Returns a new tensor.
	Equal(other Tensor) Tensor

	// GreaterThan returns a tensor with 1.0 where t > other, 0.0 otherwise.
	// Panics if shapes don't match. Returns a new tensor.
	GreaterThan(other Tensor) Tensor

	// Greater is an alias for GreaterThan (matches TensorFlow naming).
	// Returns a tensor with 1.0 where t > other, 0.0 otherwise.
	Greater(other Tensor) Tensor

	// Less returns a tensor with 1.0 where t < other, 0.0 otherwise.
	// Panics if shapes don't match. Returns a new tensor.
	Less(other Tensor) Tensor

	// Conditional Operations

	// Where performs element-wise selection: result[i] = condition[i] ? a[i] : b[i].
	// All tensors must have the same shape. Returns a new tensor.
	// Panics if shapes don't match.
	Where(condition, a, b Tensor) Tensor

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

	// Broadcasting

	// BroadcastTo broadcasts the tensor to the target shape.
	// Returns a new tensor with the target shape or an error if broadcasting is not possible.
	// Currently creates a clone if shapes match exactly.
	BroadcastTo(shape Shape) (Tensor, error)

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

	// Transpose transposes dimensions. Currently supports 2D only: [M, N] → [N, M].
	// Returns a new tensor. Panics if tensor is not 2D.
	Transpose(dims ...int) Tensor

	// TransposeTo transposes dimensions and stores result in dst.
	// If dst is nil, creates a new tensor. Returns the destination tensor.
	TransposeTo(dst Tensor, dims ...int) Tensor

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

	// Convolution Operations
	// All convolution operations use optimized primitive functions for computation.

	// Conv1D performs 1D convolution (implemented via 2D conv with width=1).
	// Input: [inChannels, length] or [batch, inChannels, length]
	// Kernel: [outChannels, inChannels, kernelLen]
	// Bias: [outChannels] (optional, can be nil)
	// Output: [outChannels, outLen] or [batch, outChannels, outLen]
	// Returns a new tensor. Panics if shapes are incompatible.
	Conv1D(kernel, bias Tensor, stride, padding int) Tensor

	// Conv1DTo performs 1D convolution and stores result in dst.
	// If dst is nil, creates a new tensor. Returns the destination tensor.
	Conv1DTo(kernel, bias Tensor, dst Tensor, stride, padding int) Tensor

	// Conv2D performs 2D convolution.
	// Input: [batch, inChannels, height, width]
	// Kernel: [outChannels, inChannels, kernelH, kernelW]
	// Bias: [outChannels] (optional, can be nil)
	// Stride: [strideH, strideW]
	// Padding: [padH, padW]
	// Output: [batch, outChannels, outHeight, outWidth]
	// Returns a new tensor. Panics if shapes are incompatible.
	Conv2D(kernel, bias Tensor, stride, padding []int) Tensor

	// Conv2DTo performs 2D convolution and stores result in dst.
	// If dst is nil, creates a new tensor. Returns the destination tensor.
	Conv2DTo(kernel, bias Tensor, dst Tensor, stride, padding []int) Tensor

	// Conv2DTransposed performs transposed 2D convolution (deconvolution).
	// Input: [batch, inChannels, height, width]
	// Kernel: [inChannels, outChannels, kernelH, kernelW] (transposed layout)
	// Bias: [outChannels] (optional, can be nil)
	// Output: [batch, outChannels, outHeight, outWidth]
	// Returns a new tensor. Panics if shapes are incompatible.
	Conv2DTransposed(kernel, bias Tensor, stride, padding []int) Tensor

	// Conv2DKernelGrad computes the gradient of the convolution kernel.
	// Used in backpropagation for training convolutional layers.
	// Returns a new tensor with kernel gradient.
	Conv2DKernelGrad(outputGrad, kernel Tensor, stride, padding []int) Tensor

	// Pooling Operations
	// All pooling operations use optimized primitive functions for computation.

	// MaxPool2D performs max pooling operation.
	// Input: [batch, channels, height, width]
	// KernelSize: [kernelH, kernelW]
	// Stride: [strideH, strideW]
	// Padding: [padH, padW]
	// Output: [batch, channels, outHeight, outWidth]
	// Returns a new tensor. Panics if shapes are incompatible.
	MaxPool2D(kernelSize, stride, padding []int) Tensor

	// AvgPool2D performs average pooling operation.
	// Same signature as MaxPool2D. Returns a new tensor.
	AvgPool2D(kernelSize, stride, padding []int) Tensor

	// GlobalAvgPool2D performs global average pooling.
	// Input: [batch, channels, height, width]
	// Output: [batch, channels]
	// Computes mean over spatial dimensions (height, width). Returns a new tensor.
	GlobalAvgPool2D() Tensor

	// AdaptiveAvgPool2D performs adaptive average pooling to fixed output size.
	// Input: [batch, channels, height, width]
	// outputSize: [outHeight, outWidth] - target output spatial dimensions
	// Output: [batch, channels, outHeight, outWidth]
	// Divides input into approximately equal regions and averages each region.
	// Returns a new tensor.
	AdaptiveAvgPool2D(outputSize []int) Tensor

	// Image/Column Conversion
	// These operations convert between image patches and column format for efficient convolution computation.

	// Im2Col converts image patches to columns for GEMM-based convolution.
	// Input: [batch, channels, height, width]
	// Output: [batch*outHeight*outWidth, channels*kernelH*kernelW]
	// Returns a new tensor. Used internally for optimized convolution computation.
	Im2Col(kernelSize, stride, padding []int) Tensor

	// Col2Im converts columns back to image (inverse of Im2Col).
	// Input: [batch*outHeight*outWidth, channels*kernelH*kernelW]
	// Output: [batch, channels, height, width]
	// Returns a new tensor. Used in backpropagation for convolution gradients.
	Col2Im(outputShape, kernelSize, stride, padding []int) Tensor

	// Activation Functions
	// Activation functions apply non-linear transformations element-wise.
	// dst parameter can be nil to create a new tensor, or provided to reuse memory.

	// ReLU applies Rectified Linear Unit activation: result[i] = max(0, t[i]).
	// If dst is nil, creates a new tensor. Returns the destination tensor.
	ReLU(dst Tensor) Tensor

	// Sigmoid applies sigmoid activation: result[i] = 1 / (1 + exp(-t[i])).
	// If dst is nil, creates a new tensor. Returns the destination tensor.
	Sigmoid(dst Tensor) Tensor

	// Tanh applies hyperbolic tangent activation: result[i] = tanh(t[i]).
	// If dst is nil, creates a new tensor. Returns the destination tensor.
	Tanh(dst Tensor) Tensor

	// Softmax applies softmax activation along the specified dimension.
	// result[i] = exp(t[i]) / sum(exp(t[j])) for all j along dimension dim.
	// If dst is nil, creates a new tensor. Returns the destination tensor.
	// Panics if dimension is out of range.
	Softmax(dim int, dst Tensor) Tensor

	// Dropout Operations
	// Dropout operations are used for regularization during training.

	// DropoutForward applies dropout mask during forward pass: result = t * mask.
	// Returns a new tensor with dropout applied.
	DropoutForward(mask Tensor) Tensor

	// DropoutMask creates a dropout mask with given probability and scale.
	// p: probability of keeping an element (0.0 to 1.0)
	// scale: scaling factor (typically 1.0 / (1.0 - p))
	// rng: random number generator (interface{} to avoid importing math/rand; actual type is *rand.Rand)
	// Returns a new tensor with the mask.
	DropoutMask(p, scale float64, rng interface{}) Tensor
}

// Helper functions that would accept Tensor interface:
// - ZerosLike(t Tensor) Tensor
// - OnesLike(t Tensor) Tensor
// - FullLike(t Tensor, value float32) Tensor
//
// NOTE: These helper functions currently return *Tensor (pointer).
// With interface-based design, they should return Tensor interface.
// However, they would need to create concrete EagerTensor instances internally.
//
// NOTE on DropoutMask: The rng parameter is typed as interface{} to avoid
// importing "math/rand" in this interface definition. In actual implementation,
// it should be *rand.Rand. This interface is for documentation purposes only.

// Must is a helper function that panics if the error is not nil.
// It is used to simplify error handling in the code.
// It is similar to the Must function in the standard library, but it is
// more general and can be used with any type.
func Must(t any, err error) any {
	if err != nil {
		panic(err)
	}
	return t
}
