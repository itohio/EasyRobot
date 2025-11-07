# Tensor Interface API Specification

## Overview

The Tensor interface defines the complete contract for tensor operations in the EasyRobot framework. The interface is composed of category interfaces for better organization and maintainability.

### Interface Composition

The main `Tensor` interface embeds the following category interfaces:
- `TensorCore` - Core properties and element access
- `TensorManipulation` - Shape manipulation and copying operations
- `TensorElementWise` - Element-wise mathematical operations
- `TensorMath` - Reduction and linear algebra operations
- `TensorActivations` - Activation functions for neural networks
- `TensorConvolutions` - Convolution operations
- `TensorPooling` - Pooling operations
- `TensorDropout` - Dropout operations

### Helper Types

#### RNG Interface
Random number generator interface used in dropout operations:
```go
type RNG interface {
    Float64() float64
    NormFloat64() float64
}
```

#### Element Interface
Represents a single tensor element with Get/Set methods:
```go
type Element interface {
    Get() float64
    Set(value float64)
}
```

### Data Types

#### DataType Constants
- `DT_UNKNOWN` - Unknown/unsupported type
- `INT64` - 64-bit integer tensors
- `FP64` - 64-bit floating point tensors
- `INT32` - 32-bit integer tensors
- `FP32` - 32-bit floating point tensors (default)
- `INT` - Native integer tensors (32-bit or 64-bit)
- `INT16` - 16-bit integer tensors
- `FP16` - 16-bit floating point tensors
- `INT8` - 8-bit integer tensors
- `INT48` - 4-bit integer tensors unpacked into 8-bit

### Shape Operations

The `Shape` type provides:
- `Rank() int` - Number of dimensions
- `Size() int` - Total number of elements (returns 1 for scalars with len=0, 0 for invalid shapes)
- `Equal(other Shape) bool` - Shape equality check
- `Strides(dst []int) []int` - Compute row-major strides. If dst is nil, allocates a new slice. If dst is provided, reuses it (must have capacity >= rank). Returns the destination slice.
- `IsContiguous(strides []int) bool` - Check if strides describe a dense row-major layout
- `ValidateAxes(axes []int) error` - Validate and normalize axes (sorts axes in-place). Returns error if axes are out of range or duplicated.
- `ToSlice() []int` - Convert to slice (returns a copy of the shape as []int). Returns nil for empty shapes.
- `Clone() Shape` - Create a copy of the shape. Returns nil if the shape is nil.

### Helper Functions

#### Must
Helper function that panics if error is not nil:
```go
func Must(t any, err error) any
```

#### Data Type Helpers
- `TypeFromData(v any) DataType` - Infer data type from value
- `MakeTensorData(dt DataType, size int) any` - Allocate tensor data
- `CloneTensorData(data any) any` - Clone tensor data
- `CloneTensorDataTo(dst DataType, data any) any` - Clone with type conversion
- `GetTensorData[T any](t Tensor) T` - Type-safe data extraction

## Tensor Interface Methods

### Core Properties and Access (TensorCore)

#### Identity and Metadata
- `ID() uintptr` - Returns unique identifier for the tensor
- `DataType() DataType` - Returns the tensor's data type (e.g., FP32, INT8)
- `Data() any` - Returns underlying data storage ([]float32, []int8, etc.)
- `Shape() Shape` - Returns a copy of the tensor's shape (dimensions)
- `Rank() int` - Returns the number of dimensions
- `Size() int` - Returns the total number of elements
- `Empty() bool` - Returns true if the tensor is empty (no shape or data)

#### Memory Layout and Stride Access
- `Strides(dst []int) []int` - Returns the tensor's strides, computing if necessary. If dst is provided and has sufficient capacity, strides are written to dst. For contiguous tensors (strides == nil), computes canonical row-major strides from shape. For non-contiguous tensors, returns stored strides. Returns nil for empty tensors or scalars (rank 0).

- `IsContiguous() bool` - Reports whether the tensor is contiguous (dense row-major layout). A tensor is contiguous if strides == nil (always contiguous, uses canonical strides) OR stored strides match canonical row-major strides for the shape.

- `Offset() int` - Returns the base offset into the data slice. Returns 0 for tensors that start at the beginning of data. Non-zero offset is used for views (e.g., slices) that reference a portion of larger array.

- `DataWithOffset() any` - Returns the data slice adjusted by the tensor's offset. For tensors with offset == 0, returns the data as-is. For tensors with offset > 0, returns a slice starting at the offset. This is a type-agnostic helper that works with all data types. Returns nil if the tensor has no data or if offset is out of bounds.

**Implementation Note**: The `eager_tensor` implementation stores strides and offset internally. Contiguous tensors have `strides = nil` (computed on-demand), while non-contiguous tensors store explicit strides. This design enables zero-copy views while maintaining backward compatibility.

#### Element Access
- `At(indices ...int) float64` - Returns element at given multi-dimensional indices. When only one index is provided and tensor rank > 1, uses linear indexing. Panics if indices are out of bounds.
- `SetAt(value float64, indices ...int)` - Sets element at given multi-dimensional indices. When only one index is provided and tensor rank > 1, uses linear indexing. Panics if indices are out of bounds.
- `Elements(fixedAxisValuePairs ...int) func(func(Element) bool)` - Creates iterator over tensor elements (Go 1.22+ range-over-function). fixedAxisValuePairs are pairs of axis index and fixed value. Iterates in row-major order. Returns Element objects with Get() and Set() methods.

### Manipulation Operations (TensorManipulation)

#### Copying and Cloning
- `Clone() Tensor` - Creates a deep copy of the tensor and returns it as a Tensor interface. The returned tensor is independent of the original.
- `Copy(src Tensor) Tensor` - Copies data from src tensor into this tensor. Both tensors must have the same shape. Supports data type conversion. Uses optimized primitive copy functions. Returns self for method chaining. Panics if shapes don't match.

#### Shape Manipulation
- `Reshape(dst Tensor, newShape Shape) Tensor` - Returns a tensor with the same data but different shape (zero-copy when possible). The total number of elements must remain the same. If dst is nil, creates a new tensor view (zero-copy when possible) that shares the underlying data and preserves strides/offset. If dst is provided, copies reshaped data to dst and returns dst. Panics if newShape is incompatible with current size or if dst shape doesn't match newShape.

- `Slice(dst Tensor, dim int, start int, length int) Tensor` - Extracts a contiguous slice along the specified dimension. Parameters: dim (0-based dimension), start (starting index), length (number of elements). Result has same rank but dimension 'dim' is reduced to 'length'. If dst is nil, creates a zero-copy view with adjusted offset and same strides (Phase 2: currently copies data, will be zero-copy in future). If dst is provided, copies sliced data to dst and returns dst. Panics if dim is out of range, if start+length exceeds dimension size, or if dst shape doesn't match.

#### Dimension Rearrangement
- `Transpose(dst Tensor, dims []int) Tensor` - Transposes dimensions (matches tf.transpose). For 2D: [M, N] → [N, M] (swaps last two dimensions if no dims provided). Uses Permute internally with optimized fp32.ElemCopy for all cases. If dst is nil, creates a zero-copy view with permuted strides (Phase 2: currently copies data, will be zero-copy in future). If dst is provided, writes result to dst and returns dst. Panics if dimensions are invalid.

- `Permute(dst Tensor, dims []int) Tensor` - Permutes dimensions according to the provided permutation. dims: permutation of [0, 1, 2, ..., rank-1]. Example: Permute([]int{1, 0, 2, 3}) swaps dimensions 0 and 1 in a 4D tensor. Uses optimized fp32.ElemCopy with stride-based copying. If dst is nil, creates a zero-copy view with permuted strides (Phase 2: currently copies data, will be zero-copy in future). If dst is provided, writes permuted result to dst and returns dst. Panics if permutation is invalid or if dst shape doesn't match permuted shape.
- `BroadcastTo(dst Tensor, shape Shape) Tensor` - Broadcasts the tensor to the target shape. If dst is nil, creates a new tensor. If dst is provided, writes result to dst and returns dst. Panics if broadcasting is not possible or if dst shape doesn't match target shape.

#### Filling and Padding
- `Fill(dst Tensor, value float64) Tensor` - Fills the tensor with a constant value. If dst is nil, operation is in-place (modifies t) and returns t. If dst is provided, writes result to dst and returns dst. Uses optimized primitive for efficient computation.
- `FillFunc(dst Tensor, f func() float64) Tensor` - Fills the tensor with a value calculated by callback. If dst is nil, operation is in-place (modifies t) and returns t. If dst is provided, writes result to dst and returns dst. Uses optimized primitive for efficient computation.
- `Pad(dst Tensor, padding []int, value float64) Tensor` - Adds padding to tensor with constant value (matches tf.pad). padding: [padBeforeDim0, padAfterDim0, padBeforeDim1, padAfterDim1, ...]. Each dimension has two padding values: before and after. value: constant value to pad with. If dst is nil, creates a new tensor. If dst is provided, writes result to dst and returns dst. Panics if padding values are invalid.
- `Unpad(dst Tensor, padding []int) Tensor` - Removes padding from tensor. padding: [padBeforeDim0, padAfterDim0, padBeforeDim1, padAfterDim1, ...]. Each dimension has two padding values: before and after. If dst is nil, creates a new tensor with padding removed. If dst is provided, copies unpadded data to dst and returns dst. Panics if padding values are invalid, result shape would be invalid, or if dst shape doesn't match unpadded shape.

### Element-Wise Operations (TensorElementWise)

#### Binary Operations (Destination-based)
All binary operations support in-place operations when dst is nil. If dst is provided, result is written to dst and dst is returned.

- `Add(dst Tensor, other Tensor) Tensor` - Element-wise addition: dst = t + other (matches tf.add). If dst is nil, operation is in-place. Panics if shapes don't match.
- `Subtract(dst Tensor, other Tensor) Tensor` - Element-wise subtraction: dst = t - other (matches tf.subtract). If dst is nil, operation is in-place. Panics if shapes don't match.
- `Multiply(dst Tensor, other Tensor) Tensor` - Element-wise multiplication: dst = t * other (matches tf.multiply). If dst is nil, operation is in-place. Panics if shapes don't match.
- `Divide(dst Tensor, other Tensor) Tensor` - Element-wise division: dst = t / other (matches tf.divide). If dst is nil, operation is in-place. Panics if shapes don't match or on division by zero.

#### Scalar Operations (Destination-based)
All scalar operations support in-place operations when dst is nil. If dst is provided, result is written to dst and dst is returned.

- `ScalarMul(dst Tensor, scalar float64) Tensor` - Multiplies the tensor by a scalar: dst = scalar * t (matches tf.scalar_mul). If dst is nil, operation is in-place.
- `AddScalar(dst Tensor, scalar float64) Tensor` - Adds a scalar value to all elements: dst[i] = t[i] + scalar. If dst is nil, operation is in-place.
- `SubScalar(dst Tensor, scalar float64) Tensor` - Subtracts a scalar value from all elements: dst[i] = t[i] - scalar. If dst is nil, operation is in-place.
- `MulScalar(dst Tensor, scalar float64) Tensor` - Multiplies all elements by a scalar: dst[i] = t[i] * scalar. If dst is nil, operation is in-place.
- `DivScalar(dst Tensor, scalar float64) Tensor` - Divides all elements by a scalar: dst[i] = t[i] / scalar. If dst is nil, operation is in-place.

#### Unary Operations (Destination-based)
All unary operations support in-place operations when dst is nil. If dst is provided, result is written to dst and dst is returned.

- `Square(dst Tensor) Tensor` - Element-wise square: dst[i] = t[i]^2 (matches tf.square). If dst is nil, operation is in-place.
- `Sqrt(dst Tensor) Tensor` - Element-wise square root: dst[i] = sqrt(t[i]) (matches tf.sqrt). If dst is nil, operation is in-place. Panics on negative values.
- `Exp(dst Tensor) Tensor` - Element-wise exponential: dst[i] = exp(t[i]) (matches tf.exp). If dst is nil, operation is in-place.
- `Log(dst Tensor) Tensor` - Element-wise natural logarithm: dst[i] = log(t[i]) (matches tf.log). If dst is nil, operation is in-place. Panics on non-positive values.
- `Pow(dst Tensor, power float64) Tensor` - Element-wise power: dst[i] = t[i]^power (matches tf.pow). If dst is nil, operation is in-place.
- `Abs(dst Tensor) Tensor` - Element-wise absolute value: dst[i] = |t[i]| (matches tf.abs). If dst is nil, operation is in-place.
- `Sign(dst Tensor) Tensor` - Element-wise sign: dst[i] = sign(t[i]) (-1, 0, or 1) (matches tf.sign). If dst is nil, operation is in-place.
- `Cos(dst Tensor) Tensor` - Element-wise cosine: dst[i] = cos(t[i]) (matches tf.cos). If dst is nil, operation is in-place.
- `Sin(dst Tensor) Tensor` - Element-wise sine: dst[i] = sin(t[i]) (matches tf.sin). If dst is nil, operation is in-place.
- `Negative(dst Tensor) Tensor` - Element-wise negation: dst[i] = -t[i] (matches tf.negative). If dst is nil, operation is in-place.

#### Comparison Operations
Comparison operations return new tensors with 1.0 where condition is true, 0.0 otherwise (matching TensorFlow behavior). All operations panic if shapes don't match.

- `Equal(other Tensor) Tensor` - Returns a tensor with 1.0 where t == other, 0.0 otherwise. Returns a new tensor.
- `Greater(other Tensor) Tensor` - Returns a tensor with 1.0 where t > other, 0.0 otherwise (matches tf.greater). Returns a new tensor.
- `Less(other Tensor) Tensor` - Returns a tensor with 1.0 where t < other, 0.0 otherwise. Returns a new tensor.
- `NotEqual(other Tensor) Tensor` - Returns a tensor with 1.0 where t != other, 0.0 otherwise (matches tf.not_equal). Returns a new tensor.
- `GreaterEqual(other Tensor) Tensor` - Returns a tensor with 1.0 where t >= other, 0.0 otherwise (matches tf.greater_equal). Returns a new tensor.
- `LessEqual(other Tensor) Tensor` - Returns a tensor with 1.0 where t <= other, 0.0 otherwise (matches tf.less_equal). Returns a new tensor.

#### Conditional Operations
- `Where(dst Tensor, condition, a, b Tensor) Tensor` - Element-wise selection: dst[i] = condition[i] ? a[i] : b[i] (matches tf.where). All tensors must have the same shape. If dst is nil, creates a new tensor. If dst is provided, writes result to dst and returns dst. Panics if shapes don't match.

### Math Operations (TensorMath)

#### Reduction Operations
Reduction operations reduce dimensions and return new tensors with reduced dimensions. All operations support destination tensor parameter. If dst is nil, creates a new tensor. If dst is provided, writes result to dst and returns dst.

- `Sum(dst Tensor, dims []int) Tensor` - Sums along specified dimensions. If no dimensions specified, sums all elements (matches tf.reduce_sum). Panics if dimensions are out of range.
- `ReduceSum(dst Tensor, dims []int) Tensor` - Alias for Sum (matches TensorFlow naming: tf.reduce_sum).
- `Mean(dst Tensor, dims []int) Tensor` - Computes mean along specified dimensions. If no dimensions specified, means all elements (matches tf.reduce_mean). Panics if dimensions are out of range.
- `ReduceMean(dst Tensor, dims []int) Tensor` - Alias for Mean (matches TensorFlow naming: tf.reduce_mean).
- `Max(dst Tensor, dims []int) Tensor` - Computes maximum along specified dimensions. If no dimensions specified, finds global maximum (matches tf.reduce_max). Panics if dimensions are out of range.
- `ReduceMax(dst Tensor, dims []int) Tensor` - Alias for Max (matches TensorFlow naming: tf.reduce_max).
- `Min(dst Tensor, dims []int) Tensor` - Computes minimum along specified dimensions. If no dimensions specified, finds global minimum (matches tf.reduce_min). Panics if dimensions are out of range.
- `ReduceMin(dst Tensor, dims []int) Tensor` - Alias for Min (matches TensorFlow naming: tf.reduce_min).
- `ArgMax(dst Tensor, dim int) Tensor` - Returns the index of the maximum element along the specified dimension (matches tf.argmax). If dst is nil, creates a new tensor. If dst is provided, writes result to dst and returns dst. Panics if dimension is out of range.
- `ArgMin(dst Tensor, dim int) Tensor` - Returns the index of the minimum element along the specified dimension (matches tf.argmin). If dst is nil, creates a new tensor. If dst is provided, writes result to dst and returns dst. Panics if dimension is out of range.

#### Linear Algebra Operations
All linear algebra operations use optimized BLAS/LAPACK operations when possible.

- `MatMul(dst Tensor, other Tensor) Tensor` - Performs matrix multiplication (matches tf.matmul). For 2D: [M, K] × [K, N] = [M, N]. For batched: [B, M, K] × [B, K, N] = [B, M, N]. Supports broadcasting: [M, K] × [B, K, N] or [B, M, K] × [K, N]. If dst is nil, creates a new tensor. If dst is provided, writes result to dst and returns dst. Panics if dimensions are incompatible.
- `MatMulTransposed(dst Tensor, other Tensor, transposeA, transposeB bool) Tensor` - Performs matrix multiplication with optional transposition. Computes: dst = (transposeA ? t^T : t) × (transposeB ? other^T : other). If dst is nil, creates a new tensor. Returns the destination tensor.
- `MatVecMulTransposed(dst Tensor, matrix, vector Tensor, alpha, beta float64) Tensor` - Performs matrix-vector multiplication with scaling: dst = alpha * matrix^T × vector + beta * dst. If dst is nil, creates a new tensor. If dst is provided, writes result to dst and returns dst.
- `Dot(other Tensor) float64` - Computes dot product (vector) or Frobenius inner product (matrix). For vectors: dot product of two 1D tensors. For matrices: Frobenius inner product (sum of element-wise products). Panics if shapes are incompatible.
- `Tensordot(other Tensor) float64` - Alias for Dot (matches TensorFlow naming: tf.tensordot).
- `Norm(ord int) float64` - Computes vector/matrix norm. ord: 0 = L1 norm, 1 = L2 norm, 2 = Frobenius norm (same as L2 for matrices). Panics if ord is invalid.
- `L2Normalize(dst Tensor, dim int) Tensor` - Performs L2 normalization along the specified dimension (matches tf.nn.l2_normalize). For 1D: normalizes entire vector. For 2D: normalizes along rows (dim=0) or columns (dim=1). If dst is nil, creates a new tensor. If dst is provided, writes result to dst and returns dst. Panics if dimension is out of range.
- `Normalize(dst Tensor, dim int) Tensor` - Alias for L2Normalize (matches TensorFlow naming).

#### Scaled Operations
- `AddScaled(dst Tensor, other Tensor, alpha float64) Tensor` - Computes dst = t + alpha * other (scaled addition). If dst is nil, operation is in-place (modifies t) and returns t. If dst is provided, writes result to dst and returns dst. Panics if shapes don't match.

#### Gradient Routing and Utility Operations
- `ScatterAdd(dst, index, value Tensor) Tensor` - Adds values to destination tensor at positions specified by indices. dst: destination tensor (modified in-place, should be zero-initialized). index: indices tensor [batch, channels, outHeight, outWidth] (as int16, linear indices into dst). value: values to add [batch, channels, outHeight, outWidth]. For each position in index, adds the corresponding value from value to dst[index[i]]. This is a general scatter operation useful for gradient routing in backpropagation. Returns the destination tensor.

### Activation Functions (TensorActivations)

Activation functions apply non-linear transformations element-wise. All operations support destination tensor parameter. If dst is nil, creates a new tensor. Returns the destination tensor.

- `ReLU(dst Tensor) Tensor` - Applies Rectified Linear Unit activation: result[i] = max(0, t[i]).
- `Sigmoid(dst Tensor) Tensor` - Applies sigmoid activation: result[i] = 1 / (1 + exp(-t[i])).
- `Tanh(dst Tensor) Tensor` - Applies hyperbolic tangent activation: result[i] = tanh(t[i]).
- `Softmax(dim int, dst Tensor) Tensor` - Applies softmax activation along the specified dimension. result[i] = exp(t[i]) / sum(exp(t[j])) for all j along dimension dim. Panics if dimension is out of range.
- `ReLU6(dst Tensor) Tensor` - Applies ReLU6 activation: result[i] = min(max(t[i], 0), 6) (matches tf.nn.relu6).
- `LeakyReLU(dst Tensor, alpha float64) Tensor` - Applies Leaky ReLU activation: result[i] = max(t[i], alpha * t[i]) (matches tf.nn.leaky_relu).
- `ELU(dst Tensor, alpha float64) Tensor` - Applies ELU activation: result[i] = t[i] > 0 ? t[i] : alpha * (exp(t[i]) - 1) (matches tf.nn.elu).
- `Softplus(dst Tensor) Tensor` - Applies softplus activation: result[i] = log(1 + exp(t[i])) (matches tf.nn.softplus).
- `Swish(dst Tensor) Tensor` - Applies Swish activation: result[i] = t[i] * sigmoid(t[i]) (matches tf.nn.swish).
- `GELU(dst Tensor) Tensor` - Applies GELU activation: result[i] = t[i] * 0.5 * (1 + erf(t[i]/sqrt(2))) (matches tf.nn.gelu).

### Convolution Operations (TensorConvolutions)

#### Forward Convolution
All forward convolution operations support destination tensor parameter. If dst is nil, creates a new tensor. If dst is provided, writes result to dst and returns dst. All operations panic if shapes are incompatible.

- `Conv1D(dst Tensor, kernel, bias Tensor, stride, padding int) Tensor` - Performs 1D convolution (implemented via 2D conv with width=1) (matches tf.nn.conv1d). Input: [inChannels, length] or [batch, inChannels, length]. Kernel: [outChannels, inChannels, kernelLen]. Bias: [outChannels] (optional, can be nil). Output: [outChannels, outLen] or [batch, outChannels, outLen].
- `Conv2D(dst Tensor, kernel, bias Tensor, stride, padding []int) Tensor` - Performs 2D convolution (matches tf.nn.conv2d). Input: [batch, inChannels, height, width]. Kernel: [outChannels, inChannels, kernelH, kernelW]. Bias: [outChannels] (optional, can be nil). Stride: [strideH, strideW]. Padding: [padH, padW]. Output: [batch, outChannels, outHeight, outWidth].
- `Conv2DTransposed(dst Tensor, kernel, bias Tensor, stride, padding []int) Tensor` - Performs transposed 2D convolution (deconvolution) (matches tf.nn.conv2d_transpose). Input: [batch, inChannels, height, width]. Kernel: [inChannels, outChannels, kernelH, kernelW] (transposed layout). Bias: [outChannels] (optional, can be nil). Output: [batch, outChannels, outHeight, outWidth].

#### Gradient Operations
Gradient operations are used in backpropagation for training convolutional layers. All return new tensors.

- `Conv2DKernelGrad(outputGrad, kernel Tensor, stride, padding []int) Tensor` - Computes the gradient of the convolution kernel. Used in backpropagation for training convolutional layers. Returns a new tensor with kernel gradient.
- `Conv1DKernelGrad(outputGrad, kernel Tensor, stride, padding int) Tensor` - Computes the gradient of the 1D convolution kernel. Used in backpropagation for training 1D convolutional layers. Returns a new tensor with kernel gradient.

#### Image/Column Conversion
These operations convert between image patches and column format for efficient convolution computation. Used internally for optimized convolution computation.

- `Im2Col(kernelSize, stride, padding []int) Tensor` - Converts image patches to columns for GEMM-based convolution. Input: [batch, channels, height, width]. Output: [batch*outHeight*outWidth, channels*kernelH*kernelW]. Returns a new tensor. Used internally for optimized convolution computation.
- `Col2Im(outputShape, kernelSize, stride, padding []int) Tensor` - Converts columns back to image (inverse of Im2Col). Input: [batch*outHeight*outWidth, channels*kernelH*kernelW]. Output: [batch, channels, height, width]. Returns a new tensor. Used in backpropagation for convolution gradients.

### Pooling Operations (TensorPooling)

Pooling operations perform downsampling on tensors. All operations panic if shapes are incompatible.

#### Max Pooling
- `MaxPool2D(dst Tensor, kernelSize, stride, padding []int) Tensor` - Performs max pooling operation. Input: [batch, channels, height, width]. KernelSize: [kernelH, kernelW]. Stride: [strideH, strideW]. Padding: [padH, padW]. Output: [batch, channels, outHeight, outWidth]. If dst is nil, creates a new tensor. If dst is provided, writes result to dst and returns dst. Panics if shapes are incompatible.
- `MaxPool2DWithIndices(dst Tensor, indicesDst Tensor, kernelSize, stride, padding []int) (Tensor, Tensor)` - Performs max pooling and returns both output and indices. Input: [batch, channels, height, width]. KernelSize: [kernelH, kernelW]. Stride: [strideH, strideW]. Padding: [padH, padW]. Output: [batch, channels, outHeight, outWidth]. Indices: [batch, channels, outHeight, outWidth] (as int32, linear indices into input). Returns: (output Tensor, indices Tensor). If dst is nil, creates a new output tensor. If dst is provided, writes result to dst. If indicesDst is nil, creates a new indices tensor. If indicesDst is provided, writes indices to indicesDst. The indices are used for efficient backward pass computation.
- `MaxPool2DBackward(dst Tensor, gradOutput, indices Tensor, kernelSize, stride, padding []int) Tensor` - Performs backward pass for max pooling using stored indices. gradOutput: input gradient [batch, channels, outHeight, outWidth]. indices: indices from forward pass [batch, channels, outHeight, outWidth] (as int32). kernelSize, stride, padding: pooling parameters. If dst is nil, creates a new tensor. If dst is provided, writes gradient to dst and returns dst. Returns: gradient w.r.t. input [batch, channels, inHeight, inWidth].

#### Average Pooling
- `AvgPool2D(dst Tensor, kernelSize, stride, padding []int) Tensor` - Performs average pooling operation. Same signature as MaxPool2D. If dst is nil, creates a new tensor. If dst is provided, writes result to dst and returns dst.
- `AvgPool2DBackward(dst Tensor, gradOutput Tensor, kernelSize, stride, padding []int) Tensor` - Performs backward pass for average pooling. gradOutput: input gradient [batch, channels, outHeight, outWidth]. kernelSize, stride, padding: pooling parameters. If dst is nil, creates a new tensor. If dst is provided, writes gradient to dst and returns dst. Returns: gradient w.r.t. input [batch, channels, inHeight, inWidth].

#### Global and Adaptive Pooling
- `GlobalAvgPool2D(dst Tensor) Tensor` - Performs global average pooling. Input: [batch, channels, height, width]. Output: [batch, channels]. Computes mean over spatial dimensions (height, width). If dst is nil, creates a new tensor. If dst is provided, writes result to dst and returns dst.
- `AdaptiveAvgPool2D(dst Tensor, outputSize []int) Tensor` - Performs adaptive average pooling to fixed output size. Input: [batch, channels, height, width]. outputSize: [outHeight, outWidth] - target output spatial dimensions. Output: [batch, channels, outHeight, outWidth]. Divides input into approximately equal regions and averages each region. If dst is nil, creates a new tensor. If dst is provided, writes result to dst and returns dst.

### Dropout Operations (TensorDropout)

Dropout operations are used for regularization during training of neural networks.

- `DropoutForward(dst Tensor, mask Tensor) Tensor` - Applies dropout mask during forward pass: dst = t * mask. If dst is nil, creates a new tensor. If dst is provided, writes result to dst and returns dst.
- `DropoutMask(p, scale float64, rng RNG) Tensor` - Creates a dropout mask with given probability and scale. p: probability of keeping an element (0.0 to 1.0). scale: scaling factor (typically 1.0 / (1.0 - p)). rng: random number generator implementing RNG interface (e.g., *rand.Rand). Returns a new tensor with the mask.
- `DropoutBackward(dst Tensor, gradOutput, mask Tensor) Tensor` - Computes dropout backward pass: dst = gradOutput * mask. gradOutput: gradient from next layer. mask: dropout mask used in forward pass. If dst is nil, creates a new tensor. If dst is provided, writes result to dst and returns dst.

## Common Patterns and Conventions

### Destination Parameter Pattern

Many operations follow a consistent pattern with a `dst` (destination) parameter:
- **If `dst` is `nil`**: The operation is typically in-place (modifies the receiver tensor) or creates a new tensor, depending on the operation semantics.
- **If `dst` is provided**: The result is written to `dst` and `dst` is returned, enabling efficient reuse of pre-allocated tensors.

This pattern allows for:
- Memory-efficient in-place operations when `dst` is `nil`
- Pre-allocated buffer reuse for performance-critical code
- Clear separation between mutation and creation

### Error Handling

Most operations use panic-based error handling for invalid inputs (e.g., shape mismatches, out-of-bounds indices). This design choice prioritizes performance and simplicity.

### Shape Conventions

- Shapes are represented as `[]int` slices (the `Shape` type)
- Dimensions are 0-indexed
- Row-major (C-style) memory layout is used by default
- Batch dimension is typically the first dimension for neural network operations

### Data Type Handling

- Operations work with the underlying data types (FP32, INT8, etc.) transparently
- Type conversion is handled automatically in operations like `Copy`
- Use `DataType()` to check the data type before type assertion on `Data()`
- Use `GetTensorData[T]()` for type-safe data extraction

### TensorFlow Compatibility

Many operations are designed to match TensorFlow behavior:
- Comparison operations return 1.0/0.0 tensors
- Reduction operations match `tf.reduce_*` semantics
- Convolution operations follow TensorFlow's NHWC/NCHW conventions
- Padding follows TensorFlow's padding format

### Performance Considerations

- **Zero-Copy Operations**: Operations attempt zero-copy when possible (e.g., `Reshape` creates views, `Slice` and `Transpose` will create views in Phase 2). Views share the underlying data and maintain their own shape/strides/offset metadata.

- **Stride Optimization**: The implementation stores strides internally to avoid recomputing them on every operation. Contiguous tensors use lazy stride computation (strides computed on-demand), while non-contiguous tensors store explicit strides.

- **Stack Allocation**: Stride computations use stack-allocated arrays (MAX_DIMS = 16) to avoid heap allocations in hot paths.

- **Memory Efficiency**: Views enable multiple tensors to share the same backing array, reducing memory usage and enabling efficient in-place operations.

- **Optimized Primitives**: Optimized primitives are used for common operations (e.g., BLAS/LAPACK for linear algebra, SIMD for element-wise operations).

- **Destination Reuse**: Prefer reusing destination tensors to reduce allocations.

- **Iterator Pattern**: Use `Elements()` iterator for efficient element-wise iteration.

### Implementation Notes

- Concrete implementations must use value receivers (not pointers) to satisfy the interface
- The Tensor interface is designed to be composable with category interfaces
- Helper types (`Element`, `RNG`) enable flexible implementations
- The interface design supports both eager and lazy evaluation backends
