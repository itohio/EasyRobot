# math/tensor - Tensor Operations Specification

## Overview

The `tensor` package provides multi-dimensional tensor operations optimized for embedded systems with minimal allocations, supporting numerical computing, computer vision, and machine learning applications. All tensors use **row-major** storage layout matching Go nested arrays.

**Foundation**: This package builds on the `math/primitive/fp32` package which provides BLAS levels 1-3 operations with zero allocations, stride-based access, and row-major storage. See [primitive/SPEC.md](../primitive/SPEC.md) for details.

**Quantization**: For INT8 quantized computation support:
- Design document: [QUANTIZATION_PLAN.md](./QUANTIZATION_PLAN.md)
- Implementation roadmap: [QUANTIZATION_IMPLEMENTATION.md](./QUANTIZATION_IMPLEMENTATION.md)

## Design Principles

1. **Row-Major Storage**: All tensors stored in row-major order (matching Go nested arrays layout `[][]float32`)
2. **Zero Allocations**: Critical paths use `primitive` package for allocation-free operations
3. **Stride-based Indexing**: Efficient multi-dimensional access with leading dimensions (ld)
4. **In-Place Operations**: Many operations modify tensors in-place for efficiency
5. **Method Chaining**: Operations return the tensor to enable method chaining
6. **Primitive Integration**: All linear algebra uses `math/primitive` BLAS/LAPACK operations
7. **Float32 Precision**: Uses `float32` for embedded-friendly precision
8. **Batch Support**: Automatic handling of batched operations
9. **Value Semantics**: Tensor arguments are passed by value (`Tensor`), not by pointer. This enables efficient zero-copy operations while maintaining clear ownership semantics. Return values are pointers (`*Tensor`) to indicate newly allocated tensors.

## Tensor Storage Layout

### Row-Major Storage

All tensors use **row-major** storage (matching Go nested arrays layout):

```go
// Tensor A: shape [M, N]
// Element A[i][j] stored at index: i*N + j
// where N = number of columns (last dimension)

// Example: 3x2 tensor A
A = [a00 a01
     a10 a11  
     a20 a21]

// Row-major storage (contiguous):
storage = [a00, a01, a10, a11, a20, a21]
//          row0    row1    row2

// Access formula:
A[i][j] = storage[i*N + j]
```

### Multi-Dimensional Storage

For higher-dimensional tensors, storage is batch-major then row-major:

```go
// Tensor A: shape [B, C, H, W] (batch, channels, height, width)
// Element A[b][c][h][w] stored at:
//   batchOffset = b * (C * H * W)
//   channelOffset = c * (H * W)
//   rowOffset = h * W
//   colOffset = w
//   index = batchOffset + channelOffset + rowOffset + colOffset

// Storage layout:
// [batch0_channel0_row0, batch0_channel0_row1, ..., batch0_channel1_row0, ..., batch1_channel0_row0, ...]
```

**Why Row-Major?**
- Matches Go nested arrays layout: `[][]float32` is row-major
- Consistent with `primitive` package conventions
- Compatible with BLAS/LAPACK operations (via stride parameters)
- Efficient for row-oriented operations

## Core Tensor Structure

### Tensor Type

```go
type Tensor struct {
    dtype DataType  // Data type (currently FP32)
    shape Shape     // Tensor dimensions
    data  []float32 // Flat data storage (single allocation)
}
```

**Characteristics:**
- Typed data storage with extensible DataType system
- Shape abstraction for dimension management
- Single contiguous backing array (`[]float32`)
- Zero-copy operations where possible
- Row-major storage layout (matching Go nested arrays)

### DataType

```go
type DataType uint8

const (
    DTFP32 DataType = iota // 32-bit floating point (default)
)
```

**Current Support:**
- `DTFP32`: 32-bit floating point tensors (primary data type)
- Extensible for future data types (INT8, FP16, etc.)

### Shape

```go
type Shape []int // Represents tensor dimensions
```

**Shape Methods:**
- `Rank() int`: Returns number of dimensions
- `Size() int`: Returns total number of elements
- `Strides() []int`: Computes row-major strides
- `IsContiguous(strides []int) bool`: Checks if strides represent contiguous layout
- `ValidateAxes(axes []int) error`: Validates dimension indices
- `Iterator(fixedAxisValuePairs ...int) func(func([]int) bool)`: Creates iterator over shape indices (Go 1.22+ range-over-function)

### Tensor Creation

```go
// New creates a new tensor with the provided data type and shape
// The underlying buffer is zero-initialized
func New(dtype DataType, shape Shape) Tensor

// FromFloat32 constructs an FP32 tensor from an existing backing slice
// If data is nil, a new buffer is allocated. The slice is used directly (no copy)
func FromFloat32(shape Shape, data []float32) Tensor
```

### Helper Functions

```go
// DataType returns the tensor's data type
func (t Tensor) DataType() DataType

// Shape returns a copy of the tensor's shape
func (t Tensor) Shape() Shape

// Rank returns the number of dimensions
func (t Tensor) Rank() int

// Size returns the total number of elements in the tensor
func (t Tensor) Size() int

// Clone creates a deep copy of the tensor
func (t Tensor) Clone() *Tensor

// At returns the element at the given indices
func (t Tensor) At(indices ...int) float32

// SetAt sets the element at the given indices
func (t *Tensor) SetAt(indices []int, value float32)

// Elements creates an iterator over tensor elements (Go 1.22+ range-over-function)
// Returns Element objects with Get() and Set() methods
func (t *Tensor) Elements(fixedAxisValuePairs ...int) func(func(Element) bool)

// Reshape returns a new tensor with the same data but different shape (zero-copy when possible)
func (t Tensor) Reshape(newShape []int) *Tensor

// Data returns the underlying data slice (zero-copy)
// Deprecated: Use Elements() for element access instead
func (t Tensor) Data() []float32

// Flat returns the underlying data slice (zero-copy)
// Deprecated: Use Elements() for element access instead
func (t Tensor) Flat() []float32
```

### Element Type

```go
// Element represents a single tensor element with Get and Set methods
type Element struct {
    tensor *Tensor
    index  int
}

// Get returns the float32 value at this element's position
func (e Element) Get() float32

// Set sets the float32 value at this element's position
func (e Element) Set(value float32)
```

### Shape Iterator

```go
// Iterator creates an iterator that fixes specified dimensions and iterates over the remaining ones
// Returns a function that can be used in Go 1.22+ range loops
// fixedAxisValuePairs are pairs of axis index and fixed value: axis1, value1, axis2, value2, ...
// The iterator yields complete indices for all dimensions that can be used with At or SetAt
func (s Shape) Iterator(fixedAxisValuePairs ...int) func(func([]int) bool)
```

**Element Access:**
- `At(indices ...int)`: Returns the element at the given multi-dimensional indices
- `SetAt(indices []int, value float32)`: Sets the element at the given indices
- Both functions validate indices and compute linear index using strides

**Element Iteration:**
- `Elements(fixedAxisValuePairs ...int)`: Creates an iterator over tensor elements (Go 1.22+ range-over-function)
  - Returns `Element` objects with `Get() float32` and `Set(value float32)` methods
  - Supports fixing dimensions: `Elements(0, 1, 2, 3)` fixes axis 0 at 1, axis 2 at 3
  - Iterates in row-major order (last dimension changes fastest)
  - Recommended for neural network operations over direct data access

**Data Access (Deprecated):**
- `Data() []float32`: ⚠️ **Deprecated** - Use `Elements()` instead
- `Flat() []float32`: ⚠️ **Deprecated** - Use `Elements()` instead
- Direct access to underlying data is discouraged for neural network operations as it bypasses tensor abstractions

**Reshaping:**
- `Reshape(newShape []int) *Tensor`: Creates a new tensor view with different shape but same data
- The total number of elements must remain the same
- Returns a zero-copy view when possible

## Tensor Operations

Beyond basic element-wise operations, the tensor package provides higher-level operations optimized for neural network computations and general numerical computing.

### Element-Wise Operations

#### In-Place Operations

All in-place operations modify the tensor and return self for method chaining.

| Function | Description | Primitive Used | Status |
|----------|-------------|----------------|--------|
| `Add(other Tensor) *Tensor` | Add tensor element-wise | `fp32.Axpy` | ✅ |
| `Sub(other Tensor) *Tensor` | Subtract tensor element-wise | `fp32.Axpy` (alpha=-1) | ✅ |
| `Mul(other Tensor) *Tensor` | Multiply element-wise | `fp32.ElemMul` | ✅ |
| `Div(other Tensor) *Tensor` | Divide element-wise | `fp32.ElemDiv` | ✅ |
| `Scale(scalar float32) *Tensor` | Multiply by scalar | `fp32.Scal` | ✅ |
| `Square() *Tensor` | Element-wise square | `fp32.ElemSquare` | ✅ |
| `Sqrt() *Tensor` | Element-wise square root | `fp32.ElemSqrt` | ✅ |
| `Exp() *Tensor` | Element-wise exponential | `fp32.ElemExp` | ✅ |
| `Log() *Tensor` | Element-wise natural logarithm | `fp32.ElemLog` | ✅ |
| `Pow(power float32) *Tensor` | Element-wise power | `fp32.ElemPow` | ✅ |
| `Abs() *Tensor` | Element-wise absolute value | `fp32.ElemAbs` | ✅ |
| `Sign() *Tensor` | Element-wise sign (-1, 0, or 1) | `fp32.ElemSign` | ✅ |
| `Cos() *Tensor` | Element-wise cosine | `fp32.ElemCos` | ✅ |
| `Sin() *Tensor` | Element-wise sine | `fp32.ElemSin` | ✅ |
| `Negative() *Tensor` | Element-wise negation | `fp32.ElemNegative` | ✅ |

**Function Signatures:**

```go
// Add: t = t + other (in-place)
func (t *Tensor) Add(other Tensor) *Tensor

// Sub: t = t - other (in-place)
func (t *Tensor) Sub(other Tensor) *Tensor

// Mul: t = t * other (element-wise, in-place)
func (t *Tensor) Mul(other Tensor) *Tensor

// Div: t = t / other (element-wise, in-place)
func (t *Tensor) Div(other Tensor) *Tensor

// Scale: t = scalar * t (in-place)
func (t *Tensor) Scale(scalar float32) *Tensor

// Square: t[i] = t[i]^2 (in-place)
func (t *Tensor) Square() *Tensor

// Sqrt: t[i] = sqrt(t[i]) (in-place)
func (t *Tensor) Sqrt() *Tensor

// Exp: t[i] = exp(t[i]) (in-place)
func (t *Tensor) Exp() *Tensor

// Log: t[i] = log(t[i]) (in-place)
func (t *Tensor) Log() *Tensor

// Pow: t[i] = t[i]^power (in-place)
func (t *Tensor) Pow(power float32) *Tensor

// Abs: t[i] = abs(t[i]) (in-place)
func (t *Tensor) Abs() *Tensor

// Sign: t[i] = sign(t[i]) (in-place, returns -1, 0, or 1)
func (t *Tensor) Sign() *Tensor

// Cos: t[i] = cos(t[i]) (in-place)
func (t *Tensor) Cos() *Tensor

// Sin: t[i] = sin(t[i]) (in-place)
func (t *Tensor) Sin() *Tensor

// Negative: t[i] = -t[i] (in-place)
func (t *Tensor) Negative() *Tensor
```

**Parameters:**
- `other`: Tensor with same shape as `t` (passed by value)
- `scalar`: Scalar multiplier

**Note**: Tensor arguments are passed by value (`Tensor`), not by pointer. This allows efficient zero-copy operations while maintaining clear ownership semantics.

#### Comparison Operations

Comparison operations return new tensors with 1.0 where the condition is true, 0.0 otherwise (matching TensorFlow behavior).

| Function | Description | Primitive Used | Status |
|----------|-------------|----------------|--------|
| `Equal(other Tensor) *Tensor` | Element-wise equality (1.0 if equal, 0.0 otherwise) | `fp32.ElemEqual` | ✅ |
| `Greater(other Tensor) *Tensor` | Element-wise greater than (1.0 if t > other, 0.0 otherwise) | `fp32.ElemGreaterThan` | ✅ |
| `GreaterThan(other Tensor) *Tensor` | Alias for Greater (matches TensorFlow naming) | `fp32.ElemGreaterThan` | ✅ |
| `Less(other Tensor) *Tensor` | Element-wise less than (1.0 if t < other, 0.0 otherwise) | `fp32.ElemLess` | ✅ |

**Function Signatures:**

```go
// Equal: Returns tensor with 1.0 where t == other, 0.0 otherwise
func (t Tensor) Equal(other Tensor) *Tensor

// Greater: Returns tensor with 1.0 where t > other, 0.0 otherwise
// Note: This is an alias for GreaterThan to match TensorFlow naming
func (t Tensor) Greater(other Tensor) *Tensor

// GreaterThan: Returns tensor with 1.0 where t > other, 0.0 otherwise
func (t Tensor) GreaterThan(other Tensor) *Tensor

// Less: Returns tensor with 1.0 where t < other, 0.0 otherwise
func (t Tensor) Less(other Tensor) *Tensor
```

**Note**: Comparison operations create new tensors (non-mutating) to match TensorFlow's behavior. They return boolean-like tensors where 1.0 represents true and 0.0 represents false.

### Operations Creating New Tensors

| Function | Description | Status |
|----------|-------------|--------|
| `AddTo(other Tensor, dst *Tensor) *Tensor` | Add to destination (or create new) | ✅ |
| `MulTo(other Tensor, dst *Tensor) *Tensor` | Multiply to destination (or create new) | ✅ |

**Function Signatures:**

```go
// AddTo: result = t + other
// If dst is nil, creates new tensor
// If dst is provided, uses it (must match shape)
func (t *Tensor) AddTo(other Tensor, dst *Tensor) *Tensor

// MulTo: result = t * other (element-wise)
// If dst is nil, creates new tensor
// If dst is provided, uses it (must match shape)
func (t *Tensor) MulTo(other Tensor, dst *Tensor) *Tensor
```

## Reduction Operations

All reduction operations return new tensors with reduced dimensions.

| Function | Description | Primitive Used | Status |
|----------|-------------|----------------|--------|
| `Sum(dims ...int) *Tensor` | Sum along dimensions | `primitive.Asum` (vector case) | ✅ |
| `Mean(dims ...int) *Tensor` | Mean along dimensions | - | ✅ |
| `Max(dims ...int) *Tensor` | Maximum along dimensions | - | ✅ |
| `Min(dims ...int) *Tensor` | Minimum along dimensions | - | ✅ |
| `ArgMax(dim int) *Tensor` | Index of maximum element | `primitive.Iamax` (vector case) | ✅ |

**Function Signatures:**

```go
// Sum: Sum along specified dimensions (all dimensions if none specified)
func (t Tensor) Sum(dims ...int) *Tensor

// Mean: Mean along specified dimensions (all dimensions if none specified)
func (t Tensor) Mean(dims ...int) *Tensor

// Max: Maximum along specified dimensions (all dimensions if none specified)
func (t Tensor) Max(dims ...int) *Tensor

// Min: Minimum along specified dimensions (all dimensions if none specified)
func (t Tensor) Min(dims ...int) *Tensor

// ArgMax: Index of maximum element along specified dimension
func (t Tensor) ArgMax(dim int) *Tensor
```

**Examples:**

```go
// Sum all elements
sum := t.Sum() // Returns scalar tensor [1]

// Sum along first dimension
sum := t.Sum(0) // Reduces first dimension

// Sum along multiple dimensions
sum := t.Sum(0, 2) // Reduces dimensions 0 and 2
```

## Broadcasting

| Function | Description | Status |
|----------|-------------|--------|
| `BroadcastTo(shape []int) (*Tensor, error)` | Broadcast tensor to new shape | ✅ |

**Function Signature:**

```go
// BroadcastTo: Broadcast tensor to target shape
// Returns error if broadcasting is not possible
func (t Tensor) BroadcastTo(shape []int) (*Tensor, error)
```

**Notes:**
- Currently creates clone if shapes match exactly
- Future: Implement efficient broadcasting without copying
- Validates that broadcasting is possible (dimensions are compatible)

## Linear Algebra Operations

All linear algebra operations use `math/primitive/fp32` functions for optimal performance.

### Matrix Operations

| Function | Description | Primitive Used | Status |
|----------|-------------|----------------|--------|
| `MatMul(other Tensor) *Tensor` | Matrix multiplication | `fp32.Gemm_*`, `GemmBatched`, `GemmStrided` | ✅ |
| `MatMulTo(other Tensor, dst *Tensor) *Tensor` | Matrix multiply to destination | - | ✅ |
| `Transpose(dims ...int) *Tensor` | Transpose dimensions | Manual implementation | ✅ (2D only) |
| `TransposeTo(dst *Tensor, dims ...int) *Tensor` | Transpose to destination | - | ✅ (2D only) |

**Function Signatures:**

```go
// MatMul: Matrix multiplication
// For 2D: [M, K] × [K, N] = [M, N]
// For batched: [B, M, K] × [B, K, N] = [B, M, N]
// Supports broadcasting: [M, K] × [B, K, N] or [B, M, K] × [K, N]
func (t Tensor) MatMul(other Tensor) *Tensor

// MatMulTo: Matrix multiply to destination
func (t Tensor) MatMulTo(other Tensor, dst *Tensor) *Tensor

// Transpose: Transpose dimensions (currently supports 2D only)
// [M, N] → [N, M]
func (t Tensor) Transpose(dims ...int) *Tensor

// TransposeTo: Transpose to destination
func (t Tensor) TransposeTo(dst *Tensor, dims ...int) *Tensor
```

**MatMul Details:**
- Automatically detects 2D vs batched cases
- Uses `fp32.Gemm_NN` for 2D case
- Uses `fp32.GemmStrided` for batched contiguous tensors
- Uses `fp32.GemmBatched` for batched strided access
- Handles broadcasting for compatible batch dimensions
- Supports three broadcasting patterns:
  - Same batch size: `[B, M, K] × [B, K, N]`
  - Broadcast first: `[M, K] × [B, K, N]`
  - Broadcast second: `[B, M, K] × [K, N]`

### Dot Products and Norms

| Function | Description | Primitive Used | Status |
|----------|-------------|----------------|--------|
| `Dot(other Tensor) float32` | Dot product | `fp32.Dot` (vector case) | ✅ |
| `Norm(ord int) float32` | Vector/matrix norm | `fp32.Nrm2`, `fp32.Asum` | ✅ |
| `Normalize(dim int) *Tensor` | Normalize along dimension | `fp32.Nrm2` + `fp32.Scal` | ✅ |

**Function Signatures:**

```go
// Dot: Dot product (vector) or Frobenius inner product (matrix)
// Vector case: dot product of two 1D tensors
// Matrix case: Frobenius inner product (sum of element-wise products)
func (t Tensor) Dot(other Tensor) float32

// Norm: Compute norm
// ord: 0 = L1 norm, 1 = L2 norm, 2 = Frobenius norm (same as L2 for matrices)
func (t Tensor) Norm(ord int) float32

// Normalize: L2 normalization along dimension
// For 1D: normalizes entire vector
// For 2D: normalizes along rows (dim=0) or columns (dim=1)
func (t Tensor) Normalize(dim int) *Tensor
```

**Norm Details:**
- L1 norm (`ord=0`): Sum of absolute values
- L2 norm (`ord=1`): Euclidean norm (square root of sum of squares)
- Frobenius norm (`ord=2`): Same as L2 norm for flattened matrices

## Convolution Operations

All convolution operations use `math/primitive/fp32` functions for optimized computation.

### 2D Convolution

| Function | Description | Primitive Used | Status |
|----------|-------------|----------------|--------|
| `Conv2D(kernel, bias Tensor, stride, padding) *Tensor` | 2D convolution | `fp32.Conv2D` | ✅ |
| `Conv2DTo(kernel, bias Tensor, dst *Tensor, stride, padding) *Tensor` | Conv to destination | - | ✅ |
| `Conv2DTransposed(kernel, bias Tensor, stride, padding) *Tensor` | Transposed 2D convolution | `fp32.Conv2DTransposed` | ✅ |

**Function Signatures:**

```go
// Conv2D: 2D convolution
// Input: [batch, inChannels, height, width]
// Kernel: [outChannels, inChannels, kernelH, kernelW]
// Bias: [outChannels] (optional, can be nil)
// Stride: [strideH, strideW]
// Padding: [padH, padW]
// Output: [batch, outChannels, outHeight, outWidth]
func (t Tensor) Conv2D(kernel, bias Tensor, stride, padding []int) *Tensor

// Conv2DTo: Conv2D to destination
func (t Tensor) Conv2DTo(kernel, bias Tensor, dst *Tensor, stride, padding []int) *Tensor

// Conv2DTransposed: Transposed 2D convolution (deconvolution)
// Input: [batch, inChannels, height, width]
// Kernel: [inChannels, outChannels, kernelH, kernelW] (transposed layout)
// Output: [batch, outChannels, outHeight, outWidth]
func (t Tensor) Conv2DTransposed(kernel, bias Tensor, stride, padding []int) *Tensor
```

**Output Dimension Calculation:**

For `Conv2D`:
- `outHeight = (inHeight + 2*padH - kernelH) / strideH + 1`
- `outWidth = (inWidth + 2*padW - kernelW) / strideW + 1`

For `Conv2DTransposed`:
- `outHeight = (inHeight - 1) * strideH - 2*padH + kernelH`
- `outWidth = (inWidth - 1) * strideW - 2*padW + kernelW`

### 1D Convolution

| Function | Description | Status |
|----------|-------------|--------|
| `Conv1D(kernel, bias, stride, padding) *Tensor` | 1D convolution | ✅ |

**Function Signature:**

```go
// Conv1D: 1D convolution (implemented via 2D conv with width=1)
// Input: [inChannels, length] or [batch, inChannels, length]
// Kernel: [outChannels, inChannels, kernelLen]
// Stride, padding: integers
// Output: [outChannels, outLen] or [batch, outChannels, outLen]
func (t Tensor) Conv1D(kernel, bias Tensor, stride, padding int) *Tensor
```

### 3D Convolution

| Function | Description | Primitive Used | Status |
|----------|-------------|----------------|--------|
| `Conv3D(kernel, bias, stride, padding) *Tensor` | 3D convolution | `fp32.Conv3D` | ✅ |

**Function Signature:**

```go
// Conv3D: 3D convolution
// Input: [batch, inChannels, depth, height, width]
// Kernel: [outChannels, inChannels, kernelD, kernelH, kernelW]
// Bias: [outChannels] (optional, can be nil)
// Stride: [strideD, strideH, strideW]
// Padding: [padD, padH, padW]
// Output: [batch, outChannels, outDepth, outHeight, outWidth]
func (t Tensor) Conv3D(kernel, bias Tensor, stride, padding []int) *Tensor
```

### Convolution Variants

**DepthwiseConv2D:**
```go
// DepthwiseConv2D: Depthwise separable 2D convolution
// Each input channel is convolved with its own kernel (depth multiplier = 1)
// Input: [batch, inChannels, height, width]
// Kernel: [inChannels, 1, kernelH, kernelW] or [inChannels, kernelH, kernelW]
// Bias: [inChannels] (optional, can be nil)
// Output: [batch, inChannels, outHeight, outWidth]
func (t Tensor) DepthwiseConv2D(kernel, bias Tensor, stride, padding []int) *Tensor
```

**GroupConv2D:**
```go
// GroupConv2D: Grouped 2D convolution
// Input: [batch, inChannels, height, width]
// Kernel: [outChannels, inChannels/groups, kernelH, kernelW]
// Bias: [outChannels] (optional, can be nil)
// groups: number of groups (inChannels must be divisible by groups)
// Output: [batch, outChannels, outHeight, outWidth]
func (t Tensor) GroupConv2D(kernel, bias Tensor, stride, padding []int, groups int) *Tensor
```

**DilatedConv2D:**
```go
// DilatedConv2D: Dilated (atrous) 2D convolution
// Input: [batch, inChannels, height, width]
// Kernel: [outChannels, inChannels, kernelH, kernelW]
// Bias: [outChannels] (optional, can be nil)
// dilation: [dilationH, dilationW] - dilation rates
// Output: [batch, outChannels, outHeight, outWidth]
func (t Tensor) DilatedConv2D(kernel, bias Tensor, stride, padding, dilation []int) *Tensor
```

**Dilated Convolution Details:**
- Effective kernel size: `(kernelH-1)*dilationH + 1` × `(kernelW-1)*dilationW + 1`
- Output dimensions: `(inHeight + 2*padH - effKernelH) / strideH + 1`

## Pooling Operations

All pooling operations use `math/primitive/fp32` functions for optimized computation.

| Function | Description | Primitive Used | Status |
|----------|-------------|----------------|--------|
| `MaxPool2D(kernelSize, stride, padding) *Tensor` | Max pooling | `fp32.MaxPool2D` | ✅ |
| `AvgPool2D(kernelSize, stride, padding) *Tensor` | Average pooling | `fp32.AvgPool2D` | ✅ |
| `GlobalAvgPool2D() *Tensor` | Global average pooling | `fp32.GlobalAvgPool2D` | ✅ |
| `AdaptiveAvgPool2D(outputSize) *Tensor` | Adaptive average pooling | `fp32.AdaptiveAvgPool2D` | ✅ |

**Function Signatures:**

```go
// MaxPool2D: Max pooling operation
// Input: [batch, channels, height, width]
// KernelSize: [kernelH, kernelW]
// Stride: [strideH, strideW]
// Padding: [padH, padW]
// Output: [batch, channels, outHeight, outWidth]
func (t Tensor) MaxPool2D(kernelSize, stride, padding []int) *Tensor

// AvgPool2D: Average pooling operation
// Same signature as MaxPool2D
func (t Tensor) AvgPool2D(kernelSize, stride, padding []int) *Tensor

// GlobalAvgPool2D: Global average pooling
// Input: [batch, channels, height, width]
// Output: [batch, channels]
// Computes mean over spatial dimensions (height, width)
func (t Tensor) GlobalAvgPool2D() *Tensor

// AdaptiveAvgPool2D: Adaptive average pooling to fixed output size
// Input: [batch, channels, height, width]
// outputSize: [outHeight, outWidth] - target output spatial dimensions
// Output: [batch, channels, outHeight, outWidth]
// Divides input into approximately equal regions and averages each region
func (t Tensor) AdaptiveAvgPool2D(outputSize []int) *Tensor
```

**Output Dimension Calculation:**

For `MaxPool2D` and `AvgPool2D`:
- `outHeight = (inHeight + 2*padH - kernelH) / strideH + 1`
- `outWidth = (inWidth + 2*padW - kernelW) / strideW + 1`

For `GlobalAvgPool2D`:
- Reduces spatial dimensions to 1×1 per channel

For `AdaptiveAvgPool2D`:
- Output dimensions exactly match `outputSize` parameters
- Uses adaptive kernel sizes to achieve target output dimensions

## Unfolding/Folding Operations

| Function | Description | Primitive Used | Status |
|----------|-------------|----------------|--------|
| `Im2Col(kernelSize, stride, padding) *Tensor` | Image to column | `fp32.Im2Col` | ✅ |
| `Col2Im(outputShape, kernelSize, stride, padding) *Tensor` | Column to image | `fp32.Col2Im` | ✅ |

**Function Signatures:**

```go
// Im2Col: Convert image patches to columns for GEMM-based convolution
// Input: [batch, channels, height, width]
// Output: [batch*outHeight*outWidth, channels*kernelH*kernelW]
func (t Tensor) Im2Col(kernelSize, stride, padding []int) *Tensor

// Col2Im: Convert columns back to image (inverse of Im2Col)
// Input: [batch*outHeight*outWidth, channels*kernelH*kernelW]
// Output: [batch, channels, height, width]
func (t Tensor) Col2Im(outputShape, kernelSize, stride, padding []int) *Tensor
```

**Use Case:**
- `Im2Col` converts convolution operations to matrix multiplication
- Enables use of optimized GEMM for convolution computation
- `Col2Im` is the inverse operation (used in backpropagation)

## Stride and Indexing

### Stride Calculation

Strides are computed row-major (from right to left):

```go
// For shape [d0, d1, ..., dn], strides[i] = product of shape[i+1:]
strides[n] = 1
strides[n-1] = shape[n]
strides[n-2] = shape[n-1] * shape[n]
// ...
strides[0] = shape[1] * shape[2] * ... * shape[n]
```

**Example:**
```go
// Shape [2, 3, 4]
// strides[2] = 1
// strides[1] = 4
// strides[0] = 3 * 4 = 12

// Element [i, j, k] stored at:
// index = i*12 + j*4 + k
```

### Leading Dimension

For matrix operations, the leading dimension (ld) is the number of columns:

```go
// For matrix [M, N]:
ld = N  // number of columns

// For tensor [B, M, N]:
// When treating as matrix, ld = N
```

### Element Index Calculation

```go
// Linear index from multi-dimensional indices:
index = offset + sum(indices[i] * strides[i])
```

## Performance Characteristics

### Zero Allocations

Operations that use `primitive` package functions are allocation-free in hot paths:
- Element-wise operations (`Add`, `Sub`, `Scale`) when tensors are contiguous
- Matrix operations (`MatMul`) via `primitive.Gemm_*`
- Dot products (`Dot`) via `primitive.Dot`
- Norms (`Norm`) via `primitive.Nrm2`, `primitive.Asum`
- Convolutions (`Conv2D`) via `primitive.Conv2D`

### Contiguous vs Strided

- **Contiguous tensors**: Operations use optimized `primitive` functions directly
- **Strided tensors**: Operations use recursive traversal (may be slower)

Check contiguity:
```go
if t.isContiguous() {
    // Use optimized path
} else {
    // Use strided path
}
```

### Batched Operations

- Batched operations automatically detect batch dimensions
- Use `primitive.GemmStrided` or `primitive.GemmBatched` when applicable
- Efficient for neural network batch processing

## Usage Examples

### Creating Tensors

```go
// Create tensor from shape (zeros)
t := tensor.New(tensor.DTFP32, tensor.NewShape(2, 3))

// Create tensor from existing data
data := []float32{1, 2, 3, 4, 5, 6}
t := tensor.FromFloat32(tensor.NewShape(2, 3), data)

// Note: Tensor arguments are passed by value, so you can pass tensors directly:
other := tensor.FromFloat32(tensor.NewShape(2, 3), []float32{1, 1, 1, 1, 1, 1})
t.Add(other) // other is passed by value to Add()
```

### Element Iteration

```go
// Iterate over all elements
t := tensor.FromFloat32(tensor.NewShape(2, 3), []float32{1, 2, 3, 4, 5, 6})
for elem := range t.Elements() {
    value := elem.Get()
    elem.Set(value * 2)
}

// Fix dimension 0 at index 1, iterate over remaining dimensions
for elem := range t.Elements(0, 1) {
    elem.Set(0.0)
}

// Fix multiple dimensions (axis 0 at 1, axis 2 at 3)
for elem := range t.Elements(0, 1, 2, 3) {
    // Modify elements
}

// Iterate over shape indices (useful for At/SetAt)
shape := tensor.NewShape(2, 3, 4)
for indices := range shape.Iterator(0, 1) {
    value := t.At(indices...)
    t.SetAt(indices, value * 2)
}
```

### Element-Wise Operations

```go
a := tensor.FromFloat32(tensor.NewShape(2, 3), []float32{1, 2, 3, 4, 5, 6})
b := tensor.FromFloat32(tensor.NewShape(2, 3), []float32{1, 1, 1, 1, 1, 1})

// In-place addition (modifies a)
a.Add(b)

// Create new tensor
c := a.AddTo(b, nil)

// Scale by scalar
a.Scale(2.0)
```

### Matrix Operations

```go
// 2D matrix multiplication
a := tensor.New(tensor.DTFP32, tensor.NewShape(32, 64))
b := tensor.New(tensor.DTFP32, tensor.NewShape(64, 128))
c := a.MatMul(b) // Result: [32, 128]

// Batched matrix multiplication
a := tensor.New(tensor.DTFP32, tensor.NewShape(8, 32, 64))
b := tensor.New(tensor.DTFP32, tensor.NewShape(8, 64, 128))
c := a.MatMul(b) // Result: [8, 32, 128]

// Transpose
a := tensor.FromFloat32(tensor.NewShape(2, 3), []float32{1, 2, 3, 4, 5, 6})
aT := a.Transpose() // Result: [3, 2]
```

### Convolution Operations

```go
// 2D Convolution
input := tensor.New(tensor.DTFP32, tensor.NewShape(1, 3, 224, 224)) // [batch, channels, height, width]
kernel := tensor.New(tensor.DTFP32, tensor.NewShape(64, 3, 3, 3))  // [outChannels, inChannels, kernelH, kernelW]
bias := tensor.New(tensor.DTFP32, tensor.NewShape(64))              // [outChannels]

output := input.Conv2D(kernel, bias, []int{1, 1}, []int{1, 1})
// Result: [1, 64, 224, 224] (with stride=1, padding=1)
```

### Pooling Operations

```go
input := &tensor.Tensor{
    Dim:  []int{1, 64, 112, 112},
    Data: make([]float32, 1*64*112*112),
}

// Max pooling: 2x2 kernel, stride 2, no padding
output := input.MaxPool2D([]int{2, 2}, []int{2, 2}, []int{0, 0})
// Result: [1, 64, 56, 56]

// Global average pooling
output := input.GlobalAvgPool2D()
// Result: [1, 64]
```

## Implementation Status

### ✅ Implemented

- **Core Operations**: Shape, Size, Clone, At, SetAt, Reshape
- **Iteration**: Elements() (Go 1.22+ range-over-function), Shape.Iterator()
- **Element-wise Operations**: Add, Sub, Mul, Div, Scale, Square, Sqrt, Exp, Log, Pow, Abs, Sign, Cos, Sin, Negative (in-place)
- **Comparison Operations**: Equal, Greater, GreaterThan, Less (return boolean-like tensors)
- **Reduction Operations**: Sum, Mean, Max, Min, ArgMax
- **Broadcasting**: BroadcastTo (basic validation)
- **Linear Algebra**: MatMul (2D and batched), Transpose (2D), Dot, Norm, Normalize
- **Convolution**: Conv2D, Conv2DTransposed, Conv1D, Conv3D, DepthwiseConv2D, GroupConv2D, DilatedConv2D
- **Pooling**: MaxPool2D, AvgPool2D, GlobalAvgPool2D, AdaptiveAvgPool2D
- **Activation Functions**: ReLU, Sigmoid, Tanh, Softmax (see activations.go)
- **Unfolding/Folding**: Im2Col, Col2Im

### 🔮 Future Work

- **Quantization**: INT8 quantized computations (see [QUANTIZATION_PLAN.md](./QUANTIZATION_PLAN.md))
- **Views**: Zero-copy slicing with offset + strides (beyond Reshape)
- **Advanced Broadcasting**: Efficient broadcasting without copying
- **General Transpose**: Support arbitrary dimension permutation
- **Advanced Pooling**: Global max pooling
- **Tensor Operations**: More specialized tensor operations

## File Organization

```
pkg/core/math/tensor/
├── dense.go                  # Core tensor structure and basic operations
├── tensor_math.go            # Element-wise operations and reductions
├── tensor_linalg.go          # Linear algebra operations
├── tensor_conv.go            # Convolution and pooling operations
├── tensor_math_test.go       # Tests for element-wise operations
├── tensor_linalg_test.go     # Tests for linear algebra
├── tensor_conv_test.go       # Tests for convolution operations
├── SPEC.md                   # This file
└── QUANTIZATION_PLAN.md      # INT8 quantization implementation plan
```

## Integration with Primitive Package

All tensor operations delegate to `math/primitive/fp32` when possible:

| Tensor Operation | Primitive Function | Use Case |
|-----------------|-------------------|----------|
| `Add`, `Sub` | `fp32.Axpy` | Contiguous element-wise add/sub |
| `Scale` | `fp32.Scal` | Contiguous scalar multiplication |
| `Sum` | `fp32.Asum` | Vector sum (L1 norm) |
| `ArgMax` | `fp32.Iamax` | Vector argmax |
| `MatMul` | `fp32.Gemm_*`, `GemmBatched`, `GemmStrided` | Matrix multiplication |
| `Dot` | `fp32.Dot` | Vector dot product |
| `Norm` (L2) | `fp32.Nrm2` | L2 norm |
| `Norm` (L1) | `fp32.Asum` | L1 norm |
| `Normalize` | `fp32.Nrm2` + `fp32.Scal` | Vector normalization |
| `Conv2D` | `fp32.Conv2D` | 2D convolution |
| `Conv2DTransposed` | `fp32.Conv2DTransposed` | Transposed convolution |
| `Conv3D` | `fp32.Conv3D` | 3D convolution |
| `MaxPool2D` | `fp32.MaxPool2D` | Max pooling |
| `AvgPool2D` | `fp32.AvgPool2D` | Average pooling |
| `GlobalAvgPool2D` | `fp32.GlobalAvgPool2D` | Global average pooling |
| `AdaptiveAvgPool2D` | `fp32.AdaptiveAvgPool2D` | Adaptive average pooling |
| `Im2Col` | `fp32.Im2Col` | Image to column |
| `Col2Im` | `fp32.Col2Im` | Column to image |
| `BroadcastTo` | `fp32.ExpandTo` | Broadcasting operations |
| `ZerosLike` | `tensor.New()` | Create zero tensor with same shape |
| `OnesLike` | Direct fill | Create ones tensor with same shape |
| `FullLike` | Direct fill | Create tensor filled with value |
| `GreaterThan`, `Greater` | `fp32.ElemGreaterThan` | Element-wise greater than comparison |
| `Less` | `fp32.ElemLess` | Element-wise less than comparison |
| `Equal` | `fp32.ElemEqual` | Element-wise equality comparison |
| `Where` | `fp32.ElemWhere` | Conditional element selection |
| `Square` | `fp32.ElemSquare` | Element-wise square |
| `Sqrt` | `fp32.ElemSqrt` | Element-wise square root |
| `Exp` | `fp32.ElemExp` | Element-wise exponential |
| `Log` | `fp32.ElemLog` | Element-wise natural logarithm |
| `Pow` | `fp32.ElemPow` | Element-wise power |
| `Abs` | `fp32.ElemAbs` | Element-wise absolute value |
| `Sign` | `fp32.ElemSign` | Element-wise sign |
| `Cos` | `fp32.ElemCos` | Element-wise cosine |
| `Sin` | `fp32.ElemSin` | Element-wise sine |
| `Negative` | `fp32.ElemNegative` | Element-wise negation |
| `Tanh` | `fp32.Tanh` | Hyperbolic tangent (see activations.go) |

## Deprecated Gradient Functions

**⚠️ IMPORTANT**: Specific gradient computation functions are deprecated in favor of composing gradients from primitives in layer implementations.

### Deprecated Tensor Gradient Functions

The following tensor-level gradient functions are **DEPRECATED** and will be removed in a future version:

| Function | Replacement |
|----------|-------------|
| `tensor.ReLUGrad(gradOutput, dst)` | `gradOutput.Mul(input.GreaterThan(ZerosLike(input)))` |
| `tensor.SigmoidGrad(gradOutput, dst)` | `gradOutput.Mul(output).Mul(OnesLike(output).Sub(output))` |
| `tensor.TanhGrad(gradOutput, dst)` | `gradOutput.Mul(OnesLike(output).Sub(output.Mul(output)))` |
| `tensor.SoftmaxGrad(gradOutput, dim, dst)` | Complex composition using `Sum()`, `BroadcastTo()`, `Mul()`, `Sub()` |
| `tensor.Conv2DKernelGrad(outputGrad, kernel, stride, padding)` | `outputGrad.Im2Col(...).MatMul(input.Im2Col(...).T())` |
| `tensor.Conv1DKernelGrad(outputGrad, kernel, stride, padding)` | Matrix operations on reshaped tensors |

### Architectural Rationale

Gradient computations should be composed from mathematical primitives rather than implemented as pre-built algorithms. This provides:

- **Better separation of concerns**: Primitives in `fp32`/`tensor`, algorithms in `layers`
- **Improved composability**: Easy to create new gradient operations
- **Enhanced testability**: Primitives tested independently
- **Greater flexibility**: Layer-specific gradient logic stays in layers

### Migration Path

Replace direct gradient function calls in layer `Backward()` methods with compositions using primitives:

```go
// OLD: Direct gradient function call
gradInput := input.ReLUGrad(&gradOutput, nil)

// NEW: Compose from primitives using utility functions
mask := input.GreaterThan(ZerosLike(input))
gradInput := gradOutput.Mul(mask)
```

## Error Handling

Operations that fail validation typically panic with descriptive error messages:

```go
// Shape mismatch
t.Add(other) // Panics if shapes don't match

// Dimension out of range
t.Sum(5) // Panics if dimension 5 doesn't exist

// Invalid operation
t.MatMul(other) // Panics if dimensions are incompatible
```

**Recommendation**: Validate tensor shapes before calling operations in performance-critical code paths.

## Performance Considerations

1. **Contiguous Tensors**: Operations on contiguous tensors are fastest (use `primitive` functions directly)
2. **Strided Tensors**: Operations on strided tensors use recursive traversal (may be slower)
3. **Batch Processing**: Use batched operations when possible (more efficient than looping)
4. **In-Place Operations**: Prefer in-place operations to minimize allocations
5. **Buffer Reuse**: Future: use buffer pool for temporary tensors

## Testing

All operations have comprehensive unit tests:
- Basic functionality tests
- Edge case tests (empty tensors, zero dimensions)
- Shape validation tests
- Numerical accuracy tests

See test files for examples:
- `tensor_math_test.go` - Element-wise and reduction operations
- `tensor_linalg_test.go` - Linear algebra operations
- `tensor_conv_test.go` - Convolution and pooling operations

## Testing

All operations have comprehensive unit tests:
- Basic functionality tests
- Edge case tests (empty tensors, zero dimensions)
- Shape validation tests
- Numerical accuracy tests

See test files for examples:
- `tensor_math_test.go` - Element-wise and reduction operations
- `tensor_linalg_test.go` - Linear algebra operations
- `tensor_conv_test.go` - Convolution and pooling operations


