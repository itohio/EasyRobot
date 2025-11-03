# math/tensor - Tensor Operations Specification

## Overview

The `tensor` package provides multi-dimensional tensor operations optimized for embedded systems with minimal allocations, supporting neural network inference, computer vision, and general numerical computing. All tensors use **row-major** storage layout matching Go nested arrays.

**Foundation**: This package builds on the `math/primitive` package which provides BLAS levels 1-3 and LAPACK operations with zero allocations, stride-based access, and row-major storage. See [primitive/SPEC.md](../primitive/SPEC.md), [primitive/BLAS.md](../primitive/BLAS.md), and [primitive/LA.md](../primitive/LA.md) for details.

**Note**: Neural network operations (Linear, ReLU, Sigmoid, Tanh, Softmax, MSE, CrossEntropy) are in the separate `math/nn` package.

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
    Dim  []int      // Dimensions (e.g., [batch, height, width, channels])
    Data []float32  // Flat data storage (single allocation)
}
```

**Characteristics:**
- Variable dimensions (runtime-determined)
- Single contiguous backing array (`[]float32`)
- Shape and data stored separately
- Strides computed on-demand from shape
- No internal copying for views (future: add offset + strides for zero-copy views)

### Helper Functions

```go
// Shape returns the shape (dimensions) of the tensor
func (t *Tensor) Shape() []int

// Size returns the total number of elements in the tensor
func (t *Tensor) Size() int

// Clone creates a deep copy of the tensor
func (t *Tensor) Clone() *Tensor

// Flat returns the underlying data slice (zero-copy)
func (t *Tensor) Flat() []float32

// At returns the element at the given indices
func (t *Tensor) At(indices ...int) float32

// SetAt sets the element at the given indices
func (t *Tensor) SetAt(indices []int, value float32)

// Reshape returns a new tensor with the same data but different shape (zero-copy)
func (t *Tensor) Reshape(newShape []int) *Tensor
```

**Element Access:**
- `At(indices ...int)`: Returns the element at the given multi-dimensional indices
- `SetAt(indices []int, value float32)`: Sets the element at the given indices
- Both functions validate indices and compute linear index using strides

**Data Access:**
- `Flat() []float32`: Returns the underlying data slice directly (zero-copy access)

**Reshaping:**
- `Reshape(newShape []int) *Tensor`: Creates a new tensor view with different shape but same data
- The total number of elements must remain the same
- Returns a zero-copy view sharing the same underlying data slice

## Element-Wise Operations

### In-Place Operations

All in-place operations modify the tensor and return self for method chaining.

| Function | Description | Primitive Used | Status |
|----------|-------------|----------------|--------|
| `Add(other *Tensor) *Tensor` | Add tensor element-wise | `primitive.Axpy` | ✅ |
| `Sub(other *Tensor) *Tensor` | Subtract tensor element-wise | `primitive.Axpy` (alpha=-1) | ✅ |
| `Mul(other *Tensor) *Tensor` | Multiply element-wise | - | ✅ |
| `Div(other *Tensor) *Tensor` | Divide element-wise | - | ✅ |
| `Scale(scalar float32) *Tensor` | Multiply by scalar | `primitive.Scal` | ✅ |

**Function Signatures:**

```go
// Add: t = t + other (in-place)
func (t *Tensor) Add(other *Tensor) *Tensor

// Sub: t = t - other (in-place)
func (t *Tensor) Sub(other *Tensor) *Tensor

// Mul: t = t * other (element-wise, in-place)
func (t *Tensor) Mul(other *Tensor) *Tensor

// Div: t = t / other (element-wise, in-place)
func (t *Tensor) Div(other *Tensor) *Tensor

// Scale: t = scalar * t (in-place)
func (t *Tensor) Scale(scalar float32) *Tensor
```

**Parameters:**
- `other`: Tensor with same shape as `t`
- `scalar`: Scalar multiplier

### Operations Creating New Tensors

| Function | Description | Status |
|----------|-------------|--------|
| `AddTo(other, dst *Tensor) *Tensor` | Add to destination (or create new) | ✅ |
| `MulTo(other, dst *Tensor) *Tensor` | Multiply to destination (or create new) | ✅ |

**Function Signatures:**

```go
// AddTo: result = t + other
// If dst is nil, creates new tensor
// If dst is provided, uses it (must match shape)
func (t *Tensor) AddTo(other *Tensor, dst *Tensor) *Tensor

// MulTo: result = t * other (element-wise)
func (t *Tensor) MulTo(other *Tensor, dst *Tensor) *Tensor
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
func (t *Tensor) Sum(dims ...int) *Tensor

// Mean: Mean along specified dimensions (all dimensions if none specified)
func (t *Tensor) Mean(dims ...int) *Tensor

// Max: Maximum along specified dimensions (all dimensions if none specified)
func (t *Tensor) Max(dims ...int) *Tensor

// Min: Minimum along specified dimensions (all dimensions if none specified)
func (t *Tensor) Min(dims ...int) *Tensor

// ArgMax: Index of maximum element along specified dimension
func (t *Tensor) ArgMax(dim int) *Tensor
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
func (t *Tensor) BroadcastTo(shape []int) (*Tensor, error)
```

**Notes:**
- Currently creates clone if shapes match exactly
- Future: Implement efficient broadcasting without copying
- Validates that broadcasting is possible (dimensions are compatible)

## Linear Algebra Operations

All linear algebra operations use `math/primitive` BLAS/LAPACK functions.

### Matrix Operations

| Function | Description | Primitive Used | Status |
|----------|-------------|----------------|--------|
| `MatMul(other *Tensor) *Tensor` | Matrix multiplication | `primitive.Gemm_*`, `GemmBatched`, `GemmStrided` | ✅ |
| `MatMulTo(other, dst *Tensor) *Tensor` | Matrix multiply to destination | - | ✅ |
| `Transpose(dims ...int) *Tensor` | Transpose dimensions | - | ✅ (2D only) |
| `TransposeTo(dst *Tensor, dims ...int) *Tensor` | Transpose to destination | - | ✅ (2D only) |

**Function Signatures:**

```go
// MatMul: Matrix multiplication
// For 2D: [M, K] × [K, N] = [M, N]
// For batched: [B, M, K] × [B, K, N] = [B, M, N]
// Supports broadcasting: [M, K] × [B, K, N] or [B, M, K] × [K, N]
func (t *Tensor) MatMul(other *Tensor) *Tensor

// MatMulTo: Matrix multiply to destination
func (t *Tensor) MatMulTo(other *Tensor, dst *Tensor) *Tensor

// Transpose: Transpose dimensions (currently supports 2D only)
// [M, N] → [N, M]
func (t *Tensor) Transpose(dims ...int) *Tensor

// TransposeTo: Transpose to destination
func (t *Tensor) TransposeTo(dst *Tensor, dims ...int) *Tensor
```

**MatMul Details:**
- Automatically detects 2D vs batched cases
- Uses `primitive.Gemm_NN` for 2D case
- Uses `primitive.GemmStrided` for batched contiguous tensors
- Uses `primitive.GemmBatched` for batched strided access
- Handles leading dimensions automatically from tensor shape

### Dot Products and Norms

| Function | Description | Primitive Used | Status |
|----------|-------------|----------------|--------|
| `Dot(other *Tensor) float32` | Dot product | `primitive.Dot` (vector case) | ✅ |
| `Norm(ord int) float32` | Vector/matrix norm | `primitive.Nrm2`, `primitive.Asum` | ✅ |
| `Normalize(dim int) *Tensor` | Normalize along dimension | `primitive.Nrm2` + `primitive.Scal` | ✅ |

**Function Signatures:**

```go
// Dot: Dot product (vector) or Frobenius inner product (matrix)
func (t *Tensor) Dot(other *Tensor) float32

// Norm: Compute norm
// ord: 0 = L1 norm, 1 = L2 norm, 2 = Frobenius norm (same as L2 for matrices)
func (t *Tensor) Norm(ord int) float32

// Normalize: L2 normalization along dimension
// For 1D: normalizes entire vector
// For 2D: normalizes along rows (dim=0) or columns (dim=1)
func (t *Tensor) Normalize(dim int) *Tensor
```

## Convolution Operations

All convolution operations use `math/primitive` tensor operations.

### 2D Convolution

| Function | Description | Primitive Used | Status |
|----------|-------------|----------------|--------|
| `Conv2D(kernel, bias, stride, padding) *Tensor` | 2D convolution | `primitive.Conv2D` | ✅ |
| `Conv2DTo(kernel, bias, dst, stride, padding) *Tensor` | Conv to destination | - | ✅ |
| `Conv2DTransposed(kernel, bias, stride, padding) *Tensor` | Transposed 2D convolution | `primitive.Conv2DTransposed` | ✅ |
| `DepthwiseConv2D(kernel, bias, stride, padding) *Tensor` | Depthwise separable convolution | - | ✅ |
| `GroupConv2D(kernel, bias, stride, padding, groups) *Tensor` | Grouped convolution | - | ✅ |
| `DilatedConv2D(kernel, bias, stride, padding, dilation) *Tensor` | Dilated (atrous) convolution | - | ✅ |

**Function Signatures:**

```go
// Conv2D: 2D convolution
// Input: [batch, inChannels, height, width]
// Kernel: [outChannels, inChannels, kernelH, kernelW]
// Bias: [outChannels] (optional, can be nil)
// Stride: [strideH, strideW]
// Padding: [padH, padW]
// Output: [batch, outChannels, outHeight, outWidth]
func (t *Tensor) Conv2D(kernel, bias *Tensor, stride, padding []int) *Tensor

// Conv2DTo: Conv2D to destination
func (t *Tensor) Conv2DTo(kernel, bias, dst *Tensor, stride, padding []int) *Tensor

// Conv2DTransposed: Transposed 2D convolution (deconvolution)
// Input: [batch, inChannels, height, width]
// Kernel: [inChannels, outChannels, kernelH, kernelW] (transposed layout)
// Output: [batch, outChannels, outHeight, outWidth]
func (t *Tensor) Conv2DTransposed(kernel, bias *Tensor, stride, padding []int) *Tensor
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
func (t *Tensor) Conv1D(kernel, bias *Tensor, stride, padding int) *Tensor
```

### 3D Convolution

| Function | Description | Status |
|----------|-------------|--------|
| `Conv3D(kernel, bias, stride, padding) *Tensor` | 3D convolution | ✅ |

**Function Signature:**

```go
// Conv3D: 3D convolution
// Input: [batch, inChannels, depth, height, width]
// Kernel: [outChannels, inChannels, kernelD, kernelH, kernelW]
// Bias: [outChannels] (optional, can be nil)
// Stride: [strideD, strideH, strideW]
// Padding: [padD, padH, padW]
// Output: [batch, outChannels, outDepth, outHeight, outWidth]
func (t *Tensor) Conv3D(kernel, bias *Tensor, stride, padding []int) *Tensor
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
func (t *Tensor) DepthwiseConv2D(kernel, bias *Tensor, stride, padding []int) *Tensor
```

**GroupConv2D:**
```go
// GroupConv2D: Grouped 2D convolution
// Input: [batch, inChannels, height, width]
// Kernel: [outChannels, inChannels/groups, kernelH, kernelW]
// Bias: [outChannels] (optional, can be nil)
// groups: number of groups (inChannels must be divisible by groups)
// Output: [batch, outChannels, outHeight, outWidth]
func (t *Tensor) GroupConv2D(kernel, bias *Tensor, stride, padding []int, groups int) *Tensor
```

**DilatedConv2D:**
```go
// DilatedConv2D: Dilated (atrous) 2D convolution
// Input: [batch, inChannels, height, width]
// Kernel: [outChannels, inChannels, kernelH, kernelW]
// Bias: [outChannels] (optional, can be nil)
// dilation: [dilationH, dilationW] - dilation rates
// Output: [batch, outChannels, outHeight, outWidth]
func (t *Tensor) DilatedConv2D(kernel, bias *Tensor, stride, padding, dilation []int) *Tensor
```

## Pooling Operations

| Function | Description | Status |
|----------|-------------|--------|
| `MaxPool2D(kernelSize, stride, padding) *Tensor` | Max pooling | ✅ |
| `AvgPool2D(kernelSize, stride, padding) *Tensor` | Average pooling | ✅ |
| `GlobalAvgPool2D() *Tensor` | Global average pooling | ✅ |
| `AdaptiveAvgPool2D(outputSize) *Tensor` | Adaptive average pooling | ✅ |

**Function Signatures:**

```go
// MaxPool2D: Max pooling operation
// Input: [batch, channels, height, width]
// KernelSize: [kernelH, kernelW]
// Stride: [strideH, strideW]
// Padding: [padH, padW]
// Output: [batch, channels, outHeight, outWidth]
func (t *Tensor) MaxPool2D(kernelSize, stride, padding []int) *Tensor

// AvgPool2D: Average pooling operation
// Same signature as MaxPool2D
func (t *Tensor) AvgPool2D(kernelSize, stride, padding []int) *Tensor

// GlobalAvgPool2D: Global average pooling
// Input: [batch, channels, height, width]
// Output: [batch, channels]
// Computes mean over spatial dimensions (height, width)
func (t *Tensor) GlobalAvgPool2D() *Tensor

// AdaptiveAvgPool2D: Adaptive average pooling to fixed output size
// Input: [batch, channels, height, width]
// outputSize: [outHeight, outWidth] - target output spatial dimensions
// Output: [batch, channels, outHeight, outWidth]
// Divides input into approximately equal regions and averages each region
func (t *Tensor) AdaptiveAvgPool2D(outputSize []int) *Tensor
```

**Output Dimension Calculation:**

- `outHeight = (inHeight + 2*padH - kernelH) / strideH + 1`
- `outWidth = (inWidth + 2*padW - kernelW) / strideW + 1`

## Unfolding/Folding Operations

| Function | Description | Primitive Used | Status |
|----------|-------------|----------------|--------|
| `Im2Col(kernelSize, stride, padding) *Tensor` | Image to column | `primitive.Im2Col` | ✅ |
| `Col2Im(outputShape, kernelSize, stride, padding) *Tensor` | Column to image | `primitive.Col2Im` | ✅ |

**Function Signatures:**

```go
// Im2Col: Convert image patches to columns for GEMM-based convolution
// Input: [batch, channels, height, width]
// Output: [batch*outHeight*outWidth, channels*kernelH*kernelW]
func (t *Tensor) Im2Col(kernelSize, stride, padding []int) *Tensor

// Col2Im: Convert columns back to image (inverse of Im2Col)
// Input: [batch*outHeight*outWidth, channels*kernelH*kernelW]
// Output: [batch, channels, height, width]
func (t *Tensor) Col2Im(outputShape, kernelSize, stride, padding []int) *Tensor
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
t := &tensor.Tensor{
    Dim:  []int{2, 3},
    Data: make([]float32, 6),
}

// Create tensor from values
t := &tensor.Tensor{
    Dim:  []int{2, 3},
    Data: []float32{1, 2, 3, 4, 5, 6},
}
```

### Element-Wise Operations

```go
a := &tensor.Tensor{Dim: []int{2, 3}, Data: []float32{1, 2, 3, 4, 5, 6}}
b := &tensor.Tensor{Dim: []int{2, 3}, Data: []float32{1, 1, 1, 1, 1, 1}}

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
a := &tensor.Tensor{Dim: []int{32, 64}, Data: make([]float32, 32*64)}
b := &tensor.Tensor{Dim: []int{64, 128}, Data: make([]float32, 64*128)}
c := a.MatMul(b) // Result: [32, 128]

// Batched matrix multiplication
a := &tensor.Tensor{Dim: []int{8, 32, 64}, Data: make([]float32, 8*32*64)}
b := &tensor.Tensor{Dim: []int{8, 64, 128}, Data: make([]float32, 8*64*128)}
c := a.MatMul(b) // Result: [8, 32, 128]

// Transpose
a := &tensor.Tensor{Dim: []int{2, 3}, Data: []float32{1, 2, 3, 4, 5, 6}}
aT := a.Transpose() // Result: [3, 2]
```

### Convolution Operations

```go
// 2D Convolution
input := &tensor.Tensor{
    Dim:  []int{1, 3, 224, 224}, // [batch, channels, height, width]
    Data: make([]float32, 1*3*224*224),
}

kernel := &tensor.Tensor{
    Dim:  []int{64, 3, 3, 3}, // [outChannels, inChannels, kernelH, kernelW]
    Data: make([]float32, 64*3*3*3),
}

bias := &tensor.Tensor{
    Dim:  []int{64},
    Data: make([]float32, 64),
}

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

- **Core Operations**: Shape, Size, Clone, Flat, At, SetAt, Reshape
- **Element-wise Operations**: Add, Sub, Mul, Div, Scale (in-place and new tensor)
- **Reduction Operations**: Sum, Mean, Max, Min, ArgMax
- **Broadcasting**: BroadcastTo (basic validation)
- **Linear Algebra**: MatMul (2D and batched), Transpose (2D), Dot, Norm, Normalize
- **Convolution**: Conv2D, Conv2DTransposed, Conv1D, Conv3D, DepthwiseConv2D, GroupConv2D, DilatedConv2D
- **Pooling**: MaxPool2D, AvgPool2D, GlobalAvgPool2D, AdaptiveAvgPool2D
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

All tensor operations delegate to `math/primitive` when possible:

| Tensor Operation | Primitive Function | Use Case |
|-----------------|-------------------|----------|
| `Add`, `Sub` | `Axpy` | Contiguous element-wise add/sub |
| `Scale` | `Scal` | Contiguous scalar multiplication |
| `Sum` | `Asum` | Vector sum (L1 norm) |
| `ArgMax` | `Iamax` | Vector argmax |
| `MatMul` | `Gemm_*`, `GemmBatched`, `GemmStrided` | Matrix multiplication |
| `Dot` | `Dot` | Vector dot product |
| `Norm` (L2) | `Nrm2` | L2 norm |
| `Norm` (L1) | `Asum` | L1 norm |
| `Normalize` | `Nrm2` + `Scal` | Vector normalization |
| `Conv2D` | `Conv2D` | 2D convolution |
| `Conv2DTransposed` | `Conv2DTransposed` | Transposed convolution |
| `Im2Col` | `Im2Col` | Image to column |
| `Col2Im` | `Col2Im` | Column to image |

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

