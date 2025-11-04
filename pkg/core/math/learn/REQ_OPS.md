# Required Tensor Operations Analysis

## Overview

This document analyzes all primitive tensor/vector/matrix operations required by the `learn` package and its dependencies. The analysis identifies operations that are currently implemented in the Tensor package, operations that are required but may be missing, and identifies duplicate operations that could potentially be consolidated.

## Analysis Methodology

The analysis was performed by examining:
1. Direct usage in `pkg/core/math/learn/` Go source files
2. Indirect usage through the `nn` package (neural network layers)
3. Test files and examples
4. Dependencies on tensor operations in layers and activations

## Required Operations by Category

### Core Tensor Operations (Required by learn package directly)

| Operation | Status | Usage Location | Purpose |
|-----------|--------|----------------|---------|
| `tensor.New(dtype, shape)` | ✅ Implemented | `optimizer.go`, `quantization.go` | Create new tensors for Adam optimizer state |
| `tensor.FromFloat32(shape, data)` | ✅ Implemented | Tests, quantization | Create tensors from float32 slices |
| `tensor.NewShape(...)` | ✅ Implemented | Throughout | Create shape specifications |
| `t.Shape()` | ✅ Implemented | `optimizer.go`, `quantization.go` | Get tensor shape |
| `t.Data()` | ✅ Implemented | `optimizer.go`, quantization | Access underlying float32 slice |
| `t.Size()` | ✅ Implemented | `training.go`, quantization | Get total number of elements |

### Element-wise Operations (Required by nn package)

| Operation | Status | Usage Location | Purpose |
|-----------|--------|----------------|---------|
| `t.Add(other)` | ✅ Implemented | Loss functions, layers | Element-wise addition |
| `t.Sub(other)` | ✅ Implemented | Loss functions, layers | Element-wise subtraction |
| `t.Mul(other)` | ✅ Implemented | Loss functions | Element-wise multiplication |
| `t.Div(other)` | ✅ Implemented | Not used in learn | Element-wise division |
| `t.Scale(scalar)` | ✅ Implemented | Loss gradients, layers | Scalar multiplication |
| `t.Clone()` | ✅ Implemented | Activations, loss functions | Create tensor copy |

### Reduction Operations (Required by nn package)

| Operation | Status | Usage Location | Purpose |
|-----------|--------|----------------|---------|
| `t.Sum(dims...)` | ✅ Implemented | Loss computation | Sum reduction along dimensions |
| `t.Mean(dims...)` | ✅ Implemented | Not used in learn | Mean reduction along dimensions |
| `t.Max(dims...)` | ✅ Implemented | Not used in learn | Max reduction along dimensions |
| `t.Min(dims...)` | ✅ Implemented | Not used in learn | Min reduction along dimensions |
| `t.ArgMax(dim)` | ✅ Implemented | Not used in learn | Argmax along dimension |

### Activation Functions (Required by nn package)

| Operation | Status | Usage Location | Purpose |
|-----------|--------|----------------|---------|
| `t.ReLU()` | ✅ Implemented | ReLU layers | Rectified Linear Unit activation |
| `t.Sigmoid()` | ✅ Implemented | Sigmoid layers | Sigmoid activation |
| `t.Tanh()` | ✅ Implemented | Tanh layers | Hyperbolic tangent activation |
| `t.Softmax(dim)` | ✅ Implemented | Softmax layers, loss functions | Softmax activation along dimension |

### Convolution and Pooling Operations (Required by nn layers)

| Operation | Status | Usage Location | Purpose |
|-----------|--------|----------------|---------|
| `t.Conv2D(kernel, bias, stride, padding)` | ✅ Implemented | Conv2D layers | 2D convolution |
| `t.Conv1D(kernel, bias, stride, padding)` | ✅ Implemented | Conv1D layers | 1D convolution |
| `t.MaxPool2D(kernel, stride, padding)` | ✅ Implemented | MaxPool2D layers | 2D max pooling |
| `t.AvgPool2D(kernel, stride, padding)` | ✅ Implemented | AvgPool2D layers | 2D average pooling |
| `t.GlobalAvgPool2D()` | ✅ Implemented | Global pooling layers | Global average pooling |

### Linear Algebra Operations (Required by nn package)

| Operation | Status | Usage Location | Purpose |
|-----------|--------|----------------|---------|
| `t.MatMul(other)` | ✅ Implemented | Linear/Dense layers | Matrix multiplication |
| `t.Transpose(dims...)` | ✅ Implemented | Utility layers | Tensor transpose |
| `t.Reshape(newShape)` | ✅ Implemented | Convolution layers (internal) | Tensor reshape |
| `t.BroadcastTo(shape)` | ✅ Implemented | Not used in learn | Broadcasting |

### Gradient Operations (Required by layers for backpropagation)

| Operation | Status | Usage Location | Purpose |
|-----------|--------|----------------|---------|
| `t.ReLUGrad(gradOutput, dst)` | ✅ Implemented | ReLU backward pass | ReLU gradient computation |
| `t.SigmoidGrad(gradOutput, dst)` | ✅ Implemented | Sigmoid backward pass | Sigmoid gradient computation |
| `t.TanhGrad(gradOutput, dst)` | ✅ Implemented | Tanh backward pass | Tanh gradient computation |
| `t.SoftmaxGrad(gradOutput, dim, dst)` | ✅ Implemented | Softmax backward pass | Softmax gradient computation |

## Operations NOT Currently Required

Based on the analysis, the following tensor operations are implemented but not used by the learn package:

- `t.Div(other)` - Element-wise division
- `t.Mean(dims...)` - Mean reduction
- `t.Max(dims...)` - Max reduction
- `t.Min(dims...)` - Min reduction
- `t.ArgMax(dim)` - Argmax reduction
- `t.BroadcastTo(shape)` - Broadcasting
- Various pooling operations beyond MaxPool2D/AvgPool2D
- Complex convolution operations (Conv2DTransposed, DepthwiseConv2D, etc.)

## Duplicate Operations Analysis

Several operations exist in multiple forms that perform essentially the same computation but with different APIs or implementations. These could potentially be consolidated:

### 1. Linear Operations

**Duplicate Operations:**
- `nn.Linear(t, weight, bias)` in `nn/nn.go` - High-level linear transformation
- `t.MatMul(other)` in `tensor/tensor_linalg.go` - Low-level matrix multiplication

**Analysis:**
- Both use `fp32.Gemm_NN` and `fp32.Gemv_T` primitives
- `nn.Linear` adds bias addition and handles batching automatically
- `t.MatMul` is more general-purpose but requires manual bias handling
- **Recommendation:** Keep separate - different abstraction levels serve different use cases

### 2. Softmax Operations

**Duplicate Operations:**
- `t.Softmax(dim)` in `tensor/activations.go` - Tensor method
- `nn.Softmax(t, dim)` in `nn/nn.go` - Function with dimension support

**Analysis:**
- Both handle 1D and 2D tensors
- `nn.Softmax` has more comprehensive dimension handling
- `t.Softmax` is more convenient for tensor method chaining
- **Recommendation:** Keep separate - different API preferences

### 3. Element-wise Operations with Different Implementations

**Duplicate Operations:**
- Contiguous path: Uses `fp32.Axpy`, `fp32.Scal` directly
- Non-contiguous path: Uses `fp32.ElemAdd`, `fp32.ElemSub`, etc.

**Analysis:**
- Different code paths for contiguous vs strided tensors
- Contiguous path is optimized for common case
- Strided path handles general tensor layouts
- **Recommendation:** Keep separate - optimization for performance

### 4. Reduction Operations

**Duplicate Operations:**
- `t.Sum()` - Sum all elements
- `t.Sum(dims...)` - Sum along specific dimensions

**Analysis:**
- Same underlying implementation, different parameter handling
- **Recommendation:** Keep as-is - different use cases

### 5. Convolution Operations

**Duplicate Operations:**
- `t.Conv2D()` - 2D convolution
- `t.Conv1D()` - 1D convolution

**Analysis:**
- Different dimensionalities but similar algorithms
- Share common fp32 primitives
- **Recommendation:** Keep separate - different tensor shapes require different handling

## BLAS Primitive Usage

The tensor operations heavily rely on low-level BLAS primitives from `fp32`:

**Core BLAS Operations Used:**
- `fp32.Gemv_T()` - Matrix-vector multiplication (transposed)
- `fp32.Gemm_NN()` - Matrix-matrix multiplication
- `fp32.Axpy()` - Vector addition with scaling
- `fp32.Scal()` - Vector scaling
- `fp32.Copy()` - Vector copy

**Specialized Operations:**
- `fp32.ReLU()`, `fp32.Sigmoid()`, etc. - Activation functions
- `fp32.Softmax1D()`, `fp32.Softmax2DRows()` - Softmax implementations
- `fp32.Conv2D()`, `fp32.MaxPool2D()` - Convolution and pooling

## Memory Management Operations

**Required for efficient memory usage:**
- `t.copyTo(dst)` - Copy tensor data to destination
- Tensor reuse through in-place operations (`Add`, `Sub`, `Scale`, etc.)
- Pre-allocated output tensors for convolution operations

## Recommendations

### For Missing Operations
No critical operations appear to be missing. All required tensor operations are implemented.

### For Consolidation
The identified duplicate operations serve different use cases and abstraction levels. Consolidation would reduce API flexibility without significant benefits.

### For Performance
The existing implementation already optimizes for:
- Contiguous tensor access patterns
- In-place operations to minimize allocations
- BLAS-level optimizations

### For Future Extensions
Consider adding if needed:
- Additional reduction operations (variance, standard deviation)
- More advanced pooling operations
- Specialized quantization-aware operations

## Summary

The `learn` package requires a comprehensive set of tensor operations that are fully implemented in the Tensor package. The operations cover:

- ✅ **Basic tensor creation and manipulation**
- ✅ **Element-wise arithmetic operations**
- ✅ **Reduction operations**
- ✅ **Activation functions**
- ✅ **Convolution and pooling operations**
- ✅ **Linear algebra operations**
- ✅ **Gradient computation operations**

All required operations are available with appropriate performance optimizations through BLAS primitives and specialized implementations for common tensor layouts.
