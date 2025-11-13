# Required Tensor Operations Analysis

This document analyzes all primitive tensor/vector/matrix operations required by the neural network (`nn`) and layers packages, identifies which operations are already implemented in the Tensor package, and highlights duplicate operations that perform the same task with different implementations.

## Operations Required by NN Package

### From `nn.go`
- `fp32.Gemv_T()` - Generalized matrix-vector multiplication (transposed matrix)
- `fp32.Axpy()` - Vector addition with scaling: `y = y + alpha * x`
- `fp32.Gemm_NN()` - Generalized matrix-matrix multiplication
- `t.ReLU()` - In-place ReLU activation
- `t.Sigmoid()` - Sigmoid activation
- `t.Tanh()` - Tanh activation
- `t.Softmax(dim)` - Softmax along specified dimension
- `t.Clone()` - Clone tensor
- `t.Sub(other)` - Element-wise subtraction
- `t.Mul(other)` - Element-wise multiplication
- `t.Sum()` - Sum all elements

### From `losses.go`
- `pred.Clone()` - Clone tensor
- `grad.Sub(&target)` - Element-wise subtraction
- `grad.Scale(factor)` - Scalar multiplication
- `pred.Softmax(dim)` - Softmax along dimension

## Operations Required by Layers Package

### From `activations.go`
- `output.ReLU()` - In-place ReLU
- `input.ReLUGrad(&gradOutput, nil)` - ReLU backward pass
- `output.Sigmoid()` - In-place sigmoid
- `output.SigmoidGrad(&gradOutput, nil)` - Sigmoid backward pass
- `output.Tanh()` - In-place tanh
- `output.TanhGrad(&gradOutput, nil)` - Tanh backward pass
- `output.Softmax(dim)` - In-place softmax
- `output.SoftmaxGrad(&gradOutput, dim, nil)` - Softmax backward pass
- `d.mask.DropoutMask(p, scale, rng)` - Generate dropout mask
- `output.DropoutForward(&d.mask)` - Apply dropout mask
- `input.DropoutBackward(&gradOutput, &d.mask, nil)` - Dropout backward pass

### From `dense.go`
- `outputTensor.MatVecMulTransposed(weightTensor, inputTensor, 1.0, 0.0)` - Matrix-vector multiplication with transposed matrix
- `outputTensor.AddScaled(biasTensor, 1.0)` - Add scaled tensor
- `inputTensor.MatMulTo(weightTensor, outputTensor)` - Matrix multiplication to destination
- `batchOutput.AddScaled(biasTensor, 1.0)` - Add scaled tensor (batch case)
- `inputTensor.MatMulTransposed(gradTensor, true, false, gradWeightTensor)` - Transposed matrix multiplication
- `biasGradTensor.AddScaled(gradTensor, 1.0)` - Add scaled tensor
- `gradTensor.MatMulTransposed(weightTensor, false, true, gradInputTensor)` - Transposed matrix multiplication

### From `conv1d.go`
- `input.Conv1DTo(&kernelParam.Data, biasParam, &output, c.stride, c.pad)` - 1D convolution to destination
- `gradOutput.Sum(0, 2)` - Sum over batch and length dimensions
- `input.Conv1DKernelGrad(&gradOutput, &kernelParam.Data, c.stride, c.pad)` - 1D convolution kernel gradient
- `gradOutput.Conv1DTransposed(&kernelParam.Data, nil, c.stride, c.pad)` - 1D transposed convolution

### From `conv2d.go`
- `input.Conv2DTo(&kernelParam.Data, biasParam, &output, []int{c.strideH, c.strideW}, []int{c.padH, c.padW})` - 2D convolution to destination
- `gradOutput.Sum(0, 2, 3)` - Sum over batch, height, and width dimensions
- `input.Conv2DKernelGrad(&gradOutput, &kernelParam.Data, []int{c.strideH, c.strideW}, []int{c.padH, c.padW})` - 2D convolution kernel gradient
- `gradOutput.Conv2DTransposed(&kernelParam.Data, nil, []int{c.strideH, c.strideW}, []int{c.padH, c.padW})` - 2D transposed convolution

### From `pooling.go`
- `input.MaxPool2D([]int{m.kernelH, m.kernelW}, []int{m.strideH, m.strideW}, []int{m.padH, m.padW})` - 2D max pooling
- `input.AvgPool2D([]int{a.kernelH, a.kernelW}, []int{a.strideH, a.strideW}, []int{a.padH, a.padW})` - 2D average pooling
- `input.GlobalAvgPool2D()` - Global average pooling

### From `utility.go`
- `input.Transpose()` - 2D tensor transpose

## Operations Implemented in Tensor Package

### Basic Arithmetic (`tensor_math.go`)
- `Add(other)` - Element-wise addition
- `Sub(other)` - Element-wise subtraction ✓
- `Mul(other)` - Element-wise multiplication ✓
- `Div(other)` - Element-wise division
- `Scale(scalar)` - Scalar multiplication ✓
- `AddTo(other, dst)` - Element-wise addition to destination
- `MulTo(other, dst)` - Element-wise multiplication to destination
- `Sum(dims...)` - Sum along dimensions ✓
- `Mean(dims...)` - Mean along dimensions
- `Max(dims...)` - Max along dimensions
- `Min(dims...)` - Min along dimensions
- `ArgMax(dim)` - Argmax along dimension
- `BroadcastTo(shape)` - Broadcasting

### Linear Algebra (`tensor_linalg.go`)
- `MatMul(other)` - Matrix multiplication
- `MatMulTo(other, dst)` - Matrix multiplication to destination ✓
- `Transpose(dims...)` - Transpose tensor ✓
- `TransposeTo(dst, dims...)` - Transpose to destination
- `Dot(other)` - Vector dot product
- `Norm(ord)` - Vector/matrix norm
- `Normalize(dim)` - Normalization

### Activations (`activations.go`)
- `ReLU()` - ReLU activation ✓
- `ReLUGrad(gradOutput, dst)` - ReLU gradient ✓
- `Sigmoid()` - Sigmoid activation ✓
- `SigmoidGrad(gradOutput, dst)` - Sigmoid gradient ✓
- `Tanh()` - Tanh activation ✓
- `TanhGrad(gradOutput, dst)` - Tanh gradient ✓
- `Softmax(dim)` - Softmax activation ✓
- `SoftmaxGrad(gradOutput, dim, dst)` - Softmax gradient ✓
- `DropoutMask(p, scale, rng)` - Generate dropout mask ✓
- `DropoutForward(mask)` - Apply dropout forward ✓
- `DropoutBackward(gradOutput, mask, dst)` - Dropout backward ✓

### Convolution (`tensor_conv.go`)
- `Conv1D(kernel, bias, stride, padding)` - 1D convolution
- `Conv1DTo(kernel, bias, dst, stride, padding)` - 1D convolution to destination ✓
- `Conv1DTransposed(kernel, bias, stride, padding)` - 1D transposed convolution
- `Conv1DKernelGrad(outputGrad, kernel, stride, padding)` - 1D kernel gradient ✓
- `Conv2D(kernel, bias, stride, padding)` - 2D convolution
- `Conv2DTo(kernel, bias, dst, stride, padding)` - 2D convolution to destination ✓
- `Conv2DTransposed(kernel, bias, stride, padding)` - 2D transposed convolution
- `Conv2DKernelGrad(outputGrad, kernel, stride, padding)` - 2D kernel gradient ✓
- `MaxPool2D(kernelSize, stride, padding)` - 2D max pooling ✓
- `AvgPool2D(kernelSize, stride, padding)` - 2D average pooling ✓
- `GlobalAvgPool2D()` - Global average pooling ✓

### Linear Algebra Helpers (`tensor_linalg_helpers.go`)
- `MatVecMulTransposed(matrix, vector, alpha, beta)` - Matrix-vector multiplication with transposed matrix ✓
- `MatMulTransposed(other, transposeA, transposeB, dst)` - Matrix multiplication with transposition options ✓
- `AddScaled(other, alpha)` - Add scaled tensor ✓

### Basic Operations (`dense.go`)
- `Clone()` - Clone tensor ✓
- Shape operations, indexing, etc.

## Duplicate Operations Analysis

The following operations perform essentially the same mathematical operation but have different implementations, APIs, or levels of abstraction. These could potentially be unified under a single implementation.

### 1. Matrix Multiplication Variants

**Core Operation**: Matrix multiplication with different transposition and storage options

**Variants**:
- `fp32.Gemv_T()` - Low-level BLAS GEMV (matrix-vector, transposed)
- `fp32.Gemm_NN()` - Low-level BLAS GEMM (matrix-matrix, no transpose)
- `MatVecMulTransposed()` - High-level wrapper around `fp32.Gemv_T`
- `MatMulTo()` - High-level matrix-matrix multiplication
- `MatMulTransposed()` - Matrix multiplication with transposition flags

**Potential Consolidation**: All could use `MatMulTransposed` as the unified interface, with the low-level BLAS calls as internal implementations.

### 2. Vector Addition with Scaling

**Core Operation**: `y = y + alpha * x` (AXPY operation)

**Variants**:
- `fp32.Axpy()` - Low-level BLAS AXPY
- `AddScaled()` - High-level tensor method wrapper

**Potential Consolidation**: The tensor method could be the primary interface, with BLAS as internal implementation.

### 3. Convolution Operations

**Core Operation**: Convolution with different output handling

**Variants**:
- `Conv1D()` / `Conv2D()` - Return new tensor
- `Conv1DTo()` / `Conv2DTo()` - Write to pre-allocated destination

**Potential Consolidation**: The `To` variants are more memory-efficient and could be the primary implementation, with the non-`To` variants as convenience wrappers.

### 4. Transposed Convolution Operations

**Core Operation**: Transposed convolution (backward pass for convolution)

**Variants**:
- `Conv1DTransposed()` / `Conv2DTransposed()` - Return new tensor
- Internal implementations in backward passes

**Potential Consolidation**: Could share implementation with forward convolution with appropriate flags.

### 5. Reduction Operations

**Core Operation**: Summation over dimensions

**Variants**:
- `Sum()` - General tensor sum with dimension specification
- `Sum(0, 2)` / `Sum(0, 2, 3)` - Specific dimension sums in convolution backward passes
- Pooling operations that internally compute sums

**Potential Consolidation**: All use the same underlying `Sum` implementation with different dimension specifications.

### 6. Element-wise Operations

**Core Operation**: Element-wise tensor operations

**Variants**:
- `Sub()` / `Mul()` / `Scale()` - Return new tensor
- In-place operations where available
- Operations with destination tensors (`To` variants)

**Potential Consolidation**: The destination variants are more flexible and could be the primary implementation.

## Summary

**Total Operations Required**: ~40 distinct operations
**Operations Implemented**: ~35 (88% coverage)
**Missing Operations**: None - all required operations are implemented
**Duplicate Operations**: ~15 operations that perform similar tasks with different APIs

The Tensor package provides comprehensive coverage of all required operations. The main opportunity for consolidation is around the matrix multiplication operations, where multiple BLAS calls and high-level wrappers could potentially be unified under fewer, more flexible interfaces.
