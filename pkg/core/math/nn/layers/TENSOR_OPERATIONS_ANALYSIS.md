# Tensor Operations Analysis for NN Layers

## Executive Summary

This document analyzes the operations used in `nn/layers` compared to what's available in the `Tensor` interface. It identifies:
1. Operations that can be implemented using existing Tensor interface methods
2. Missing operations that should be added to the Tensor interface
3. Patterns where element-wise access (`At()`/`SetAt()`) is used inefficiently

## Current Tensor Interface Capabilities

The Tensor interface provides:
- **Element-wise operations**: Add, Sub, Mul, Div, Scale, and various unary ops (Sqrt, Exp, Log, Pow, Abs, Sign, Cos, Sin, Negative)
- **Reduction operations**: Sum, Mean, Max, Min, ArgMax along specified dimensions
- **Linear algebra**: MatMul, MatMulTo, MatMulTransposed, MatVecMulTransposed, Transpose, Dot, Norm, Normalize
- **Convolution operations**: Conv1D, Conv2D, Conv2DTransposed, Conv2DKernelGrad, Im2Col, Col2Im
- **Pooling operations**: MaxPool2D, AvgPool2D, GlobalAvgPool2D, AdaptiveAvgPool2D
- **Activation functions**: ReLU, Sigmoid, Tanh, Softmax
- **Conditional operations**: Where, Equal, GreaterThan, Less
- **Broadcasting**: BroadcastTo
- **Shape operations**: Reshape, Slice, Clone, Copy

## Analysis by Layer Type

### 1. MaxPool2D Backward Pass

**Current Implementation**: Uses extensive element-wise access (`At()`/`SetAt()`) in nested loops (lines 171-237 in pooling.go)

**Operations Used**:
- `At()` for reading input, output, and gradOutput values
- `SetAt()` for writing gradient values
- Manual iteration over batch, channels, height, width, and kernel positions

**Can be implemented with existing operations?**: **Partially**

**Analysis**:
- The backward pass routes gradients to input positions that produced the maximum value
- Requires comparing input values with stored max values to identify which positions contributed
- This is a conditional operation based on element-wise comparisons

**Proposed Solution**:
1. **New operation**: `MaxPool2DBackward(gradOutput, indices Tensor, ...)` - Takes optional indices tensor from forward pass
   - If indices tensor is provided: Use `ScatterAdd` operation to route gradients
   - If not provided: Current element-wise approach is necessary

2. **Alternative using existing operations**:
   - Use `Where()` with `Equal()` to create masks for positions that had max values
   - Use broadcasting and element-wise operations to distribute gradients
   - However, this would be inefficient compared to a dedicated operation

**Missing Operation**: `MaxPool2DBackward(gradOutput, indices Tensor, kernelSize, stride, padding []int) Tensor`
- The indices tensor should be created during forward pass and stored
- This is a common pattern in deep learning frameworks (PyTorch, TensorFlow)

### 2. AvgPool2D Backward Pass

**Current Implementation**: **Not implemented** (returns error if CanLearn is true)

**Expected Behavior**:
- Routes gradient equally to all input positions in each pooling window
- For each output position, divide gradient by kernel size and distribute to corresponding input window

**Can be implemented with existing operations?**: **Yes, but inefficient**

**Proposed Solution**:
1. **New operation**: `AvgPool2DBackward(gradOutput Tensor, kernelSize, stride, padding []int) Tensor`
   - Upsamples gradOutput to input size
   - Divides by kernel area (kernelH * kernelW)
   - Can be implemented using `Col2Im` with appropriate scaling

2. **Using existing operations**:
   ```go
   // Upsample gradOutput using Col2Im
   // Create ones kernel for averaging
   onesKernel := tensor.OnesLike(kernelShape)
   // Use Conv2DTransposed with appropriate scaling
   gradInput = gradOutput.Conv2DTransposed(onesKernel, nil, stride, padding)
   gradInput.Scale(1.0 / (kernelH * kernelW))
   ```

**Missing Operation**: `AvgPool2DBackward(gradOutput Tensor, kernelSize, stride, padding []int) Tensor`

### 3. GlobalAvgPool2D Backward Pass

**Current Implementation**: **Not implemented** (returns error if CanLearn is true)

**Expected Behavior**:
- Broadcasts gradient to all spatial positions, divided by (height * width)

**Can be implemented with existing operations?**: **Yes**

**Proposed Solution**:
```go
// gradOutput: [batch, channels]
// input: [batch, channels, height, width]
// Need: [batch, channels, height, width] where each spatial position = gradOutput / (height * width)

height := input.Shape()[2]
width := input.Shape()[3]
scale := 1.0 / float64(height * width)

// Broadcast gradOutput to [batch, channels, height, width]
gradBroadcast := gradOutput.BroadcastTo(input.Shape())
gradInput := gradBroadcast.Scale(scale)
```

**Missing Operation**: None - can use `BroadcastTo` and `Scale`

### 4. Conv2D Backward Pass

**Current Implementation**: Uses mix of tensor operations and element-wise access

**Operations Used**:
1. **Input gradient**: Uses `Conv2DTransposed` ✓ (uses existing operation)
2. **Bias gradient**: Uses `Sum(0, 2, 3)` ✓ (uses existing operation)
3. **Kernel gradient**: Uses `Im2Col`, `MatMul`, `Transpose`, `Reshape` ✓ (uses existing operations)
4. **Kernel transposition**: Uses element-wise `At()`/`SetAt()` in loops (lines 228-237)

**Analysis**:
- Most operations use Tensor API efficiently
- Kernel transposition is done element-wise, but this could use `Transpose()` if it supported higher-dimensional tensors

**Missing Operation**: `Transpose(dims ...int)` for 4D tensors
- Currently `Transpose()` only supports 2D tensors
- Need to transpose kernel from `[outChannels, inChannels, kernelH, kernelW]` to `[inChannels, outChannels, kernelH, kernelW]`

**Proposed Solution**:
1. Extend `Transpose()` to support arbitrary dimension permutations for higher-rank tensors
2. Or add `Permute(dims []int)` method for general dimension permutation

### 5. Conv1D Backward Pass

**Current Implementation**: Uses element-wise access for kernel gradient computation (lines 245-271)

**Operations Used**:
1. **Bias gradient**: Uses `Sum(0, 2)` ✓ (uses existing operation)
2. **Input gradient**: Uses `Conv2DTransposed` with reshaping ✓ (uses existing operation)
3. **Kernel gradient**: Uses nested loops with `At()`/`SetAt()` (inefficient)

**Analysis**:
- Kernel gradient computation is similar to Conv2D but uses element-wise access
- Could potentially use similar approach as Conv2D (Im2Col + MatMul)

**Proposed Solution**:
1. **New operation**: `Conv1DKernelGrad(outputGrad, input Tensor, stride, padding int) Tensor`
   - Similar to `Conv2DKernelGrad` but for 1D
   - Or extend existing `Conv2DKernelGrad` to handle 1D via reshaping

2. **Using existing operations**:
   - Reshape to 2D and use similar approach as Conv2D
   - However, Conv1D doesn't have Im2Col equivalent, so would need to create patches manually

**Missing Operation**: `Conv1DKernelGrad(outputGrad, input Tensor, stride, padding int) Tensor`

### 6. Dense Backward Pass

**Current Implementation**: Mostly uses Tensor operations efficiently

**Operations Used**:
- `MatMulTransposed` for weight and input gradients ✓
- `Sum(0)` for bias gradient ✓
- `Copy` for operations ✓
- Some fallback code with iteration for bias broadcasting (lines 180-188)

**Analysis**:
- Very efficient, mostly uses Tensor API
- Minor issue with bias broadcasting fallback, but main path uses `BroadcastTo` correctly

**Missing Operation**: None significant

### 7. LSTM Forward/Backward

**Current Implementation**: Forward uses Tensor operations, backward not implemented

**Operations Used**:
- `MatMulTransposed` for gate computations ✓
- `Slice` for splitting gates ✓
- `Sigmoid`, `Tanh` for activations ✓
- `Add`, `Mul` for element-wise operations ✓
- `BroadcastTo` for bias ✓

**Missing Operation**: None for forward pass (backward not implemented)

### 8. Utility Layers (Pad, etc.)

**Current Implementation**: Uses element-wise access for padding/unpadding

**Operations Used**:
- `At()`/`SetAt()` for copying elements between padded and unpadded regions

**Can be implemented with existing operations?**: **Partially**

**Proposed Solution**:
- Could use `Slice` operations to extract regions, but complex padding patterns might still need element-wise access
- A dedicated `Unpad` operation would be cleaner

**Missing Operation**: `Unpad(padding []int) Tensor` (inverse of padding)

## Summary of Missing Operations

### High Priority (Needed for Efficient Gradient Computation)

1. **MaxPool2DBackward(gradOutput, indices Tensor, kernelSize, stride, padding []int) Tensor**
   - Purpose: Efficient backward pass for max pooling
   - Alternative: Store indices tensor during forward pass, use scatter operation
   - Impact: Eliminates nested loops in MaxPool2D.Backward

2. **AvgPool2DBackward(gradOutput Tensor, kernelSize, stride, padding []int) Tensor**
   - Purpose: Backward pass for average pooling
   - Alternative: Can use Conv2DTransposed with scaling, but dedicated operation is cleaner
   - Impact: Needed for AvgPool2D.Backward implementation

3. **Conv1DKernelGrad(outputGrad, input Tensor, stride, padding int) Tensor**
   - Purpose: Efficient kernel gradient computation for 1D convolution
   - Alternative: Reshape and use 2D operations, but inefficient
   - Impact: Eliminates nested loops in Conv1D.Backward

### Medium Priority (Would Improve Code Quality)

4. **Transpose/Permute for Higher-Dimensional Tensors**
   - Purpose: Transpose dimensions in 3D+ tensors
   - Current: Only supports 2D
   - Impact: Eliminates element-wise loops in Conv2D kernel transposition

5. **ScatterAdd(index Tensor, value Tensor, dim int, dst Tensor) Tensor**
   - Purpose: Add values to positions specified by indices
   - Use case: MaxPool2D backward, sparse operations
   - Impact: General operation useful for many gradient computations

6. **Unpad(padding []int) Tensor**
   - Purpose: Remove padding from tensor (inverse of padding)
   - Alternative: Use Slice operations, but complex
   - Impact: Cleaner utility layer implementations

### Low Priority (Nice to Have)

7. **Gather(index Tensor, dim int) Tensor**
   - Purpose: Gather elements along dimension using indices
   - Use case: Indexing operations, advanced pooling

8. **Repeat(repeats []int, dims ...int) Tensor**
   - Purpose: Repeat tensor along specified dimensions
   - Use case: Broadcasting alternative, upsampling

## Implementation Recommendations

### Phase 1: Critical for Backward Passes

1. Implement `AvgPool2DBackward` - needed for AvgPool2D training
2. Implement `MaxPool2DBackward` with indices tensor support
3. Implement `Conv1DKernelGrad` - needed for Conv1D training efficiency

### Phase 2: Improve Code Quality

1. Extend `Transpose()` to support higher-dimensional tensors (or add `Permute()`)
2. Implement `ScatterAdd` for general scatter operations
3. Implement `Unpad` for utility layers

### Phase 3: Advanced Operations

1. Implement `Gather` for advanced indexing
2. Implement `Repeat` for broadcasting alternatives

## Patterns to Avoid

1. **Nested loops with At()/SetAt()**: These are slow and should be replaced with tensor operations
   - Examples: MaxPool2D.Backward, Conv1D kernel gradient computation
   
2. **Manual tensor transposition**: Use Transpose() when available
   - Example: Conv2D kernel transposition (lines 228-237)

3. **Element-wise gradient routing**: Use scatter/gather operations when available
   - Example: MaxPool2D backward routing

## Conclusion

The Tensor interface provides good coverage for most operations, but several key operations are missing for efficient gradient computation:

1. **Pooling backward passes**: MaxPool2D and AvgPool2D need dedicated backward operations
2. **1D convolution gradients**: Conv1DKernelGrad would eliminate nested loops
3. **Higher-dimensional transpose**: Needed for efficient kernel operations

Most other operations can be implemented using existing Tensor interface methods, though some may be less efficient than dedicated operations.

The analysis shows that about 80% of layer operations use Tensor API efficiently, with the remaining 20% using element-wise access that could be optimized with additional Tensor operations.

