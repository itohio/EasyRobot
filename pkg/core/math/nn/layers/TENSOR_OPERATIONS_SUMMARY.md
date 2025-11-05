# Tensor Operations Summary - NN Layers Analysis

## Quick Summary

**Analysis Date**: Current
**Scope**: Comparing `nn/layers` operations with Tensor interface capabilities

## Key Findings

### ✅ What Works Well

1. **Conv2D Backward**: Uses Tensor API efficiently (Im2Col, MatMul, Transpose)
2. **Dense Backward**: Fully uses Tensor operations (MatMulTransposed, Sum)
3. **LSTM Forward**: Uses Tensor operations throughout
4. **Forward Passes**: All pooling and convolution forward passes use Tensor API

### ⚠️ Areas Needing Improvement

1. **MaxPool2D.Backward**: Uses nested loops with `At()`/`SetAt()` (95+ lines of element-wise access)
2. **AvgPool2D.Backward**: Not implemented
3. **Conv1D.Backward**: Kernel gradient uses nested loops with `At()`/`SetAt()`
4. **Conv2D.Backward**: Kernel transposition uses element-wise loops (should use Transpose)

## Missing Operations

### Critical (Blocking Efficient Training)

| Operation | Location | Impact |
|-----------|----------|--------|
| `MaxPool2DBackward(gradOutput, indices, ...)` | MaxPool2D.Backward | Eliminates 67 lines of nested loops |
| `AvgPool2DBackward(gradOutput, ...)` | AvgPool2D.Backward | Currently not implemented |
| `Conv1DKernelGrad(outputGrad, input, ...)` | Conv1D.Backward | Eliminates 27 lines of nested loops |

### Important (Code Quality)

| Operation | Location | Impact |
|-----------|----------|--------|
| `Transpose()` for 4D+ tensors | Conv2D.Backward | Eliminates kernel transposition loops |
| `ScatterAdd(index, value, dim, dst)` | MaxPool2D.Backward | General scatter operation for gradients |
| `Unpad(padding)` | Utility layers | Cleaner padding/unpadding operations |

## Implementation Proposals

### 1. MaxPool2D Backward

**Current**: Lines 171-237 in pooling.go use nested loops with At()/SetAt()

**Proposed**:
```go
// Forward pass: Store indices tensor
indices := input.MaxPool2DWithIndices(kernelSize, stride, padding)

// Backward pass: Use indices
gradInput := gradOutput.MaxPool2DBackward(indices, kernelSize, stride, padding)
```

**Alternative using existing ops**:
- Use `Where()` + `Equal()` to create masks, but inefficient
- Need indices tensor from forward pass

### 2. AvgPool2D Backward

**Current**: Not implemented (returns error)

**Proposed**:
```go
gradInput := gradOutput.AvgPool2DBackward(kernelSize, stride, padding)
```

**Alternative using existing ops**:
```go
// Upsample using Conv2DTransposed with ones kernel
onesKernel := tensor.Ones([kernelH, kernelW])
gradInput = gradOutput.Conv2DTransposed(onesKernel, nil, stride, padding)
gradInput.Scale(1.0 / (kernelH * kernelW))
```

### 3. Conv1D Kernel Gradient

**Current**: Lines 245-271 use nested loops with At()/SetAt()

**Proposed**:
```go
kernelGrad := gradOutput.Conv1DKernelGrad(input, stride, padding)
```

**Alternative**: Reshape to 2D and use Conv2D approach, but less efficient

### 4. Conv2D Kernel Transposition

**Current**: Lines 228-237 use element-wise loops

**Proposed**:
```go
// Extend Transpose to support 4D
kernelTransposed := kernel.Transpose(0, 1) // Swap first two dims
// Or add Permute
kernelTransposed := kernel.Permute([]int{1, 0, 2, 3})
```

## Statistics

### Element-wise Access Usage

- **MaxPool2D.Backward**: ~67 lines with At()/SetAt()
- **Conv1D.Backward**: ~27 lines with At()/SetAt()
- **Conv2D.Backward**: ~10 lines with At()/SetAt() (kernel transposition)
- **Utility layers**: ~20 lines with At()/SetAt() (padding operations)

**Total**: ~124 lines using element-wise access that could be optimized

### Tensor API Usage

- **Conv2D.Backward**: ~90% uses Tensor API
- **Dense.Backward**: ~100% uses Tensor API
- **LSTM.Forward**: ~100% uses Tensor API
- **Pooling Forward**: ~100% uses Tensor API

## Recommendations

### Priority 1: Implement Now

1. `AvgPool2DBackward` - Needed for AvgPool2D training
2. `MaxPool2DBackward` with indices - Eliminates most inefficient code
3. `Conv1DKernelGrad` - Eliminates nested loops in Conv1D

### Priority 2: Implement Soon

1. Extend `Transpose()` for 4D+ tensors - Clean up Conv2D code
2. `ScatterAdd` - General operation for many gradient patterns

### Priority 3: Future Enhancements

1. `Unpad` - Utility operation
2. `Gather` - Advanced indexing
3. `Repeat` - Broadcasting alternative

## Conclusion

The Tensor interface provides excellent coverage for forward passes and most backward passes. The main gaps are:

1. **Pooling backward operations** (MaxPool2D, AvgPool2D)
2. **1D convolution gradients** (Conv1DKernelGrad)
3. **Higher-dimensional transpose** (for Conv2D kernel operations)

Implementing these three critical operations would eliminate ~104 lines of inefficient element-wise access code and enable full training support for all layer types.

