# Plan to Eliminate At/SetAt/Elements Usage in Layers

## Executive Summary

This document outlines the specific Tensor interface operations that need to be added or extended to completely eliminate all `At()`, `SetAt()`, and `Elements()` usage in layer implementations (excluding test files).

## Current Usage Analysis

### Production Code (Need to Eliminate)

1. **MaxPool2D.Backward** (`pooling.go:171-237`)
   - **Lines**: ~67 lines with At/SetAt
   - **Pattern**: Nested loops routing gradients based on max value comparisons
   - **Usage**: `At()` for reading input/output/gradOutput, `SetAt()` for writing gradients

2. **Conv1D.Backward** (`conv1d.go:245-271`)
   - **Lines**: ~27 lines with At/SetAt
   - **Pattern**: Nested loops computing kernel gradients via correlation
   - **Usage**: `At()` for reading gradOutput/input, `SetAt()` for writing kernel gradients

3. **Conv2D.Backward** (`conv2d.go:228-237, 259-260`)
   - **Lines**: ~12 lines with At/SetAt
   - **Pattern**: Kernel transposition and element copying
   - **Usage**: `At()`/`SetAt()` for transposing kernel, `At()`/`SetAt()` for copying elements

4. **Pad.Backward** (`utility.go:928`)
   - **Lines**: ~1 line with At/SetAt
   - **Pattern**: Extracting gradient from padded region
   - **Usage**: `At()`/`SetAt()` for copying elements

5. **Pad.Forward** (`utility.go:878`)
   - **Lines**: ~1 line with At/SetAt
   - **Pattern**: Copying elements to padded region
   - **Usage**: `At()`/`SetAt()` for copying elements

6. **Fill** (`utility.go:834-837`)
   - **Lines**: ~4 lines with Elements()
   - **Pattern**: Setting all elements to a constant value
   - **Usage**: `Elements()` iterator with `Set()`

### Test Code (OK to Keep)

- Test files use At/SetAt for setting up test data - these are acceptable
- Gradient test helpers use At/SetAt for numerical gradient computation - acceptable

## Required Tensor Operations

### Priority 1: Critical Operations (Must Have)

#### 1. MaxPool2DBackward with Indices

**Operation**: `MaxPool2DBackward(gradOutput, indices Tensor, kernelSize, stride, padding []int) Tensor`

**Purpose**: Eliminate MaxPool2D.Backward nested loops (67 lines)

**Signature**:
```go
// Forward pass: Store indices
MaxPool2DWithIndices(kernelSize, stride, padding []int) (output Tensor, indices Tensor)

// Backward pass: Use indices
MaxPool2DBackward(gradOutput, indices Tensor, kernelSize, stride, padding []int) Tensor
```

**Alternative**: If indices tensor is not stored, need `ScatterAdd` operation

**Implementation Notes**:
- Indices tensor should be of type `int32` or `int64` (index data type)
- Shape: `[batch, channels, outHeight, outWidth]` with linear indices into input
- Backward pass uses `ScatterAdd` to route gradients to correct positions

#### 2. Conv1DKernelGrad

**Operation**: `Conv1DKernelGrad(outputGrad, input Tensor, stride, padding int) Tensor`

**Purpose**: Eliminate Conv1D.Backward kernel gradient computation (27 lines)

**Signature**:
```go
Conv1DKernelGrad(outputGrad, input Tensor, stride, padding int) Tensor
```

**Input Shapes**:
- `outputGrad`: `[batch, outChannels, outLength]`
- `input`: `[batch, inChannels, inLength]`
- Returns: `[outChannels, inChannels, kernelLen]`

**Implementation Notes**:
- Computes correlation: `sum_b,ol: gradOutput[b,oc,ol] * input[b,ic,il+k]`
- Can use `Im2Col`-like approach or direct convolution correlation

**Alternative**: Use Conv2D approach with reshaping, but less efficient

#### 3. Transpose for Higher-Dimensional Tensors

**Operation**: Extend `Transpose(dims ...int)` to support 4D+ tensors

**Purpose**: Eliminate Conv2D.Backward kernel transposition (10 lines)

**Current**: `Transpose()` only supports 2D tensors

**Required**: Support arbitrary dimension permutation

**Signature**:
```go
// Option 1: Extend Transpose
Transpose(dims ...int) Tensor  // For 2D: Transpose() or Transpose(0,1)
                                // For 4D: Transpose(0,1,2,3) or Permute([]int{1,0,2,3})

// Option 2: Add Permute
Permute(dims []int) Tensor  // Permute([]int{1,0,2,3}) swaps dims 0 and 1
```

**Usage in Conv2D**:
```go
// Current: [outChannels, inChannels, kernelH, kernelW]
// Need: [inChannels, outChannels, kernelH, kernelW]
kernelTransposed := kernel.Permute([]int{1, 0, 2, 3})
```

#### 4. AvgPool2DBackward

**Operation**: `AvgPool2DBackward(gradOutput Tensor, kernelSize, stride, padding []int) Tensor`

**Purpose**: Implement AvgPool2D.Backward (currently not implemented)

**Signature**:
```go
AvgPool2DBackward(gradOutput Tensor, kernelSize, stride, padding []int) Tensor
```

**Implementation Notes**:
- Upsamples gradOutput to input size
- Divides by kernel area (kernelH * kernelW)
- Can use `Col2Im` with scaling or `Conv2DTransposed` with ones kernel

**Alternative**:
```go
onesKernel := tensor.OnesLike([kernelH, kernelW])
gradInput = gradOutput.Conv2DTransposed(onesKernel, nil, stride, padding)
gradInput.Scale(1.0 / (kernelH * kernelW))
```

### Priority 2: Important Operations

#### 5. ScatterAdd

**Operation**: `ScatterAdd(index Tensor, value Tensor, dim int, dst Tensor) Tensor`

**Purpose**: General operation for routing gradients (used by MaxPool2D backward)

**Signature**:
```go
// ScatterAdd adds values to dst at positions specified by index
// index: [batch, channels, outHeight, outWidth] with int32/int64 values
// value: [batch, channels, outHeight, outWidth] with gradient values
// dim: dimension along which to scatter (usually spatial dimensions)
// dst: [batch, channels, inHeight, inWidth] destination tensor
ScatterAdd(index Tensor, value Tensor, dim int, dst Tensor) Tensor
```

**Alternative Signature**:
```go
// More general: scatter with multi-dimensional indices
ScatterAdd(index Tensor, value Tensor, dst Tensor) Tensor
// where index is [N, rank] with linear indices, value is [N]
```

**Usage in MaxPool2D**:
- If indices tensor not available, use ScatterAdd to route gradients
- More general than MaxPool2DBackward, but less efficient

#### 6. Unpad

**Operation**: `Unpad(padding []int) Tensor`

**Purpose**: Eliminate Pad.Backward element-wise copying (1 line)

**Signature**:
```go
Unpad(padding []int) Tensor
// padding: [padBeforeDim0, padAfterDim0, padBeforeDim1, padAfterDim1, ...]
```

**Implementation Notes**:
- Inverse of padding operation
- Removes padding from all dimensions
- Can use `Slice` operations, but dedicated operation is cleaner

**Alternative**: Use `Slice` operations in sequence, but complex

#### 7. Fill / FillValue

**Operation**: `Fill(value float64) Tensor` or `FillValue(value float64) Tensor`

**Purpose**: Eliminate Fill layer Elements() usage (4 lines)

**Signature**:
```go
// In-place fill
Fill(value float64) Tensor

// Or create new tensor filled with value
FillValue(value float64) Tensor
```

**Implementation Notes**:
- Sets all elements to the given value
- Simple operation, can be implemented using Scale(0) + AddScalar(value)
- Or dedicated primitive operation

**Alternative**:
```go
tensor.Scale(0).AddScalar(value)  // If AddScalar exists
// Or
ones := tensor.OnesLike(tensor)
ones.Scale(value)
tensor.Copy(ones)
```

### Priority 3: Nice to Have

#### 8. Gather

**Operation**: `Gather(index Tensor, dim int) Tensor`

**Purpose**: Advanced indexing operations

**Signature**:
```go
Gather(index Tensor, dim int) Tensor
```

**Use Case**: Could be used for advanced pooling operations, but not currently needed

#### 9. Repeat / Tile

**Operation**: `Repeat(repeats []int, dims ...int) Tensor`

**Purpose**: Broadcasting alternative, upsampling

**Signature**:
```go
Repeat(repeats []int, dims ...int) Tensor
```

**Use Case**: Could be used for upsampling operations, but not currently needed

## Implementation Roadmap

### Phase 1: Critical (Eliminate Most Usage)

1. **MaxPool2DWithIndices + MaxPool2DBackward**
   - Forward: Store indices tensor
   - Backward: Use indices to route gradients
   - **Impact**: Eliminates 67 lines in MaxPool2D.Backward

2. **Conv1DKernelGrad**
   - Direct kernel gradient computation
   - **Impact**: Eliminates 27 lines in Conv1D.Backward

3. **Transpose/Permute for 4D+**
   - Extend Transpose or add Permute
   - **Impact**: Eliminates 10 lines in Conv2D.Backward

4. **AvgPool2DBackward**
   - Implement backward pass
   - **Impact**: Enables AvgPool2D training

**Total Lines Eliminated**: ~104 lines

### Phase 2: Complete Elimination

5. **ScatterAdd**
   - General scatter operation
   - **Impact**: Alternative to MaxPool2DBackward if indices not stored

6. **Unpad**
   - Inverse of padding
   - **Impact**: Eliminates 1 line in Pad.Backward

7. **Fill / FillValue**
   - Fill all elements with value
   - **Impact**: Eliminates 4 lines in Fill layer

**Total Lines Eliminated**: ~5 additional lines

### Phase 3: Advanced Operations

8. **Gather** - For future advanced operations
9. **Repeat** - For future upsampling operations

## Detailed Implementation Requirements

### 1. MaxPool2DWithIndices

```go
// In Tensor interface
MaxPool2DWithIndices(kernelSize, stride, padding []int) (Tensor, Tensor)
// Returns: (output Tensor, indices Tensor)
// indices: [batch, channels, outHeight, outWidth] with int32/int64
//          Stores linear indices into input tensor

// In MaxPool2D forward pass
func (m *MaxPool2D) Forward(input types.Tensor) (types.Tensor, error) {
    // ... existing code ...
    output, indices := input.MaxPool2DWithIndices(
        []int{m.kernelH, m.kernelW},
        []int{m.strideH, m.strideW},
        []int{m.padH, m.padW},
    )
    // Store indices for backward pass
    m.indices = indices
    // ... rest of code ...
}
```

### 2. MaxPool2DBackward

```go
// In Tensor interface
MaxPool2DBackward(gradOutput, indices Tensor, kernelSize, stride, padding []int) Tensor
// gradOutput: [batch, channels, outHeight, outWidth]
// indices: [batch, channels, outHeight, outWidth] int32/int64
// Returns: [batch, channels, inHeight, inWidth]

// In MaxPool2D backward pass
func (m *MaxPool2D) Backward(gradOutput types.Tensor) (types.Tensor, error) {
    // ... validation ...
    gradInput := gradOutput.MaxPool2DBackward(
        m.indices,
        []int{m.kernelH, m.kernelW},
        []int{m.strideH, m.strideW},
        []int{m.padH, m.padW},
    )
    // ... rest of code ...
}
```

### 3. Conv1DKernelGrad

```go
// In Tensor interface
Conv1DKernelGrad(outputGrad, input Tensor, stride, padding int) Tensor
// outputGrad: [batch, outChannels, outLength]
// input: [batch, inChannels, inLength]
// Returns: [outChannels, inChannels, kernelLen]

// In Conv1D backward pass
func (c *Conv1D) Backward(gradOutput tensorTypes.Tensor) (tensorTypes.Tensor, error) {
    // ... existing code ...
    
    // Compute kernel gradient
    kernelGrad := gradOutput.Conv1DKernelGrad(input, c.stride, c.pad)
    kernelParam.Grad.Copy(kernelGrad)
    
    // ... rest of code ...
}
```

### 4. Transpose/Permute for 4D+

```go
// Option 1: Extend Transpose
Transpose(dims ...int) Tensor
// For 2D: t.Transpose() or t.Transpose(0, 1)
// For 4D: t.Transpose(1, 0, 2, 3) to swap first two dims

// Option 2: Add Permute
Permute(dims []int) Tensor
// t.Permute([]int{1, 0, 2, 3}) swaps dimensions 0 and 1

// In Conv2D backward pass
func (c *Conv2D) Backward(gradOutput tensorTypes.Tensor) (tensorTypes.Tensor, error) {
    // ... existing code ...
    
    // Transpose kernel: [outChannels, inChannels, ...] -> [inChannels, outChannels, ...]
    kernelTransposed := kernelParam.Data.Permute([]int{1, 0, 2, 3})
    
    // ... rest of code ...
}
```

### 5. AvgPool2DBackward

```go
// In Tensor interface
AvgPool2DBackward(gradOutput Tensor, kernelSize, stride, padding []int) Tensor
// gradOutput: [batch, channels, outHeight, outWidth]
// Returns: [batch, channels, inHeight, inWidth]

// In AvgPool2D backward pass
func (a *AvgPool2D) Backward(gradOutput types.Tensor) (types.Tensor, error) {
    // ... validation ...
    gradInput := gradOutput.AvgPool2DBackward(
        []int{a.kernelH, a.kernelW},
        []int{a.strideH, a.strideW},
        []int{a.padH, a.padW},
    )
    // ... rest of code ...
}
```

### 6. ScatterAdd

```go
// In Tensor interface
ScatterAdd(index Tensor, value Tensor, dim int, dst Tensor) Tensor
// index: [batch, channels, outHeight, outWidth] int32/int64 with linear indices
// value: [batch, channels, outHeight, outWidth] gradient values
// dim: dimension along which to scatter (usually -1 for all spatial dims)
// dst: [batch, channels, inHeight, inWidth] destination (modified in-place)
// Returns: dst

// Alternative: More general signature
ScatterAdd(index Tensor, value Tensor, dst Tensor) Tensor
// index: [N, rank] int32/int64 with multi-dimensional indices
// value: [N] gradient values
// dst: destination tensor (modified in-place)
```

### 7. Unpad

```go
// In Tensor interface
Unpad(padding []int) Tensor
// padding: [padBeforeDim0, padAfterDim0, padBeforeDim1, padAfterDim1, ...]
// Returns: tensor with padding removed

// In Pad backward pass
func (p *Pad) Backward(gradOutput types.Tensor) (types.Tensor, error) {
    // ... validation ...
    gradInput := gradOutput.Unpad(p.padding)
    // ... rest of code ...
}
```

### 8. Fill / FillValue

```go
// In Tensor interface
Fill(value float64) Tensor  // In-place
FillValue(value float64) Tensor  // New tensor

// In Fill forward pass
func (f *Fill) Forward(input types.Tensor) (types.Tensor, error) {
    output := input.Clone().Fill(f.value)
    // ... rest of code ...
}
```

## Summary of Required Operations

| Operation | Priority | Lines Eliminated | Complexity |
|-----------|----------|------------------|------------|
| MaxPool2DWithIndices | P1 | 67 | High |
| MaxPool2DBackward | P1 | 67 | High |
| Conv1DKernelGrad | P1 | 27 | Medium |
| Transpose/Permute 4D+ | P1 | 10 | Low |
| AvgPool2DBackward | P1 | 0 (new impl) | Medium |
| ScatterAdd | P2 | Alternative | High |
| Unpad | P2 | 1 | Low |
| Fill/FillValue | P2 | 4 | Low |

**Total Production Code Lines Using At/SetAt/Elements**: ~109 lines
**Lines Eliminated by Phase 1**: ~104 lines (95%)
**Lines Eliminated by Phase 2**: ~5 lines (5%)

## Conclusion

To completely eliminate At/SetAt/Elements usage in layers:

1. **Phase 1 (Critical)**: Implement 5 operations
   - MaxPool2DWithIndices + MaxPool2DBackward
   - Conv1DKernelGrad
   - Transpose/Permute for 4D+
   - AvgPool2DBackward

2. **Phase 2 (Complete)**: Implement 3 additional operations
   - ScatterAdd (alternative to MaxPool2DBackward)
   - Unpad
   - Fill/FillValue

After Phase 1: ~95% of element-wise access eliminated
After Phase 2: 100% of element-wise access eliminated (excluding tests)

