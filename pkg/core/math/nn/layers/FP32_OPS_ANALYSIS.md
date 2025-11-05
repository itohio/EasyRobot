# FP32 Operations Analysis for Eliminating At/SetAt/Elements

## Executive Summary

This document analyzes which operations from `ELIMINATE_AT_SETAT_PLAN.md` can be implemented using existing `fp32` primitives and which require new primitive implementations.

## Current FP32 Primitive Capabilities

From `primitive/fp32/OPS.md`, available operations include:

### Available Operations
- **Element-wise**: `ElemAdd`, `ElemSub`, `ElemMul`, `ElemDiv`, `ElemScale`, `ElemCopy` (with stride/shape support)
- **BLAS Level 1-3**: Vector and matrix operations
- **Convolution**: `Conv2D`, `Conv2DTransposed`, `Im2Col`, `Col2Im`
- **Pooling (Forward)**: `MaxPool2D`, `AvgPool2D`, `GlobalAvgPool2D`, `AdaptiveAvgPool2D`
- **Reduction**: `ReduceSum`, `ReduceMean`, `ReduceMax`, `ReduceMin`, `Argmax`
- **Broadcasting**: `ExpandTo` (broadcast tensor to target shape)
- **Utilities**: `ComputeStrides`, `IndexLinear`, `BroadcastStrides`

### Missing Operations
- **Pooling Backward**: No `MaxPool2DBackward`, `AvgPool2DBackward`
- **1D Convolution Gradients**: `Conv1DKernelGrad` is deprecated (should compose from primitives)
- **Scatter/Gather**: No `ScatterAdd`, `Gather`
- **Transpose**: Only 2D supported (manual loops), no higher-dimensional transpose primitive
- **Fill**: No direct fill primitive (can use `ElemScale` + `ElemAdd` with scalar)

## Analysis by Required Operation

### 1. Higher-Dimensional Transpose (Priority 1)

**Required**: `Transpose(dims ...int)` or `Permute(dims []int)` for 4D+ tensors

**Current Status**: 
- Only 2D transpose implemented (manual loops in `tensor_linalg.go:283-303`)
- Uses element-by-element copy with manual indexing

**Can Use Existing Primitives?**: **YES** - `ElemCopy` with stride calculation

**Implementation Strategy**:
```go
// Use ElemCopy with computed strides
// For 4D tensor [A, B, C, D] permuting to [B, A, C, D]:
// 1. Compute source strides: [B*C*D, C*D, D, 1]
// 2. Compute destination strides for permuted shape: [A*C*D, C*D, D, 1]
// 3. Use ElemCopy with shape and both stride arrays

func Permute(dims []int) Tensor {
    // Compute new shape
    newShape := computePermutedShape(shape, dims)
    
    // Compute strides for source (original) and destination (permuted)
    srcStrides := ComputeStrides(shape)
    dstStrides := ComputeStrides(newShape)
    
    // Use ElemCopy to copy elements according to permutation
    result := New(t.DataType(), newShape)
    fp32.ElemCopy(
        result.Data().([]float32),
        t.Data().([]float32),
        newShape,
        dstStrides,
        srcStrides, // Need to permute these strides too
    )
    return result
}
```

**Complexity**: Medium - Need to compute permuted strides correctly

**FP32 Primitive Needed**: None - can use `ElemCopy` with stride calculation

**Alternative**: Add `Permute` primitive to fp32 for efficiency, but not strictly necessary

---

### 2. MaxPool2DWithIndices + MaxPool2DBackward (Priority 1)

**Required**: 
- Forward: `MaxPool2DWithIndices` - returns output and indices tensor
- Backward: `MaxPool2DBackward` - uses indices to route gradients

**Current Status**:
- `MaxPool2D` exists (forward only)
- No indices output
- No backward pass

**Can Use Existing Primitives?**: **PARTIALLY**

**MaxPool2DWithIndices Implementation**:
```go
// Can extend existing MaxPool2D to also return indices
// Need to modify fp32.MaxPool2D to also write indices

// Current: fp32.MaxPool2D(dst, src, ...)
// Need: fp32.MaxPool2DWithIndices(dst, indices, src, ...)
```

**MaxPool2DBackward Implementation Options**:

**Option A: Use ScatterAdd (if implemented)**
```go
// If ScatterAdd exists:
gradInput = gradOutput.ScatterAdd(indices, dim, gradInput)
```

**Option B: Use ElemCopy with conditional routing**
```go
// Need to route gradients based on indices
// This requires element-wise conditional operations
// Could use:
// 1. Create mask tensor using indices
// 2. Use Where() to conditionally add gradients
// But this is inefficient
```

**Option C: Direct primitive implementation**
```go
// New fp32 primitive: MaxPool2DBackward
fp32.MaxPool2DBackward(
    gradInput, gradOutput, indices,
    batchSize, channels, inHeight, inWidth,
    outHeight, outWidth, kernelH, kernelW,
    strideH, strideW, padH, padW,
)
```

**FP32 Primitive Needed**: 
1. **MaxPool2DWithIndices** - Extend existing `MaxPool2D` to output indices
2. **MaxPool2DBackward** - New primitive for efficient gradient routing

**Complexity**: High - Requires new primitive implementation

---

### 3. Conv1DKernelGrad (Priority 1)

**Required**: `Conv1DKernelGrad(outputGrad, input, stride, padding)`

**Current Status**:
- `Conv1DKernelGrad` exists but is **DEPRECATED** (line 114 in OPS.md)
- Should be composed from primitives

**Can Use Existing Primitives?**: **YES** - Compose from existing operations

**Implementation Strategy**:
```go
// Conv1D kernel gradient: sum_b,ol: gradOutput[b,oc,ol] * input[b,ic,il+k]
// This is a correlation operation

// Option A: Use Im2Col approach (similar to Conv2D)
// 1. Convert input to column format (need 1D Im2Col)
// 2. Reshape gradOutput
// 3. Use GEMM: gradOutput^T @ inputCols

// Option B: Use Conv2D approach with reshaping
// Reshape to 2D, use Conv2DKernelGrad approach, reshape back

// Option C: Direct correlation using element-wise operations
// Loop over kernel positions, use ElemMul + ReduceSum
```

**FP32 Primitives Needed**: 
- **Option 1**: `Im2Col1D` (1D version of Im2Col) - **NEW**
- **Option 2**: Use existing `Im2Col` with width=1 (reshape to 2D) - **EXISTING**

**Recommended**: Use Option 2 (reshape to 2D, use existing Im2Col + GEMM)

**Complexity**: Medium - Can compose from existing primitives

---

### 4. AvgPool2DBackward (Priority 1)

**Required**: `AvgPool2DBackward(gradOutput, kernelSize, stride, padding)`

**Current Status**:
- `AvgPool2D` exists (forward only)
- No backward pass

**Can Use Existing Primitives?**: **YES** - Compose from existing operations

**Implementation Strategy**:
```go
// AvgPool2D backward: Upsample gradOutput and divide by kernel area
// Option A: Use Conv2DTransposed with ones kernel
onesKernel := tensor.OnesLike([kernelH, kernelW])
gradInput = gradOutput.Conv2DTransposed(onesKernel, nil, stride, padding)
gradInput.Scale(1.0 / (kernelH * kernelW))
```

**FP32 Primitives Needed**: None - can use existing:
- `Conv2DTransposed` (exists)
- `ElemScale` (exists) for scaling

**Alternative**: Direct primitive for efficiency:
```go
fp32.AvgPool2DBackward(
    gradInput, gradOutput,
    batchSize, channels, inHeight, inWidth,
    outHeight, outWidth, kernelH, kernelW,
    strideH, strideW, padH, padW,
)
```

**Complexity**: Low (using existing) or Medium (if new primitive)

**Recommendation**: Start with existing primitives, add dedicated primitive later if needed

---

### 5. ScatterAdd (Priority 2)

**Required**: `ScatterAdd(index Tensor, value Tensor, dim int, dst Tensor)`

**Current Status**: Does not exist

**Can Use Existing Primitives?**: **NO** - Requires new primitive

**Implementation Strategy**:
```go
// ScatterAdd adds values to dst at positions specified by index
// This is a sparse update operation

// Need primitive:
fp32.ScatterAdd(
    dst, index, value,
    dstShape, dstStrides,
    indexShape, indexStrides,
    valueShape, valueStrides,
    dim,
)
```

**FP32 Primitive Needed**: **NEW** - `ScatterAdd`

**Complexity**: High - Requires efficient sparse update implementation

**Note**: This is a general operation, useful for many gradient patterns beyond MaxPool2D

---

### 6. Unpad (Priority 2)

**Required**: `Unpad(padding []int) Tensor`

**Current Status**: Does not exist

**Can Use Existing Primitives?**: **YES** - Use `ElemCopy` with stride calculation

**Implementation Strategy**:
```go
// Unpad: Remove padding from tensor
// Similar to Slice but for multiple dimensions

// Use ElemCopy with:
// - Source: padded tensor
// - Destination: unpadded tensor
// - Strides: computed to skip padding regions

fp32.ElemCopy(
    dst, src,
    dstShape, dstStrides,
    srcShape, srcStrides, // Adjusted for padding offset
)
```

**FP32 Primitive Needed**: None - can use `ElemCopy`

**Complexity**: Medium - Need to compute correct strides with padding offsets

**Alternative**: Dedicated primitive for clarity:
```go
fp32.Unpad(
    dst, src,
    dstShape, srcShape,
    padding, // [padBeforeDim0, padAfterDim0, ...]
)
```

---

### 7. Fill/FillValue (Priority 2)

**Required**: `Fill(value float64) Tensor` or `FillValue(value float64) Tensor`

**Current Status**: Does not exist as single primitive

**Can Use Existing Primitives?**: **YES** - Compose from existing

**Implementation Strategy**:
```go
// Option A: Use ElemScale(0) + ElemAdd scalar
// But ElemAdd doesn't support scalar directly

// Option B: Use ElemCopy from constant tensor
ones := tensor.OnesLike(tensor)
ones.Scale(value)
tensor.Copy(ones)

// Option C: Direct primitive
fp32.Fill(dst, value, shape, strides)
```

**FP32 Primitive Needed**: **NEW** - `Fill` (simple, but useful)

**Complexity**: Low - Very simple operation

**Alternative**: Can work around with existing ops, but dedicated primitive is cleaner

---

## Summary Table

| Operation | Use Existing? | FP32 Primitive Needed | Complexity | Priority |
|-----------|---------------|----------------------|------------|----------|
| **Transpose 4D+** | ✅ Yes | None (use `ElemCopy`) | Medium | P1 |
| **MaxPool2DWithIndices** | ⚠️ Partial | Extend `MaxPool2D` | Medium | P1 |
| **MaxPool2DBackward** | ❌ No | **NEW**: `MaxPool2DBackward` | High | P1 |
| **Conv1DKernelGrad** | ✅ Yes | None (use `Im2Col` + GEMM) | Medium | P1 |
| **AvgPool2DBackward** | ✅ Yes | None (use `Conv2DTransposed` + `Scale`) | Low | P1 |
| **ScatterAdd** | ❌ No | **NEW**: `ScatterAdd` | High | P2 |
| **Unpad** | ✅ Yes | None (use `ElemCopy`) | Medium | P2 |
| **Fill** | ⚠️ Partial | **NEW**: `Fill` (optional) | Low | P2 |

## Implementation Roadmap

### Phase 1: Can Use Existing Primitives (No New FP32 Primitives)

1. **Transpose 4D+** - Use `ElemCopy` with stride calculation
2. **Conv1DKernelGrad** - Compose from `Im2Col` + GEMM
3. **AvgPool2DBackward** - Use `Conv2DTransposed` + `ElemScale`
4. **Unpad** - Use `ElemCopy` with padding offset strides

### Phase 2: Need New FP32 Primitives

1. **MaxPool2DWithIndices** - Extend existing `MaxPool2D`
2. **MaxPool2DBackward** - New primitive
3. **ScatterAdd** - New primitive (general purpose)
4. **Fill** - New primitive (optional, simple)

## Detailed Implementation: Higher-Dimensional Transpose

### Current Implementation (2D only)

```go
// tensor_linalg.go:283-303
func (t Tensor) transpose2D() types.Tensor {
    M, N := t.shape[0], t.shape[1]
    result := New(t.DataType(), types.NewShape(N, M))
    resultData := types.GetTensorData[[]float32](resultPtr)
    tData := types.GetTensorData[[]float32](&t)
    for i := 0; i < M; i++ {
        for j := 0; j < N; j++ {
            resultData[j*M+i] = tData[i*N+j]
        }
    }
    return resultPtr
}
```

### Proposed Implementation (4D+ using ElemCopy)

```go
func (t Tensor) Permute(dims []int) types.Tensor {
    if t.shape == nil {
        return nil
    }
    
    shape := t.Shape()
    rank := shape.Rank()
    
    // Validate permutation
    if len(dims) != rank {
        panic(fmt.Sprintf("Permute: permutation length %d must match tensor rank %d", len(dims), rank))
    }
    
    // Compute permuted shape
    newShape := make(types.Shape, rank)
    for i, d := range dims {
        if d < 0 || d >= rank {
            panic(fmt.Sprintf("Permute: invalid dimension %d in permutation", d))
        }
        newShape[i] = shape[d]
    }
    
    // Compute source and destination strides
    srcStrides := shape.Strides()
    dstStrides := types.NewShape(newShape...).Strides()
    
    // Create result tensor
    result := New(t.DataType(), types.NewShape(newShape...))
    
    // Use ElemCopy with permuted strides
    // Need to compute mapping: srcStrides[perm[dims[i]]] -> dstStrides[i]
    permutedSrcStrides := make([]int, rank)
    for i, d := range dims {
        permutedSrcStrides[i] = srcStrides[d]
    }
    
    // Use fp32.ElemCopy
    srcData := types.GetTensorData[[]float32](&t)
    dstData := types.GetTensorData[[]float32](&result)
    
    fp32.ElemCopy(
        dstData,
        srcData,
        newShape.ToSlice(),
        dstStrides,
        permutedSrcStrides,
    )
    
    return result
}

// Convenience method for common transpose cases
func (t Tensor) Transpose(dims ...int) types.Tensor {
    if len(dims) == 0 {
        // Default: transpose last two dimensions
        rank := t.Shape().Rank()
        if rank < 2 {
            panic("Transpose: tensor must have at least 2 dimensions")
        }
        dims = []int{}
        for i := 0; i < rank-2; i++ {
            dims = append(dims, i)
        }
        dims = append(dims, rank-1, rank-2) // Swap last two
    }
    
    return t.Permute(dims)
}
```

### Example Usage

```go
// 4D tensor: [outChannels, inChannels, kernelH, kernelW]
// Need: [inChannels, outChannels, kernelH, kernelW]
kernelTransposed := kernel.Permute([]int{1, 0, 2, 3})

// Or using Transpose (swaps last two dims by default)
// For 2D: [M, N] -> [N, M]
matrixTransposed := matrix.Transpose()
```

## Conclusion

**Operations that can use existing fp32 primitives** (4 out of 8):
- Transpose 4D+ (ElemCopy)
- Conv1DKernelGrad (Im2Col + GEMM)
- AvgPool2DBackward (Conv2DTransposed + Scale)
- Unpad (ElemCopy)

**Operations needing new fp32 primitives** (4 out of 8):
- MaxPool2DWithIndices (extend existing)
- MaxPool2DBackward (new)
- ScatterAdd (new)
- Fill (new, optional)

**Recommendation**: 
1. Start with operations that use existing primitives (Phase 1)
2. Implement new primitives for MaxPool2D backward (Phase 2)
3. Add ScatterAdd and Fill as general-purpose operations (Phase 3)

