# New FP32 Primitives Implementation

## Summary

This document describes the new FP32 primitive operations implemented to support eliminating At/SetAt/Elements usage in neural network layers.

## Implemented Primitives

### 1. MaxPool2DWithIndices

**Location**: `tensor.go:495-560`

**Signature**:
```go
func MaxPool2DWithIndices(
    dst, src []float32,
    indices []int32,
    batchSize, channels, height, width int,
    kernelH, kernelW, strideH, strideW, padH, padW int,
)
```

**Description**:
- Performs 2D max pooling and stores indices of max elements
- `dst`: output `[batchSize, channels, outHeight, outWidth]`
- `indices`: output indices `[batchSize, channels, outHeight, outWidth]` as `int32` (linear indices into `src`)
- `src`: input `[batchSize, channels, height, width]`
- Indices are stored as linear indices into the flattened input tensor for efficient backward pass

**Usage**:
- Used in forward pass to store which input positions produced the maximum values
- Enables efficient backward pass via `MaxPool2DBackward`

---

### 2. MaxPool2DBackward

**Location**: `tensor.go:562-659`

**Signature**:
```go
func MaxPool2DBackward(
    gradInput, gradOutput []float32,
    indices []int32,
    src []float32,
    batchSize, channels, inHeight, inWidth int,
    outHeight, outWidth int,
    kernelH, kernelW, strideH, strideW, padH, padW int,
)
```

**Description**:
- Performs backward pass for 2D max pooling
- `gradInput`: output gradient `[batchSize, channels, inHeight, inWidth]` (accumulated, should be zero-initialized)
- `gradOutput`: input gradient `[batchSize, channels, outHeight, outWidth]`
- `indices`: indices from forward pass `[batchSize, channels, outHeight, outWidth]` as `int32`
- `src`: original input `[batchSize, channels, inHeight, inWidth]` (used to resolve ties when multiple positions have same max)
- Routes gradients to input positions that produced the maximum value during forward pass
- If multiple positions had the same max value, the gradient is divided equally among them

**Algorithm**:
1. For each output position, get the gradient value
2. Use stored index to find the max value from original input
3. Count how many input positions had the same max value (for tie-breaking)
4. Route gradient equally to all positions that had the max value

**Usage**:
- Replaces element-wise loops in `MaxPool2D.Backward()` layer implementation
- Eliminates ~67 lines of nested loops with At/SetAt calls

---

### 3. AvgPool2DBackward

**Location**: `tensor.go:721-779`

**Signature**:
```go
func AvgPool2DBackward(
    gradInput, gradOutput []float32,
    batchSize, channels, inHeight, inWidth int,
    outHeight, outWidth int,
    kernelH, kernelW, strideH, strideW, padH, padW int,
)
```

**Description**:
- Performs backward pass for 2D average pooling
- `gradInput`: output gradient `[batchSize, channels, inHeight, inWidth]` (accumulated, should be zero-initialized)
- `gradOutput`: input gradient `[batchSize, channels, outHeight, outWidth]`
- Routes gradient equally to all input positions in each pooling window, divided by kernel area

**Algorithm**:
1. Compute kernel area: `kernelH * kernelW`
2. For each output position, get the gradient value
3. Divide gradient by kernel area: `gradPerPosition = gradVal / kernelArea`
4. Distribute gradient equally to all valid positions in the corresponding input window

**Usage**:
- Enables `AvgPool2D.Backward()` implementation (currently not implemented)
- Can also be used with `Conv2DTransposed` + scaling as an alternative

---

### 4. ScatterAdd

**Location**: `tensor.go:1077-1123`

**Signature**:
```go
func ScatterAdd(
    dst []float32,
    index []int32,
    value []float32,
    batchSize, channels, inHeight, inWidth int,
    outHeight, outWidth int,
)
```

**Description**:
- Adds values to destination tensor at positions specified by indices
- `dst`: destination tensor `[batchSize, channels, inHeight, inWidth]` (modified in-place, should be zero-initialized)
- `index`: indices tensor `[batchSize, channels, outHeight, outWidth]` as `int32` (linear indices into `dst`)
- `value`: values to add `[batchSize, channels, outHeight, outWidth]`
- For each position in `index`, adds the corresponding value from `value` to `dst[index[i]]`
- This is a general scatter operation useful for gradient routing in backpropagation

**Algorithm**:
1. Iterate over all output positions
2. Get index and value for each position
3. Add value to `dst[index]` if index is valid

**Usage**:
- General-purpose operation for sparse gradient updates
- Alternative to `MaxPool2DBackward` if indices are stored differently
- Useful for other gradient routing patterns beyond max pooling

---

### 5. Fill

**Location**: `tensor.go:1125-1144`

**Signature**:
```go
func Fill(
    dst []float32,
    value float32,
    num, stride int,
)
```

**Description**:
- Fills a tensor with a constant value
- `dst`: destination tensor (modified in-place)
- `value`: value to fill
- `num`: number of elements to fill
- `stride`: access stride (for non-contiguous tensors)

**Algorithm**:
1. Iterate over `num` elements with given `stride`
2. Set each element to `value`

**Usage**:
- Replaces `Elements()` iterator usage in `Fill` layer
- Can be used for zero-initialization, one-initialization, or any constant fill
- Simple but useful utility operation

---

## Implementation Details

### Indices Storage

- **Data Type**: `int32` for indices (sufficient for tensor sizes up to ~2 billion elements)
- **Format**: Linear indices into flattened tensor (row-major order)
- **Benefits**: 
  - Efficient lookup during backward pass
  - No need to reconstruct multi-dimensional indices
  - Compatible with existing tensor layout

### Tie-Breaking in MaxPool2DBackward

- When multiple positions have the same max value, gradients are divided equally
- Uses epsilon-based floating point comparison (`1e-6`) to handle floating point precision
- Matches behavior of PyTorch and TensorFlow

### Performance Considerations

- All operations use nested loops for simplicity and clarity
- No memory allocations in hot paths (except for temporary buffers in some operations)
- Operations are designed to be called from tensor layer implementations, not directly from user code

## Integration with Tensor Layer

These primitives will be called from:
- `MaxPool2D.Forward()` - calls `MaxPool2DWithIndices`
- `MaxPool2D.Backward()` - calls `MaxPool2DBackward`
- `AvgPool2D.Backward()` - calls `AvgPool2DBackward`
- General gradient routing - uses `ScatterAdd`
- Fill operations - uses `Fill`

## Testing Recommendations

1. **MaxPool2DWithIndices**: Test that indices correctly point to max values
2. **MaxPool2DBackward**: Test gradient routing with and without ties
3. **AvgPool2DBackward**: Test that gradients are correctly distributed
4. **ScatterAdd**: Test with various index patterns and verify accumulation
5. **Fill**: Test with different strides and values

## Next Steps

1. Integrate these primitives into tensor layer implementations
2. Update `OPS.md` documentation to include new primitives
3. Add unit tests for each primitive
4. Benchmark performance compared to element-wise access

