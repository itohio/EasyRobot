# Tensor Interface Recommendations for Layers Efficiency

This document provides recommendations for improving the `types.Tensor` interface to make layer implementations more efficient. These recommendations are based on patterns observed in the layers code.

## Table of Contents

1. [In-Place Operations with Destination Tensors](#in-place-operations-with-destination-tensors)
2. [Fused Operations](#fused-operations)
3. [Shape and Dimension Utilities](#shape-and-dimension-utilities)
4. [Broadcasting Improvements](#broadcasting-improvements)
5. [Memory Management](#memory-management)
6. [Comparison Operations](#comparison-operations)
7. [Operation Variants](#operation-variants)

---

## 1. In-Place Operations with Destination Tensors

### Problem

Many operations in layers need to write results to pre-allocated tensors, but current interface often creates new tensors. For example:

```go
// Current: Creates new tensor
result := input.AvgPool2D(kernel, stride, pad)
output.Copy(result)

// Desired: Write directly to output
input.AvgPool2DTo(output, kernel, stride, pad)
```

### Recommendations

**Add `*To` variants for operations that currently create new tensors:**

1. **Pooling Operations:**
   - `MaxPool2DTo(dst Tensor, kernel, stride, pad []int) Tensor`
   - `MaxPool2DWithIndicesTo(dst Tensor, dstIndices Tensor, kernel, stride, pad []int) (Tensor, Tensor)`
   - `AvgPool2DTo(dst Tensor, kernel, stride, pad []int) Tensor`
   - `GlobalAvgPool2DTo(dst Tensor) Tensor`
   - `MaxPool2DBackwardTo(dst Tensor, gradOutput, indices Tensor, kernel, stride, pad []int) Tensor`
   - `AvgPool2DBackwardTo(dst Tensor, gradOutput Tensor, kernel, stride, pad []int) Tensor`

2. **Convolution Operations:**
   - `Conv1DTo(dst Tensor, kernel, bias Tensor, stride, pad int) Tensor`
   - `Conv2DTo(dst Tensor, kernel, bias Tensor, stride, pad []int) Tensor`
   - `Conv1DBackwardTo(dst Tensor, gradOutput, kernel Tensor, stride, pad int) Tensor`
   - `Conv2DBackwardTo(dst Tensor, gradOutput, kernel Tensor, stride, pad []int) Tensor`
   - `Conv1DKernelGradTo(dst Tensor, gradOutput, input Tensor, stride, pad int) Tensor`
   - `Conv2DKernelGradTo(dst Tensor, gradOutput, input Tensor, stride, pad []int) Tensor`

3. **Activation Operations:**
   - `ReLUTo(dst Tensor) Tensor`
   - `SigmoidTo(dst Tensor) Tensor`
   - `TanhTo(dst Tensor) Tensor`
   - `SoftmaxTo(dst Tensor, dim int) Tensor`

4. **Linear Algebra:**
   - `MatMulTo(dst Tensor, other Tensor) Tensor`
   - `MatMulTransposedTo(dst Tensor, other Tensor, transposeA, transposeB bool) Tensor`
   - `MatVecMulTransposedTo(dst Tensor, weight, vec Tensor, alpha, beta float64) Tensor`

**Benefits:**
- Eliminates intermediate tensor allocations
- Reduces memory usage
- Improves cache locality
- Allows layers to pre-allocate all tensors during Init

**Implementation Notes:**
- `dst` parameter should be the first parameter (convention)
- Return `dst` for method chaining
- Panic if `dst` shape doesn't match expected output shape
- Panic if `dst` is nil (or handle nil by creating new tensor - but this defeats the purpose)

---

## 2. Fused Operations

### Problem

Layers often perform sequences of operations that could be fused for better performance. For example:

```go
// Current: Multiple operations
term1 := ones.Clone().Sub(output)  // (1 - output)
term2 := output.Clone().Mul(term1) // output * (1 - output)
gradInput := gradOutput.Clone().Mul(term2) // gradOutput * output * (1 - output)
```

### Recommendations

**Add fused operations for common layer patterns:**

1. **Sigmoid Gradient:**
   ```go
   // Computes: gradOutput * output * (1 - output)
   SigmoidGradientTo(dst Tensor, gradOutput, output Tensor) Tensor
   ```

2. **Tanh Gradient:**
   ```go
   // Computes: gradOutput * (1 - output^2)
   TanhGradientTo(dst Tensor, gradOutput, output Tensor) Tensor
   ```

3. **ReLU Gradient:**
   ```go
   // Computes: gradOutput * (input > 0 ? 1 : 0)
   ReLUGradientTo(dst Tensor, gradOutput, input Tensor) Tensor
   // Or more efficiently:
   ReLUGradientMaskTo(dst Tensor, gradOutput Tensor, mask Tensor) Tensor
   ```

4. **Softmax Gradient:**
   ```go
   // Computes: output * (gradOutput - sum(gradOutput * output))
   SoftmaxGradientTo(dst Tensor, gradOutput, output Tensor, dim int) Tensor
   ```

5. **Linear with Bias:**
   ```go
   // Computes: input @ weight + bias (with broadcasting)
   LinearTo(dst Tensor, input, weight, bias Tensor) Tensor
   ```

6. **Bias Addition:**
   ```go
   // Adds bias with automatic broadcasting
   AddBiasTo(dst Tensor, bias Tensor) Tensor
   ```

**Benefits:**
- Reduces intermediate tensor allocations
- Better cache locality
- Potentially faster (single kernel vs multiple)

**Implementation Notes:**
- These should be implemented using primitives when possible
- Can be implemented as convenience methods that call existing operations internally
- Priority: Implement as primitives if they're used frequently enough

---

## 3. Shape and Dimension Utilities

### Problem

Layers frequently compute sizes and validate shapes. Current interface requires manual calculations:

```go
outputSize := 1
for _, dim := range outputShape {
    outputSize *= dim
}
```

### Recommendations

**Add shape utilities:**

1. **Shape Size:**
   ```go
   // Shape interface should have:
   type Shape interface {
       // ... existing methods ...
       Size() int  // Returns product of all dimensions
   }
   ```

2. **Shape Validation:**
   ```go
   // On Tensor interface:
   CompatibleShape(other Tensor) bool  // Checks if shapes are compatible for operations
   BroadcastableTo(targetShape Shape) bool  // Checks if can broadcast to target
   ```

3. **Dimension Helpers:**
   ```go
   // On Tensor interface:
   DimSize(dim int) int  // Returns size of dimension (avoid Shape()[dim])
   LeadingDimSize() int  // Returns size of first dimension (batch size)
   LastDimSize() int     // Returns size of last dimension
   ```

4. **Shape Comparison:**
   ```go
   // On Shape interface:
   Equal(other Shape) bool  // Already exists, but verify it's efficient
   CompatibleForOp(other Shape, op string) bool  // Checks compatibility for specific operation
   ```

**Benefits:**
- Reduces redundant calculations
- Makes code more readable
- Centralizes shape logic

---

## 4. Broadcasting Improvements

### Problem

Broadcasting is used frequently but creates intermediate tensors:

```go
biasBroadcast := bias.Reshape(tensor.NewShape(1, outFeatures))
biasFull, err := biasBroadcast.BroadcastTo(output.Shape())
output.Add(biasFull)
```

### Recommendations

**Add broadcasting variants:**

1. **Broadcast and Operate:**
   ```go
   // Adds bias with automatic broadcasting (no intermediate tensor)
   AddBroadcastTo(dst Tensor, other Tensor) Tensor
   MulBroadcastTo(dst Tensor, other Tensor) Tensor
   // etc. for other operations
   ```

2. **In-Place Broadcast:**
   ```go
   // Broadcasts other to match self, then operates in-place
   AddBroadcast(other Tensor) Tensor
   MulBroadcast(other Tensor) Tensor
   ```

3. **Broadcast Check:**
   ```go
   // Returns true if other can be broadcast to self's shape
   CanBroadcastFrom(other Tensor) bool
   ```

**Alternative Approach:**

Add a `BroadcastTo` variant that accepts a destination:
```go
BroadcastToTo(dst Tensor, targetShape Shape) Tensor
```

But this is less useful since broadcasting usually happens as part of an operation.

**Benefits:**
- Eliminates intermediate broadcast tensors
- More efficient for common patterns
- Cleaner API

---

## 5. Memory Management

### Problem

Layers create many temporary tensors. Better memory management could help:

### Recommendations

**Add memory management utilities:**

1. **Tensor Pool:**
   ```go
   // Interface for tensor pooling (optional, advanced optimization)
   type TensorPool interface {
       Get(dtype DataType, shape Shape) Tensor
       Put(t Tensor)
   }
   ```

2. **Reuse Tensors:**
   ```go
   // Method to check if tensor can be reused (same shape, dtype)
   CanReuseAs(shape Shape, dtype DataType) bool
   ```

3. **Zero-Copy Operations:**
   ```go
   // Document which operations are zero-copy (views)
   // Reshape, Slice, Transpose might be views
   IsView() bool  // Returns true if tensor is a view of another
   ```

**Note:** These are advanced optimizations. Start with `*To` variants first.

---

## 6. Comparison Operations

### Problem

Current comparison operations might create new tensors. For ReLU gradient, we do:

```go
zeros := tensor.ZerosLike(input)
mask := input.GreaterThan(zeros)
```

### Recommendations

**Add scalar comparison operations:**

1. **Scalar Comparisons:**
   ```go
   // Compare with scalar (no need to create zero tensor)
   GreaterThanScalar(scalar float64) Tensor  // Returns mask tensor
   LessThanScalar(scalar float64) Tensor
   EqualScalar(scalar float64) Tensor
   GreaterEqualScalar(scalar float64) Tensor
   LessEqualScalar(scalar float64) Tensor
   ```

2. **In-Place Comparison Masks:**
   ```go
   // Write comparison result to existing tensor (for reuse)
   GreaterThanScalarTo(dst Tensor, scalar float64) Tensor
   ```

**Benefits:**
- Eliminates need for `ZerosLike` in ReLU gradient
- More efficient
- Cleaner code

---

## 7. Operation Variants

### Problem

Some operations need variants for different use cases.

### Recommendations

**Add operation variants:**

1. **1D/2D Compatibility:**
   ```go
   // Operations that handle both 1D and 2D inputs
   // Current: Layers reshape 1D to 2D, operate, reshape back
   // Better: Operations handle both natively
   
   MatMul1DTo(dst Tensor, weight Tensor, vec Tensor) Tensor  // For [N] @ [M,N] -> [M]
   MatMul2DTo(dst Tensor, input Tensor, weight Tensor) Tensor // For [B,N] @ [N,M] -> [B,M]
   ```

2. **Batch-Aware Operations:**
   ```go
   // Operations that automatically handle batch dimension
   // Many operations already do this, but make it explicit
   ```

3. **Strided Operations:**
   ```go
   // For operations that work with views/slices
   // Might be lower-level, but useful for advanced optimizations
   ```

---

## Implementation Priority

### Phase 1: High Impact, Easy to Implement

1. **Add `*To` variants** for pooling operations (MaxPool2D, AvgPool2D, GlobalAvgPool2D)
2. **Add `*To` variants** for convolution operations (Conv1D, Conv2D)
3. **Add scalar comparison operations** (GreaterThanScalar, etc.)
4. **Add `Size()` to Shape interface**

### Phase 2: High Impact, Moderate Effort

1. **Add fused gradient operations** (SigmoidGradientTo, TanhGradientTo, ReLUGradientTo)
2. **Add `*To` variants** for activations (ReLUTo, SigmoidTo, TanhTo, SoftmaxTo)
3. **Add broadcasting variants** (AddBroadcastTo, MulBroadcastTo)
4. **Add LinearTo** for linear transformation with bias

### Phase 3: Advanced Optimizations

1. **Tensor pooling** (if memory pressure is an issue)
2. **Fused operations as primitives** (if performance profiling shows they're bottlenecks)
3. **Strided operations** (for advanced use cases)

---

## Backward Compatibility

All new methods should be **additions** to the interface, not modifications to existing methods. This ensures:

- Existing code continues to work
- New code can use optimized variants
- Gradual migration path

**Migration Strategy:**

1. Add new `*To` methods
2. Update layers to use new methods
3. Keep old methods for compatibility
4. Eventually deprecate old methods (if desired)

---

## Example: Before and After

### Before (Current Code):

```go
// Sigmoid.Backward
ones := tensor.OnesLike(output)
term1 := ones.Clone().Sub(output)
term2 := output.Clone().Mul(term1)
gradInput := gradOutput.Clone().Mul(term2)
```

**Issues:**
- 4 tensor allocations (ones, term1, term2, gradInput clone)
- Multiple intermediate tensors
- Unclear ownership

### After (With Recommendations):

```go
// Option 1: Using fused operation
gradInput := r.Base.Grad() // Pre-allocated
output.SigmoidGradientTo(gradInput, gradOutput)

// Option 2: Using *To variants
gradInput := r.Base.Grad() // Pre-allocated
temp := r.Base.Intermediate1() // Pre-allocated
ones := r.Base.Ones() // Pre-allocated during Init
ones.SubTo(temp, output) // temp = ones - output
output.MulTo(temp, temp) // temp = output * temp
gradOutput.MulTo(gradInput, temp) // gradInput = gradOutput * temp
```

**Benefits:**
- Zero allocations in backward pass
- All tensors pre-allocated
- Clear ownership
- Better cache locality

---

## Notes for Implementation

1. **Performance First**: These recommendations prioritize performance for layers use case
2. **Backward Compatible**: All additions, no breaking changes
3. **Gradual Adoption**: Layers can migrate gradually to new methods
4. **Test Coverage**: Each new method needs comprehensive tests
5. **Documentation**: Clear documentation on when to use which variant

---

## Related Work

- See `OPTIMIZATION_RECOMMENDATIONS.md` for layer-specific optimizations
- See `tensor/SPEC.md` for current tensor interface design
- Consider performance profiling before implementing advanced optimizations

