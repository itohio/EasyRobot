# Tensor Interface Improvements for Learning/Training Efficiency

This document proposes improvements to the Tensor interface (`pkg/core/math/tensor/types/tensor.go`) specifically to make learning/training operations more efficient. These improvements address common patterns in optimizer implementations, gradient computations, and training loops.

## Table of Contents
1. [In-Place Scalar Operations](#in-place-scalar-operations)
2. [Scratch Tensor Operations](#scratch-tensor-operations)
3. [Shape Access Optimization](#shape-access-optimization)
4. [Gradient-Specific Operations](#gradient-specific-operations)
5. [Batch Operations](#batch-operations)
6. [Memory Pool Integration](#memory-pool-integration)

---

## 1. In-Place Scalar Operations

### Problem
Current Tensor interface requires cloning tensors to perform scalar operations like adding a constant or scaling, then using element-wise operations. This leads to unnecessary allocations.

**Current Pattern (Inefficient)**:
```go
// Adding epsilon in Adam optimizer
epsilonTensor := sqrtVHat.Clone()
for elem := range epsilonTensor.Elements() {
    elem.Set(a.epsilon)
}
sqrtVHat = sqrtVHat.Add(epsilonTensor)  // Creates another tensor
```

### Proposed Solution

Add scalar operation methods to Tensor interface:

```go
// AddScalar adds a scalar value to all elements in-place: t[i] = t[i] + scalar
// Returns self for method chaining.
AddScalar(scalar float64) Tensor

// SubScalar subtracts a scalar value from all elements in-place: t[i] = t[i] - scalar
// Returns self for method chaining.
SubScalar(scalar float64) Tensor

// MulScalar multiplies all elements by a scalar in-place: t[i] = t[i] * scalar
// Returns self for method chaining.
MulScalar(scalar float64) Tensor

// DivScalar divides all elements by a scalar in-place: t[i] = t[i] / scalar
// Returns self for method chaining.
DivScalar(scalar float64) Tensor
```

**Improved Usage**:
```go
// Adding epsilon in Adam optimizer
sqrtVHat = sqrtVHat.AddScalar(a.epsilon)  // No allocation, direct operation
```

**Benefits**:
- Eliminates tensor allocations for scalar operations
- Reduces memory bandwidth (no intermediate tensors)
- Simpler, more readable code
- Better cache locality (single tensor traversal)

**Implementation Notes**:
- These should use optimized BLAS-like primitives from `pkg/core/math/primitive`
- Can be implemented as `axpy`-style operations: `y = alpha * x + beta` where `alpha = 1.0`, `beta = scalar`

---

## 2. Scratch Tensor Operations

### Problem
Many training operations require temporary tensors for intermediate computations. Currently, this requires cloning or creating new tensors, leading to frequent allocations.

**Current Pattern (Inefficient)**:
```go
// In Adam optimizer
scaledGrad1 := param.Grad.Clone()  // Allocation 1
scaledGrad1 = scaledGrad1.Scale(1 - a.beta1)
state.m = state.m.Add(scaledGrad1)

gradSquared := param.Grad.Clone()  // Allocation 2
gradSquared = gradSquared.Mul(param.Grad)
scaledGrad2 := gradSquared.Clone()  // Allocation 3
scaledGrad2 = scaledGrad2.Scale(1 - a.beta2)
state.v = state.v.Add(scaledGrad2)
```

### Proposed Solution

Add "To" methods that write results to a destination tensor:

```go
// ScaleTo scales the tensor by a scalar and writes result to dst: dst = scalar * t
// If dst is nil, creates a new tensor. If dst is provided, uses it (must match shape).
// Returns the destination tensor.
ScaleTo(scalar float64, dst Tensor) Tensor

// AddScalarTo adds a scalar to the tensor and writes result to dst: dst = t + scalar
// If dst is nil, creates a new tensor. If dst is provided, uses it (must match shape).
// Returns the destination tensor.
AddScalarTo(scalar float64, dst Tensor) Tensor

// MulTo computes element-wise multiplication and writes result to dst: dst = t * other
// If dst is nil, creates a new tensor. If dst is provided, uses it (must match shape).
// Returns the destination tensor.
MulTo(other Tensor, dst Tensor) Tensor

// AddTo computes element-wise addition and writes result to dst: dst = t + other
// Already exists, but document it better for training use cases
AddTo(other Tensor, dst Tensor) Tensor
```

**Improved Usage**:
```go
// Pre-allocate scratch tensors once
if scratch.scaledGrad1 == nil {
    scratch.scaledGrad1 = tensor.New(tensor.DTFP32, param.Data.Shape())
    scratch.scaledGrad2 = tensor.New(tensor.DTFP32, param.Data.Shape())
    scratch.gradSquared = tensor.New(tensor.DTFP32, param.Data.Shape())
}

// Reuse scratch tensors (zero allocations)
param.Grad.ScaleTo(1 - a.beta1, scratch.scaledGrad1)
state.m.Add(scratch.scaledGrad1)

param.Grad.MulTo(param.Grad, scratch.gradSquared)
scratch.gradSquared.ScaleTo(1 - a.beta2, scratch.scaledGrad2)
state.v.Add(scratch.scaledGrad2)
```

**Benefits**:
- Enables scratch tensor reuse patterns
- Eliminates allocations in hot paths
- Better performance for large tensors
- Clearer intent (destination explicitly specified)

**Implementation Notes**:
- Should validate that dst shape matches source shape
- Should support nil dst for backward compatibility
- Can use existing `AddTo` pattern as template

---

## 3. Shape Access Optimization

### Problem
`Shape()` method returns a copy of the shape slice. While this is safe, frequent calls create unnecessary allocations, especially in validation-heavy code paths.

**Current Pattern**:
```go
// In optimizer validation
if len(param.Data.Shape()) == 0 {  // Shape() call 1
    return fmt.Errorf("empty data")
}
if len(param.Grad.Shape()) == 0 {  // Shape() call 2
    return nil
}
if !param.Data.Shape().Equal(param.Grad.Shape()) {  // Shape() calls 3 & 4
    return fmt.Errorf("shape mismatch: %v vs %v", 
        param.Data.Shape(), param.Grad.Shape())  // Shape() calls 5 & 6
}
```

### Proposed Solution

Add shape accessor methods that avoid copies where possible:

```go
// ShapeLen returns the length of the shape (rank) without allocating
// Equivalent to len(t.Shape()) but more efficient
ShapeLen() int

// ShapeAt returns the dimension at index i without allocating the full shape
// Panics if i is out of range
ShapeAt(i int) int

// ShapeEqual checks if this tensor's shape equals another tensor's shape
// More efficient than t.Shape().Equal(other.Shape()) as it avoids shape copies
ShapeEqual(other Tensor) bool

// ShapeSize returns the total number of elements (product of dimensions)
// Equivalent to t.Shape().Size() but may be cached
ShapeSize() int
```

**Improved Usage**:
```go
// More efficient validation
if param.Data.ShapeSize() == 0 {
    return fmt.Errorf("empty data")
}
if param.Grad.ShapeSize() == 0 {
    return nil
}
if !param.Data.ShapeEqual(param.Grad) {
    return fmt.Errorf("shape mismatch")
}
```

**Alternative**: Keep `Shape()` but document that it returns a slice reference (not a copy), and that callers should not modify it. However, this breaks encapsulation.

**Benefits**:
- Eliminates shape slice allocations in validation paths
- Faster shape comparisons
- Better performance for frequently validated tensors
- Clearer intent (shape comparison vs shape access)

**Implementation Notes**:
- `ShapeSize()` might already be cached in tensor implementation
- `ShapeEqual()` should compare shapes directly without creating copies
- Consider adding `ShapeRef()` that returns shape without copying (if safe)

---

## 4. Gradient-Specific Operations

### Problem
Training operations often involve gradient accumulation, scaling, and clipping. Current interface requires multiple operations and allocations.

**Current Pattern**:
```go
// Gradient accumulation
param.Grad = param.Grad.Add(batchGrad)  // May allocate if not optimized

// Gradient clipping
gradNorm := param.Grad.Norm()
if gradNorm > maxNorm {
    scale := maxNorm / gradNorm
    param.Grad = param.Grad.Scale(scale)  // May allocate
}
```

### Proposed Solution

Add gradient-specific convenience methods:

```go
// Accumulate adds another tensor to this tensor in-place: t = t + other
// Optimized for gradient accumulation patterns
// Returns self for method chaining.
Accumulate(other Tensor) Tensor

// ScaleAccumulate scales other tensor and adds to this: t = t + alpha * other
// Common pattern in optimizers: param = param + lr * grad
// Returns self for method chaining.
ScaleAccumulate(alpha float64, other Tensor) Tensor

// ClipNorm clips the tensor to have maximum norm maxNorm
// If norm > maxNorm, scales tensor by maxNorm / norm
// Returns self for method chaining.
ClipNorm(maxNorm float64) Tensor

// Normalize normalizes the tensor to unit norm (L2)
// Returns the original norm before normalization
Normalize() float64
```

**Improved Usage**:
```go
// Gradient accumulation (no allocation)
param.Grad.Accumulate(batchGrad)

// Gradient clipping (no allocation)
param.Grad.ClipNorm(1.0)

// Optimizer update pattern
param.Data.ScaleAccumulate(-lr, param.Grad)  // param = param - lr * grad
```

**Benefits**:
- Optimized for common training patterns
- Reduces allocations in gradient operations
- More readable training code
- Can use optimized BLAS primitives internally

**Implementation Notes**:
- `ScaleAccumulate` is essentially `axpy`: `y = y + alpha * x`
- `ClipNorm` requires computing norm first, then scaling if needed
- `Normalize` computes norm and scales in one pass

---

## 5. Batch Operations

### Problem
Training often involves batch processing where the same operation is applied to multiple tensors. Current interface requires looping.

**Current Pattern**:
```go
// Zero gradients for all parameters
for _, param := range params {
    param.Grad.Zero()  // Assuming Zero() exists, or use Fill(0)
}
```

### Proposed Solution

Add batch operation helpers (these might be in a separate utility package):

```go
// BatchZero zeros multiple tensors efficiently
// Can use parallel processing for large batches
BatchZero(tensors ...Tensor)

// BatchScale scales multiple tensors by the same scalar
BatchScale(scalar float64, tensors ...Tensor)

// BatchAccumulate accumulates gradients from multiple sources
// result[i] = result[i] + source[i] for all i
BatchAccumulate(results []Tensor, sources []Tensor) error
```

**Note**: These might be better as package-level functions rather than methods:

```go
package tensor

// BatchZero zeros multiple tensors
func BatchZero(tensors ...Tensor) {
    for _, t := range tensors {
        if t != nil && !t.Empty() {
            t.Fill(0)
        }
    }
}

// BatchScale scales multiple tensors by the same scalar
func BatchScale(scalar float64, tensors ...Tensor) {
    for _, t := range tensors {
        if t != nil && !t.Empty() {
            t.Scale(scalar)
        }
    }
}
```

**Benefits**:
- Can be parallelized for large batches
- Clearer intent for batch operations
- Potential for optimization (SIMD, parallel processing)

**Implementation Notes**:
- Consider parallelization for large batches
- Should handle nil/empty tensors gracefully
- May want to add context support for cancellation

---

## 6. Memory Pool Integration

### Problem
Frequent tensor allocations in training loops cause GC pressure. A memory pool could reuse tensor buffers.

### Proposed Solution

Add tensor pool interface and integration:

```go
// TensorPool manages a pool of reusable tensors
type TensorPool interface {
    // Get returns a tensor from the pool with the given shape and dtype
    // If no tensor is available, creates a new one
    Get(dtype DataType, shape Shape) Tensor
    
    // Put returns a tensor to the pool for reuse
    // The tensor should be zeroed or its contents will be undefined
    Put(t Tensor)
    
    // Clear clears all tensors from the pool
    Clear()
}

// NewTensorPool creates a new tensor pool
func NewTensorPool() TensorPool {
    return &tensorPool{
        pools: make(map[poolKey][]Tensor),
    }
}

// Tensor creation with pool support
func NewFromPool(pool TensorPool, dtype DataType, shape Shape) Tensor {
    if pool != nil {
        return pool.Get(dtype, shape)
    }
    return New(dtype, shape)
}
```

**Usage in Optimizers**:
```go
type Adam struct {
    // ... existing fields ...
    pool TensorPool
}

func (a *Adam) Update(param types.Parameter) error {
    // Get scratch tensors from pool
    if scratch.scaledGrad1 == nil {
        scratch.scaledGrad1 = tensor.NewFromPool(a.pool, tensor.DTFP32, param.Data.Shape())
        // ...
    }
    
    // Use scratch tensors...
    
    // Optionally return to pool (or reuse across iterations)
}
```

**Benefits**:
- Reduces GC pressure
- Faster tensor allocation for common shapes
- Better memory locality

**Implementation Notes**:
- Pool should key tensors by (dtype, shape) tuple
- Consider size limits to prevent unbounded growth
- May want separate pools for different tensor sizes
- Thread-safety considerations for concurrent training

---

## Priority Recommendations

### High Priority (Immediate Impact)
1. **AddScalar, SubScalar, MulScalar, DivScalar** - Eliminates allocations in optimizers
2. **ScaleTo, AddScalarTo** - Enables scratch tensor reuse
3. **ShapeEqual, ShapeSize** - Faster validation in hot paths

### Medium Priority (Performance Improvement)
4. **Accumulate, ScaleAccumulate** - Optimized gradient operations
5. **ClipNorm** - Common training operation

### Low Priority (Nice to Have)
6. **Batch operations** - Can be implemented as package functions
7. **TensorPool** - Requires more design work for integration

---

## Migration Strategy

1. **Phase 1**: Add new methods to Tensor interface with default implementations that call existing methods (for backward compatibility)
2. **Phase 2**: Optimize implementations in `eager_tensor` package
3. **Phase 3**: Update optimizers to use new methods
4. **Phase 4**: Deprecate old patterns (if any)

---

## Example: Optimized Adam Update

**Before (Current)**:
```go
func (a *Adam) Update(param types.Parameter) error {
    // ... validation ...
    
    // 8 allocations per update
    scaledGrad1 := param.Grad.Clone()
    scaledGrad1 = scaledGrad1.Scale(1 - a.beta1)
    state.m = state.m.Add(scaledGrad1)
    
    gradSquared := param.Grad.Clone()
    gradSquared = gradSquared.Mul(param.Grad)
    scaledGrad2 := gradSquared.Clone()
    scaledGrad2 = scaledGrad2.Scale(1 - a.beta2)
    state.v = state.v.Add(scaledGrad2)
    
    mHat := state.m.Clone()
    mHat = mHat.Scale(1.0 / biasCorrection1)
    vHat := state.v.Clone()
    vHat = vHat.Scale(1.0 / biasCorrection2)
    
    sqrtVHat := vHat.Clone()
    sqrtVHat = sqrtVHat.Sqrt(nil)
    
    epsilonTensor := sqrtVHat.Clone()
    for elem := range epsilonTensor.Elements() {
        elem.Set(a.epsilon)
    }
    sqrtVHat = sqrtVHat.Add(epsilonTensor)
    
    update := mHat.Clone()
    update = update.Div(sqrtVHat)
    update = update.Scale(a.lr)
    param.Data.Sub(update)
    
    return nil
}
```

**After (With Improvements)**:
```go
func (a *Adam) Update(param types.Parameter) error {
    // Fast validation
    if param.Data.ShapeSize() == 0 {
        return fmt.Errorf("Adam.Update: empty parameter data")
    }
    if param.Grad.ShapeSize() == 0 || !param.Data.ShapeEqual(param.Grad) {
        return nil
    }
    
    // ... get/create state and scratch ...
    
    // Zero allocations - reuse scratch tensors
    param.Grad.ScaleTo(1 - a.beta1, scratch.scaledGrad1)
    state.m.Add(scratch.scaledGrad1)
    
    param.Grad.MulTo(param.Grad, scratch.gradSquared)
    scratch.gradSquared.ScaleTo(1 - a.beta2, scratch.scaledGrad2)
    state.v.Add(scratch.scaledGrad2)
    
    state.m.ScaleTo(1.0 / biasCorrection1, scratch.mHat)
    state.v.ScaleTo(1.0 / biasCorrection2, scratch.vHat)
    
    scratch.vHat.SqrtTo(scratch.sqrtVHat)  // Need SqrtTo method
    scratch.sqrtVHat.AddScalar(a.epsilon)  // New method
    
    scratch.mHat.DivTo(scratch.sqrtVHat, scratch.update)
    scratch.update.ScaleTo(a.lr, scratch.update)
    param.Data.Sub(scratch.update)
    
    return nil
}
```

**Improvements**:
- **8 allocations → 0 allocations** (after first update)
- **Faster validation** using ShapeSize() and ShapeEqual()
- **Clearer code** with explicit scratch tensor reuse
- **Better performance** with optimized operations

---

## Conclusion

These Tensor interface improvements will significantly improve training performance by:
1. Reducing allocations in hot paths (optimizers)
2. Enabling scratch tensor reuse patterns
3. Providing optimized operations for common training patterns
4. Improving validation performance

The improvements are backward compatible and can be implemented incrementally.

