# Tensor API Transition Plan

## Overview

This document outlines the transition plan for improving the Tensor interface API to enable efficient scratch tensor reuse and eliminate temporary tensor allocations in training loops. The goal is to unify all operations to use a consistent pattern: `Operation(dst, ...args) self` where `dst` is the first parameter.

## Design Goals

1. **Unified API Pattern**: All operations follow `Operation(dst, ...args) Tensor`
   - `dst` is always the first parameter
   - If `dst` is `nil` or `self`, operation is in-place
   - If `dst` is provided, writes result to `dst`
   - **Return behavior**: Returns `self` if `dst` is `nil` (in-place), returns `dst` if `dst` is provided (enables chaining)
   - **No internal allocation**: Operations never allocate new tensors internally - they either use provided `dst` or operate in-place

2. **No Method Renaming**: Keep existing method names (`Add`, `Scale`, `ReLU`, etc.)

3. **No Dual Versions**: Only one version of each method (no `Add` and `AddTo`)

4. **Efficiency**: Enable scratch tensor reuse and eliminate temporary allocations

5. **Operations Creating New Tensors**: Some operations must create tensors of different data types (e.g., indices as INT tensors). These operations:
   - Reuse `dst` if provided and has correct DataType
   - Create new tensor internally if `dst` is `nil` or has wrong DataType
   - Accept only specific DataType for `dst` (e.g., INT for indices operations)
   - Return the created or reused tensor (not `self`)

## Target API Pattern

### Core Pattern

```go
// Unified pattern for standard operations
func (t Tensor) Operation(dst Tensor, ...args) Tensor {
    // If dst is nil or self, operate in-place
    if dst == nil || dst == t {
        // ... perform in-place operation on t ...
        return t  // Return self for chaining
    }
    
    // If dst is provided, write to dst
    // ... perform operation, write result to dst ...
    return dst  // Return dst for chaining
}
```

### Return Value Behavior

**Standard Operations** (same data type, same or compatible shape):
- Returns `self` if `dst == nil` (in-place operation)
- Returns `dst` if `dst != nil` (write to destination, enables chaining)

**Operations Creating Different Data Types** (e.g., indices as INT tensors):
- If `dst` is provided and has correct DataType: reuses `dst`, returns `dst`
- If `dst` is `nil` or has wrong DataType: creates new tensor with appropriate DataType, returns new tensor
- These operations **must** create a new tensor if `dst` is not suitable

### Examples

```go
// In-place operations (dst is nil or self) - returns self
result := t.Add(nil, other)        // t = t + other, returns t
result := t.Add(t, other)          // Same as above, returns t
result := t.Scale(nil, 2.0)        // t = 2.0 * t, returns t
result := t.ReLU(nil)              // t = ReLU(t), returns t

// Scratch tensor operations - returns dst (enables chaining)
scratch := tensor.New(tensor.DTFP32, t.Shape())
result := t.Add(scratch, other)    // scratch = t + other, returns scratch
result := t.Scale(scratch, 2.0)    // scratch = 2.0 * t, returns scratch
result := t.ReLU(scratch)          // scratch = ReLU(t), returns scratch

// Chain operations - in-place
t.Add(nil, other).Scale(nil, 2.0).ReLU(nil)  // All in-place, returns t

// Chain operations - with scratch tensors
scratch1 := tensor.New(tensor.DTFP32, t.Shape())
scratch2 := tensor.New(tensor.DTFP32, t.Shape())
t.Add(scratch1, other).Scale(scratch2, 2.0).ReLU(scratch1)  // Returns scratch1
```

### Operations Creating Different Data Types

```go
// MaxPool2DWithIndices - creates indices tensor (INT type)
outputDst := tensor.New(tensor.DTFP32, outputShape)
indicesDst := tensor.New(tensor.DTINT32, outputShape)  // Must be INT type

// Reuse provided tensors
outputResult, indicesResult := input.MaxPool2DWithIndices(outputDst, indicesDst, kernelSize, stride, padding)
// outputResult == outputDst (reused)
// indicesResult == indicesDst (reused)

// Create new tensors if dst is nil or wrong type
outputResult, indicesResult := input.MaxPool2DWithIndices(nil, nil, kernelSize, stride, padding)
// outputResult is new FP32 tensor (created internally)
// indicesResult is new INT tensor (created internally)

// Reuse output but create new indices
outputResult, indicesResult := input.MaxPool2DWithIndices(outputDst, nil, kernelSize, stride, padding)
// outputResult == outputDst (reused)
// indicesResult is new INT tensor (created internally)

// ArgMax - creates INT tensor for indices
indices := tensor.New(tensor.DTINT32, resultShape)
result := t.ArgMax(indices, dim)  // Reuses indices if provided and correct type
// result == indices (same tensor, reused)
// If indices is nil or wrong type, creates new INT tensor internally and returns it
```

## Method Categories and Transition

### 1. Element-wise Binary Operations

**Current**:
```go
Add(other Tensor) Tensor              // In-place
AddTo(other Tensor, dst Tensor) Tensor  // To destination
Mul(other Tensor) Tensor
MulTo(other Tensor, dst Tensor) Tensor
```

**Target**:
```go
// Unified: dst is first parameter
Add(dst Tensor, other Tensor) Tensor
Mul(dst Tensor, other Tensor) Tensor
Sub(dst Tensor, other Tensor) Tensor
Div(dst Tensor, other Tensor) Tensor
```

**Behavior**:
- If `dst == nil` or `dst == self`: operation is in-place, returns `self`
- If `dst != nil` and `dst != self`: writes result to `dst`, returns `dst` (enables chaining)

### 2. Element-wise Unary Operations

**Current**:
```go
Square(dst Tensor) Tensor    // dst is last parameter
Sqrt(dst Tensor) Tensor
Exp(dst Tensor) Tensor
Log(dst Tensor) Tensor
Pow(dst Tensor, power float64) Tensor
Abs(dst Tensor) Tensor
Sign(dst Tensor) Tensor
Cos(dst Tensor) Tensor
Sin(dst Tensor) Tensor
Negative(dst Tensor) Tensor
```

**Target**:
```go
// dst is first parameter, keep same names
Square(dst Tensor) Tensor
Sqrt(dst Tensor) Tensor
Exp(dst Tensor) Tensor
Log(dst Tensor) Tensor
Pow(dst Tensor, power float64) Tensor
Abs(dst Tensor) Tensor
Sign(dst Tensor) Tensor
Cos(dst Tensor) Tensor
Sin(dst Tensor) Tensor
Negative(dst Tensor) Tensor
```

**Note**: These already have `dst` but as last parameter. Move to first.

### 3. Scalar Operations

**Current**:
```go
Scale(scalar float64) Tensor  // In-place only
Fill(value float64) Tensor    // In-place only
```

**Target**:
```go
// Add dst parameter for scratch tensor reuse
Scale(dst Tensor, scalar float64) Tensor
Fill(dst Tensor, value float64) Tensor
RandomFill(dst Tensor, rng RNG) Tensor
FillWithCallback(dst Tensor, callback func() float64) Tensor
```

**Additional Operations** (new):
```go
// Scalar arithmetic operations
AddScalar(dst Tensor, scalar float64) Tensor  // dst = t + scalar
SubScalar(dst Tensor, scalar float64) Tensor  // dst = t - scalar
MulScalar(dst Tensor, scalar float64) Tensor  // dst = t * scalar
DivScalar(dst Tensor, scalar float64) Tensor  // dst = t / scalar
```

**RandomFill Behavior**:
- Fills tensor with random values in range [0, 1) using `rng.Float64()`
- If `dst == nil`: operates in-place on self, returns `self`
- If `dst != nil`: writes random values to `dst`, returns `dst` (enables chaining)
- Uses `rng.Float64()` for uniform random values in [0, 1)
- **Note**: `rng` parameter must not be `nil` (panics if nil)

**FillWithCallback Behavior**:
- Fills tensor by calling `callback()` for each element
- Callback is invoked once per element in the tensor (in iteration order)
- If `dst == nil`: operates in-place on self, returns `self`
- If `dst != nil`: writes callback results to `dst`, returns `dst` (enables chaining)
- **Note**: `callback` parameter must not be `nil` (panics if nil)

### 4. Scaled Operations (New)

Common patterns in training:

```go
// (1 + value) * another
AddScaledMul(dst Tensor, scalar float64, other Tensor) Tensor
// Computes: dst = (1 + scalar) * other
// If dst == nil: self = (1 + scalar) * other, returns self
// If dst != nil: dst = (1 + scalar) * other, returns dst

// (1 + value^2) * another
AddScaledSquareMul(dst Tensor, scalar float64, other Tensor) Tensor
// Computes: dst = (1 + scalar * other^2) * other
// If dst == nil: self = (1 + scalar * other^2) * other, returns self
// If dst != nil: dst = (1 + scalar * other^2) * other, returns dst
```

### 5. Activation Functions

**Current**:
```go
ReLU(dst Tensor) Tensor
Sigmoid(dst Tensor) Tensor
Tanh(dst Tensor) Tensor
Softmax(dim int, dst Tensor) Tensor
```

**Target**:
```go
// dst is first parameter
ReLU(dst Tensor) Tensor
Sigmoid(dst Tensor) Tensor
Tanh(dst Tensor) Tensor
Softmax(dst Tensor, dim int) Tensor  // Move dim after dst
```

**New Gradient Operations**:
```go
// Gradient operations that write to destination
ReLUGrad(dst Tensor, input Tensor, output Tensor) Tensor
// Computes: dst = gradOutput * (input > 0)
// If dst == nil: operates in-place on self, returns self
// If dst != nil: writes to dst, returns dst

SigmoidGrad(dst Tensor, output Tensor) Tensor
// Computes: dst = gradOutput * output * (1 - output)
// If dst == nil: operates in-place on self, returns self
// If dst != nil: writes to dst, returns dst

TanhGrad(dst Tensor, output Tensor) Tensor
// Computes: dst = gradOutput * (1 - output^2)
// If dst == nil: operates in-place on self, returns self
// If dst != nil: writes to dst, returns dst
```

### 6. Linear Algebra Operations

**Current**:
```go
MatMul(other Tensor) Tensor
MatMulTo(other Tensor, dst Tensor) Tensor
Transpose(dims ...int) Tensor
TransposeTo(dst Tensor, dims ...int) Tensor
```

**Target**:
```go
// Unified: dst is first parameter
MatMul(dst Tensor, other Tensor) Tensor
Transpose(dst Tensor, dims ...int) Tensor
Permute(dst Tensor, dims []int) Tensor
```

### 7. Convolution Operations

**Current**:
```go
Conv2D(kernel, bias Tensor, stride, padding []int) Tensor
Conv2DTo(kernel, bias Tensor, dst Tensor, stride, padding []int) Tensor
```

**Target**:
```go
// dst is first parameter
Conv2D(dst Tensor, kernel, bias Tensor, stride, padding []int) Tensor
Conv1D(dst Tensor, kernel, bias Tensor, stride, padding int) Tensor
Conv2DTransposed(dst Tensor, kernel, bias Tensor, stride, padding []int) Tensor
```

### 8. Pooling Operations

**Current**:
```go
MaxPool2D(kernelSize, stride, padding []int) Tensor
MaxPool2DWithIndices(kernelSize, stride, padding []int) (Tensor, Tensor)
AvgPool2D(kernelSize, stride, padding []int) Tensor
```

**Target**:
```go
// Standard pooling operations
MaxPool2D(dst Tensor, kernelSize, stride, padding []int) Tensor
AvgPool2D(dst Tensor, kernelSize, stride, padding []int) Tensor
GlobalAvgPool2D(dst Tensor) Tensor
AdaptiveAvgPool2D(dst Tensor, outputSize []int) Tensor

// MaxPool2DWithIndices - special case (returns two tensors)
MaxPool2DWithIndices(outputDst Tensor, indicesDst Tensor, kernelSize, stride, padding []int) (Tensor, Tensor)
```

**Behavior** (Standard Pooling):
- If `dst == nil`: creates new tensor with same DataType as source, returns new tensor
- If `dst != nil`: writes result to `dst`, returns `dst` (enables chaining)

**Behavior** (MaxPool2DWithIndices - Special Case):
- Creates two tensors: output (same DataType as input) and indices (INT DataType)
- `outputDst`: If `nil`, creates new tensor; if provided, reuses it. Returns output tensor.
- `indicesDst`: If `nil` or wrong DataType, creates new INT tensor; if provided with INT DataType, reuses it. Returns indices tensor.
- **Note**: This operation must create new tensors if `dst` parameters are not suitable

### 9. Reduction Operations

**Current**:
```go
Sum(dims ...int) Tensor      // Returns new tensor
Mean(dims ...int) Tensor     // Returns new tensor
Max(dims ...int) Tensor      // Returns new tensor
Min(dims ...int) Tensor      // Returns new tensor
ArgMax(dim int) Tensor       // Returns new tensor (INT type for indices)
```

**Target**:
```go
// dst is first parameter
Sum(dst Tensor, dims ...int) Tensor
Mean(dst Tensor, dims ...int) Tensor
Max(dst Tensor, dims ...int) Tensor
Min(dst Tensor, dims ...int) Tensor
ArgMax(dst Tensor, dim int) Tensor
```

**Behavior** (Standard Reductions: Sum, Mean, Max, Min):
- If `dst == nil`: creates new tensor with same DataType as source, returns new tensor
- If `dst != nil`: writes result to `dst`, returns `dst` (enables chaining)

**Behavior** (ArgMax - Special Case):
- ArgMax creates indices tensor (INT DataType)
- If `dst` is provided and has INT DataType: reuses `dst`, returns `dst`
- If `dst` is `nil` or has wrong DataType: creates new INT tensor, returns new tensor
- **Note**: This operation must create a new tensor if `dst` is not suitable (wrong DataType)

### 10. Comparison Operations

**Current**:
```go
Equal(other Tensor) Tensor
GreaterThan(other Tensor) Tensor
Less(other Tensor) Tensor
```

**Target**:
```go
// dst is first parameter
Equal(dst Tensor, other Tensor) Tensor
GreaterThan(dst Tensor, other Tensor) Tensor
Less(dst Tensor, other Tensor) Tensor
```

### 11. Helper Functions for Common Patterns

**Fill Operations**:
```go
// Fill with zeros: t.Fill(nil, 0.0) or scratch.Fill(scratch, 0.0)
// Fill with ones: t.Fill(nil, 1.0) or scratch.Fill(scratch, 1.0)
// Fill with random: t.RandomFill(nil, rng) or scratch.RandomFill(scratch, rng)
// Fill with callback: t.FillWithCallback(nil, func() float64 { return ... }) or scratch.FillWithCallback(scratch, callback)
// If dst == nil: operates in-place on self, returns self
// If dst != nil: writes to dst, returns dst
```

**Behavior**:
- `Fill(dst, value)` follows standard pattern: returns `self` if `dst == nil`, returns `dst` if `dst != nil`
- `RandomFill(dst, rng)` follows standard pattern: returns `self` if `dst == nil`, returns `dst` if `dst != nil`
- `FillWithCallback(dst, callback)` follows standard pattern: returns `self` if `dst == nil`, returns `dst` if `dst != nil`
- Fills tensor with uniform random values in range [0, 1) using `rng.Float64()`
- Fills tensor by calling `callback()` once per element (callback invoked for each element in iteration order)
- No need for separate `Zeros()` or `Ones()` methods - use `Fill()` with appropriate values

### 12. Operations Creating Different Data Types

**Special Cases**: Some operations must create tensors with different data types than the source tensor.

**Examples**:
- `ArgMax`: Creates indices tensor (INT DataType)
- `MaxPool2DWithIndices`: Creates indices tensor (INT DataType)
- Any operation that produces indices or integer results

**Pattern**:
```go
// Operation that creates INT tensor
func (t Tensor) OperationWithIndices(dst Tensor, ...args) Tensor {
    // If dst is provided and has correct DataType (e.g., INT), reuse it
    if dst != nil && dst.DataType() == INT {
        // ... perform operation, write to dst ...
        return dst
    }
    
    // If dst is nil or has wrong DataType, create new tensor
    // with appropriate DataType (e.g., INT for indices)
    result := tensor.New(INT, resultShape)
    // ... perform operation, write to result ...
    return result
}
```

**Key Principles**:
1. **Reuse if suitable**: If `dst` is provided with correct DataType, reuse it
2. **Create if needed**: If `dst` is `nil` or has wrong DataType, create new tensor with appropriate DataType
3. **Return what was written**: Return the tensor that contains the result (either reused `dst` or newly created tensor)
4. **Never allocate unnecessarily**: If `dst` is suitable, always reuse it

## Transition Phases

### Phase 0: Planning (Current)

**Status**: ✅ Complete

- Analyze current API
- Design target API
- Document transition plan
- Identify all affected methods

### Phase 1: Interface Update

**Goal**: Update `types.Tensor` interface with new signatures

**Tasks**:
1. Update method signatures in `types/tensor.go`
2. Update interface documentation
3. Add deprecation comments for old patterns (if any remain)
4. Ensure interface is backward compatible during transition

**Key Changes**:
- Move `dst` parameter to first position for all operations
- Update return type documentation to specify "returns self or dst"
- Add new methods (AddScalar, AddScaledMul, RandomFill, FillWithCallback, etc.)
- Add gradient methods (ReLUGrad, SigmoidGrad, etc.)

**Interface Example**:
```go
// Add adds another tensor element-wise.
// If dst is nil or self, operation is in-place: self = self + other
// If dst is provided, writes result to dst: dst = self + other
// Returns self if dst is nil, returns dst if dst is provided (enables chaining).
Add(dst Tensor, other Tensor) Tensor

// Scale multiplies the tensor by a scalar.
// If dst is nil or self, operation is in-place: self = scalar * self
// If dst is provided, writes result to dst: dst = scalar * self
// Returns self if dst is nil, returns dst if dst is provided (enables chaining).
Scale(dst Tensor, scalar float64) Tensor

// ArgMax returns indices of maximum elements along dimension.
// Creates INT tensor for indices. If dst is provided with INT DataType, reuses it.
// If dst is nil or has wrong DataType, creates new INT tensor.
// Returns the tensor containing indices (either reused dst or newly created).
ArgMax(dst Tensor, dim int) Tensor

// RandomFill fills the tensor with random values in range [0, 1) using rng.Float64().
// If dst is nil or self, operation is in-place: self = random values
// If dst is provided, writes random values to dst: dst = random values
// Returns self if dst is nil, returns dst if dst is provided (enables chaining).
// Panics if rng is nil.
RandomFill(dst Tensor, rng RNG) Tensor

// FillWithCallback fills the tensor by calling callback() for each element.
// Callback is invoked once per element in iteration order.
// If dst is nil or self, operation is in-place: self = callback() for each element
// If dst is provided, writes callback results to dst: dst = callback() for each element
// Returns self if dst is nil, returns dst if dst is provided (enables chaining).
// Panics if callback is nil.
FillWithCallback(dst Tensor, callback func() float64) Tensor
```

### Phase 2: Test Update

**Goal**: Update all unit tests to use new signatures

**Tasks**:
1. Update test files in `types/` package
2. Update test files in `eager_tensor/` package
3. Update test files in `tensor/` package (if any)
4. Ensure all tests pass with new signatures

**Test Patterns**:
```go
// Old test pattern
func TestAdd(t *testing.T) {
    a := tensor.New(tensor.DTFP32, shape)
    b := tensor.New(tensor.DTFP32, shape)
    result := a.Add(b)  // In-place
    // ...
}

// New test pattern
func TestAdd(t *testing.T) {
    a := tensor.New(tensor.DTFP32, shape)
    b := tensor.New(tensor.DTFP32, shape)
    result := a.Add(nil, b)  // In-place, explicit
    // ...
    
    // Test with scratch tensor
    scratch := tensor.New(tensor.DTFP32, shape)
    a.Add(scratch, b)  // scratch = a + b
    // ...
}
```

### Phase 3: Implementation Update

**Goal**: Update all implementations in `eager_tensor` package

**Tasks**:
1. Update method implementations to match new signatures
2. Implement new methods (AddScalar, AddScaledMul, etc.)
3. Implement gradient methods (ReLUGrad, etc.)
4. Update internal helpers to use new patterns
5. Ensure all operations correctly handle `dst` parameter

**Implementation Pattern** (Standard Operations):
```go
func (t Tensor) Add(dst Tensor, other Tensor) Tensor {
    // Determine if in-place or write to dst
    target := t
    if dst != nil && dst != t {
        target = dst
        // Validate dst shape matches t shape
        if !dst.Shape().Equal(t.Shape()) {
            panic("Add: dst shape mismatch")
        }
    }
    
    // Perform operation on target
    // ... implementation ...
    
    // Return what was written to (enables chaining)
    if dst != nil && dst != t {
        return dst  // Return dst for chaining
    }
    return t  // Return self for in-place operations
}
```

**Implementation Pattern** (Operations Creating Different Data Types):
```go
func (t Tensor) ArgMax(dst Tensor, dim int) Tensor {
    // Check if dst is suitable (INT DataType)
    if dst != nil && dst.DataType() == INT && dst.Shape().Equal(resultShape) {
        // Reuse provided dst
        // ... perform operation, write indices to dst ...
        return dst
    }
    
    // Create new INT tensor
    result := tensor.New(INT, resultShape)
    // ... perform operation, write indices to result ...
    return result
}
```

## Migration Examples

### Example 1: Simple Addition

**Before**:
```go
result := a.Add(b)  // In-place, modifies a
```

**After**:
```go
result := a.Add(nil, b)  // In-place, explicit
// Or with scratch tensor:
scratch := tensor.New(tensor.DTFP32, a.Shape())
a.Add(scratch, b)  // scratch = a + b, a unchanged
```

### Example 2: Optimizer Update (Adam)

**Before**:
```go
scaledGrad := param.Grad.Clone()
scaledGrad = scaledGrad.Scale(1 - beta)
state.m = state.m.Add(scaledGrad)
```

**After**:
```go
// Pre-allocate scratch tensors once
if scratch.scaledGrad == nil {
    scratch.scaledGrad = tensor.New(tensor.DTFP32, param.Grad.Shape())
}

// Reuse scratch tensor (zero allocations)
// Scale returns scratch.scaledGrad, Add returns state.m (in-place)
param.Grad.Scale(scratch.scaledGrad, 1 - beta)  // Returns scratch.scaledGrad
state.m.Add(nil, scratch.scaledGrad)            // Returns state.m (in-place)
```

### Example 5: ArgMax with Indices

**Before**:
```go
indices := t.ArgMax(dim)  // Creates new INT tensor
```

**After**:
```go
// Option 1: Let it create new INT tensor
indices := t.ArgMax(nil, dim)  // Creates new INT tensor, returns it

// Option 2: Reuse provided INT tensor
indices := tensor.New(tensor.DTINT32, resultShape)
result := t.ArgMax(indices, dim)  // Reuses indices, returns indices
// result == indices (same tensor)
```

### Example 3: ReLU Activation

**Before**:
```go
output := input.ReLU(nil)  // In-place, modifies input
```

**After**:
```go
// Same signature, but dst is first parameter
output := input.ReLU(nil)  // In-place
// Or with scratch tensor:
scratch := tensor.New(tensor.DTFP32, input.Shape())
input.ReLU(scratch)  // scratch = ReLU(input), input unchanged
```

### Example 4: Common Pattern (1 + value) * another

**Before**:
```go
temp1 := tensor.OnesLike(other)
temp1 = temp1.AddScalar(value)
temp2 := temp1.Mul(other)
result := temp2.Clone()
```

**After**:
```go
// Single operation
result := tensor.New(tensor.DTFP32, other.Shape())
other.AddScaledMul(result, value, other)
// result = (1 + value) * other
```

### Example 6: Random Fill

**Before**:
```go
// Manual random fill
rng := rand.New(rand.NewSource(time.Now().UnixNano()))
for indices := range t.Shape().Iterator() {
    t.SetAt(rng.Float64(), indices...)
}
```

**After**:
```go
rng := rand.New(rand.NewSource(time.Now().UnixNano()))

// In-place random fill
t.RandomFill(nil, rng)  // Fills t with random values, returns t

// Random fill to scratch tensor
scratch := tensor.New(tensor.DTFP32, t.Shape())
t.RandomFill(scratch, rng)  // Fills scratch with random values, returns scratch
```

### Example 7: Fill With Callback

**Before**:
```go
// Manual fill with callback
counter := 0.0
for indices := range t.Shape().Iterator() {
    t.SetAt(counter, indices...)
    counter += 1.0
}

// Or with custom computation
rng := rand.New(rand.NewSource(time.Now().UnixNano()))
for indices := range t.Shape().Iterator() {
    val := rng.Float64() * 2.0 - 1.0  // Range [-1, 1)
    t.SetAt(val, indices...)
}
```

**After**:
```go
// Fill with simple counter
counter := 0.0
callback := func() float64 {
    val := counter
    counter += 1.0
    return val
}
t.FillWithCallback(nil, callback)  // Fills t with 0, 1, 2, ..., returns t

// Fill with random values in custom range
rng := rand.New(rand.NewSource(time.Now().UnixNano()))
randomCallback := func() float64 {
    return rng.Float64()*2.0 - 1.0  // Range [-1, 1)
}
scratch := tensor.New(tensor.DTFP32, t.Shape())
t.FillWithCallback(scratch, randomCallback)  // Fills scratch, returns scratch
```

## Breaking Changes

### Method Signature Changes

All methods that currently don't have `dst` parameter will have it added as first parameter:

- `Add(other)` → `Add(dst, other)`
- `Scale(scalar)` → `Scale(dst, scalar)`
- `Fill(value)` → `Fill(dst, value)`
- `MatMul(other)` → `MatMul(dst, other)`
- `Sum(dims...)` → `Sum(dst, dims...)`
- etc.

### Return Type Behavior

**Standard Operations**:
- If `dst == nil`: returns `self` (in-place operation)
- If `dst != nil`: returns `dst` (enables chaining: `t.Add(scratch, other).Scale(scratch, 2.0)`)

**Operations Creating Different Data Types**:
- If `dst` is provided and suitable: reuses `dst`, returns `dst`
- If `dst` is `nil` or unsuitable: creates new tensor with appropriate DataType, returns new tensor
- These operations **must** create a new tensor if `dst` is not suitable (cannot operate in-place due to DataType mismatch)

**Migration Note**:
- Callers using `dst` parameter should use the returned value for chaining
- Callers using in-place operations (`dst == nil`) will receive `self` for chaining
- This enables method chaining in both scenarios

### Migration Path

For callers:
1. Add `nil` as first parameter for in-place operations
2. Use scratch tensors for non-in-place operations
3. Update code to use `dst` parameter instead of return value when `dst != nil`

## Testing Strategy

### Unit Tests

- Test in-place operations (`dst == nil`)
- Test scratch tensor operations (`dst != nil` and `dst != self`)
- Test error cases (shape mismatch, etc.)
- Test method chaining

### Integration Tests

- Test optimizer updates with scratch tensors
- Test activation functions with scratch tensors
- Test backward pass operations with scratch tensors
- Verify zero allocations in hot paths

### Performance Tests

- Measure allocation reduction in training loops
- Compare performance before/after transition
- Verify scratch tensor reuse works correctly

## Documentation Updates

### SPEC.md Updates

1. Update all method signatures
2. Document new unified pattern
3. Add migration examples
4. Update usage examples

### Code Comments

1. Update interface docstrings
2. Add examples for common patterns
3. Document `dst` parameter behavior clearly

## Implementation Checklist

### Phase 1: Interface
- [ ] Update `types/tensor.go` interface
- [ ] Add new methods (AddScalar, AddScaledMul, RandomFill, FillWithCallback, etc.)
- [ ] Add gradient methods (ReLUGrad, etc.)
- [ ] Update interface documentation

### Phase 2: Tests
- [ ] Update `types/tensor_test.go`
- [ ] Update `eager_tensor/*_test.go` files
- [ ] Add tests for scratch tensor patterns
- [ ] Add tests for new methods

### Phase 3: Implementation
- [ ] Update `eager_tensor/tensor.go` core methods
- [ ] Update `eager_tensor/tensor_math.go`
- [ ] Update `eager_tensor/tensor_linalg.go`
- [ ] Update `eager_tensor/tensor_conv.go`
- [ ] Update `eager_tensor/activations.go`
- [ ] Implement new methods
- [ ] Implement gradient methods

### Phase 4: Documentation
- [ ] Update SPEC.md
- [ ] Update package documentation
- [ ] Add migration guide
- [ ] Update examples

## Success Criteria

1. ✅ All operations follow unified `Operation(dst, ...args) Tensor` pattern
   - Returns `self` if `dst == nil` (in-place)
   - Returns `dst` if `dst != nil` (enables chaining)
   - Operations creating different data types return created/reused tensor
2. ✅ No method renaming (same names, different signatures)
3. ✅ No dual versions (single method per operation)
4. ✅ Zero allocations in training hot paths (with scratch tensors)
5. ✅ All tests pass
6. ✅ Performance improvements measurable
7. ✅ Documentation updated

## Timeline

- **Phase 0**: ✅ Complete (Planning)
- **Phase 1**: TBD (Interface Update)
- **Phase 2**: TBD (Test Update)
- **Phase 3**: TBD (Implementation Update)
- **Phase 4**: TBD (Documentation Update)

## Notes

- This transition is a breaking change but necessary for efficiency
- Callers must update to use new signatures
- Scratch tensor patterns should be documented clearly
- Consider providing migration tooling/scripts if needed

