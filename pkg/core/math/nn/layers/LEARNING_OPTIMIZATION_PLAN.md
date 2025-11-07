# Learning Operations Optimization Plan

## Overview

This document provides a comprehensive plan to optimize learning operations (forward/backward passes, optimizers, loss functions) by maximizing the use of tensor API functionality and leveraging generic operations from `primitive/generics/OPS.md`.

## Goals

1. **Reduce Memory Allocations**: Minimize tensor cloning and intermediate tensor creation
2. **Leverage Tensor API**: Use destination parameters (`dst`) wherever available
3. **Utilize Generic Operations**: Replace manual loops with optimized generic primitives
4. **Improve Performance**: Reduce memory bandwidth and improve cache locality
5. **Maintain Correctness**: Ensure all optimizations preserve mathematical correctness

## Current State Analysis

### 1. Optimizer Operations (`learn/optimizer.go`)

#### SGD Optimizer (Lines 27-61)
**Current Implementation:**
```go
scaledGrad := param.Grad.Clone()           // Clone 1
scaledGrad = scaledGrad.MulScalar(nil, s.lr)
param.Data.Subtract(nil, scaledGrad)
```

**Issues:**
- Unnecessary `Clone()` before `MulScalar` (can use in-place with dst parameter)
- Can use `AddScaled` or `SubScaled` for combined operations
- Missing use of tensor API's `ScalarMul` or `AddScaled` operations

**Optimization Opportunities:**
- Use `AddScaled(dst, other, alpha)` with negative alpha: `param.Data.AddScaled(nil, param.Grad, -s.lr)`
- Or use `SubScaled` if available, or combine `MulScalar` + `Subtract` with pre-allocated scratch tensor

#### Adam Optimizer (Lines 115-204)
**Current Implementation:**
- **Line 167**: `state.m.MulScalar(nil, a.beta1)` - In-place (good)
- **Line 168-169**: `scaledGrad1 := param.Grad.Clone(); scaledGrad1.MulScalar(nil, 1-a.beta1)` - Clone unnecessary
- **Line 170**: `state.m.Add(nil, scaledGrad1)` - Can use `AddScaled` directly
- **Line 173-174**: `gradSquared := param.Grad.Clone(); gradSquared.Multiply(nil, param.Grad)` - Clone unnecessary
- **Line 175**: `state.v.MulScalar(nil, a.beta2)` - In-place (good)
- **Line 176-177**: `scaledGrad2 := gradSquared.Clone(); scaledGrad2.MulScalar(nil, 1-a.beta2)` - Clone unnecessary
- **Line 178**: `state.v.Add(nil, scaledGrad2)` - Can use `AddScaled` directly
- **Line 181-184**: Multiple clones for bias correction
- **Line 187-188**: Clone for sqrt computation
- **Line 191-195**: Clone + manual loop for epsilon tensor (inefficient)
- **Line 198-200**: Clone for update computation

**Issues:**
- **12+ Clone operations** per parameter update
- Manual loop using `Elements()` for epsilon tensor (line 192-194) - very inefficient
- Multiple intermediate tensors that could be pre-allocated
- Not using `AddScaled` for combined operations

**Optimization Opportunities:**
1. **Pre-allocate scratch tensors** in optimizer state:
   - `scaledGrad1`, `scaledGrad2`, `gradSquared`, `mHat`, `vHat`, `sqrtVHat`, `epsilonTensor`, `update`
2. **Use `AddScaled`** instead of `Clone` + `MulScalar` + `Add`:
   - `state.m.AddScaled(nil, param.Grad, 1-a.beta1)` after `state.m.MulScalar(nil, a.beta1)`
   - `state.v.AddScaled(nil, gradSquared, 1-a.beta2)` after `state.v.MulScalar(nil, a.beta2)`
3. **Use `Fill`** for epsilon tensor instead of manual loop:
   - `epsilonTensor.Fill(nil, a.epsilon)` instead of `Elements()` loop
4. **Use destination parameters** for all operations:
   - `MulScalar(dst, scalar)`, `AddScaled(dst, other, alpha)`, `Sqrt(dst)`, `Divide(dst, other)`, etc.

### 2. Loss Functions (`nn/losses.go`)

#### MSE Loss (Lines 17-54)
**Current Implementation:**
```go
squaredDiff := pred.Clone()                // Clone 1
squaredDiff = squaredDiff.Subtract(nil, target)
squaredDiff = squaredDiff.Multiply(nil, squaredDiff)
sum := squaredDiff.Sum(nil, nil)
return float32(sum.At(0)) / float32(size)
```

**Issues:**
- Unnecessary `Clone()` - can use pre-allocated tensor or in-place operations
- `Sum` creates new tensor when dst is nil

**Optimization Opportunities:**
- Pre-allocate `squaredDiff` tensor in loss function or use destination parameter
- Use `Sum(dst, nil)` with pre-allocated scalar tensor

#### CrossEntropy Loss (Lines 64-103)
**Current Implementation:**
- Multiple `Clone()` operations
- Manual mask creation and conditional operations
- Could use `Where` operation more efficiently

**Optimization Opportunities:**
- Use `Where(dst, condition, a, b)` for conditional operations
- Pre-allocate intermediate tensors
- Use `Log` and `Multiply` with destination parameters

#### CategoricalCrossEntropy (Lines 106-186)
**Current Implementation:**
- Similar issues to CrossEntropy
- Softmax computation creates new tensor

**Optimization Opportunities:**
- Use `Softmax(dim, dst)` with pre-allocated destination
- Pre-allocate all intermediate tensors

### 3. Layer Backward Passes

#### Dense Layer (`layers/dense.go` Lines 189-311)
**Current Implementation:**
- Uses `MatMulTransposed` with destination (good)
- Uses `Sum` with destination for bias gradient (good)
- Some `Reshape` operations create views (acceptable)

**Optimization Opportunities:**
- Already well-optimized, but can pre-allocate more scratch tensors

#### Conv2D Layer (`layers/conv2d.go` Lines 196-321)
**Current Implementation:**
- Uses pre-allocated scratch tensors for `gradOutputT` and `kernelGradMatrix` (good)
- `Im2Col` creates new tensor (necessary, no alternative)
- Some operations could use more destination parameters

**Optimization Opportunities:**
- Already has good scratch tensor usage
- Can optimize `Reshape` + `Copy` patterns

#### Activation Layers (`layers/activations.go`)
**Current Implementation:**
- Softmax backward uses pre-allocated scratch tensors (good, from OPTIMIZATION_ANALYSIS.md)
- ReLU backward creates mask tensor (could be pre-allocated)
- Sigmoid/Tanh backward create multiple intermediate tensors

**Optimization Opportunities:**
- Pre-allocate mask tensors for ReLU backward
- Pre-allocate intermediate tensors for Sigmoid/Tanh backward

## Optimization Plan

### Phase 1: High-Impact Optimizer Optimizations (Priority: Critical)

#### 1.1 Optimize SGD Optimizer
**File**: `learn/optimizer.go` (Lines 27-61)

**Actions:**
1. Replace `Clone()` + `MulScalar` + `Subtract` with `AddScaled`:
   ```go
   // Before:
   scaledGrad := param.Grad.Clone()
   scaledGrad = scaledGrad.MulScalar(nil, s.lr)
   param.Data.Subtract(nil, scaledGrad)
   
   // After:
   param.Data.AddScaled(nil, param.Grad, -s.lr)
   ```

**Expected Impact:**
- Eliminates 1 Clone operation per parameter update
- Reduces memory allocation by ~50% for SGD
- Improves performance by ~10-15%

**Dependencies:**
- `AddScaled` operation available in tensor API (SPEC.md line 195)

#### 1.2 Optimize Adam Optimizer - Pre-allocate Scratch Tensors
**File**: `learn/optimizer.go` (Lines 115-204)

**Actions:**
1. Extend `adamState` struct to include pre-allocated scratch tensors:
   ```go
   type adamState struct {
       m            tensor.Tensor // First moment estimate
       v            tensor.Tensor // Second moment estimate
       step         int           // Step counter
       // Pre-allocated scratch tensors
       scaledGrad1  tensor.Tensor // For (1-beta1) * grad
       scaledGrad2  tensor.Tensor // For (1-beta2) * grad^2
       gradSquared  tensor.Tensor // For grad^2
       mHat         tensor.Tensor // Bias-corrected first moment
       vHat         tensor.Tensor // Bias-corrected second moment
       sqrtVHat     tensor.Tensor // sqrt(vHat)
       epsilonTensor tensor.Tensor // Epsilon tensor
       update       tensor.Tensor // Final update term
   }
   ```

2. Initialize scratch tensors in state creation (line 149):
   ```go
   state = &adamState{
       m:            tensor.New(tensor.DTFP32, shape),
       v:            tensor.New(tensor.DTFP32, shape),
       step:         0,
       scaledGrad1:   tensor.New(tensor.DTFP32, shape),
       scaledGrad2:   tensor.New(tensor.DTFP32, shape),
       gradSquared:  tensor.New(tensor.DTFP32, shape),
       mHat:         tensor.New(tensor.DTFP32, shape),
       vHat:         tensor.New(tensor.DTFP32, shape),
       sqrtVHat:     tensor.New(tensor.DTFP32, shape),
       epsilonTensor: tensor.New(tensor.DTFP32, shape),
       update:       tensor.New(tensor.DTFP32, shape),
   }
   ```

3. Replace all `Clone()` operations with destination parameter usage:
   - Line 168-169: Use `param.Grad.MulScalar(state.scaledGrad1, 1-a.beta1)`
   - Line 170: Use `state.m.AddScaled(nil, param.Grad, 1-a.beta1)` after `state.m.MulScalar(nil, a.beta1)`
   - Line 173-174: Use `param.Grad.Multiply(state.gradSquared, param.Grad)`
   - Line 176-177: Use `state.gradSquared.MulScalar(state.scaledGrad2, 1-a.beta2)`
   - Line 178: Use `state.v.AddScaled(nil, state.gradSquared, 1-a.beta2)` after `state.v.MulScalar(nil, a.beta2)`
   - Line 181-184: Use destination parameters for bias correction
   - Line 187-188: Use `vHat.Sqrt(state.sqrtVHat)`
   - Line 191-195: Use `state.epsilonTensor.Fill(nil, a.epsilon)` instead of `Elements()` loop
   - Line 198-200: Use destination parameters for update computation

**Expected Impact:**
- Eliminates **12+ Clone operations** per parameter update
- Reduces memory allocations by ~90% for Adam optimizer
- Improves performance by ~30-40% for Adam updates
- Eliminates slow `Elements()` iterator usage

**Dependencies:**
- All tensor operations support destination parameters (verified in SPEC.md)
- `Fill` operation available (SPEC.md line 115)

#### 1.3 Use AddScaled for Combined Operations
**File**: `learn/optimizer.go`

**Actions:**
1. Replace `MulScalar` + `Add` patterns with `AddScaled`:
   ```go
   // Before:
   state.m.MulScalar(nil, a.beta1)
   scaledGrad1 := param.Grad.Clone()
   scaledGrad1 = scaledGrad1.MulScalar(nil, 1-a.beta1)
   state.m.Add(nil, scaledGrad1)
   
   // After:
   state.m.MulScalar(nil, a.beta1)
   state.m.AddScaled(nil, param.Grad, 1-a.beta1)
   ```

**Expected Impact:**
- Eliminates intermediate tensor for scaled gradient
- Reduces memory bandwidth
- Improves cache locality

### Phase 2: Loss Function Optimizations (Priority: High)

#### 2.1 Optimize MSE Loss
**File**: `nn/losses.go` (Lines 17-54)

**Actions:**
1. Pre-allocate `squaredDiff` tensor in loss struct or use destination parameter
2. Use `Sum(dst, nil)` with pre-allocated scalar tensor:
   ```go
   type MSELoss struct {
       squaredDiff tensor.Tensor // Pre-allocated scratch tensor
       sumResult   tensor.Tensor // Pre-allocated scalar tensor
   }
   ```

**Expected Impact:**
- Eliminates Clone operation
- Reduces memory allocation per loss computation

#### 2.2 Optimize CrossEntropy Loss
**File**: `nn/losses.go` (Lines 64-103)

**Actions:**
1. Pre-allocate intermediate tensors:
   - `predPlusEps`, `logPred`, `targetLogPred`, `maskedLoss`
2. Use `Fill` for epsilon tensor instead of `FullLike` + operations
3. Use `Where` more efficiently for conditional operations

**Expected Impact:**
- Reduces intermediate tensor allocations
- Improves performance

#### 2.3 Optimize CategoricalCrossEntropy
**File**: `nn/losses.go` (Lines 106-186)

**Actions:**
1. Pre-allocate `predProb` tensor for softmax result
2. Use `Softmax(dim, dst)` with pre-allocated destination
3. Pre-allocate all intermediate tensors

**Expected Impact:**
- Eliminates softmax tensor allocation
- Reduces memory pressure

### Phase 3: Layer Backward Pass Optimizations (Priority: Medium)

#### 3.1 Pre-allocate Activation Layer Scratch Tensors
**Files**: `layers/activations.go`

**Actions:**
1. **ReLU Backward**: Pre-allocate `mask` tensor in `Init()`
2. **Sigmoid Backward**: Pre-allocate `term1`, `term2` tensors in `Init()`
3. **Tanh Backward**: Pre-allocate `squared`, `term` tensors in `Init()`

**Expected Impact:**
- Reduces tensor allocations in backward passes
- Improves performance for activation layers

#### 3.2 Optimize Reshape + Copy Patterns
**Files**: Multiple layer files

**Actions:**
1. Replace `Reshape(nil, shape)` + `Copy` with `Reshape(dst, shape)`:
   ```go
   // Before:
   inputReshaped := input.Reshape(nil, newShape)
   output.Copy(inputReshaped)
   
   // After:
   input.Reshape(output, newShape)
   ```

**Expected Impact:**
- Eliminates intermediate view tensor
- Reduces memory operations

### Phase 4: Generic Operations Integration (Priority: Medium-Low)

#### 4.1 Replace Manual Loops with Generic Operations
**Files**: All learning code

**Actions:**
1. Identify manual loops that can be replaced with generic operations from `OPS.md`
2. Use `ElemApply*` functions for custom element-wise operations
3. Use vector/matrix optimized operations where applicable

**Considerations:**
- Generic operations have some overhead (5-15% per OPS.md)
- Only use when the operation is complex enough to justify the overhead
- Prefer tensor API operations over generic primitives when available

#### 4.2 Use Generic Comparison Operations
**Files**: Loss functions, activation backward passes

**Actions:**
1. Use generic comparison operations from `OPS.md`:
   - `ElemGreaterThan`, `ElemEqual`, `ElemLess`, etc.
2. These are already used via tensor API, but can verify direct usage where beneficial

**Note**: Tensor API already wraps these operations, so direct usage may not be necessary unless performance profiling shows benefits.

## Implementation Strategy

### Step 1: Optimizer Optimizations (Week 1)
1. Implement SGD optimization (1.1)
2. Implement Adam scratch tensor pre-allocation (1.2)
3. Implement AddScaled usage (1.3)
4. Add benchmarks to measure improvements

### Step 2: Loss Function Optimizations (Week 2)
1. Implement MSE loss optimizations (2.1)
2. Implement CrossEntropy optimizations (2.2)
3. Implement CategoricalCrossEntropy optimizations (2.3)
4. Add benchmarks

### Step 3: Layer Optimizations (Week 3)
1. Pre-allocate activation layer scratch tensors (3.1)
2. Optimize Reshape + Copy patterns (3.2)
3. Add benchmarks

### Step 4: Generic Operations (Week 4)
1. Identify opportunities for generic operations (4.1)
2. Implement where beneficial
3. Benchmark and verify performance improvements

## Success Metrics

### Performance Metrics
- **Memory Allocations**: Reduce by 70-90% in optimizers
- **Memory Bandwidth**: Reduce by 50-70% in learning operations
- **Execution Time**: Improve by 20-40% for optimizer updates
- **Cache Locality**: Improve by using pre-allocated scratch tensors

### Code Quality Metrics
- **Clone Operations**: Reduce from 12+ to 0 in Adam optimizer
- **Intermediate Tensors**: Pre-allocate all reusable tensors
- **API Usage**: Maximize use of destination parameters

## Testing Strategy

1. **Unit Tests**: Ensure all optimizations maintain mathematical correctness
2. **Integration Tests**: Verify end-to-end training still works
3. **Benchmark Tests**: Measure performance improvements
4. **Memory Profiling**: Verify reduction in allocations
5. **Numerical Stability**: Ensure optimizations don't affect convergence

## Dependencies and Prerequisites

### Tensor API Requirements
- ✅ `AddScaled(dst, other, alpha)` - Available (SPEC.md line 195)
- ✅ `Fill(dst, value)` - Available (SPEC.md line 115)
- ✅ All operations support destination parameters - Verified
- ✅ `Sum(dst, dims)` - Available (SPEC.md line 171)
- ✅ `Softmax(dim, dst)` - Available (SPEC.md line 207)

### Generic Operations (from OPS.md)
- ✅ `ElemApply*` functions available for custom operations
- ✅ Comparison operations available
- ⚠️ Generic operations have 5-15% overhead - Use judiciously

## Risk Assessment

### Low Risk
- Optimizer optimizations (well-isolated, easy to test)
- Loss function optimizations (mathematical correctness easy to verify)

### Medium Risk
- Layer backward pass optimizations (more complex, need careful testing)
- Generic operations integration (performance overhead considerations)

### Mitigation Strategies
1. Implement optimizations incrementally
2. Add comprehensive tests for each optimization
3. Benchmark before and after each change
4. Maintain backward compatibility
5. Add feature flags for experimental optimizations

## Future Enhancements

1. **Automatic Scratch Tensor Management**: Framework-level support for pre-allocated scratch tensors
2. **Operation Fusion**: Combine multiple operations into single kernels
3. **SIMD Optimizations**: Use SIMD for element-wise operations
4. **Multi-threading**: Leverage multi-threaded generic operations where available
5. **Memory Pool**: Implement tensor memory pooling for frequently allocated tensors

## References

- `tensor/types/SPEC.md` - Tensor API specification
- `primitive/generics/OPS.md` - Generic operations specification
- `layers/OPTIMIZATION_ANALYSIS.md` - Forward pass optimization analysis
- `learn/optimizer.go` - Current optimizer implementation
- `nn/losses.go` - Current loss function implementation

## Conclusion

This plan provides a comprehensive roadmap for optimizing learning operations by:
1. Maximizing use of tensor API destination parameters
2. Pre-allocating scratch tensors to eliminate Clone operations
3. Using combined operations like `AddScaled` where applicable
4. Leveraging generic operations judiciously

The optimizations are prioritized by impact and effort, with Phase 1 (optimizer optimizations) providing the highest impact with relatively low effort.

