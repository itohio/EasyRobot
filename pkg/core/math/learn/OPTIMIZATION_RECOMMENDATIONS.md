# Learning Code Optimization Recommendations

This document identifies optimization opportunities in the learning/training code (`pkg/core/math/learn`). Each recommendation includes:
- **Location**: File and line numbers
- **Issue**: Description of the inefficiency
- **Impact**: Performance/memory impact
- **Fix**: How to improve it

## Table of Contents
1. [Optimizer Optimizations](#optimizer-optimizations)
2. [Tensor Interface Usage](#tensor-interface-usage)
3. [Shape Operations](#shape-operations)
4. [Quantization Optimizations](#quantization-optimizations)
5. [Parameter Handling](#parameter-handling)
6. [Code Quality Issues](#code-quality-issues)

---

## 1. Optimizer Optimizations

### 1.1 Excessive Cloning in SGD.Update (optimizer.go:55-58)

**Location**: `optimizer.go:55-58`

**Issue**: 
```go
scaledGrad := param.Grad.Clone()
scaledGrad = scaledGrad.Scale(s.lr)
// Sub() modifies param.Data in place, so the tensor data is updated
param.Data.Sub(scaledGrad)
```

The gradient is cloned unnecessarily. Since `Scale()` modifies in-place and returns self, we can avoid the clone.

**Impact**: 
- Memory: One unnecessary tensor allocation per update
- Performance: O(n) copy operation where n is parameter size

**Fix**:
```go
// Option 1: Use AddTo with scaled gradient (if AddTo supports scalar scaling)
// Option 2: Pre-allocate a scratch tensor and reuse it
// Option 3: Use in-place operations directly on param.Grad (if it's safe to modify)
// Best: Add a ScaleTo method to Tensor interface that writes to a destination

// Temporary workaround:
if s.scratchGrad == nil || !s.scratchGrad.Shape().Equal(param.Grad.Shape()) {
    s.scratchGrad = tensor.NewAs(param.Grad)
}
s.scratchGrad.Copy(param.Grad).Scale(s.lr)
param.Data.Sub(s.scratchGrad)
```

**Recommendation**: Add `ScaleTo(dst Tensor, scalar float64)` method to Tensor interface for zero-allocation scaling into destination.

---

### 1.2 Excessive Cloning in Adam.Update (optimizer.go:168-202)

**Location**: `optimizer.go:168-202`

**Issue**: Multiple unnecessary clones in Adam update:
- Line 168: `scaledGrad1 := param.Grad.Clone()`
- Line 173: `gradSquared := param.Grad.Clone()`
- Line 176: `scaledGrad2 := gradSquared.Clone()`
- Line 181: `mHat := state.m.Clone()`
- Line 183: `vHat := state.v.Clone()`
- Line 187: `sqrtVHat := vHat.Clone()`
- Line 191: `epsilonTensor := sqrtVHat.Clone()`
- Line 198: `update := mHat.Clone()`

**Impact**: 
- Memory: 8 tensor allocations per parameter update
- Performance: 8 O(n) copy operations per update
- For a model with 1M parameters, this is 8M allocations per step

**Fix**: Reuse scratch tensors stored in optimizer state:

```go
type Adam struct {
    // ... existing fields ...
    scratch map[uintptr]*adamScratch // Per-parameter scratch space
}

type adamScratch struct {
    scaledGrad1 tensor.Tensor
    scaledGrad2 tensor.Tensor
    gradSquared tensor.Tensor
    mHat        tensor.Tensor
    vHat        tensor.Tensor
    sqrtVHat    tensor.Tensor
    epsilonT    tensor.Tensor
    update      tensor.Tensor
}

func (a *Adam) Update(param types.Parameter) error {
    // ... validation ...
    
    key := param.Data.ID()
    state := a.state[key]
    scratch := a.scratch[key]
    
    // Allocate scratch space if needed
    if scratch == nil {
        shape := param.Data.Shape()
        scratch = &adamScratch{
            scaledGrad1: tensor.New(tensor.DTFP32, shape),
            scaledGrad2: tensor.New(tensor.DTFP32, shape),
            gradSquared: tensor.New(tensor.DTFP32, shape),
            mHat:        tensor.New(tensor.DTFP32, shape),
            vHat:        tensor.New(tensor.DTFP32, shape),
            sqrtVHat:    tensor.New(tensor.DTFP32, shape),
            epsilonT:    tensor.New(tensor.DTFP32, shape),
            update:      tensor.New(tensor.DTFP32, shape),
        }
        a.scratch[key] = scratch
    }
    
    // Reuse scratch tensors
    scratch.scaledGrad1.Copy(param.Grad).Scale(1 - a.beta1)
    state.m.Scale(a.beta1).Add(scratch.scaledGrad1)
    
    scratch.gradSquared.Copy(param.Grad).Mul(param.Grad)
    scratch.scaledGrad2.Copy(scratch.gradSquared).Scale(1 - a.beta2)
    state.v.Scale(a.beta2).Add(scratch.scaledGrad2)
    
    // ... continue with reused tensors ...
}
```

**Recommendation**: Implement scratch tensor reuse pattern for all optimizers.

---

### 1.3 Inefficient Epsilon Tensor Creation (optimizer.go:191-195)

**Location**: `optimizer.go:191-195`

**Issue**:
```go
epsilonTensor := sqrtVHat.Clone()
for elem := range epsilonTensor.Elements() {
    elem.Set(a.epsilon)
}
sqrtVHat = sqrtVHat.Add(epsilonTensor)
```

Creating a full tensor copy just to fill it with epsilon, then adding it.

**Impact**: 
- Memory: One unnecessary tensor allocation
- Performance: O(n) copy + O(n) fill + O(n) add operations

**Fix**:
```go
// Option 1: Use Fill() method if available
// sqrtVHat = sqrtVHat.AddScalar(a.epsilon)

// Option 2: Add epsilon in-place using ScaleAdd pattern
// sqrtVHat = sqrtVHat.AddScalar(a.epsilon) // Need AddScalar method

// Option 3: Use scratch tensor pre-filled with epsilon
// if scratch.epsilonT == nil {
//     scratch.epsilonT = tensor.New(tensor.DTFP32, shape).Fill(a.epsilon)
// }
// sqrtVHat = sqrtVHat.Add(scratch.epsilonT)

// Best: Add AddScalar method to Tensor interface
sqrtVHat = sqrtVHat.AddScalar(a.epsilon)
```

**Recommendation**: Add `AddScalar(scalar float64) Tensor` method to Tensor interface for efficient scalar addition.

---

### 1.4 Shape Comparison Inefficiency (optimizer.go:47, 135)

**Location**: `optimizer.go:47, 135`

**Issue**:
```go
if !shapesEqual(param.Data.Shape(), param.Grad.Shape()) {
    return fmt.Errorf("SGD.Update: parameter and gradient shapes mismatch: %v vs %v", param.Data.Shape(), param.Grad.Shape())
}
```

`Shape()` is called multiple times, and `shapesEqual` creates copies. Also, `param.Data.Shape()` and `param.Grad.Shape()` are called again in the error message.

**Impact**: 
- Memory: Multiple shape slice allocations
- Performance: Redundant shape comparisons

**Fix**:
```go
// Cache shapes once
dataShape := param.Data.Shape()
gradShape := param.Grad.Shape()
if !shapesEqual(dataShape, gradShape) {
    return fmt.Errorf("SGD.Update: parameter and gradient shapes mismatch: %v vs %v", dataShape, gradShape)
}
```

**Better Fix**: Use Shape.Equal() method if available:
```go
if !param.Data.Shape().Equal(param.Grad.Shape()) {
    return fmt.Errorf("SGD.Update: parameter and gradient shapes mismatch: %v vs %v", 
        param.Data.Shape(), param.Grad.Shape())
}
```

**Recommendation**: Use `Shape.Equal()` method for shape comparisons instead of custom `shapesEqual` function.

---

## 2. Tensor Interface Usage

### 2.1 Shape() Called Multiple Times (optimizer.go:38, 42, 47, 126, 130, 135, 148)

**Location**: Throughout `optimizer.go`

**Issue**: `Shape()` is called multiple times for the same tensor in validation checks.

**Impact**: 
- Memory: Multiple slice allocations (though Shape is a slice, so underlying array is shared)
- Performance: Redundant method calls

**Fix**: Cache shape values:
```go
func (s *SGD) Update(param types.Parameter) error {
    // ... validation ...
    
    dataShape := param.Data.Shape()
    gradShape := param.Grad.Shape()
    
    if len(dataShape) == 0 {
        return fmt.Errorf("SGD.Update: empty parameter data")
    }
    
    if len(gradShape) == 0 {
        return nil
    }
    
    if !dataShape.Equal(gradShape) {
        return fmt.Errorf("SGD.Update: parameter and gradient shapes mismatch: %v vs %v", dataShape, gradShape)
    }
    
    // ... rest of update ...
}
```

**Recommendation**: Cache frequently accessed tensor properties within function scope.

---

### 2.2 Parameter Passing by Value (optimizer.go:27, 115)

**Location**: `optimizer.go:27, 115`

**Issue**: 
```go
func (s *SGD) Update(param types.Parameter) error {
```

Parameter is passed by value, but contains Tensor interfaces which are reference types. This is actually fine, but the comment suggests confusion.

**Current Code**:
```go
// Note: param is passed by value, but Data and Grad are tensor references
// We modify the underlying tensor data in place
```

**Impact**: 
- Code clarity: Confusing comments
- Performance: Minor - Parameter struct is small (2 Tensor interfaces + 1 bool)

**Fix**: Clarify comments and ensure consistency:
```go
// Update applies SGD update: param.Data = param.Data - lr * param.Grad.
// Parameter is passed by value, but Tensor interfaces are reference types,
// so modifications to param.Data and param.Grad affect the underlying tensors.
func (s *SGD) Update(param types.Parameter) error {
```

**Recommendation**: Update comments to clarify Tensor interface semantics.

---

## 3. Shape Operations

### 3.1 Redundant Shape() Calls in Quantization (quantization.go:327, 340, 343, 363, 368, 372)

**Location**: `quantization.go:327, 340, 343, 363, 368, 372`

**Issue**: `Shape()` is called multiple times for the same tensor:
```go
func QuantizeTensor(t tensor.Tensor, params *QuantizationParams, scheme QuantizationScheme, bits int) (tensor.Tensor, *QuantizationParams, error) {
    if len(t.Shape()) == 0 {
        return tensor.EmptyLike(t), nil, fmt.Errorf("quantization: empty tensor")
    }
    // ...
    quantized := tensor.New(tensor.DTFP32, t.Shape())
    // ...
    for indices := range t.Shape().Iterator() {
```

**Impact**: 
- Performance: Redundant method calls
- Code clarity: Less readable

**Fix**: Cache shape:
```go
func QuantizeTensor(t tensor.Tensor, params *QuantizationParams, scheme QuantizationScheme, bits int) (tensor.Tensor, *QuantizationParams, error) {
    shape := t.Shape()
    if len(shape) == 0 {
        return tensor.EmptyLike(t), nil, fmt.Errorf("quantization: empty tensor")
    }
    // ...
    quantized := tensor.New(tensor.DTFP32, shape)
    // ...
    for indices := range shape.Iterator() {
```

**Recommendation**: Always cache Shape() results when used multiple times.

---

## 4. Quantization Optimizations

### 4.1 Inefficient Element-wise Access (quantization.go:343-356, 372-377)

**Location**: `quantization.go:343-356, 372-377`

**Issue**: Using `Shape().Iterator()` and `At()/SetAt()` for element-wise operations is inefficient:
```go
for indices := range t.Shape().Iterator() {
    val := t.At(indices...)
    q := int32(math.Round(val/params.Scale)) + params.ZeroPoint
    // ...
    quantized.SetAt(float64(q), indices...)
}
```

**Impact**: 
- Performance: O(n) index calculations per element
- Overhead: Iterator creation, index conversion

**Fix**: Use `Elements()` iterator for direct element access:
```go
// Quantize using Elements() iterator
tElems := t.Elements()
qElems := quantized.Elements()

for {
    tElem, ok := <-tElems
    if !ok {
        break
    }
    qElem, ok := <-qElems
    if !ok {
        break
    }
    
    val := tElem.Get()
    q := int32(math.Round(val/params.Scale)) + params.ZeroPoint
    
    // Clamp
    if q < quantMin {
        q = quantMin
    }
    if q > quantMax {
        q = quantMax
    }
    
    qElem.Set(float64(q))
}
```

**Better Fix**: If `Elements()` supports parallel iteration, use that:
```go
// Use range-over-func for Elements()
tIter := t.Elements()
qIter := quantized.Elements()

for tElem := range tIter {
    qElem := <-qIter
    val := tElem.Get()
    q := int32(math.Round(val/params.Scale)) + params.ZeroPoint
    
    if q < quantMin {
        q = quantMin
    }
    if q > quantMax {
        q = quantMax
    }
    
    qElem.Set(float64(q))
}
```

**Recommendation**: Use `Elements()` iterator instead of `Shape().Iterator()` + `At()/SetAt()` for element-wise operations.

---

### 4.2 Redundant Tensor Allocation Check (quantization.go:327-329)

**Location**: `quantization.go:327-329`

**Issue**: 
```go
if len(t.Shape()) == 0 {
    return tensor.EmptyLike(t), nil, fmt.Errorf("quantization: empty tensor")
}
```

`EmptyLike()` is called even when we're returning an error. Also, `t.Empty()` method exists.

**Impact**: 
- Memory: Unnecessary tensor allocation on error path
- Performance: Redundant check

**Fix**:
```go
if t.Empty() {
    return nil, nil, fmt.Errorf("quantization: empty tensor")
}
```

**Recommendation**: Use `t.Empty()` method instead of `len(t.Shape()) == 0`.

---

## 5. Parameter Handling

### 5.1 Parameter Map Copying (base.go:406-419, model.go:186-212)

**Location**: `base.go:406-419`, referenced in `model.go:186-212`

**Issue**: `Parameters()` returns a copy of the parameter map:
```go
func (b *Base) Parameters() map[types.ParamIndex]types.Parameter {
    result := make(map[types.ParamIndex]types.Parameter)
    for idx, param := range b.params {
        result[idx] = param
    }
    return result
}
```

This is called in `QuantizeModel()` and creates a full copy of all parameters.

**Impact**: 
- Memory: Full parameter map copy (though Parameter structs are small)
- Performance: O(n) copy operation where n is number of parameters

**Note**: This might be intentional for encapsulation. However, if the caller only needs to iterate, we could provide an iterator instead.

**Fix Options**:
1. **Keep current design** if encapsulation is important
2. **Add iterator method**:
```go
type ParameterIterator interface {
    Next() (types.ParamIndex, types.Parameter, bool)
}

func (b *Base) ParameterIterator() ParameterIterator {
    // Return iterator over parameters
}
```
3. **Pre-allocate and reuse** in callers:
```go
// In QuantizeModel, reuse the same map
params := layer.Parameters()
// Process params without creating new maps
```

**Recommendation**: Keep current design if encapsulation is required, but document the allocation cost. Consider adding iterator interface for large-scale operations.

---

## 6. Code Quality Issues

### 6.1 Non-Idiomatic Error Handling (optimizer.go:38-44, 126-132)

**Location**: `optimizer.go:38-44, 126-132`

**Issue**: Multiple early returns with similar patterns. Could be consolidated.

**Current**:
```go
if param.Grad == nil || tensor.IsNil(param.Grad) || len(param.Grad.Shape()) == 0 {
    return nil
}
if param.Data == nil || tensor.IsNil(param.Data) || len(param.Data.Shape()) == 0 {
    return fmt.Errorf("SGD.Update: empty parameter data")
}
```

**Impact**: 
- Code clarity: Repetitive checks
- Performance: Multiple shape calls

**Fix**: Extract validation helper:
```go
func validateParameter(param types.Parameter, name string) error {
    if param.Data == nil || tensor.IsNil(param.Data) || param.Data.Empty() {
        return fmt.Errorf("%s: empty parameter data", name)
    }
    if param.Grad != nil && !tensor.IsNil(param.Grad) && !param.Grad.Empty() {
        if !param.Data.Shape().Equal(param.Grad.Shape()) {
            return fmt.Errorf("%s: parameter and gradient shapes mismatch: %v vs %v", 
                name, param.Data.Shape(), param.Grad.Shape())
        }
    }
    return nil
}

func (s *SGD) Update(param types.Parameter) error {
    if s == nil {
        return fmt.Errorf("SGD.Update: nil optimizer")
    }
    if !param.RequiresGrad {
        return nil
    }
    if err := validateParameter(param, "SGD.Update"); err != nil {
        return err
    }
    if param.Grad == nil || tensor.IsNil(param.Grad) || param.Grad.Empty() {
        return nil
    }
    // ... update logic ...
}
```

**Recommendation**: Extract common validation patterns into helper functions.

---

### 6.2 Inconsistent Shape Validation (optimizer.go:47-48, 135-136)

**Location**: `optimizer.go:47-48, 135-136`

**Issue**: Custom `shapesEqual` function instead of using `Shape.Equal()` method.

**Current**:
```go
func shapesEqual(a, b []int) bool {
    if len(a) != len(b) {
        return false
    }
    for i := range a {
        if a[i] != b[i] {
            return false
        }
    }
    return true
}
```

**Impact**: 
- Code duplication: Reimplementing existing functionality
- Maintenance: Two places to update if shape comparison logic changes

**Fix**: Use `Shape.Equal()` method:
```go
if !param.Data.Shape().Equal(param.Grad.Shape()) {
    return fmt.Errorf("SGD.Update: parameter and gradient shapes mismatch: %v vs %v", 
        param.Data.Shape(), param.Grad.Shape())
}
```

**Recommendation**: Remove `shapesEqual` function and use `Shape.Equal()` everywhere.

---

### 6.3 Inefficient Epsilon Addition Pattern (optimizer.go:191-195)

**Location**: `optimizer.go:191-195`

**Issue**: Creating a full tensor copy just to add a scalar constant:
```go
epsilonTensor := sqrtVHat.Clone()
for elem := range epsilonTensor.Elements() {
    elem.Set(a.epsilon)
}
sqrtVHat = sqrtVHat.Add(epsilonTensor)
```

**Impact**: 
- Memory: Full tensor allocation
- Performance: O(n) copy + O(n) fill + O(n) add

**Fix**: Use scalar addition if available, or pre-allocate scratch tensor:
```go
// Option 1: Add AddScalar method to Tensor interface
sqrtVHat = sqrtVHat.AddScalar(a.epsilon)

// Option 2: Pre-allocate and reuse epsilon tensor
if state.epsilonTensor == nil {
    state.epsilonTensor = tensor.New(tensor.DTFP32, param.Data.Shape()).Fill(a.epsilon)
}
sqrtVHat = sqrtVHat.Add(state.epsilonTensor)
```

**Recommendation**: Add `AddScalar(scalar float64) Tensor` to Tensor interface.

---

## Summary of Priority Fixes

### High Priority (Performance Critical)
1. **Adam optimizer cloning** (1.2) - 8 allocations per update
2. **SGD optimizer cloning** (1.1) - 1 allocation per update
3. **Epsilon tensor creation** (1.3, 6.3) - 1 allocation per update

### Medium Priority (Code Quality)
4. **Shape caching** (2.1, 3.1) - Redundant calls
5. **Use Elements() iterator** (4.1) - Better performance for element-wise ops
6. **Use Shape.Equal()** (6.2) - Remove duplication

### Low Priority (Nice to Have)
7. **Extract validation helpers** (6.1) - Code clarity
8. **Parameter map copying** (5.1) - May be intentional for encapsulation

---

## Implementation Notes

1. **Scratch Tensor Reuse**: Consider adding a `ScratchTensorPool` interface for optimizers to request reusable tensors.

2. **Tensor Interface Extensions**: The following methods should be added to Tensor interface:
   - `AddScalar(scalar float64) Tensor`
   - `ScaleTo(dst Tensor, scalar float64) Tensor` 
   - `MulScalar(scalar float64) Tensor`

3. **Shape Optimization**: Consider making Shape() return a const reference when possible, or provide `ShapeRef()` that returns without copying.

4. **Benchmarking**: Before and after benchmarks should be created for each optimization to measure impact.

