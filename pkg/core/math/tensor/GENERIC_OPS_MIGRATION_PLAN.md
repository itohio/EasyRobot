# Tensor Operations Generic Migration Plan

**Date**: Generated from code analysis  
**Purpose**: Plan for migrating tensor eager operations from fp32-specific primitives to generic operations where applicable  
**Scope**: Eager tensor operations in `tensor/eager_tensor` package

---

## Executive Summary

This document identifies which eager tensor operations can be migrated to use generic operations from `primitive/generics` instead of fp32-specific primitives. The migration enables:

1. **Type flexibility**: Support for multiple numeric types (float32, float64, int8, int16, int32, int64, int)
2. **Code reuse**: Reduce duplication between fp32 and future type-specific implementations
3. **Maintainability**: Single implementation for generic operations
4. **Performance**: Generic operations are highly optimized and may outperform helper-based fp32 implementations

**Key Principle**: Operations that are type-agnostic and don't rely on float32-specific precision or behavior should use generics. Operations that require float32-specific algorithms (e.g., BLAS, convolutions, pooling) must remain fp32-specific.

---

## Decision Framework

### ✅ Use Generic Ops When:
- Operation is type-agnostic (works for all numeric types)
- Operation doesn't rely on float32-specific precision
- Generic equivalent exists and is well-tested
- Operation is simple element-wise transformation
- Operation is comparison/boolean (works for all numeric types)

### ❌ Keep FP32-Specific When:
- Operation requires float32-specific algorithms (BLAS, LAPACK)
- Operation involves floating-point precision requirements
- Operation is specialized (convolutions, pooling, activations with special math)
- Operation is in hot path with direct loop implementation
- Operation has SIMD optimizations or platform-specific code

---

## Migration Analysis by Operation Category

### 1. Element-wise Binary Operations

#### ✅ **CAN MIGRATE TO GENERICS**

| Tensor Operation | Current FP32 Primitive | Generic Equivalent | Status |
|-----------------|----------------------|-------------------|--------|
| `Add` (strided) | `fp32.ElemAdd` | `generics.ElemApplyBinaryStrided[T](dst, a, b, shape, stridesDst, stridesA, stridesB, func(a, b T) T { return a + b })` | ⏳ **MIGRATE** |
| `Sub` (strided) | `fp32.ElemSub` | `generics.ElemApplyBinaryStrided[T](dst, a, b, shape, stridesDst, stridesA, stridesB, func(a, b T) T { return a - b })` | ⏳ **MIGRATE** |
| `Mul` (strided) | `fp32.ElemMul` | `generics.ElemApplyBinaryStrided[T](dst, a, b, shape, stridesDst, stridesA, stridesB, func(a, b T) T { return a * b })` | ⏳ **MIGRATE** |
| `Div` (strided) | `fp32.ElemDiv` | `generics.ElemApplyBinaryStrided[T](dst, a, b, shape, stridesDst, stridesA, stridesB, func(a, b T) T { return a / b })` | ⏳ **MIGRATE** |

**Note**: Contiguous versions use `fp32.Axpy` for Add/Sub which is BLAS-optimized. Keep contiguous fast paths as-is, migrate strided versions to generics.

**Migration Strategy**:
- Keep contiguous fast paths using `fp32.Axpy` (BLAS-optimized)
- Migrate strided versions to `generics.ElemApplyBinaryStrided` with appropriate operation functions
- Add type dispatch based on tensor DataType

---

### 2. Element-wise Unary Operations

#### ✅ **CAN MIGRATE TO GENERICS**

| Tensor Operation | Current FP32 Primitive | Generic Equivalent | Status |
|-----------------|----------------------|-------------------|--------|
| `Copy` (strided) | `fp32.ElemCopy` | `generics.ElemCopyStrided[T]` | ⏳ **MIGRATE** |
| `Sign` | `fp32.ElemSign` | `generics.ElemSignStrided[T]` | ⏳ **MIGRATE** |
| `Negative` | `fp32.ElemNegative` | `generics.ElemNegativeStrided[T]` | ⏳ **MIGRATE** |

#### ❌ **MUST REMAIN FP32-SPECIFIC**

| Tensor Operation | Current FP32 Primitive | Reason |
|-----------------|----------------------|--------|
| `Square` | `fp32.ElemSquare` | Relies on float32 precision for overflow handling |
| `Sqrt` | `fp32.ElemSqrt` | Requires float32 math library (`math32.Sqrt`) |
| `Exp` | `fp32.ElemExp` | Requires float32 math library (`math32.Exp`) |
| `Log` | `fp32.ElemLog` | Requires float32 math library (`math32.Log`) |
| `Pow` | `fp32.ElemPow` | Requires float32 math library (`math32.Pow`) |
| `Abs` | `fp32.ElemAbs` | Relies on float32 precision (though could be generic) |
| `Cos` | `fp32.ElemCos` | Requires float32 math library (`math32.Cos`) |
| `Sin` | `fp32.ElemSin` | Requires float32 math library (`math32.Sin`) |

**Note**: `Abs` could potentially be generic, but fp32 version may have optimizations. Keep as-is for now.

**Migration Strategy**:
- Migrate `Copy`, `Sign`, `Negative` to generics immediately
- Keep math library operations (Square, Sqrt, Exp, Log, Pow, Cos, Sin) as fp32-specific
- Consider generic `Abs` in future if performance is acceptable

---

### 3. Scalar Operations

#### ✅ **CAN MIGRATE TO GENERICS**

| Tensor Operation | Current FP32 Primitive | Generic Equivalent | Status |
|-----------------|----------------------|-------------------|--------|
| `Fill` | `fp32.Fill` | `generics.ElemFillStrided[T]` | ⏳ **MIGRATE** |
| `Scale` (strided) | `fp32.ElemScaleInPlace` | `generics.ElemApplyUnaryScalarStrided[T](dst, src, scalar, shape, stridesDst, stridesSrc, func(x, s T) T { return x * s })` | ⏳ **MIGRATE** |

**Note**: Contiguous `Scale` uses `fp32.Scal` (BLAS-optimized). Keep contiguous fast path as-is.

**Migration Strategy**:
- Keep contiguous `Scale` using `fp32.Scal` (BLAS-optimized)
- Migrate strided `Scale` to generics
- Migrate `Fill` to generics

---

### 4. Comparison Operations

#### ✅ **CAN MIGRATE TO GENERICS**

| Tensor Operation | Current FP32 Primitive | Generic Equivalent | Status |
|-----------------|----------------------|-------------------|--------|
| `Equal` | `fp32.ElemEqual` | `generics.ElemEqualStrided[T]` | ⏳ **MIGRATE** |
| `GreaterThan` | `fp32.ElemGreaterThan` | `generics.ElemGreaterThanStrided[T]` | ⏳ **MIGRATE** |
| `Less` | `fp32.ElemLess` | `generics.ElemLessStrided[T]` | ⏳ **MIGRATE** |

**Note**: All comparison operations work for all numeric types. Generic versions return numeric 0/1 which is compatible.

**Migration Strategy**:
- Migrate all comparison operations to generics
- Add type dispatch based on tensor DataType

---

### 5. Ternary Operations

#### ✅ **CAN MIGRATE TO GENERICS**

| Tensor Operation | Current FP32 Primitive | Generic Equivalent | Status |
|-----------------|----------------------|-------------------|--------|
| `Where` | `fp32.ElemWhere` | `generics.ElemWhere[T]` | ⏳ **MIGRATE** |

**Migration Strategy**:
- Migrate `Where` to generics
- Add type dispatch based on tensor DataType

---

### 6. Reduction Operations

#### ❌ **MUST REMAIN FP32-SPECIFIC**

| Tensor Operation | Current FP32 Primitive | Reason |
|-----------------|----------------------|--------|
| `Sum` | `fp32.Asum` (contiguous), `fp32.ReduceSum` (strided) | BLAS-optimized (contiguous), specialized reduction algorithm |
| `Mean` | `fp32.ReduceMean` | Specialized reduction algorithm with division |
| `Max` | `fp32.ReduceMax` | Specialized reduction algorithm |
| `Min` | `fp32.ReduceMin` | Specialized reduction algorithm |
| `ArgMax` | `fp32.Iamax` (contiguous), `fp32.Argmax` (strided) | BLAS-optimized (contiguous), specialized reduction algorithm |

**Rationale**: Reduction operations are specialized algorithms that handle multi-dimensional reductions efficiently. They are not simple element-wise operations and require careful handling of axes and strides.

---

### 7. Linear Algebra Operations

#### ❌ **MUST REMAIN FP32-SPECIFIC**

| Tensor Operation | Current FP32 Primitive | Reason |
|-----------------|----------------------|--------|
| `MatMul` | `fp32.Gemm_NN`, `fp32.GemmBatched`, `fp32.GemmStrided` | BLAS-optimized matrix multiplication |
| `Dot` | `fp32.Dot` | BLAS-optimized dot product |
| `Norm` (L1) | `fp32.Asum` | BLAS-optimized L1 norm |
| `Norm` (L2) | `fp32.Nrm2` | BLAS-optimized L2 norm |
| `Normalize` | `fp32.Nrm2` + `fp32.Scal` | Uses BLAS operations |
| `AddScaled` | `fp32.Axpy` | BLAS-optimized scaled addition |
| `Transpose` / `Permute` | `fp32.ElemCopy` (with stride manipulation) | Can use generic `ElemCopyStrided` but keep as-is for now (complex stride logic) |

**Rationale**: All linear algebra operations use BLAS primitives which are highly optimized for performance. These should remain fp32-specific.

**Note**: `Transpose`/`Permute` use `fp32.ElemCopy` but could potentially use `generics.ElemCopyStrided`. However, the stride manipulation logic is complex and keeping it as-is is safer.

---

### 8. Convolution Operations

#### ❌ **MUST REMAIN FP32-SPECIFIC**

| Tensor Operation | Current FP32 Primitive | Reason |
|-----------------|----------------------|--------|
| `Conv1D` | `fp32.Conv2D` (via reshape) | Specialized convolution algorithm |
| `Conv2D` | `fp32.Conv2D` | Specialized convolution algorithm (Im2Col + GEMM) |
| `Conv2DTransposed` | `fp32.Conv2DTransposed` | Specialized transposed convolution algorithm |
| `Conv3D` | `fp32.Conv3D` | Specialized 3D convolution algorithm |
| `Conv2DKernelGrad` | `fp32.Conv2DKernelGrad` | Specialized gradient computation |
| `Conv1DKernelGrad` | `fp32.Conv1DKernelGrad` | Specialized gradient computation |
| `DepthwiseConv2D` | `fp32.DepthwiseConv2D` | Specialized depthwise convolution |
| `GroupConv2D` | `fp32.GroupConv2D` | Specialized grouped convolution |
| `DilatedConv2D` | `fp32.DilatedConv2D` | Specialized dilated convolution |

**Rationale**: Convolution operations are highly specialized algorithms that use Im2Col transformations and GEMM operations. They are not simple element-wise operations.

---

### 9. Pooling Operations

#### ❌ **MUST REMAIN FP32-SPECIFIC**

| Tensor Operation | Current FP32 Primitive | Reason |
|-----------------|----------------------|--------|
| `MaxPool2D` | `fp32.MaxPool2D` | Specialized pooling algorithm |
| `MaxPool2DWithIndices` | `fp32.MaxPool2DWithIndices` | Specialized pooling with index tracking |
| `MaxPool2DBackward` | `fp32.MaxPool2DBackward` | Specialized backward pass |
| `AvgPool2D` | `fp32.AvgPool2D` | Specialized pooling algorithm |
| `AvgPool2DBackward` | `fp32.AvgPool2DBackward` | Specialized backward pass |
| `GlobalAvgPool2D` | `fp32.GlobalAvgPool2D` | Specialized global pooling |
| `AdaptiveAvgPool2D` | `fp32.AdaptiveAvgPool2D` | Specialized adaptive pooling |

**Rationale**: Pooling operations are specialized algorithms that require careful handling of kernel windows, strides, and padding. They are not simple element-wise operations.

---

### 10. Image/Column Conversion

#### ❌ **MUST REMAIN FP32-SPECIFIC**

| Tensor Operation | Current FP32 Primitive | Reason |
|-----------------|----------------------|--------|
| `Im2Col` | `fp32.Im2Col` | Specialized image-to-column transformation |
| `Col2Im` | `fp32.Col2Im` | Specialized column-to-image transformation |

**Rationale**: These are specialized transformations for convolution optimization. They are not simple element-wise operations.

---

### 11. Activation Functions

#### ❌ **MUST REMAIN FP32-SPECIFIC**

| Tensor Operation | Current FP32 Primitive | Reason |
|-----------------|----------------------|--------|
| `ReLU` | `fp32.ReLU` | Simple but may have optimizations |
| `Sigmoid` | `fp32.Sigmoid` | Requires float32 math (`math32.Exp`) |
| `Tanh` | `fp32.Tanh` | Requires float32 math (`math32.Tanh`) |
| `Softmax` | `fp32.Softmax1D`, `fp32.Softmax2DRows`, `fp32.Softmax2DCols` | Specialized normalization algorithm |

**Rationale**: Activation functions use float32 math libraries and specialized algorithms. `ReLU` could potentially be generic, but keeping it as-is maintains consistency.

---

### 12. Utility Operations

#### ✅ **CAN MIGRATE TO GENERICS** (Partial)

| Tensor Operation | Current Implementation | Generic Equivalent | Status |
|-----------------|----------------------|-------------------|--------|
| `Copy` (data copy) | `fp32.Copy` (contiguous), `fp32.ElemCopy` (strided) | `generics.Copy[T]` (contiguous), `generics.ElemCopyStrided[T]` (strided) | ⏳ **MIGRATE** |

**Note**: `copyTensorData` function uses `fp32.Copy` for contiguous float32 tensors and `primitive.CopyWithConversion` for type conversion. This should remain as-is for type conversion, but can use generics for same-type copies.

**Migration Strategy**:
- Keep type conversion logic using `primitive.CopyWithConversion`
- Use generics for same-type copies (both contiguous and strided)

---

## Migration Implementation Plan

### Phase 1: Simple Operations (Low Risk)

**Target**: Operations that have direct generic equivalents

1. **Copy Operations**
   - Migrate `fp32.ElemCopy` → `generics.ElemCopyStrided[T]`
   - Migrate `fp32.Copy` → `generics.Copy[T]` (for same-type copies)
   - Files: `tensor_math.go`, `tensor_linalg.go`, `tensor_conv.go`

2. **Sign and Negative**
   - Migrate `fp32.ElemSign` → `generics.ElemSignStrided[T]`
   - Migrate `fp32.ElemNegative` → `generics.ElemNegativeStrided[T]`
   - File: `tensor_math.go`

3. **Fill**
   - Migrate `fp32.Fill` → `generics.ElemFillStrided[T]`
   - File: `tensor_math.go`

**Estimated Effort**: 2-3 days  
**Risk**: Low  
**Testing**: Unit tests for each operation with different data types

---

### Phase 2: Binary Operations (Medium Risk)

**Target**: Element-wise binary operations with strided support

1. **Add, Sub, Mul, Div (strided only)**
   - Migrate strided versions to `generics.ElemApplyBinaryStrided[T]`
   - Keep contiguous fast paths using `fp32.Axpy` (BLAS-optimized)
   - Add type dispatch based on `t.DataType()`
   - File: `tensor_math.go`

2. **Scale (strided only)**
   - Migrate strided version to `generics.ElemApplyUnaryScalarStrided[T]`
   - Keep contiguous fast path using `fp32.Scal` (BLAS-optimized)
   - File: `tensor_math.go`

**Estimated Effort**: 3-4 days  
**Risk**: Medium (need to ensure type dispatch works correctly)  
**Testing**: Unit tests with different data types, performance benchmarks

---

### Phase 3: Comparison Operations (Low Risk)

**Target**: All comparison operations

1. **Equal, GreaterThan, Less**
   - Migrate to `generics.ElemEqualStrided[T]`, `generics.ElemGreaterThanStrided[T]`, `generics.ElemLessStrided[T]`
   - Add type dispatch based on `t.DataType()`
   - File: `tensor_math.go`

**Estimated Effort**: 1-2 days  
**Risk**: Low  
**Testing**: Unit tests with different data types

---

### Phase 4: Ternary Operations (Low Risk)

**Target**: Where operation

1. **Where**
   - Migrate to `generics.ElemWhere[T]`
   - Add type dispatch based on `t.DataType()`
   - File: `tensor_math.go`

**Estimated Effort**: 1 day  
**Risk**: Low  
**Testing**: Unit tests with different data types

---

## Implementation Details

### Type Dispatch Pattern

For operations that need to work with multiple types, use a type dispatch pattern:

```go
func (t Tensor) Add(other types.Tensor) types.Tensor {
    // ... validation ...
    
    dtype := t.DataType()
    switch dtype {
    case types.FP32:
        // Use fp32.Axpy for contiguous (fast path)
        // Use generics for strided
        return t.addFP32(other)
    case types.FP64:
        return t.addGeneric[float64](other)
    case types.INT64:
        return t.addGeneric[int64](other)
    // ... other types ...
    default:
        panic(fmt.Sprintf("unsupported dtype: %v", dtype))
    }
}

func (t Tensor) addGeneric[T types.Numeric](other types.Tensor) types.Tensor {
    // Use generics.ElemApplyBinaryStrided[T]
}
```

### Contiguous Fast Paths

Keep BLAS-optimized fast paths for contiguous tensors:

```go
if t.isContiguous() && isTensorContiguous(other) {
    // Use BLAS-optimized fp32.Axpy for float32
    if t.DataType() == types.FP32 {
        fp32.Axpy(...)
        return t
    }
    // For other types, use generics
}
// Strided case: use generics
```

---

## Testing Strategy

### Unit Tests
- Test each migrated operation with different data types (float32, float64, int32, int64)
- Test contiguous and strided cases
- Test edge cases (empty tensors, single element, etc.)

### Performance Benchmarks
- Compare performance of generic vs fp32 implementations
- Ensure no significant performance regression
- Benchmark with different tensor sizes and shapes

### Integration Tests
- Test tensor operations in realistic scenarios
- Test with different data types in the same computation graph

---

## Rollback Plan

If issues arise during migration:

1. **Keep fp32 implementations**: Don't remove fp32 code immediately
2. **Feature flag**: Use build tags or feature flags to switch between implementations
3. **Gradual rollout**: Migrate one operation at a time, test thoroughly before moving to next
4. **Performance monitoring**: Monitor performance metrics to catch regressions early

---

## Summary

### Operations to Migrate (✅ Generic-Compatible)

**Element-wise Operations:**
- `Copy` (strided) → `generics.ElemCopyStrided[T]`
- `Sign` → `generics.ElemSignStrided[T]`
- `Negative` → `generics.ElemNegativeStrided[T]`
- `Fill` → `generics.ElemFillStrided[T]`
- `Add`, `Sub`, `Mul`, `Div` (strided) → `generics.ElemApplyBinaryStrided[T]`
- `Scale` (strided) → `generics.ElemApplyUnaryScalarStrided[T]`

**Comparison Operations:**
- `Equal` → `generics.ElemEqualStrided[T]`
- `GreaterThan` → `generics.ElemGreaterThanStrided[T]`
- `Less` → `generics.ElemLessStrided[T]`

**Ternary Operations:**
- `Where` → `generics.ElemWhere[T]`

**Total: 13 operations** can be migrated to generics

---

### Operations to Keep FP32-Specific (❌ Algorithm-Specific)

**Math Library Operations:**
- `Square`, `Sqrt`, `Exp`, `Log`, `Pow`, `Abs`, `Cos`, `Sin`

**BLAS Operations:**
- `MatMul`, `Dot`, `Norm`, `Normalize`, `AddScaled`
- Contiguous fast paths for `Add`, `Sub`, `Scale` (using `Axpy`, `Scal`)

**Reduction Operations:**
- `Sum`, `Mean`, `Max`, `Min`, `ArgMax`

**Specialized Algorithms:**
- All convolution operations (Conv1D, Conv2D, Conv3D, etc.)
- All pooling operations (MaxPool2D, AvgPool2D, etc.)
- Image/column conversion (Im2Col, Col2Im)
- Activation functions (ReLU, Sigmoid, Tanh, Softmax)

**Total: 40+ operations** must remain fp32-specific

---

## Final Lists

### FP32-Specific Algorithms (Must Remain FP32)

1. **BLAS Operations** (Performance-critical, optimized)
   - `Axpy`, `Scal`, `Dot`, `Nrm2`, `Asum`, `Iamax`
   - `Gemm_NN`, `Gemm_NT`, `Gemm_TN`, `Gemm_TT`
   - `GemmBatched`, `GemmStrided`
   - `Gemv_N`, `Gemv_T`, `Ger`, `Symv`, `Trmv`

2. **Math Library Operations** (Require float32 math)
   - `ElemSquare`, `ElemSqrt`, `ElemExp`, `ElemLog`, `ElemPow`
   - `ElemAbs`, `ElemCos`, `ElemSin`

3. **Reduction Operations** (Specialized algorithms)
   - `ReduceSum`, `ReduceMean`, `ReduceMax`, `ReduceMin`
   - `Argmax`, `Argmin`

4. **Convolution Operations** (Specialized algorithms)
   - `Conv1D`, `Conv2D`, `Conv3D`
   - `Conv2DTransposed`, `Conv3DTransposed`
   - `Conv2DKernelGrad`, `Conv1DKernelGrad`
   - `DepthwiseConv2D`, `GroupConv2D`, `DilatedConv2D`
   - `SeparableConv2D`

5. **Pooling Operations** (Specialized algorithms)
   - `MaxPool1D`, `MaxPool2D`, `MaxPool3D`
   - `MaxPool2DWithIndices`, `MaxPool2DBackward`
   - `AvgPool1D`, `AvgPool2D`, `AvgPool3D`
   - `AvgPool2DBackward`
   - `GlobalMaxPool1D`, `GlobalMaxPool2D`, `GlobalMaxPool3D`
   - `GlobalAvgPool2D`
   - `AdaptiveMaxPool1D`, `AdaptiveMaxPool2D`, `AdaptiveMaxPool3D`
   - `AdaptiveAvgPool2D`

6. **Image/Column Conversion** (Specialized transformations)
   - `Im2Col`, `Col2Im`

7. **Activation Functions** (Specialized algorithms)
   - `ReLU`, `ReLUGrad`, `ReLUGradStride`
   - `Sigmoid`, `SigmoidGrad`, `SigmoidGradStride`
   - `Tanh`, `TanhGrad`, `TanhGradStride`
   - `Softmax1D`, `Softmax2DRows`, `Softmax2DCols`
   - `Softmax1DGrad`, `Softmax2DRowsGrad`, `Softmax2DColsGrad`

8. **Linear Algebra Operations** (LAPACK-style)
   - `Getrf`, `Getrf_IP`, `Getri`
   - `Geqrf`, `Orgqr`
   - `Gepseu`, `Gesvd`
   - `Gnnls`
   - Givens and Householder transformations

9. **Utility Operations** (Type conversion)
   - `CopyWithConversion` (handles type conversion between different types)

---

### Generic-Compatible Operations (Can Use Generics)

1. **Copy Operations**
   - `ElemCopy` / `Copy` (strided and contiguous versions)
   - `ElemSwap` / `Swap` (strided and contiguous versions)
   - `ElemConvert` / `Convert` (type conversion between compatible types)

2. **Unary Operations**
   - `ElemSign` / `Sign`
   - `ElemNegative` / `Negative`

3. **Binary Operations** (via Apply functions)
   - `Add`, `Sub`, `Mul`, `Div` (strided versions)
   - Can use `ElemApplyBinaryStrided[T]` with operation functions

4. **Scalar Operations**
   - `ElemFill` / `Fill`
   - `Scale` (strided version, via `ElemApplyUnaryScalarStrided`)

5. **Comparison Operations**
   - `ElemEqual` / `Equal`
   - `ElemGreaterThan` / `GreaterThan`
   - `ElemLess` / `Less`
   - `ElemNotEqual` / `NotEqual`
   - `ElemLessEqual` / `LessEqual`
   - `ElemGreaterEqual` / `GreaterEqual`

6. **Ternary Operations**
   - `ElemWhere` / `Where`

7. **Apply Operations** (Generic framework)
   - `ElemApplyBinary`, `ElemApplyUnary`, `ElemApplyTernary`
   - `ElemApplyUnaryScalar`, `ElemApplyBinaryScalar`, `ElemApplyTernaryScalar`
   - All strided and contiguous versions

---

## Conclusion

The migration plan identifies **13 tensor operations** that can be migrated to generics, while **40+ operations** must remain fp32-specific due to:

1. **Performance requirements**: BLAS operations are highly optimized
2. **Algorithm specialization**: Convolutions, pooling, reductions require specialized implementations
3. **Math library dependencies**: Operations requiring float32 math libraries
4. **Type-specific behavior**: Some operations have type-specific optimizations

The migration should be done gradually, starting with low-risk operations (Copy, Sign, Negative, Fill) and progressing to more complex operations (binary operations, comparisons). Each phase should be thoroughly tested before proceeding to the next.

