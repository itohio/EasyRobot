# Generic Operations Migration Plan for FP32

**Date**: Generated from code analysis  
**Purpose**: Plan for replacing manual loops and own implementations with generic operations from `primitive/generics`  
**Performance Principle**: If existing operations already use `applyElem*` helper functions, migrate them to generics since generics are highly optimized. For direct loop implementations, only migrate if generics provide clear benefits.

---

## Executive Summary

This document identifies where and how to use generic operations from `primitive/generics` instead of manual loops or custom implementations in the `fp32` package. The migration should prioritize:

1. **Direct generic ops** (e.g., `ElemCopy`, `ElemSign`) that replace manual implementations
2. **Helper-based functions** that already use `applyElem*` helpers should be migrated to generics since generics are highly optimized and will improve performance
3. **BLAS operations** and specialized algorithms should remain as-is (performance-critical or specialized)
4. **Code reuse** and type safety benefits from using generics

**Key Update**: If existing operations in fp32 already use `applyElem*` helper loopers, they can definitely be migrated to generics since generics are highly optimized and will provide better performance than the current helper-based approach.

---

## Decision Framework

### When to Use Generic Ops

✅ **USE generic ops when:**
- Operation is simple and matches a generic op exactly (e.g., `ElemCopy`, `ElemSign`, `ElemNegative`)
- **Operation already uses `applyElem*` helper functions** (generics are highly optimized and will improve performance)
- Code duplication can be reduced significantly
- Operation needs to work with multiple types (future-proofing)

❌ **DO NOT use generic ops when:**
- Operation is in a hot path with direct loop implementation (no helper overhead)
- Operation has special optimizations (e.g., BLAS operations)
- Operation requires type-specific behavior that cannot be expressed generically
- Operation is a specialized algorithm (e.g., reductions, convolutions)

### Key Principle

**If an operation already uses `applyElemBinary`, `applyElemUnary`, `applyElemTernary`, or `applyElemUnaryScalar` helpers, it should be migrated to generics since generics are highly optimized and will provide better performance than the current helper-based approach.**

---

## File-by-File Analysis

### 1. `level1.go` - BLAS Level 1 Operations

**Status**: ✅ **KEEP AS-IS** (BLAS operations, performance-critical)

**Functions**:
- `Axpy`, `Dot`, `Nrm2`, `Asum`, `Scal`, `Copy`, `Swap`, `Iamax`

**Rationale**:
- These are BLAS-standard operations optimized for performance
- Generic equivalents exist (`Copy`, `Swap` in `generics/blas.go`) but BLAS versions are specialized
- These are hot paths and should remain direct implementations

**Recommendation**: No changes needed.

---

### 2. `level2.go` - BLAS Level 2 Operations

**Status**: ✅ **KEEP AS-IS** (BLAS operations, performance-critical)

**Functions**:
- `Gemv_N`, `Gemv_T`, `Ger`, `Symv`, `Trmv`

**Rationale**:
- BLAS matrix-vector operations are highly optimized
- Complex operations with specific memory access patterns
- Hot paths that should remain direct implementations

**Recommendation**: No changes needed.

---

### 3. `level3.go` - BLAS Level 3 Operations

**Status**: ✅ **KEEP AS-IS** (BLAS operations, performance-critical)

**Functions**:
- `Gemm_NN`, `Gemm_NT`, `Gemm_TN`, `Gemm_TT`, `Syrk`, `Trmm`

**Rationale**:
- BLAS matrix-matrix operations are the most performance-critical
- Highly optimized for cache efficiency
- Hot paths that should remain direct implementations

**Recommendation**: No changes needed.

---

### 4. `array.go` - Array Utility Functions

**Status**: 🔄 **PARTIAL MIGRATION** (Some can use generics, some should stay)

#### Functions to Migrate:

**`Sum`** (lines 13-27)
- **Current**: Manual loop with stride
- **Generic Op**: Could use `generics` reduction, but this is a simple sum
- **Recommendation**: ✅ **KEEP AS-IS** - Simple loop, likely inlined, no benefit from generic

**`SqrSum`** (lines 29-46)
- **Current**: Manual loop with stride
- **Generic Op**: Could use `generics` reduction, but this is a simple sum of squares
- **Recommendation**: ✅ **KEEP AS-IS** - Simple loop, likely inlined, no benefit from generic

**`StatsArr`** (lines 48-104)
- **Current**: Complex one-pass algorithm with multiple accumulators
- **Generic Op**: Too complex for generic ops
- **Recommendation**: ✅ **KEEP AS-IS** - Specialized algorithm

**`PercentileArr`** (lines 106-159)
- **Current**: Requires sorting, complex logic
- **Generic Op**: Not applicable
- **Recommendation**: ✅ **KEEP AS-IS** - Specialized algorithm

**`DiffArrInPlace`** (lines 161-171)
- **Current**: Simple loop: `dst[i] -= c`
- **Generic Op**: Could use `ElemSubScalar` but in-place
- **Recommendation**: ✅ **KEEP AS-IS** - Simple loop, in-place operation

**`DiffArrScalar`** (lines 173-188)
- **Current**: Simple loop with strides: `dst[pd] = src[ps] - c`
- **Generic Op**: Could use `ElemSubScalarStrided` from generics
- **Recommendation**: 🔄 **CONSIDER MIGRATION** - If not in hot path, could use `generics.ElemSubScalarStrided` (but note: this doesn't exist yet, would need to use `ElemApplyUnaryScalarStrided` which we should avoid per user instruction)
- **Decision**: ✅ **KEEP AS-IS** - Simple loop, avoid `ElemApply` overhead

**Recommendation**: Keep all functions as-is. They are either too simple to benefit from generics or too specialized.

---

### 5. `vector.go` - Vector Operations

**Status**: 🔄 **PARTIAL MIGRATION** (Some can use generics)

#### Functions to Migrate:

**`HadamardProductAdd`** (lines 3-20)
- **Current**: Simple loop: `dst[pd] += a[pa] * b[pb]`
- **Generic Op**: Could use `ElemApplyBinaryStrided` but that's `ElemApply` (avoid per user instruction)
- **Recommendation**: ✅ **KEEP AS-IS** - Simple loop, avoid `ElemApply` overhead

**`DotProduct2D`** (lines 22-42)
- **Current**: Specialized 2D dot product
- **Generic Op**: Not applicable (specialized operation)
- **Recommendation**: ✅ **KEEP AS-IS** - Specialized operation

**`NormalizeVecInPlace`** (lines 44-58)
- **Current**: Uses `Nrm2` + `Scal` (BLAS operations)
- **Generic Op**: Not applicable (uses BLAS)
- **Recommendation**: ✅ **KEEP AS-IS** - Uses optimized BLAS operations

**`SumArrInPlace`** (lines 60-71)
- **Current**: Simple loop: `dst[i] += c`
- **Generic Op**: Could use `ElemAddScalar` but in-place
- **Recommendation**: ✅ **KEEP AS-IS** - Simple loop, in-place operation

**`HadamardProduct`** (lines 73-90)
- **Current**: Simple loop with strides: `dst[pd] = a[pa] * b[pb]`
- **Generic Op**: Could use `ElemMulStrided` from generics (but doesn't exist, would need `ElemApplyBinaryStrided`)
- **Recommendation**: ✅ **KEEP AS-IS** - Simple loop, avoid `ElemApply` overhead

**`NormalizeVec`** (lines 92-109)
- **Current**: Uses `Nrm2` + `Copy` + `Scal` (BLAS operations)
- **Generic Op**: Not applicable (uses BLAS)
- **Recommendation**: ✅ **KEEP AS-IS** - Uses optimized BLAS operations

**`SumArrScalar`** (lines 111-126)
- **Current**: Simple loop with strides: `dst[pd] = src[ps] + c`
- **Generic Op**: Could use `ElemAddScalarStrided` from generics (but doesn't exist, would need `ElemApplyUnaryScalarStrided`)
- **Recommendation**: ✅ **KEEP AS-IS** - Simple loop, avoid `ElemApply` overhead

**Recommendation**: Keep all functions as-is. They are simple loops that would require `ElemApply` variants, which we should avoid per user instruction.

---

### 6. `batched.go` - Batched BLAS Operations

**Status**: ✅ **KEEP AS-IS** (BLAS operations, performance-critical)

**Functions**:
- `GemmBatched`, `GemmStrided`, `GemvBatched`

**Rationale**:
- BLAS batched operations are performance-critical
- These are wrappers around BLAS Level 2/3 operations
- Hot paths that should remain direct implementations

**Recommendation**: No changes needed.

---

### 7. `tensor_elementwise.go` - Element-wise Tensor Operations

**Status**: 🔄 **ALREADY USES HELPERS** (Some can migrate to generics)

#### Current Implementation Pattern:

Most functions already use helper functions (`applyElemBinary`, `applyElemUnary`, `applyElemTernary`, `applyElemUnaryScalar`). These helpers are similar to generic ops but are float32-specific.

#### Functions to Migrate and Deprecate (Thin Wrappers):

These functions are thin wrappers that just pass all parameters directly to generics. They should be migrated and then deprecated, with callers using generics directly.

**`ElemCopy`** (lines 97-122)
- **Current**: Uses manual stride iteration
- **Generic Op**: ✅ **MIGRATE** to `generics.ElemCopyStrided[float32]`
- **Deprecation**: ⚠️ **DEPRECATE** - Thin wrapper, callers should use `generics.ElemCopyStrided[float32]` directly
- **Recommendation**: Migrate implementation, then deprecate function

**`ElemFill`** (lines 267-293)
- **Current**: Manual stride iteration
- **Generic Op**: ✅ **MIGRATE** to `generics.ElemFillStrided[float32]`
- **Deprecation**: ⚠️ **DEPRECATE** - Thin wrapper, callers should use `generics.ElemFillStrided[float32]` directly
- **Recommendation**: Migrate implementation, then deprecate function

**`ElemSign`** (lines 226-237)
- **Current**: Uses `applyElemUnary`
- **Generic Op**: ✅ **MIGRATE** to `generics.ElemSignStrided[float32]`
- **Deprecation**: ⚠️ **DEPRECATE** - Thin wrapper, callers should use `generics.ElemSignStrided[float32]` directly
- **Recommendation**: Migrate implementation, then deprecate function

**`ElemNegative`** (lines 260-265)
- **Current**: Uses `applyElemUnary`
- **Generic Op**: ✅ **MIGRATE** to `generics.ElemNegativeStrided[float32]`
- **Deprecation**: ⚠️ **DEPRECATE** - Thin wrapper, callers should use `generics.ElemNegativeStrided[float32]` directly
- **Recommendation**: Migrate implementation, then deprecate function

**`ElemWhere`** (lines 124-133)
- **Current**: Uses `applyElemTernary`
- **Generic Op**: ✅ **MIGRATE** to `generics.ElemWhere[float32]`
- **Deprecation**: ⚠️ **DEPRECATE** - Thin wrapper, callers should use `generics.ElemWhere[float32]` directly
- **Recommendation**: Migrate implementation, then deprecate function

**Comparison Operations** (lines 135-379):
- `ElemGreaterThan`, `ElemEqual`, `ElemLess`, `ElemNotEqual`, `ElemLessEqual`, `ElemGreaterEqual`
- **Current**: Uses `applyElemBinary`
- **Generic Op**: ✅ **MIGRATE** to `generics.ElemGreaterThanStrided[float32]`, etc.
- **Deprecation**: ⚠️ **DEPRECATE** - Thin wrappers, callers should use `generics.Elem*Strided[float32]` directly
- **Recommendation**: Migrate implementation, then deprecate functions
- **Note**: According to `OPS.md`, comparison ops "roll out their own loops for optimization (not using `ElemApplyBinary`)", so these are already optimized in generics

#### Functions to Migrate (Keep - Have Operation Functions):

These functions pass operation functions to generics, so they are NOT thin wrappers and should be kept.

**`ElemAdd`**, **`ElemSub`**, **`ElemMul`** (lines 9-28)
- **Current**: Uses `applyElemBinary`
- **Generic Op**: ✅ **MIGRATE** to `generics.ElemApplyBinaryStrided[float32]` with appropriate operation function
- **Deprecation**: ✅ **KEEP** - Not a thin wrapper, has operation function logic
- **Recommendation**: Migrate implementation, keep function (provides convenient API with operation function)

**`ElemDiv`** (lines 30-67)
- **Current**: Manual implementation with zero-division check
- **Generic Op**: Not applicable (requires special zero-division handling)
- **Recommendation**: ✅ **KEEP AS-IS** - Specialized behavior

**`ElemScaleInPlace`** (lines 69-95)
- **Current**: Uses `Scal` (BLAS) for contiguous, manual for strided
- **Generic Op**: Could use `generics.ElemApplyUnaryScalarStrided` but in-place
- **Recommendation**: ✅ **KEEP AS-IS** - Already optimized (uses BLAS for contiguous)

**`ElemSquare`**, **`ElemSqrt`**, **`ElemExp`**, **`ElemLog`**, **`ElemPow`**, **`ElemAbs`**, **`ElemCos`**, **`ElemSin`**, **`ElemTanh`** (lines 165-258)
- **Current**: Uses `applyElemUnary` with type-specific math functions
- **Generic Op**: ✅ **MIGRATE** to `generics.ElemApplyUnaryStrided[float32]` with appropriate math function
- **Deprecation**: ✅ **KEEP** - Not thin wrappers, have math function logic
- **Recommendation**: Migrate implementation, keep functions (provide convenient API with math functions)

**`ElemAddScalar`**, **`ElemSubScalar`**, **`ElemScale`** (lines 295-314)
- **Current**: Uses `applyElemUnaryScalar`
- **Generic Op**: ✅ **MIGRATE** to `generics.ElemApplyUnaryScalarStrided[float32]` with appropriate operation
- **Deprecation**: ✅ **KEEP** - Not thin wrappers, have operation function logic
- **Recommendation**: Migrate implementation, keep functions (provide convenient API with operation functions)

**`ElemDivScalar`** (lines 316-349)
- **Current**: Manual implementation with zero-division check
- **Generic Op**: Not applicable (requires special zero-division handling)
- **Recommendation**: ✅ **KEEP AS-IS** - Specialized behavior

**`ElemAddScaledMul`**, **`ElemAddScaledSquareMul`** (lines 381-393)
- **Current**: Uses `applyElemUnaryScalar`
- **Generic Op**: ✅ **MIGRATE** to `generics.ElemApplyUnaryScalarStrided[float32]` with appropriate operation
- **Deprecation**: ✅ **KEEP** - Not thin wrappers, have operation function logic
- **Recommendation**: Migrate implementation, keep functions (provide convenient API with operation functions)

**Recommendation Summary for `tensor_elementwise.go`**:
- ⚠️ **MIGRATE & DEPRECATE** (thin wrappers): `ElemCopy`, `ElemFill`, `ElemSign`, `ElemNegative`, `ElemWhere`, all comparison ops (6 functions)
- ✅ **MIGRATE & KEEP** (have operation functions): `ElemAdd`, `ElemSub`, `ElemMul`, all unary math ops (`ElemSquare`, `ElemSqrt`, `ElemExp`, etc.), scalar ops (`ElemAddScalar`, `ElemSubScalar`, `ElemScale`, `ElemAddScaledMul`, `ElemAddScaledSquareMul`)
- ✅ **KEEP AS-IS**: `ElemDiv`, `ElemDivScalar` (zero-division handling), `ElemScaleInPlace` (uses BLAS)

---

### 8. `tensor_reduction.go` - Reduction Operations

**Status**: ✅ **KEEP AS-IS** (Specialized algorithms)

**Functions**:
- `ReduceSum`, `ReduceMean`, `ReduceMax`, `ReduceMin`, `Argmax`, `Argmin`

**Rationale**:
- Complex reduction algorithms with axis handling
- Specialized logic for different reduction types
- `HOT_PATH_INEFFICIENCIES.md` notes allocations but these are algorithmic requirements
- Not suitable for generic element-wise operations

**Recommendation**: No changes needed. These are specialized reduction algorithms, not element-wise operations.

---

### 9. `activations.go` - Activation Functions

**Status**: 🔄 **PARTIAL MIGRATION** (Some can use generics for stride versions)

#### Functions to Migrate:

**`ReLU`** (lines 9-23)
- **Current**: Simple loop for contiguous arrays
- **Generic Op**: Could use `generics.ElemVecApplyUnaryStrided` but that's `ElemApply` (avoid)
- **Recommendation**: ✅ **KEEP AS-IS** - Simple loop, avoid `ElemApply` overhead

**`ReLUGrad`** (lines 25-39)
- **Current**: Simple loop for contiguous arrays
- **Generic Op**: Could use `generics.ElemVecApplyTernaryStrided` but that's `ElemApply` (avoid)
- **Recommendation**: ✅ **KEEP AS-IS** - Simple loop, avoid `ElemApply` overhead

**`Sigmoid`** (lines 41-58)
- **Current**: Simple loop with overflow checks
- **Generic Op**: Would need `ElemApplyUnaryStrided` (which we should avoid)
- **Recommendation**: ✅ **KEEP AS-IS** - Type-specific math, avoid `ElemApply` overhead

**`SigmoidGrad`** (lines 60-70)
- **Current**: Simple loop
- **Generic Op**: Would need `ElemApplyBinaryStrided` (which we should avoid)
- **Recommendation**: ✅ **KEEP AS-IS** - Simple loop, avoid `ElemApply` overhead

**`Tanh`** (lines 72-83)
- **Current**: Simple loop
- **Generic Op**: Would need `ElemApplyUnaryStrided` (which we should avoid)
- **Recommendation**: ✅ **KEEP AS-IS** - Type-specific math, avoid `ElemApply` overhead

**`TanhGrad`** (lines 85-95)
- **Current**: Simple loop
- **Generic Op**: Would need `ElemApplyBinaryStrided` (which we should avoid)
- **Recommendation**: ✅ **KEEP AS-IS** - Simple loop, avoid `ElemApply` overhead

**`Softmax1D`**, **`Softmax2DRows`**, **`Softmax2DCols`** (lines 97-207)
- **Current**: Complex algorithms with max-finding, exp, normalization
- **Generic Op**: Not applicable (complex multi-step algorithms)
- **Recommendation**: ✅ **KEEP AS-IS** - Specialized algorithms

**`Softmax1DGrad`**, **`Softmax2DRowsGrad`**, **`Softmax2DColsGrad`** (lines 209-274)
- **Current**: Complex gradient algorithms
- **Generic Op**: Not applicable (complex multi-step algorithms)
- **Recommendation**: ✅ **KEEP AS-IS** - Specialized algorithms

**`ReLUGradStride`**, **`SigmoidGradStride`**, **`TanhGradStride`** (lines 276-391)
- **Current**: Stride-aware versions with contiguous fast path
- **Generic Op**: Could use `generics.ElemVecApplyTernaryStrided` but that's `ElemApply` (avoid)
- **Recommendation**: ✅ **KEEP AS-IS** - Already optimized with contiguous fast path, avoid `ElemApply` overhead

**Recommendation**: Keep all functions as-is. They are either simple loops (avoid `ElemApply` overhead) or complex specialized algorithms.

---

### 10. `tensor.go` - Tensor Operations

**Status**: ✅ **KEEP AS-IS** (Complex tensor operations)

**Functions**:
- `Im2Col`, `Col2Im`, `Conv2D`, and other convolution operations

**Rationale**:
- Complex tensor transformations and convolution operations
- Specialized algorithms for neural network operations
- Not suitable for generic element-wise operations

**Recommendation**: No changes needed.

---

### 11. `tensor_helpers.go` - Helper Functions

**Status**: ✅ **KEEP AS-IS** (Utility functions)

**Functions**:
- `ComputeStrides`, `SizeFromShape`, `EnsureStrides`, `IsContiguous`, `advanceOffsets`, etc.

**Rationale**:
- These are utility functions used by other operations
- Some are already generic (work with any numeric type via slice indexing)
- `HOT_PATH_INEFFICIENCIES.md` notes optimizations needed but these are implementation details, not migration targets

**Recommendation**: No changes needed for migration. Optimize per `HOT_PATH_INEFFICIENCIES.md` separately.

---

## Migration Priority

### Priority 1: Direct Generic Ops - Migrate and Deprecate (Thin Wrappers)

These are direct generic ops that replace manual implementations. Since they are thin wrappers (just pass all params to generics), they should be deprecated after migration:

1. ⚠️ **`ElemCopy`** → `generics.ElemCopyStrided[float32]` → **DEPRECATE**
2. ⚠️ **`ElemFill`** → `generics.ElemFillStrided[float32]` → **DEPRECATE**
3. ⚠️ **`ElemSign`** → `generics.ElemSignStrided[float32]` → **DEPRECATE**
4. ⚠️ **`ElemNegative`** → `generics.ElemNegativeStrided[float32]` → **DEPRECATE**
5. ⚠️ **`ElemWhere`** → `generics.ElemWhere[float32]` → **DEPRECATE**
6. ⚠️ **Comparison ops** (6 functions) → `generics.Elem*Strided[float32]` → **DEPRECATE**

**Expected Impact**: 
- Code reuse, type safety, no performance penalty (direct generic ops)
- Deprecation reduces API surface, encourages direct use of generics
- Callers should migrate to `generics.Elem*Strided[float32]` directly

### Priority 2: Migrate Helper-Based Functions - Keep (Have Operation Functions)

These already use `applyElem*` helpers and should be migrated to generics for better performance. They are NOT thin wrappers (they pass operation functions), so they should be kept:

1. ✅ **`ElemAdd`**, **`ElemSub`**, **`ElemMul`** → `generics.ElemApplyBinaryStrided[float32]` → **KEEP**
2. ✅ **Unary math ops** (`ElemSquare`, `ElemSqrt`, `ElemExp`, `ElemLog`, `ElemPow`, `ElemAbs`, `ElemCos`, `ElemSin`, `ElemTanh`) → `generics.ElemApplyUnaryStrided[float32]` → **KEEP**
3. ✅ **Scalar ops** (`ElemAddScalar`, `ElemSubScalar`, `ElemScale`, `ElemAddScaledMul`, `ElemAddScaledSquareMul`) → `generics.ElemApplyUnaryScalarStrided[float32]` → **KEEP**

**Expected Impact**: 
- Better performance than current helper-based approach (generics are highly optimized)
- Functions provide convenient API with operation functions, so they should be kept

### Priority 3: No Migration Needed

All other functions should remain as-is because:
- They are BLAS operations (performance-critical)
- They are specialized algorithms (reductions, convolutions, activations)
- They have special handling (zero-division checks)
- They already use optimized BLAS operations

---

## Implementation Guidelines

### Migration Pattern for Thin Wrappers (Deprecate)

```go
// Before (tensor_elementwise.go)
func ElemCopy(dst, src []float32, shape []int, stridesDst, stridesSrc []int) {
    stridesDst = EnsureStrides(stridesDst, shape)
    stridesSrc = EnsureStrides(stridesSrc, shape)
    size := SizeFromShape(shape)
    if size == 0 {
        return
    }
    if IsContiguous(stridesDst, shape) && IsContiguous(stridesSrc, shape) {
        Copy(dst, src, 1, 1, size)
        return
    }
    // ... manual stride iteration
}

// After - Migrate implementation
import "github.com/.../primitive/generics"

// Deprecated: Use generics.ElemCopyStrided[float32] directly instead.
// This function is a thin wrapper that just passes all parameters to generics.
func ElemCopy(dst, src []float32, shape []int, stridesDst, stridesSrc []int) {
    generics.ElemCopyStrided[float32](dst, src, shape, stridesDst, stridesSrc)
}
```

**Note**: After migration, these functions should be marked as deprecated with a comment directing callers to use the generic function directly.

### Migration Pattern for Helper-Based Functions

```go
// Before (tensor_elementwise.go)
func ElemAdd(dst, a, b []float32, shape []int, stridesDst, stridesA, stridesB []int) {
    applyElemBinary(dst, a, b, shape, stridesDst, stridesA, stridesB, func(av, bv float32) float32 {
        return av + bv
    })
}

// After
import "github.com/.../primitive/generics"

func ElemAdd(dst, a, b []float32, shape []int, stridesDst, stridesA, stridesB []int) {
    generics.ElemApplyBinaryStrided[float32](dst, a, b, shape, stridesDst, stridesA, stridesB, func(av, bv float32) float32 {
        return av + bv
    })
}

// Example for unary operations
func ElemSquare(dst, src []float32, shape []int, stridesDst, stridesSrc []int) {
    generics.ElemApplyUnaryStrided[float32](dst, src, shape, stridesDst, stridesSrc, func(v float32) float32 {
        return v * v
    })
}

// Example for scalar operations
func ElemAddScalar(dst, src []float32, scalar float32, shape []int, stridesDst, stridesSrc []int) {
    generics.ElemApplyUnaryScalarStrided[float32](dst, src, scalar, shape, stridesDst, stridesSrc, func(v, s float32) float32 {
        return v + s
    })
}
```

---

## Testing Requirements

For each migrated function:

1. **Unit Tests**: Ensure behavior matches original implementation
2. **Performance Tests**: Benchmark to ensure no regression
3. **Type Safety**: Verify generic type constraints work correctly
4. **Edge Cases**: Test with various shapes, strides, and edge cases

---

## Summary

### Functions to Migrate to Generics (~28 functions)

#### Thin Wrappers - Migrate and Deprecate (Priority 1, ~11 functions):
These are thin wrappers that just pass all parameters to generics. They should be deprecated after migration:

1. ⚠️ `ElemCopy` → `generics.ElemCopyStrided[float32]` → **DEPRECATE**
2. ⚠️ `ElemFill` → `generics.ElemFillStrided[float32]` → **DEPRECATE**
3. ⚠️ `ElemSign` → `generics.ElemSignStrided[float32]` → **DEPRECATE**
4. ⚠️ `ElemNegative` → `generics.ElemNegativeStrided[float32]` → **DEPRECATE**
5. ⚠️ `ElemWhere` → `generics.ElemWhere[float32]` → **DEPRECATE**
6. ⚠️ `ElemGreaterThan` → `generics.ElemGreaterThanStrided[float32]` → **DEPRECATE**
7. ⚠️ `ElemEqual` → `generics.ElemEqualStrided[float32]` → **DEPRECATE**
8. ⚠️ `ElemLess` → `generics.ElemLessStrided[float32]` → **DEPRECATE**
9. ⚠️ `ElemNotEqual` → `generics.ElemNotEqualStrided[float32]` → **DEPRECATE**
10. ⚠️ `ElemLessEqual` → `generics.ElemLessEqualStrided[float32]` → **DEPRECATE**
11. ⚠️ `ElemGreaterEqual` → `generics.ElemGreaterEqualStrided[float32]` → **DEPRECATE**

#### Helper-Based Functions - Migrate and Keep (Priority 2, ~17 functions):
These have operation functions, so they are NOT thin wrappers and should be kept:

12. ✅ `ElemAdd` → `generics.ElemApplyBinaryStrided[float32]` → **KEEP**
13. ✅ `ElemSub` → `generics.ElemApplyBinaryStrided[float32]` → **KEEP**
14. ✅ `ElemMul` → `generics.ElemApplyBinaryStrided[float32]` → **KEEP**
15. ✅ `ElemSquare` → `generics.ElemApplyUnaryStrided[float32]` → **KEEP**
16. ✅ `ElemSqrt` → `generics.ElemApplyUnaryStrided[float32]` → **KEEP**
17. ✅ `ElemExp` → `generics.ElemApplyUnaryStrided[float32]` → **KEEP**
18. ✅ `ElemLog` → `generics.ElemApplyUnaryStrided[float32]` → **KEEP**
19. ✅ `ElemPow` → `generics.ElemApplyUnaryScalarStrided[float32]` → **KEEP**
20. ✅ `ElemAbs` → `generics.ElemApplyUnaryStrided[float32]` → **KEEP**
21. ✅ `ElemCos` → `generics.ElemApplyUnaryStrided[float32]` → **KEEP**
22. ✅ `ElemSin` → `generics.ElemApplyUnaryStrided[float32]` → **KEEP**
23. ✅ `ElemTanh` → `generics.ElemApplyUnaryStrided[float32]` → **KEEP**
24. ✅ `ElemAddScalar` → `generics.ElemApplyUnaryScalarStrided[float32]` → **KEEP**
25. ✅ `ElemSubScalar` → `generics.ElemApplyUnaryScalarStrided[float32]` → **KEEP**
26. ✅ `ElemScale` → `generics.ElemApplyUnaryScalarStrided[float32]` → **KEEP**
27. ✅ `ElemAddScaledMul` → `generics.ElemApplyUnaryScalarStrided[float32]` → **KEEP**
28. ✅ `ElemAddScaledSquareMul` → `generics.ElemApplyUnaryScalarStrided[float32]` → **KEEP**

**Total**: ~28 functions to migrate
- **~11 functions to deprecate** (thin wrappers: 5 direct ops + 6 comparison ops)
- **~17 functions to keep** (have operation functions: 3 binary + 9 unary math + 5 scalar)

### Functions to Keep As-Is

- All BLAS operations (Level 1, 2, 3)
- All specialized algorithms (reductions, activations, convolutions)
- Functions with special handling (`ElemDiv`, `ElemDivScalar` - zero-division checks)
- Functions already using optimized BLAS (`ElemScaleInPlace`)
- All utility/helper functions

---

## Next Steps

1. **Phase 1**: Migrate Priority 1 functions (thin wrappers: `ElemCopy`, `ElemFill`, `ElemSign`, `ElemNegative`, `ElemWhere`, comparison ops) and mark as deprecated
2. **Phase 2**: Migrate Priority 2 functions (helper-based: `ElemAdd`, `ElemSub`, `ElemMul`, unary math ops, scalar ops) and keep them (not thin wrappers)
3. **Phase 3**: Update all callers of deprecated functions to use generics directly
4. **Phase 4**: Benchmark and validate improvements (should see performance gains for helper-based functions)
5. **Phase 5**: Remove deprecated functions in next major version
6. **Phase 6**: Address `HOT_PATH_INEFFICIENCIES.md` optimizations separately (allocations in `IsContiguous`, `EnsureStrides`, etc.)

---

## References

- `primitive/generics/OPS.md` - Generic operations specification
- `primitive/fp32/HOT_PATH_INEFFICIENCIES.md` - Performance inefficiencies report
- `primitive/fp32/OPS.md` - FP32 operations reference

