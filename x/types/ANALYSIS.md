# Convert Package Analysis: Function Reduction Impact

## Executive Summary

This analysis examines the `x/convert` package to understand the impact of reducing exported functions to a minimal set and eliminating duplication:
- `Copy`, `CopyStrided` (unified from BLAS and ElemCopy versions)
- `Swap`, `SwapStrided` (unified from BLAS and ElemSwap versions)
- `Cast`, `CastStrided`, `CastClamped`, `CastClampedStrided`
- `CastFromComplex`, `CastToComplex`
- `CastStridedAny`
- `CastValue`

**Primary Goal**: Reduce duplication by unifying similar functions with different signatures or names.

## Current Exported Functions

### 1. Value Clamping Functions (4 functions)
- `ClampToInt8Value[U ClampableToInt8](v U) int8`
- `ClampToInt16Value[U ClampableToInt16](v U) int16`
- `ClampToInt32Value[U ClampableToInt32](v U) int32`
- `ClampToInt64Value[U ClampableToInt64](v U) int64`

**Usage**: 0 external usages found
- Only used internally within the convert package
- Used by `ValueConvert` and internal clamping functions

**Impact**: **LOW** - Can be made internal or removed if `CastValue` handles clamping

---

### 2. Type-Agnostic Conversion Functions (2 functions)
- `CopyWithConversion(dstData, srcData any) any`
- `CopyWithStrides(dstData, srcData any, shape []int, dstStrides, srcStrides []int)`

**Usage Count**: 7 usages
- `x/math/tensor/types/dtype.go`: 1 usage (`CopyWithConversion`)
- `x/math/tensor/types/dtype_test.go`: 1 usage (`CopyWithConversion`)
- `x/math/tensor/eager/tensor.go`: 2 usages (`CopyWithStrides`)
- `x/math/tensor/eager/conv.go`: 2 usages (`CopyWithStrides`)
- `x/math/tensor/eager/linalg.go`: 1 usage (`CopyWithStrides`)

**Impact**: **HIGH** - These are actively used and would need to be replaced with `Cast`/`CastStrided`/`CastClamped`/`CastClampedStrided`

---

### 3. Generic Type-Safe Conversion Functions (2 functions)
- `ElemConvert[T, U Numeric](dst []T, src []U, n int)`
- `ElemConvertStrided[T, U Numeric](dst []T, src []U, shape []int, stridesDst, stridesSrc []int)`

**Usage Count**: 0 external usages found
- These are wrapper functions that delegate to `st` or `mt` implementations
- Not directly called from outside the package

**Impact**: **MEDIUM** - These would become `Cast` and `CastStrided` in the new API

---

### 4. Complex Number Conversion Functions (3 functions)
- `FromComplexToT[T NumericReal](dst []T, src []complex64)`
- `FromTToComplex[T NumericReal](dst []complex64, src []T)`
- `FromUToT[T NumericReal, U NumericReal](dst []T, src []U)`

**Usage Count**: 0 external usages found
- Only used internally within the convert package

**Impact**: **MEDIUM** - Should be exported but renamed:
- `FromComplexToT` → `CastFromComplex`
- `FromTToComplex` → `CastToComplex`
- `FromUToT` → `Cast`

---

### 5. Vector/Matrix Copy Functions (2 functions)
- `ElemVecCopyStrided[T Numeric](dst, src []T, n int, strideDst, strideSrc int)`
- `ElemMatCopyStrided[T Numeric](dst, src []T, rows, cols int, ldDst, ldSrc int)`

**Usage Count**: 0 external usages found

**Impact**: **LOW** - These are specialized versions that could be replaced with `CopyStrided`

---

### 6. Element Copy Functions (6 functions)
- `ElemCopy[T Numeric](dst, src []T, n int)`
- `ElemCopyStrided[T Numeric](dst, src []T, shape []int, stridesDst, stridesSrc []int)`
- `ElemSwap[T Numeric](dst, src []T, n int)`
- `ElemSwapStrided[T Numeric](dst, src []T, shape []int, stridesDst, stridesSrc []int)`
- `ElemCopyAny(dst, src any, n int)`
- `ElemCopyStridedAny(dst, src any, shape []int, stridesDst, stridesSrc []int)`

**Usage Count**: 24 usages
- `x/math/tensor/eager/tensor.go`: 1 usage (`ElemCopyStridedAny`)
- `x/math/tensor/eager/linalg.go`: 2 usages (`ElemCopyStrided`)
- `x/math/tensor/eager/math.go`: 8 usages (`ElemCopyStrided`)
- `x/math/tensor/eager/linalg_helpers.go`: 3 usages (`ElemCopyStrided`)
- `x/math/tensor/eager/activations.go`: 10 usages (`ElemCopyStrided`)

**Impact**: **HIGH** - `ElemCopyStrided` and `ElemCopyStridedAny` are heavily used
- `ElemCopy`/`ElemCopyStrided` would map to `Copy`/`CopyStrided`
- `ElemCopyAny`/`ElemCopyStridedAny` would need to be replaced with type-agnostic versions
- `ElemSwap`/`ElemSwapStrided` are not in the target API - need to determine if they should be kept

---

### 7. Single Value Conversion (1 function)
- `ValueConvert[T, U Numeric](value T) (zeroU U)`

**Usage Count**: 0 external usages found

**Impact**: **LOW** - This would become `CastValue` in the new API

---

### 8. BLAS Functions (4 functions)
- `Copy[T Numeric](y, x []T, n int)`
- `CopyStrided[T Numeric](y, x []T, strideY, strideX, n int)`
- `Swap[T Numeric](x, y []T, n int)`
- `SwapStrided[T Numeric](x, y []T, strideX, strideY, n int)`

**Usage Count**: 0 external usages found

**Impact**: **LOW** - `Copy` and `CopyStrided` are already in the target API
- `Swap` and `SwapStrided` are not in the target API - need to determine if they should be kept

---

## Usage Summary by Package

| Package | Function | Count |
|---------|----------|-------|
| `x/math/tensor/types` | `CopyWithConversion` | 2 |
| `x/math/tensor/eager` | `CopyWithStrides` | 5 |
| `x/math/tensor/eager` | `ElemCopyStrided` | 23 |
| `x/math/tensor/eager` | `ElemCopyStridedAny` | 1 |
| **Total External Usages** | | **31** |

---

## Proposed API Mapping

### Target Functions → Current Functions

1. **`Copy[T Numeric](dst, src []T, n int)`**
   - Maps to: Unify `Copy` (BLAS) and `ElemCopy`
   - Status: ⚠️ Need to analyze implementation differences (see Implementation Analysis below)

2. **`CopyStrided[T Numeric](dst, src []T, shape []int, stridesDst, stridesSrc []int)`**
   - Maps to: Unify `CopyStrided` (BLAS) and `ElemCopyStrided`
   - Status: ⚠️ **CRITICAL**: BLAS uses `(strideY, strideX, n)` while ElemCopy uses `(shape, stridesDst, stridesSrc)`
   - **Decision Needed**: Extend BLAS `CopyStrided` to support shape-based strides, or create unified implementation

3. **`Swap[T Numeric](dst, src []T, n int)`**
   - Maps to: Unify `Swap` (BLAS) and `ElemSwap`
   - Status: ⚠️ Need to analyze implementation differences

4. **`SwapStrided[T Numeric](dst, src []T, shape []int, stridesDst, stridesSrc []int)`**
   - Maps to: Unify `SwapStrided` (BLAS) and `ElemSwapStrided`
   - Status: ⚠️ **CRITICAL**: Same signature difference as `CopyStrided`

5. **`Cast[T, U Numeric](dst []T, src []U, n int)`**
   - Maps to: `ElemConvert` (rename from `FromUToT`)
   - Status: ⚠️ Needs to be created/renamed

6. **`CastStrided[T, U Numeric](dst []T, src []U, shape []int, stridesDst, stridesSrc []int)`**
   - Maps to: `ElemConvertStrided`
   - Status: ⚠️ Needs to be created/renamed

7. **`CastClamped[T, U Numeric](dst []T, src []U, n int)`**
   - Maps to: Part of `ElemConvert` (when clamping needed)
   - Status: ❌ Needs to be created

8. **`CastClampedStrided[T, U Numeric](dst []T, src []U, shape []int, stridesDst, stridesSrc []int)`**
   - Maps to: Part of `ElemConvertStrided` (when clamping needed)
   - Status: ❌ Needs to be created

9. **`CastFromComplex[T NumericReal](dst []T, src []complex64)`**
   - Maps to: `FromComplexToT` (rename)
   - Status: ⚠️ Needs to be renamed

10. **`CastToComplex[T NumericReal](dst []complex64, src []T)`**
    - Maps to: `FromTToComplex` (rename)
    - Status: ⚠️ Needs to be renamed

11. **`CastStridedAny(dst, src any, shape []int, stridesDst, stridesSrc []int)`**
    - Maps to: `ElemCopyStridedAny` (rename, handles conversion with clamping)
    - Status: ⚠️ Needs to be renamed

12. **`CastValue[T, U Numeric](value T) (zeroU U)`**
    - Maps to: `ValueConvert`
    - Status: ⚠️ Needs to be renamed

---

## Migration Impact Analysis

### High Impact Changes (31 usages)

#### 1. `CopyWithConversion` → `Cast`/`CastClamped` (2 usages)
- **Location**: `x/math/tensor/types/dtype.go`, `dtype_test.go`
- **Change**: Replace with appropriate `Cast` or `CastClamped` based on type compatibility
- **Complexity**: Medium - Need to determine if clamping is required

#### 2. `CopyWithStrides` → `CastStrided`/`CastClampedStrided` (5 usages)
- **Locations**: 
  - `x/math/tensor/eager/tensor.go` (2)
  - `x/math/tensor/eager/conv.go` (2)
  - `x/math/tensor/eager/linalg.go` (1)
- **Change**: Replace with `CastStrided` or `CastClampedStrided` based on type compatibility
- **Complexity**: Medium - Need to determine if clamping is required

#### 3. `ElemCopyStrided` → `CopyStrided` (23 usages)
- **Locations**:
  - `x/math/tensor/eager/linalg.go` (2)
  - `x/math/tensor/eager/math.go` (8)
  - `x/math/tensor/eager/linalg_helpers.go` (3)
  - `x/math/tensor/eager/activations.go` (10)
- **Change**: Direct replacement with `CopyStrided` (same-type copy)
- **Complexity**: Low - Direct mapping

#### 4. `ElemCopyStridedAny` → `CastStridedAny` (1 usage)
- **Location**: `x/math/tensor/eager/tensor.go`
- **Change**: Rename to `CastStridedAny` (handles type conversion with clamping when needed)
- **Complexity**: Low - Direct rename, functionality already handles clamping internally

---

## Multithreading Support Opportunities

### Functions Suitable for Multithreading

1. **`Cast` / `CastStrided`**
   - ✅ Good candidate - Element-wise operations, independent
   - Current: Single-threaded (`st`) and multi-threaded (`mt`) implementations exist
   - Implementation: Already has `convert_st.go` and `convert_mt.go` with build tags

2. **`CastClamped` / `CastClampedStrided`**
   - ✅ Good candidate - Element-wise operations with clamping, independent
   - Current: Only single-threaded implementation
   - Implementation: Can follow same pattern as `Cast`/`CastStrided`

3. **`Copy` / `CopyStrided`**
   - ⚠️ Moderate candidate - Memory copy operations
   - Current: Single-threaded only
   - Implementation: Could benefit from parallelization for large arrays

### Implementation Pattern

The package already uses build tags for single-threaded vs multi-threaded:
- `//go:build !use_mt` for single-threaded
- `//go:build use_mt` for multi-threaded

This pattern can be extended to:
- `Cast` / `CastStrided` - Already supports this
- `CastClamped` / `CastClampedStrided` - Needs implementation

---

## Duplication Reduction Strategy

### Identified Duplications

1. **Copy Functions** (3 implementations doing similar things):
   - `Copy` (BLAS) - 1D contiguous
   - `ElemCopy` - 1D contiguous (identical to Copy)
   - `CopyStrided` (BLAS) - 1D strided
   - `ElemCopyStrided` - Multi-dimensional strided
   - **Reduction**: Unify to single internal implementation, expose as `Copy`/`CopyStrided`

2. **Swap Functions** (3 implementations doing similar things):
   - `Swap` (BLAS) - 1D contiguous
   - `ElemSwap` - 1D contiguous (identical to Swap)
   - `SwapStrided` (BLAS) - 1D strided
   - `ElemSwapStrided` - Multi-dimensional strided
   - **Reduction**: Unify to single internal implementation, expose as `Swap`/`SwapStrided`

3. **Conversion Functions** (multiple overlapping names):
   - `FromUToT` → `Cast`
   - `ElemConvert` → `Cast` (same functionality)
   - `ElemConvertStrided` → `CastStrided`
   - **Reduction**: Single `Cast`/`CastStrided` API

4. **Type-Agnostic Functions**:
   - `CopyWithConversion` → Replace with `Cast`/`CastClamped`
   - `CopyWithStrides` → Replace with `CastStrided`/`CastClampedStrided`
   - `ElemCopyStridedAny` → `CastStridedAny`
   - **Reduction**: Unified type-agnostic API

5. **Complex Number Functions**:
   - `FromComplexToT` → `CastFromComplex`
   - `FromTToComplex` → `CastToComplex`
   - **Reduction**: Clearer naming, keep exported

### Unification Approach

**Principle**: Create unified internal implementations, expose through consistent API.

**Pattern**:
1. Create internal unified implementation (handles all cases)
2. Expose public API functions as thin wrappers or direct calls
3. Deprecate old functions, migrate callers
4. Remove deprecated functions after migration

## Implementation Analysis: BLAS vs ElemCopy

### Key Differences

#### 1. **Copy vs ElemCopy**
- **BLAS `Copy`**: Simple 1D copy with `(y, x []T, n int)`
- **ElemCopy**: Same signature, identical implementation
- **Verdict**: ✅ Can be unified - they're already identical

#### 2. **CopyStrided vs ElemCopyStrided**
- **BLAS `CopyStrided`**: 
  - Signature: `(y, x []T, strideY, strideX, n int)`
  - 1D strided copy (BLAS-style)
  - Simple loop: `y[py] = x[px]` where `py += strideY, px += strideX`
  
- **ElemCopyStrided**:
  - Signature: `(dst, src []T, shape []int, stridesDst, stridesSrc []int)`
  - Multi-dimensional strided copy
  - Uses `AdvanceOffsets` pattern for multi-dimensional iteration
  - Has contiguous detection optimization
  - Handles arbitrary rank tensors

- **Verdict**: ⚠️ **Cannot directly unify** - Different use cases:
  - BLAS: 1D vectors with simple strides
  - ElemCopy: Multi-dimensional arrays with shape-based strides
  - **Solution**: Keep both, or extend BLAS to support shape-based version

#### 3. **Swap vs ElemSwap**
- **BLAS `Swap`**: Simple 1D swap with `(x, y []T, n int)`
- **ElemSwap**: Same signature, identical implementation
- **Verdict**: ✅ Can be unified - they're already identical

#### 4. **SwapStrided vs ElemSwapStrided**
- **BLAS `SwapStrided`**: 
  - Signature: `(x, y []T, strideX, strideY, n int)`
  - 1D strided swap (BLAS-style)
  
- **ElemSwapStrided**:
  - Signature: `(dst, src []T, shape []int, stridesDst, stridesSrc []int)`
  - Multi-dimensional strided swap
  - Uses `AdvanceOffsets` pattern

- **Verdict**: ⚠️ **Same situation as CopyStrided** - Different use cases

### Recommended Approach

**Option 1: Keep Both (Recommended)**
- Keep BLAS functions for 1D operations (BLAS compatibility)
- Keep ElemCopy functions for multi-dimensional operations
- **Problem**: Still have duplication

**Option 2: Extend BLAS Functions**
- Add overloaded versions that accept `shape []int, stridesDst, stridesSrc []int`
- Internally dispatch to appropriate implementation
- **Benefit**: Single API, no duplication
- **Challenge**: Go doesn't support function overloading

**Option 3: Unified Implementation with Wrappers**
- Create unified internal implementation
- BLAS functions become thin wrappers for 1D case
- ElemCopy functions use full implementation
- **Benefit**: Single implementation, multiple APIs
- **Recommended**: This approach reduces duplication while maintaining API compatibility

### Implementation Strategy for Option 3

```go
// Internal unified implementation
func copyStridedInternal[T Numeric](
    dst, src []T, 
    shape []int, 
    stridesDst, stridesSrc []int,
) {
    // Unified implementation using AdvanceOffsets
}

// BLAS wrapper (1D case)
func CopyStrided[T Numeric](y, x []T, strideY, strideX, n int) {
    shape := []int{n}
    stridesDst := []int{strideY}
    stridesSrc := []int{strideX}
    copyStridedInternal(y, x, shape, stridesDst, stridesSrc)
}

// Multi-dimensional version
func CopyStrided[T Numeric](
    dst, src []T, 
    shape []int, 
    stridesDst, stridesSrc []int,
) {
    copyStridedInternal(dst, src, shape, stridesDst, stridesSrc)
}
```

**Note**: Go doesn't support function overloading, so we'd need different names or use a different approach.

**Alternative**: Use the same name but different parameter patterns, or create a unified function that detects the case.

## Recommendations

### 1. Immediate Actions
1. **Rename existing functions**:
   - `FromUToT` → `Cast`
   - `FromComplexToT` → `CastFromComplex`
   - `FromTToComplex` → `CastToComplex`
   - `ValueConvert` → `CastValue`
   - `ElemCopyStridedAny` → `CastStridedAny`

2. **Create new API functions**:
   - `CastStrided` (rename from `ElemConvertStrided`)
   - `CastClamped` (extract from `ElemConvert` when clamping needed)
   - `CastClampedStrided` (extract from `ElemConvertStrided` when clamping needed)

3. **Unify Copy/Swap functions**:
   - **Option A**: Keep both BLAS and ElemCopy versions (maintains API compatibility)
   - **Option B**: Create unified internal implementation, make BLAS functions wrappers
   - **Recommended**: Option B to reduce duplication while maintaining compatibility

4. **Deprecate old functions** (keep for backward compatibility initially):
   - `CopyWithConversion` → Use `Cast`/`CastClamped`
   - `CopyWithStrides` → Use `CastStrided`/`CastClampedStrided`
   - `ElemCopy` → Use `Copy` (after unification)
   - `ElemCopyStrided` → Use `CopyStrided` (after unification)
   - `ElemSwap` → Use `Swap` (after unification)
   - `ElemSwapStrided` → Use `SwapStrided` (after unification)

5. **Make internal**:
   - All `ClampTo*Value` functions
   - `ElemVecCopyStrided`, `ElemMatCopyStrided`

### 2. Migration Strategy
1. **Phase 1**: Add new functions alongside old ones
2. **Phase 2**: Update all 31 usages to new API
3. **Phase 3**: Remove deprecated functions

### 3. Multithreading Implementation
1. **Extend build tag pattern** to `CastClamped`/`CastClampedStrided`
2. **Use existing `st` and `mt` subpackages** for implementation
3. **Consider adding runtime detection** for automatic MT/ST selection

### 4. Implementation Details for Unification

#### Copy/Swap Unification Strategy

Since Go doesn't support function overloading, we have several options:

**Option 1: Keep Separate Functions (No Unification)**
- Keep `Copy`/`CopyStrided` (BLAS) for 1D operations
- Keep `ElemCopy`/`ElemCopyStrided` for multi-dimensional operations
- **Pros**: Clear API, no breaking changes
- **Cons**: Code duplication

**Option 2: Unified Internal Implementation**
```go
// Internal unified implementation
func copyStridedUnified[T Numeric](
    dst, src []T,
    shape []int,
    stridesDst, stridesSrc []int,
) {
    // Single implementation using AdvanceOffsets
}

// BLAS version (1D)
func CopyStrided[T Numeric](y, x []T, strideY, strideX, n int) {
    copyStridedUnified(y, x, []int{n}, []int{strideY}, []int{strideX})
}

// Multi-dimensional version (keep ElemCopyStrided name or rename)
func ElemCopyStrided[T Numeric](
    dst, src []T,
    shape []int,
    stridesDst, stridesSrc []int,
) {
    copyStridedUnified(dst, src, shape, stridesDst, stridesSrc)
}
```

**Option 3: Single Function with Variadic Shape Parameter**
```go
// If shape is nil or empty, treat as 1D with strideY, strideX, n
// Otherwise, use shape-based strides
func CopyStrided[T Numeric](
    dst, src []T,
    shape []int,  // nil or empty for 1D case
    stridesDst, stridesSrc []int,
    strideY, strideX, n int,  // Used when shape is nil
) {
    if len(shape) == 0 {
        // 1D BLAS case
    } else {
        // Multi-dimensional case
    }
}
```
**Note**: This is awkward in Go - not recommended.

**Recommended**: Option 2 - Unified internal implementation with wrapper functions

### 5. Open Questions
1. **Should `Swap`/`SwapStrided` be kept?** - Yes, they're useful and should follow same unification pattern as Copy
2. **Should `CastStridedAny` handle clamping automatically?** - Yes, it should detect when clamping is needed (current `ElemCopyStridedAny` already does this via `ElemConvertStrided`)
3. **How to handle the signature difference between BLAS and multi-dimensional versions?** - Use unified internal implementation with wrapper functions

---

## Summary Statistics

- **Total Exported Functions**: ~25
- **Functions to Keep/Rename**: 12
  - `Copy`, `CopyStrided` (unify with ElemCopy versions)
  - `Swap`, `SwapStrided` (unify with ElemSwap versions)
  - `Cast`, `CastStrided`, `CastClamped`, `CastClampedStrided`
  - `CastFromComplex`, `CastToComplex`
  - `CastStridedAny`
  - `CastValue`
- **Functions to Remove/Internalize**: ~13
- **External Usages to Migrate**: 31
- **Packages Affected**: 2 (`x/math/tensor/types`, `x/math/tensor/eager`)
- **Migration Complexity**: Medium (mostly straightforward replacements, but need to handle BLAS vs multi-dimensional signature differences)

---

## Files Requiring Changes

### High Priority (31 usages)
1. `x/math/tensor/types/dtype.go` (1 usage)
2. `x/math/tensor/types/dtype_test.go` (1 usage)
3. `x/math/tensor/eager/tensor.go` (3 usages)
4. `x/math/tensor/eager/conv.go` (2 usages)
5. `x/math/tensor/eager/linalg.go` (3 usages)
6. `x/math/tensor/eager/math.go` (8 usages)
7. `x/math/tensor/eager/linalg_helpers.go` (3 usages)
8. `x/math/tensor/eager/activations.go` (10 usages)

