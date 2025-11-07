# Optimization Plan for generics/copy.go

## Overview

This document identifies all memory allocations and bounds check opportunities in the generics copy operations. The goal is to eliminate allocations in hot paths and reduce bounds checking overhead.

## 1. Memory Allocations

### 1.1 Allocations in `copy.go`

#### `ElemCopyStrided` (lines 17-43)
- **Line 32**: `indices := make([]int, len(shape))` - Allocates slice for multi-dimensional indices
  - **Impact**: High - allocated on every strided copy operation
  - **Frequency**: Every call to `ElemCopyStrided` when arrays are not contiguous
  - **Size**: `len(shape) * 8 bytes` (assuming 64-bit ints)

- **Line 33**: `offsets := make([]int, 2)` - Allocates slice for destination and source offsets
  - **Impact**: High - allocated on every strided copy operation
  - **Frequency**: Every call to `ElemCopyStrided` when arrays are not contiguous
  - **Size**: `2 * 8 = 16 bytes`

- **Line 34**: `strideSet := [][]int{stridesDst, stridesSrc}` - Composite literal creates slice of slices
  - **Impact**: Medium - allocates backing array for slice of slices
  - **Frequency**: Every call to `ElemCopyStrided` when arrays are not contiguous
  - **Size**: `2 * 8 = 16 bytes` (slice headers) + backing array overhead

#### `ElemSwapStrided` (lines 59-87)
- **Line 76**: `indices := make([]int, len(shape))` - Same as above
- **Line 77**: `offsets := make([]int, 2)` - Same as above
- **Line 78**: `strideSet := [][]int{stridesDst, stridesSrc}` - Same as above

### 1.2 Allocations in Helper Functions (`helpers/helpers.go`)

#### `ComputeStrides` (lines 84-97)
- **Line 89**: `strides := make([]int, len(shape))` - Allocates when computing canonical strides
  - **Impact**: High - called by `EnsureStrides` and `IsContiguous`
  - **Frequency**: 
    - Every call to `EnsureStrides` when `len(strides) != len(shape)`
    - Every call to `IsContiguous` (always allocates for comparison)
  - **Size**: `len(shape) * 8 bytes`

#### `IsContiguous` (lines 127-144)
- **Line 137**: `canonical := ComputeStrides(shape)` - Always allocates to compare strides
  - **Impact**: High - called on every strided operation to check fast path
  - **Frequency**: Every call to `ElemCopyStrided`, `ElemSwapStrided`, `ElemConvertStrided`
  - **Size**: `len(shape) * 8 bytes`
  - **Note**: This allocation happens even when the result is `false` early in the function

#### `IterateOffsets` (lines 176-190)
- **Line 181**: `indices := make([]int, len(shape))` - Allocates indices slice
- **Line 182**: `offsets := make([]int, len(strides))` - Allocates offsets slice
  - **Impact**: Medium - only used by convenience wrappers, not in hot paths

#### `IterateOffsetsWithIndices` (lines 195-209)
- **Line 200**: `indices := make([]int, len(shape))` - Allocates indices slice
- **Line 201**: `offsets := make([]int, len(strides))` - Allocates offsets slice
  - **Impact**: Medium - only used by convenience wrappers, not in hot paths

### 1.3 Allocations in Conversion Functions (`st/convert.go`)

#### `elemConvertStridedGeneric` (lines 286-327)
- **Line 293**: `indices := make([]int, ndims)` - Allocates indices for strided conversion
  - **Impact**: High - allocated on every strided conversion
  - **Frequency**: Every call to `ElemConvertStrided` when arrays are not contiguous or types differ

#### `clampToInt8Strided` (lines 589-636)
- **Line 595**: `indices := make([]int, ndims)` - Allocates indices for strided clamping
  - **Impact**: High - allocated on every strided int8 conversion with clamping

#### `clampToInt16Strided` (lines 642-689)
- **Line 648**: `indices := make([]int, ndims)` - Allocates indices for strided clamping
  - **Impact**: High - allocated on every strided int16 conversion with clamping

#### `clampToInt32Strided` (lines 739-781)
- **Line 745**: `indices := make([]int, ndims)` - Allocates indices for strided clamping
  - **Impact**: High - allocated on every strided int32 conversion with clamping

#### `clampToInt64Strided` (lines 693-735)
- **Line 699**: `indices := make([]int, ndims)` - Allocates indices for strided clamping
  - **Impact**: High - allocated on every strided int64 conversion with clamping

### 1.4 Allocation Summary

**Total allocations per strided operation:**
- `ElemCopyStrided`: 3 allocations (indices, offsets, strideSet)
- `ElemSwapStrided`: 3 allocations (indices, offsets, strideSet)
- `ElemConvertStrided`: 1-2 allocations (indices, potentially ComputeStrides in IsContiguous)
- `IsContiguous`: 1 allocation (ComputeStrides) - **called on every strided operation**

**Critical path allocations:**
1. `IsContiguous` always allocates via `ComputeStrides` even when checking fails early
2. `indices` and `offsets` arrays allocated on every strided operation
3. `strideSet` composite literal creates unnecessary slice structure

## 2. Bounds Check Analysis

### 2.1 Bounds Checks in `copy.go`

#### `ElemCopy` (lines 8-13)
- **Line 12**: `copy(dst[:n], src[:n])` - Slice expressions may require bounds checks
  - **Impact**: Low - `copy` is optimized, but slice expressions still checked
  - **Mitigation**: Compiler may eliminate if `n <= len(dst)` and `n <= len(src)` are provable

#### `ElemCopyStrided` (lines 17-43)
- **Line 27**: `copy(dst[:size], src[:size])` - Slice expressions in fast path
  - **Impact**: Low - compiler may eliminate
- **Line 36**: `dIdx := offsets[0]` - Array access, bounds check
  - **Impact**: Medium - checked on every iteration
  - **Frequency**: Once per element in strided path
- **Line 37**: `sIdx := offsets[1]` - Array access, bounds check
  - **Impact**: Medium - checked on every iteration
- **Line 38**: `dst[dIdx] = src[sIdx]` - Slice access, bounds check
  - **Impact**: High - checked on every iteration, cannot be eliminated without proving bounds
  - **Frequency**: Once per element (hot path)

#### `ElemSwap` (lines 48-55)
- **Line 52**: `for i := 0; i < n; i++` - Loop with bounds check potential
  - **Impact**: Medium - bounds checks on each iteration
- **Line 53**: `dst[i], src[i] = src[i], dst[i]` - Four slice accesses per iteration
  - **Impact**: High - 4 bounds checks per iteration
  - **Mitigation**: Compiler may eliminate with `_ = dst[n-1]` and `_ = src[n-1]` hints

#### `ElemSwapStrided` (lines 59-87)
- **Line 69**: `for i := 0; i < size; i++` - Loop in fast path
  - **Impact**: Medium - bounds checks on each iteration
- **Line 70**: `dst[i], src[i] = src[i], dst[i]` - Four slice accesses per iteration
  - **Impact**: High - 4 bounds checks per iteration
- **Line 80**: `dIdx := offsets[0]` - Array access, bounds check
  - **Impact**: Medium - checked on every iteration
- **Line 81**: `sIdx := offsets[1]` - Array access, bounds check
  - **Impact**: Medium - checked on every iteration
- **Line 82**: `dst[dIdx], src[sIdx] = src[sIdx], dst[dIdx]` - Four slice accesses with computed indices
  - **Impact**: High - 4 bounds checks per iteration, cannot be eliminated

### 2.2 Bounds Checks in Helper Functions

#### `AdvanceOffsets` (lines 149-171 in `helpers.go`)
- **Line 154**: `for dim := len(shape) - 1; dim >= 0; dim--` - Loop iteration
  - **Impact**: Low - `len(shape)` is checked once
- **Line 155**: `indices[dim]++` - Array access, bounds check
  - **Impact**: Medium - checked on every dimension iteration
  - **Frequency**: Once per dimension per element (nested loop)
- **Line 157**: `offsets[buf] += strides[buf][dim]` - Nested array access
  - **Impact**: High - 2 bounds checks per iteration
    - `offsets[buf]` - checked
    - `strides[buf][dim]` - checked (2D array access)
  - **Frequency**: Once per buffer per dimension per element
- **Line 160**: `indices[dim] < shape[dim]` - Two array accesses
  - **Impact**: Medium - 2 bounds checks
- **Line 165**: `offsets[buf] -= strides[buf][dim] * shape[dim]` - Multiple array accesses
  - **Impact**: High - 3 bounds checks (`offsets[buf]`, `strides[buf][dim]`, `shape[dim]`)
- **Line 167**: `indices[dim] = 0` - Array access, bounds check
  - **Impact**: Medium - checked on dimension reset

#### `ComputeStrideOffset` (lines 212-218 in `helpers.go`)
- **Line 214**: `for i := range indices` - Loop iteration
  - **Impact**: Low
- **Line 215**: `offset += indices[i] * strides[i]` - Two array accesses
  - **Impact**: Medium - 2 bounds checks per iteration
  - **Frequency**: Called on every element access in strided operations

### 2.3 Bounds Checks in Conversion Functions

#### `elemConvertStridedGeneric` (lines 286-327 in `st/convert.go`)
- **Line 299**: `sIdx := ComputeStrideOffset(indices, srcStrides)` - Function call with bounds checks
- **Line 300**: `dIdx := ComputeStrideOffset(indices, dstStrides)` - Function call with bounds checks
- **Line 302**: `dst[dIdx] = T(src[sIdx])` - Two slice accesses with computed indices
  - **Impact**: High - 2 bounds checks per element, cannot be eliminated
- **Line 314**: `indices[dim] >= shape[dim]` - Two array accesses
  - **Impact**: Medium - 2 bounds checks per dimension check

#### `clampToInt8Strided` and similar clamping functions
- **Line 602**: `sIdx := ComputeStrideOffset(indices, srcStrides)` - Function call
- **Line 603**: `dIdx := ComputeStrideOffset(indices, dstStrides)` - Function call
- **Line 604**: `val := src[sIdx]` - Slice access, bounds check
  - **Impact**: High - checked on every iteration
- **Line 605-610**: Clamping logic with `dst[dIdx]` access
  - **Impact**: High - bounds check on every iteration
- **Line 623**: `indices[dim] >= shape[dim]` - Two array accesses
  - **Impact**: Medium - 2 bounds checks per dimension check

### 2.4 Bounds Check Summary

**Hot path bounds checks per element:**
- Strided copy/swap: 4-6 bounds checks per element
  - `offsets[0]`, `offsets[1]` (2 checks)
  - `dst[dIdx]`, `src[sIdx]` (2 checks)
  - Additional checks in `AdvanceOffsets` (2-3 per dimension)
- Strided conversion: 4-8 bounds checks per element
  - `ComputeStrideOffset` calls (2-4 checks per call)
  - `dst[dIdx]`, `src[sIdx]` (2 checks)
  - Dimension checks in loop (2 checks per dimension)

**Critical bounds check issues:**
1. Computed indices (`dIdx`, `sIdx`) cannot have bounds checks eliminated by compiler
2. Nested array access in `AdvanceOffsets` (`strides[buf][dim]`) requires multiple checks
3. `ComputeStrideOffset` called twice per element in strided operations

## 3. Optimization Strategies

### 3.1 Eliminate Allocations

#### Strategy 1: Stack-allocated arrays for small dimensions
- Use fixed-size arrays when `len(shape) <= MAX_DIMS` (already defined as 16)
- Fall back to heap allocation only for very large dimensions
- **Implementation**: 
  ```go
  var indices [MAX_DIMS]int
  var offsets [2]int
  // Use slice of fixed array: indices[:len(shape)]
  ```

#### Strategy 2: Reuse stride computation in `IsContiguous`
- Cache canonical strides or compute inline without allocation
- Compare strides element-by-element without allocating full array
- **Implementation**: Inline comparison loop instead of `ComputeStrides` call

#### Strategy 3: Eliminate `strideSet` composite literal
- Pass `stridesDst` and `stridesSrc` directly to `AdvanceOffsets`
- Modify `AdvanceOffsets` signature to accept two stride slices instead of `[][]int`
- **Implementation**: 
  ```go
  func AdvanceOffsets(shape []int, indices []int, offsets []int, stridesDst, stridesSrc []int) bool
  ```

#### Strategy 4: Object pooling for large dimensions
- Use `sync.Pool` for `indices` and `offsets` slices when `len(shape) > MAX_DIMS`
- Reuse slices across operations
- **Trade-off**: Adds complexity, only beneficial for very large tensors

### 3.2 Reduce Bounds Checks

#### Strategy 1: Bounds check elimination hints
- Add `_ = dst[n-1]` and `_ = src[n-1]` before loops (already done in some conversion functions)
- Helps compiler prove bounds for simple loops

#### Strategy 2: Inline `ComputeStrideOffset`
- Inline the offset computation to allow better optimization
- Reduces function call overhead and may help bounds check elimination
- **Implementation**: Replace function call with inline computation

#### Strategy 3: Restructure `AdvanceOffsets` to reduce nested access
- Pre-compute stride values to reduce nested array access
- Use local variables for frequently accessed values
- **Implementation**: 
  ```go
  strideDst := stridesDst[dim]
  strideSrc := stridesSrc[dim]
  offsets[0] += strideDst
  offsets[1] += strideSrc
  ```

#### Strategy 4: Unroll loops for small dimensions
- Specialize for common small dimensions (1D, 2D, 3D)
- Eliminates loop overhead and some bounds checks
- **Implementation**: Type switch on `len(shape)` with specialized functions

### 3.3 Combined Optimizations

#### Strategy 1: Fast path specialization
- Create specialized functions for common cases:
  - 1D contiguous
  - 2D contiguous
  - 3D contiguous
  - 1D strided
  - 2D strided
- Reduces allocations and bounds checks for common cases

#### Strategy 2: Inline strided iteration
- Inline the iteration logic directly in copy/swap/convert functions
- Eliminates function call overhead and allows better optimization
- **Trade-off**: Code duplication, but significant performance gain

## 4. Priority Ranking

### High Priority (Biggest Impact)
1. **Eliminate `IsContiguous` allocation** - Called on every strided operation
2. **Stack-allocate indices/offsets for small dimensions** - Eliminates 2-3 allocations per operation
3. **Eliminate `strideSet` composite literal** - Simple refactor, eliminates allocation
4. **Inline `ComputeStrideOffset`** - Reduces bounds checks and function call overhead

### Medium Priority
5. **Restructure `AdvanceOffsets`** - Reduces nested array bounds checks
6. **Add bounds check elimination hints** - Simple, helps compiler
7. **Specialize for small dimensions** - Significant win for common cases

### Low Priority (Diminishing Returns)
8. **Object pooling for large dimensions** - Only helps edge cases
9. **Full loop unrolling** - High complexity, moderate benefit

## 5. Implementation Notes

### Backward Compatibility
- All optimizations should maintain the same function signatures
- Internal implementation changes only

### Testing Requirements
- Benchmark before and after each optimization
- Verify correctness with existing tests
- Test edge cases: empty shapes, single elements, large dimensions

### Performance Targets
- **Allocation reduction**: Eliminate 100% of allocations for dimensions <= 16
- **Bounds check reduction**: Reduce bounds checks by 30-50% in hot paths
- **Overall performance**: Target 10-20% improvement for strided operations

## 6. Measurement Plan

### Benchmarks to Add/Update
1. Benchmark `ElemCopyStrided` with various shapes (1D, 2D, 3D, 4D+)
2. Benchmark `ElemSwapStrided` with various shapes
3. Benchmark `ElemConvertStrided` with various shapes and type combinations
4. Measure allocations using `go test -benchmem`
5. Measure bounds checks using `go test -gcflags=-d=ssa/check_bce`

### Success Criteria
- Zero allocations for operations with `len(shape) <= 16`
- Reduced bounds check warnings from compiler
- 10-20% performance improvement in strided operations
- No regression in contiguous fast paths

