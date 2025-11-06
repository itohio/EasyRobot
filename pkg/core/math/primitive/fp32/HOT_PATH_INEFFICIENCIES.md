# Hot Path Inefficiencies Report

**Date**: Generated from code analysis  
**Based on**: Performance Report + Code Review  
**Focus**: Identifying inefficient hot paths in FP32 primitive operations

## Executive Summary

This report identifies inefficient hot paths in the FP32 primitive operations codebase. While Phase 6 operations achieve zero allocations, several tensor operations (`ElemAdd`, `ElemScale`, `ReduceSum`) still allocate memory and have performance bottlenecks that could be optimized.

## Critical Inefficiencies

### 1. ⚠️ **IsContiguous() Allocates on Every Call** (HIGH IMPACT)

**Location**: `tensor_helpers.go:53-67`

**Problem**:
```go
func IsContiguous(strides []int, shape []int) bool {
    // ...
    canonical := ComputeStrides(shape)  // ❌ ALLOCATES every time
    // ...
}
```

**Impact**:
- Called **3 times** per `ElemAdd` operation (for dst, a, b)
- Called **2 times** per `ElemScale` operation
- **Allocates 16-24 bytes** per call (depending on shape rank)
- Even for **contiguous** tensors, this allocation happens before the fast path

**Evidence from Benchmarks**:
- `ElemAdd` (contiguous): 48 B/op (3 allocs) - likely 3× `ComputeStrides` allocations
- `ElemScale` (contiguous): 32 B/op (2 allocs) - likely 2× `ComputeStrides` allocations

**Root Cause**: `ComputeStrides` always allocates a new slice, even when checking if existing strides match.

**Recommendation**:
- Cache canonical strides or compute inline without allocation
- Use a fast-path check: `strides[len(strides)-1] == 1` before full comparison
- Consider passing pre-computed canonical strides from tensor objects

**Estimated Improvement**: Eliminate 32-48 B/op allocations, reduce latency by 5-10% for contiguous tensors

---

### 2. ⚠️ **EnsureStrides() May Allocate** (MEDIUM IMPACT)

**Location**: `tensor_helpers.go:42-50`

**Problem**:
```go
func EnsureStrides(strides []int, shape []int) []int {
    if len(strides) != len(shape) {
        return ComputeStrides(shape)  // ❌ ALLOCATES if mismatch
    }
    return strides
}
```

**Impact**:
- Called **3 times** per `ElemAdd` (dst, a, b strides)
- Called **2 times** per `ElemScale` (dst, src strides)
- Allocates if stride length doesn't match shape (defensive check)

**Evidence**: Allocations in benchmarks suggest this path is taken, even for valid inputs.

**Recommendation**:
- Validate strides at tensor creation time, not in hot paths
- Return early if strides are already valid (length check is cheap)
- Consider removing this defensive check in hot paths if tensor API guarantees valid strides

**Estimated Improvement**: Eliminate defensive allocations, reduce latency by 2-5%

---

### 3. ⚠️ **StrideSet Array Creation in Hot Paths** (MEDIUM IMPACT)

**Location**: `tensor_elementwise.go:414`, `tensor_elementwise.go:446`, etc.

**Problem**:
```go
strideSet := [][]int{stridesDst, stridesA, stridesB}  // ❌ Creates slice header array
```

**Impact**:
- Creates a new slice header array (3-4 pointers) on every strided operation
- Not measured in allocations (slice headers are small), but adds overhead
- Called in **every strided access path** (non-contiguous tensors)

**Evidence**: Strided operations are 3.6x slower than contiguous, suggesting overhead beyond cache misses.

**Recommendation**:
- Pass strides directly to `advanceOffsets` instead of creating intermediate array
- Or use a pre-allocated strideSet buffer (if operation is called repeatedly)

**Estimated Improvement**: Reduce strided access overhead by 5-10%

---

### 4. ⚠️ **Indices/Offsets Arrays Allocated in Hot Paths** (MEDIUM IMPACT)

**Location**: `tensor_elementwise.go:412-414`, `tensor_reduction.go:20-21`

**Problem**:
```go
indices := make([]int, len(shape))      // ❌ ALLOCATES
offsets := make([]int, 3)              // ❌ ALLOCATES
```

**Impact**:
- Allocated on **every strided operation** (non-contiguous path)
- For 2D tensors: 2×4 bytes (indices) + 3×4 bytes (offsets) = 20 bytes
- For 3D tensors: 3×4 bytes (indices) + 3×4 bytes (offsets) = 24 bytes
- These allocations happen **inside the hot path**, not once per operation

**Evidence**: Strided operations show allocations even though they shouldn't need them for the computation itself.

**Recommendation**:
- Use stack-allocated arrays for small ranks (up to 4-5 dimensions)
- For larger ranks, use sync.Pool to reuse buffers
- Consider passing pre-allocated buffers from tensor objects

**Estimated Improvement**: Eliminate 20-24 B/op allocations for strided operations, reduce latency by 3-7%

---

### 5. ⚠️ **Function Call Overhead in applyElem* Helpers** (LOW-MEDIUM IMPACT)

**Location**: `tensor_elementwise.go:395-424`

**Problem**:
```go
func ElemAdd(...) {
    applyElemBinary(..., func(av, bv float32) float32 {
        return av + bv  // ❌ Closure allocation + indirect call
    })
}
```

**Impact**:
- Closure allocation for each operation (though Go may optimize some)
- Indirect function call overhead (virtual dispatch)
- Prevents inlining of simple operations

**Evidence**: Simple operations like `HadamardProduct` (direct implementation) are 15x faster than `ElemAdd` (helper-based).

**Recommendation**:
- For simple operations (Add, Sub, Mul), inline the operation directly
- Keep helper functions only for complex operations or when code duplication is excessive
- Consider using build-time code generation for common operations

**Estimated Improvement**: 5-15% latency reduction for simple element-wise operations

---

### 6. ⚠️ **ReduceSum: Multiple Allocations in Hot Path** (MEDIUM IMPACT)

**Location**: `tensor_reduction.go:5-30`

**Problem**:
```go
func ReduceSum(...) {
    dstStrides = EnsureStrides(dstStrides, dstShape)  // ❌ May allocate
    srcStrides = EnsureStrides(srcStrides, srcShape)  // ❌ May allocate
    reduceMask := makeReduceMask(len(srcShape), axes)   // ❌ ALLOCATES
    mapped := mapStridesToSource(...)                  // ❌ ALLOCATES
    indices := make([]int, len(srcShape))              // ❌ ALLOCATES
    offsets := make([]int, 2)                          // ❌ ALLOCATES
}
```

**Impact**:
- **6 allocations** per `ReduceSum` call:
  1. `EnsureStrides` (dst) - may allocate
  2. `EnsureStrides` (src) - may allocate
  3. `makeReduceMask` - always allocates `[]bool`
  4. `mapStridesToSource` - always allocates `[]int`
  5. `indices` - always allocates `[]int`
  6. `offsets` - always allocates `[]int`

**Evidence**: Benchmark shows 34 B/op (3 allocs) - likely some allocations are combined or optimized away, but still significant.

**Recommendation**:
- Pre-compute `reduceMask` and `mapped` strides at tensor creation or operation setup
- Use stack-allocated arrays for small ranks
- Cache intermediate results if operation is called repeatedly on same tensor shapes

**Estimated Improvement**: Eliminate 30-50 B/op allocations, reduce latency by 10-20%

---

### 7. ⚠️ **advanceOffsets Loop Structure** (LOW IMPACT)

**Location**: `tensor_helpers.go:111-133`

**Problem**:
```go
func advanceOffsets(...) bool {
    for dim := len(shape) - 1; dim >= 0; dim-- {
        indices[dim]++
        for buf := range offsets {  // ❌ Nested loop, cache-unfriendly
            offsets[buf] += strides[buf][dim]
        }
        // ...
    }
}
```

**Impact**:
- Nested loop structure may not be optimal for cache locality
- Multiple stride array accesses per iteration
- Called once per element in strided operations (32,768 times for benchmark)

**Evidence**: Strided operations are significantly slower, though cache misses are the primary cause.

**Recommendation**:
- Unroll loops for small numbers of buffers (2-4 buffers)
- Consider SIMD for offset computation if multiple buffers
- Profile to confirm this is a bottleneck (may be dominated by memory access)

**Estimated Improvement**: 2-5% latency reduction for strided operations (if not memory-bound)

---

## Performance Impact Summary

| Issue | Allocations | Latency Impact | Priority |
|-------|-------------|----------------|----------|
| `IsContiguous` allocates | 32-48 B/op | 5-10% | **HIGH** |
| `EnsureStrides` may allocate | 16-32 B/op | 2-5% | **MEDIUM** |
| StrideSet array creation | 0 B (slice headers) | 5-10% | **MEDIUM** |
| Indices/Offsets arrays | 20-24 B/op | 3-7% | **MEDIUM** |
| Function call overhead | 0 B (closures) | 5-15% | **LOW-MEDIUM** |
| `ReduceSum` multiple allocs | 30-50 B/op | 10-20% | **MEDIUM** |
| `advanceOffsets` loop | 0 B | 2-5% | **LOW** |

## Comparison: Phase 6 vs Tensor Operations

**Why Phase 6 operations are fast** (zero allocations):
- ✅ Direct implementations (no helper function overhead)
- ✅ No `IsContiguous` checks (assume caller knows layout)
- ✅ No `EnsureStrides` calls (strides passed directly)
- ✅ Stack-allocated or reused buffers
- ✅ Inlined operations

**Why tensor operations are slower** (allocations):
- ❌ `IsContiguous` called multiple times (allocates each time)
- ❌ `EnsureStrides` defensive checks (may allocate)
- ❌ Helper function overhead (closures, indirect calls)
- ❌ Allocations in strided paths (indices, offsets, strideSets)

## Recommendations by Priority

### Priority 1: Eliminate IsContiguous Allocations (HIGH)
1. **Fast-path optimization**: Check `strides[len(strides)-1] == 1` first
2. **Inline comparison**: Compare strides directly without allocating canonical strides
3. **Cache canonical strides**: Pre-compute at tensor creation, pass to operations

**Expected Impact**: Eliminate 32-48 B/op, 5-10% latency reduction

### Priority 2: Optimize ReduceSum Allocations (MEDIUM)
1. **Pre-compute masks**: Compute `reduceMask` and `mapped` at tensor/operation setup
2. **Stack-allocated arrays**: Use fixed-size arrays for small ranks (≤4 dimensions)
3. **Reuse buffers**: Use `sync.Pool` for larger ranks

**Expected Impact**: Eliminate 30-50 B/op, 10-20% latency reduction

### Priority 3: Eliminate Strided Path Allocations (MEDIUM)
1. **Stack-allocated indices/offsets**: Use fixed-size arrays for common ranks
2. **Remove StrideSet arrays**: Pass strides directly to `advanceOffsets`
3. **Pre-validate strides**: Move validation out of hot paths

**Expected Impact**: Eliminate 20-24 B/op, 3-7% latency reduction

### Priority 4: Reduce Function Call Overhead (LOW-MEDIUM)
1. **Inline simple operations**: Direct implementation for Add, Sub, Mul
2. **Code generation**: Generate optimized versions for common operations
3. **Profile-guided optimization**: Measure actual overhead before optimizing

**Expected Impact**: 5-15% latency reduction for simple operations

## Implementation Notes

### Fast-Path IsContiguous (Example)
```go
func IsContiguousFast(strides []int, shape []int) bool {
    if len(shape) == 0 {
        return true
    }
    if len(strides) != len(shape) {
        return false
    }
    // Fast path: check last stride first (most common case)
    if strides[len(strides)-1] != 1 {
        return false
    }
    // Inline comparison without allocation
    stride := 1
    for i := len(shape) - 1; i >= 0; i-- {
        if strides[i] != stride {
            return false
        }
        stride *= shape[i]
    }
    return true
}
```

### Stack-Allocated Indices/Offsets (Example)
```go
// For common 2D case
var indices [4]int  // stack-allocated, supports up to 4D
var offsets [4]int  // stack-allocated, supports up to 4 buffers
if len(shape) <= 4 {
    // Use stack arrays
} else {
    // Fall back to heap allocation
}
```

### 8. ✅ **Convolution Operations: Expected Allocations** (NOT AN INEFFICIENCY)

**Location**: `tensor.go:145`, `tensor.go:171`

**Note**:
```go
im2col := make([]float32, im2colSize*kernelSize)      // Expected
gemmOutput := make([]float32, outChannels*im2colSize) // Expected
```

**Analysis**:
- These allocations are **algorithmic requirements** for GEMM-based convolution
- `Im2Col` transformation is necessary for efficient matrix multiplication
- Temporary buffers are expected and acceptable for convolution operations
- **Not considered an inefficiency** - this is the standard approach for convolution

**Comparison**:
- Pooling operations (MaxPool, AvgPool) have **zero allocations** ✅
- Convolution operations allocate temporary buffers (expected) ✅
- Element-wise operations should have zero allocations (currently don't) ❌

---

## Conclusion

The primary inefficiencies in hot paths are:

1. **Memory allocations** in `IsContiguous` and `EnsureStrides` (32-48 B/op)
2. **Allocations in strided paths** for indices/offsets arrays (20-24 B/op)
3. **Multiple allocations in reduction operations** (30-50 B/op)
4. **Function call overhead** in helper-based operations (5-15% latency)

**Estimated Total Improvement**:
- Eliminate **82-122 B/op** allocations
- Reduce latency by **15-35%** for tensor operations
- Bring tensor operations closer to Phase 6 operation performance

**Next Steps**:
1. Implement fast-path `IsContiguous` (Priority 1)
2. Optimize `ReduceSum` allocations (Priority 2)
3. Profile to validate improvements
4. Consider SIMD optimizations for contiguous paths

