# Generic Operations Specification

This document specifies all generic operations for the `primitive/generics` package. These operations work with any numeric type (float32, float64, int8, int16, int32, int64, int) and do not rely on underlying datatype precision.

**Note on Multi-threading**: Building with the `use_mt` build tag enables multi-threaded implementations for operations that support parallelization. Operations marked with 🔀 support both single-threaded and multi-threaded execution. Operations marked with 🔒 are single-threaded only.

Multi-threaded code paths now share the typed worker pool defined in `primitive/generics/helpers`. The pool is re-exported from the `generics` package as `WorkerPool`, `WorkerPoolOption`, and related option helpers so higher-level packages can compose the same concurrency primitives. The `mt` subpackage initialises a single global worker pool during `init`, mirroring the previous bespoke implementation while centralising the concurrency plumbing.

**Status Legend:**
- ⏳ **Not Implemented** - Specified but not yet implemented
- 🚧 **In Progress** - Currently being implemented
- ✅ **Implemented** - Fully implemented and tested
- ❌ **Excluded** - Intentionally excluded (relies on precision or type-specific behavior)
- 🔀 **Multi-threaded Support** - Operation supports both single-threaded and multi-threaded execution (in `st/` subpackage)
- 🔒 **Single-threaded Only** - Operation is single-threaded only (only in main `generics` package)

**Performance Impact:**
- **None/Minimal** - Generic implementation has same or better performance
- **Low** - Small overhead (<5%), acceptable for code reuse
- **Medium** - Moderate overhead (5-15%), may need type-specific fast paths
- **High** - Significant overhead (>15%), may not be worth genericizing

**Test Coverage:**
- **None** - No tests yet
- **Partial** - Some tests exist
- **Complete** - Full test coverage

---

## Table of Contents

1. [Tensor Operations](#tensor-operations) - Multi-dimensional tensor operations (contiguous and strided)
2. [Vector Operations](#vector-operations) - Optimized 1D vector operations (strided only)
3. [Matrix Operations](#matrix-operations) - Optimized 2D matrix operations (strided only)
4. [Scalar Operations](#scalar-operations) - Operations with scalar values
5. [Boolean Operations](#boolean-operations) - Comparison and conditional operations
6. [BLAS Operations](#blas-operations) - BLAS Level 1 vector operations
7. [Helper Operations](#helper-operations) - Utility functions for stride iteration, shape manipulation, and iterators

---

## Implementation Requirements

### Specialized and Optimized Implementations

**Requirement**: Similar functions must have specialized and optimized implementations for different use cases:
- **Contiguous versions**: Take `n int` for simple, direct loops (e.g., `ElemCopy[T](dst, src []T, n int)`)
- **Strided versions**: Take `shape []int, strides... []int` and include fast path checks for contiguous arrays
- **Vector versions**: Optimized for 1D data with stride parameters (`n int, strideDst, strideSrc int`)
- **Matrix versions**: Optimized for 2D data with leading dimensions (`rows, cols int, ldDst, ldSrc int`)

**Rationale**: 
- Contiguous operations are the common case and should have simple, fast implementations
- Strided operations should handle both cases with optimized fast paths for contiguous arrays
- Vector/matrix operations provide better performance for 1D/2D data by avoiding general stride iteration overhead
- This allows callers to choose the appropriate function based on their knowledge of memory layout
- Compiler optimizations (inlining, bounds check elimination) work better with simpler, specialized functions

### Optimization Patterns

**Boundary Checks**: Avoid requiring boundary checks where possible. The Go compiler can optimize these away when:
- Array lengths are known at compile time or checked once before loops
- Loop bounds are explicit and within slice capacity
- Slice operations use `[:n]` patterns that the compiler can prove are safe

---

## Tensor Operations

Multi-dimensional tensor operations that work with arbitrary shapes and strides. Contiguous versions are optimized for the common case of contiguous memory, while strided versions handle both contiguous and strided cases.

### Copy Operations

| Operation | Generic Function | Status | Performance | Tests | Threading | Source |
|-----------|------------------|--------|-------------|-------|-----------|--------|
| Copy (Contiguous) | `ElemCopy[T](dst, src []T, n int)` | ✅ | None/Minimal | Complete | 🔒 | `fp32/tensor_elementwise.go:98` |
| Copy (Strided) | `ElemCopyStrided[T](dst, src []T, shape []int, stridesDst, stridesSrc []int)` | ✅ | None/Minimal | Complete | 🔒 | `primitive/copy.go:567` |
| Swap (Contiguous) | `ElemSwap[T](dst, src []T, n int)` | ✅ | None/Minimal | Complete | 🔒 | New |
| Swap (Strided) | `ElemSwapStrided[T](dst, src []T, shape []int, stridesDst, stridesSrc []int)` | ✅ | None/Minimal | Complete | 🔒 | New |
| Convert (Contiguous) | `ElemConvert[T, U](dst []T, src []U, n int)` | ✅ | Low | None | 🔀 | `primitive/copy.go:14` |
| Convert (Strided) | `ElemConvertStrided[T, U](dst []T, src []U, shape []int, stridesDst, stridesSrc []int)` | ✅ | Low | None | 🔀 | `primitive/copy.go:508` |

### Unary Operations

| Operation | Generic Function | Status | Performance | Tests | Threading | Source |
|-----------|------------------|--------|-------------|-------|-----------|--------|
| Sign (Contiguous) | `ElemSign[T](dst, src []T, n int)` | ✅ | None/Minimal | Complete | 🔒 | `fp32/tensor_elementwise.go:227` |
| Sign (Strided) | `ElemSignStrided[T](dst, src []T, shape []int, stridesDst, stridesSrc []int)` | ✅ | None/Minimal | Complete | 🔒 | `fp32/tensor_elementwise.go:227` |
| Negative (Contiguous) | `ElemNegative[T](dst, src []T, n int)` | ✅ | None/Minimal | Complete | 🔒 | `fp32/tensor_elementwise.go:261` |
| Negative (Strided) | `ElemNegativeStrided[T](dst, src []T, shape []int, stridesDst, stridesSrc []int)` | ✅ | None/Minimal | Complete | 🔒 | `fp32/tensor_elementwise.go:261` |

**Excluded**: `ElemSquare`, `ElemAbs` - rely on type precision

### Ternary Operations

| Operation | Generic Function | Status | Performance | Tests | Threading | Source |
|-----------|------------------|--------|-------------|-------|-----------|--------|
| Where (Strided) | `ElemWhere[T](dst, condition, a, b []T, shape []int, stridesDst, stridesCond, stridesA, stridesB []int)` | ✅ | None/Minimal | None | 🔒 | `fp32/tensor_elementwise.go:126` |

**Note**: `ElemWhere` selects `a[i]` if `condition[i] > 0`, else `b[i]`. Works for all numeric types.

### Apply Operations

Generic apply functions that accept custom operation functions for element-wise processing. These are intended for complex custom operations that cannot be expressed as simple rolled-out loops.

| Operation | Generic Function | Status | Performance | Tests | Threading | Source |
|-----------|------------------|--------|-------------|-------|-----------|--------|
| Apply Binary (Contiguous) | `ElemApplyBinary[T](dst, a, b []T, n int, op func(T, T) T)` | ✅ | Low | Complete | 🔀 | `fp32/tensor_elementwise.go:395` |
| Apply Binary (Strided) | `ElemApplyBinaryStrided[T](dst, a, b []T, shape []int, stridesDst, stridesA, stridesB []int, op func(T, T) T)` | ✅ | Low | Complete | 🔀 | `fp32/tensor_elementwise.go:395` |
| Apply Unary (Contiguous) | `ElemApplyUnary[T](dst, src []T, n int, op func(T) T)` | ✅ | Low | Complete | 🔀 | `fp32/tensor_elementwise.go:460` |
| Apply Unary (Strided) | `ElemApplyUnaryStrided[T](dst, src []T, shape []int, stridesDst, stridesSrc []int, op func(T) T)` | ✅ | Low | Complete | 🔀 | `fp32/tensor_elementwise.go:460` |
| Apply Ternary (Contiguous) | `ElemApplyTernary[T](dst, condition, a, b []T, n int, op func(T, T, T) T)` | ✅ | Low | Complete | 🔀 | `fp32/tensor_elementwise.go:426` |
| Apply Ternary (Strided) | `ElemApplyTernaryStrided[T](dst, condition, a, b []T, shape []int, stridesDst, stridesCond, stridesA, stridesB []int, op func(T, T, T) T)` | ✅ | Low | Complete | 🔀 | `fp32/tensor_elementwise.go:426` |
| Apply Unary Scalar (Contiguous) | `ElemApplyUnaryScalar[T](dst, src []T, scalar T, n int, op func(T, T) T)` | ✅ | Low | Complete | 🔀 | `fp32/tensor_elementwise.go:490` |
| Apply Unary Scalar (Strided) | `ElemApplyUnaryScalarStrided[T](dst, src []T, scalar T, shape []int, stridesDst, stridesSrc []int, op func(T, T) T)` | ✅ | Low | Complete | 🔀 | `fp32/tensor_elementwise.go:490` |
| Apply Binary Scalar (Contiguous) | `ElemApplyBinaryScalar[T](dst, a []T, scalar T, n int, op func(T, T) T)` | ✅ | Low | Complete | 🔀 | New |
| Apply Binary Scalar (Strided) | `ElemApplyBinaryScalarStrided[T](dst, a []T, scalar T, shape []int, stridesDst, stridesA []int, op func(T, T) T)` | ✅ | Low | Complete | 🔀 | New |
| Apply Ternary Scalar (Contiguous) | `ElemApplyTernaryScalar[T](dst, condition []T, scalar T, n int, op func(T, T, T) T)` | ✅ | Low | Complete | 🔀 | New |
| Apply Ternary Scalar (Strided) | `ElemApplyTernaryScalarStrided[T](dst, condition []T, scalar T, shape []int, stridesDst, stridesCond []int, op func(T, T, T) T)` | ✅ | Low | Complete | 🔀 | New |

---

## Vector Operations

Optimized 1D vector operations with stride support. These functions are optimized for vector operations by using stride-based iteration instead of general multi-dimensional stride iteration. Only strided versions are provided since contiguous operations are already covered by tensor operations.

### Copy Operations

| Operation | Generic Function | Status | Performance | Tests | Threading | Source |
|-----------|------------------|--------|-------------|-------|-----------|--------|
| Copy Strided | `ElemVecCopyStrided[T](dst, src []T, n int, strideDst, strideSrc int)` | ✅ | None/Minimal | Complete | 🔒 | New |
| Convert Strided | `ElemVecConvertStrided[T, U](dst []T, src []U, n int, strideDst, strideSrc int)` | ✅ | Low | None | 🔀 | New |

### Unary Operations

| Operation | Generic Function | Status | Performance | Tests | Threading | Source |
|-----------|------------------|--------|-------------|-------|-----------|--------|
| Sign Strided | `ElemVecSignStrided[T](dst, src []T, n int, strideDst, strideSrc int)` | ✅ | None/Minimal | None | 🔒 | New |
| Negative Strided | `ElemVecNegativeStrided[T](dst, src []T, n int, strideDst, strideSrc int)` | ✅ | None/Minimal | None | 🔒 | New |

### Apply Operations

| Operation | Generic Function | Status | Performance | Tests | Threading | Source |
|-----------|------------------|--------|-------------|-------|-----------|--------|
| Apply Unary Strided | `ElemVecApplyUnaryStrided[T](dst, src []T, n int, strideDst, strideSrc int, op func(T) T)` | ✅ | None/Minimal | Complete | 🔀 | New |
| Apply Binary Strided | `ElemVecApplyBinaryStrided[T](dst, a, b []T, n int, strideDst, strideA, strideB int, op func(T, T) T)` | ✅ | None/Minimal | Complete | 🔀 | New |
| Apply Ternary Strided | `ElemVecApplyTernaryStrided[T](dst, condition, a, b []T, n int, strideDst, strideCond, strideA, strideB int, op func(T, T, T) T)` | ✅ | None/Minimal | Complete | 🔀 | New |
| Apply Unary Scalar Strided | `ElemVecApplyUnaryScalarStrided[T](dst, src []T, scalar T, n int, strideDst, strideSrc int, op func(T, T) T)` | ✅ | None/Minimal | Complete | 🔀 | New |
| Apply Binary Scalar Strided | `ElemVecApplyBinaryScalarStrided[T](dst, a []T, scalar T, n int, strideDst, strideA int, op func(T, T) T)` | ✅ | None/Minimal | Complete | 🔀 | New |
| Apply Ternary Scalar Strided | `ElemVecApplyTernaryScalarStrided[T](dst, condition []T, scalar T, n int, strideDst, strideCond int, op func(T, T, T) T)` | ✅ | None/Minimal | Complete | 🔀 | New |

---

## Matrix Operations

Optimized 2D matrix operations with leading dimension support. These functions are optimized for matrix operations by using row-by-row iteration instead of general multi-dimensional stride iteration. Only strided versions are provided since contiguous operations are already covered by tensor operations.

### Copy Operations

| Operation | Generic Function | Status | Performance | Tests | Threading | Source |
|-----------|------------------|--------|-------------|-------|-----------|--------|
| Copy Strided | `ElemMatCopyStrided[T](dst, src []T, rows, cols int, ldDst, ldSrc int)` | ✅ | None/Minimal | Complete | 🔒 | New |
| Convert Strided | `ElemMatConvertStrided[T, U](dst []T, src []U, rows, cols int, ldDst, ldSrc int)` | ✅ | Low | None | 🔀 | New |

### Unary Operations

| Operation | Generic Function | Status | Performance | Tests | Threading | Source |
|-----------|------------------|--------|-------------|-------|-----------|--------|
| Sign Strided | `ElemMatSignStrided[T](dst, src []T, rows, cols int, ldDst, ldSrc int)` | ✅ | None/Minimal | None | 🔒 | New |
| Negative Strided | `ElemMatNegativeStrided[T](dst, src []T, rows, cols int, ldDst, ldSrc int)` | ✅ | None/Minimal | None | 🔒 | New |

### Apply Operations

| Operation | Generic Function | Status | Performance | Tests | Threading | Source |
|-----------|------------------|--------|-------------|-------|-----------|--------|
| Apply Unary Strided | `ElemMatApplyUnaryStrided[T](dst, src []T, rows, cols int, ldDst, ldSrc int, op func(T) T)` | ✅ | None/Minimal | Complete | 🔀 | New |
| Apply Binary Strided | `ElemMatApplyBinaryStrided[T](dst, a, b []T, rows, cols int, ldDst, ldA, ldB int, op func(T, T) T)` | ✅ | None/Minimal | Complete | 🔀 | New |
| Apply Ternary Strided | `ElemMatApplyTernaryStrided[T](dst, condition, a, b []T, rows, cols int, ldDst, ldCond, ldA, ldB int, op func(T, T, T) T)` | ✅ | None/Minimal | Complete | 🔀 | New |
| Apply Unary Scalar Strided | `ElemMatApplyUnaryScalarStrided[T](dst, src []T, scalar T, rows, cols int, ldDst, ldSrc int, op func(T, T) T)` | ✅ | None/Minimal | Complete | 🔀 | New |
| Apply Binary Scalar Strided | `ElemMatApplyBinaryScalarStrided[T](dst, a []T, scalar T, rows, cols int, ldDst, ldA int, op func(T, T) T)` | ✅ | None/Minimal | Complete | 🔀 | New |
| Apply Ternary Scalar Strided | `ElemMatApplyTernaryScalarStrided[T](dst, condition []T, scalar T, rows, cols int, ldDst, ldCond int, op func(T, T, T) T)` | ✅ | None/Minimal | Complete | 🔀 | New |

---

## Scalar Operations

Operations that apply a scalar value element-wise or fill arrays with constant values.

| Operation | Generic Function | Status | Performance | Tests | Threading | Source |
|-----------|------------------|--------|-------------|-------|-----------|--------|
| Fill (Contiguous) | `ElemFill[T](dst []T, value T, n int)` | ✅ | None/Minimal | None | 🔒 | `fp32/tensor_elementwise.go:268` |
| Fill (Strided) | `ElemFillStrided[T](dst []T, value T, shape []int, stridesDst []int)` | ✅ | None/Minimal | None | 🔒 | `fp32/tensor_elementwise.go:268` |
| Value Convert | `ValueConvert[T, U](value T) U` | ✅ | Low | None | 🔒 | `primitive/copy.go:14` |

**Note**: 
- `ElemFill` writes `value` to all elements of `dst` according to `shape` and `stridesDst`.
- `ValueConvert` converts a single scalar value from type `T` to type `U` with appropriate clamping for down-conversions. This is a scalar operation (operates on a single value, not an array), so it does not support multithreading.

**Excluded**: Arithmetic scalar operations (`AddScalar`, `SubScalar`, `MulScalar`, `DivScalar`) - rely on type precision and clamping.

---

## Boolean Operations

Comparison operations that return numeric values (0 or 1) representing boolean results. These operations work for all numeric types.

### Tensor Comparison Operations

| Operation | Generic Function | Status | Performance | Tests | Threading | Source |
|-----------|------------------|--------|-------------|-------|-----------|--------|
| Greater Than (Contiguous) | `ElemGreaterThan[T](dst, a, b []T, n int)` | ✅ | None/Minimal | Complete | 🔀 | `fp32/tensor_elementwise.go:136` |
| Greater Than (Strided) | `ElemGreaterThanStrided[T](dst, a, b []T, shape []int, stridesDst, stridesA, stridesB []int)` | ✅ | None/Minimal | Complete | 🔀 | `fp32/tensor_elementwise.go:136` |
| Equal (Contiguous) | `ElemEqual[T](dst, a, b []T, n int)` | ✅ | None/Minimal | Complete | 🔀 | `fp32/tensor_elementwise.go:146` |
| Equal (Strided) | `ElemEqualStrided[T](dst, a, b []T, shape []int, stridesDst, stridesA, stridesB []int)` | ✅ | None/Minimal | Complete | 🔀 | `fp32/tensor_elementwise.go:146` |
| Less (Contiguous) | `ElemLess[T](dst, a, b []T, n int)` | ✅ | None/Minimal | Complete | 🔀 | `fp32/tensor_elementwise.go:156` |
| Less (Strided) | `ElemLessStrided[T](dst, a, b []T, shape []int, stridesDst, stridesA, stridesB []int)` | ✅ | None/Minimal | Complete | 🔀 | `fp32/tensor_elementwise.go:156` |
| Not Equal (Contiguous) | `ElemNotEqual[T](dst, a, b []T, n int)` | ✅ | None/Minimal | Complete | 🔀 | `fp32/tensor_elementwise.go:352` |
| Not Equal (Strided) | `ElemNotEqualStrided[T](dst, a, b []T, shape []int, stridesDst, stridesA, stridesB []int)` | ✅ | None/Minimal | Complete | 🔀 | `fp32/tensor_elementwise.go:352` |
| Less Equal (Contiguous) | `ElemLessEqual[T](dst, a, b []T, n int)` | ✅ | None/Minimal | Complete | 🔀 | `fp32/tensor_elementwise.go:362` |
| Less Equal (Strided) | `ElemLessEqualStrided[T](dst, a, b []T, shape []int, stridesDst, stridesA, stridesB []int)` | ✅ | None/Minimal | Complete | 🔀 | `fp32/tensor_elementwise.go:362` |
| Greater Equal (Contiguous) | `ElemGreaterEqual[T](dst, a, b []T, n int)` | ✅ | None/Minimal | Complete | 🔀 | `fp32/tensor_elementwise.go:372` |
| Greater Equal (Strided) | `ElemGreaterEqualStrided[T](dst, a, b []T, shape []int, stridesDst, stridesA, stridesB []int)` | ✅ | None/Minimal | Complete | 🔀 | `fp32/tensor_elementwise.go:372` |

### Vector Comparison Operations

| Operation | Generic Function | Status | Performance | Tests | Threading | Source |
|-----------|------------------|--------|-------------|-------|-----------|--------|
| Greater Than Strided | `ElemVecGreaterThanStrided[T](dst, a, b []T, n int, strideDst, strideA, strideB int)` | ✅ | None/Minimal | None | 🔒 | New |
| Equal Strided | `ElemVecEqualStrided[T](dst, a, b []T, n int, strideDst, strideA, strideB int)` | ✅ | None/Minimal | None | 🔒 | New |
| Less Strided | `ElemVecLessStrided[T](dst, a, b []T, n int, strideDst, strideA, strideB int)` | ✅ | None/Minimal | None | 🔒 | New |
| Not Equal Strided | `ElemVecNotEqualStrided[T](dst, a, b []T, n int, strideDst, strideA, strideB int)` | ✅ | None/Minimal | None | 🔒 | New |
| Less Equal Strided | `ElemVecLessEqualStrided[T](dst, a, b []T, n int, strideDst, strideA, strideB int)` | ✅ | None/Minimal | None | 🔒 | New |
| Greater Equal Strided | `ElemVecGreaterEqualStrided[T](dst, a, b []T, n int, strideDst, strideA, strideB int)` | ✅ | None/Minimal | None | 🔒 | New |

### Matrix Comparison Operations

| Operation | Generic Function | Status | Performance | Tests | Threading | Source |
|-----------|------------------|--------|-------------|-------|-----------|--------|
| Greater Than Strided | `ElemMatGreaterThanStrided[T](dst, a, b []T, rows, cols int, ldDst, ldA, ldB int)` | ✅ | None/Minimal | None | 🔒 | New |
| Equal Strided | `ElemMatEqualStrided[T](dst, a, b []T, rows, cols int, ldDst, ldA, ldB int)` | ✅ | None/Minimal | None | 🔒 | New |
| Less Strided | `ElemMatLessStrided[T](dst, a, b []T, rows, cols int, ldDst, ldA, ldB int)` | ✅ | None/Minimal | None | 🔒 | New |
| Not Equal Strided | `ElemMatNotEqualStrided[T](dst, a, b []T, rows, cols int, ldDst, ldA, ldB int)` | ✅ | None/Minimal | None | 🔒 | New |
| Less Equal Strided | `ElemMatLessEqualStrided[T](dst, a, b []T, rows, cols int, ldDst, ldA, ldB int)` | ✅ | None/Minimal | None | 🔒 | New |
| Greater Equal Strided | `ElemMatGreaterEqualStrided[T](dst, a, b []T, rows, cols int, ldDst, ldA, ldB int)` | ✅ | None/Minimal | None | 🔒 | New |

### Scalar Comparison Operations

| Operation | Generic Function | Status | Performance | Tests | Threading | Source |
|-----------|------------------|--------|-------------|-------|-----------|--------|
| Equal Scalar (Contiguous) | `ElemEqualScalar[T](dst, src []T, scalar T, n int)` | ✅ | None/Minimal | None | 🔒 | New |
| Equal Scalar (Strided) | `ElemEqualScalarStrided[T](dst, src []T, scalar T, shape []int, stridesDst, stridesSrc []int)` | ✅ | None/Minimal | None | 🔒 | New |
| Greater Scalar (Contiguous) | `ElemGreaterScalar[T](dst, src []T, scalar T, n int)` | ✅ | None/Minimal | None | 🔒 | New |
| Greater Scalar (Strided) | `ElemGreaterScalarStrided[T](dst, src []T, scalar T, shape []int, stridesDst, stridesSrc []int)` | ✅ | None/Minimal | None | 🔒 | New |
| Less Scalar (Contiguous) | `ElemLessScalar[T](dst, src []T, scalar T, n int)` | ✅ | None/Minimal | None | 🔒 | New |
| Less Scalar (Strided) | `ElemLessScalarStrided[T](dst, src []T, scalar T, shape []int, stridesDst, stridesSrc []int)` | ✅ | None/Minimal | None | 🔒 | New |
| Not Equal Scalar (Contiguous) | `ElemNotEqualScalar[T](dst, src []T, scalar T, n int)` | ✅ | None/Minimal | None | 🔒 | New |
| Not Equal Scalar (Strided) | `ElemNotEqualScalarStrided[T](dst, src []T, scalar T, shape []int, stridesDst, stridesSrc []int)` | ✅ | None/Minimal | None | 🔒 | New |
| Less Equal Scalar (Contiguous) | `ElemLessEqualScalar[T](dst, src []T, scalar T, n int)` | ✅ | None/Minimal | None | 🔒 | New |
| Less Equal Scalar (Strided) | `ElemLessEqualScalarStrided[T](dst, src []T, scalar T, shape []int, stridesDst, stridesSrc []int)` | ✅ | None/Minimal | None | 🔒 | New |
| Greater Equal Scalar (Contiguous) | `ElemGreaterEqualScalar[T](dst, src []T, scalar T, n int)` | ✅ | None/Minimal | None | 🔒 | New |
| Greater Equal Scalar (Strided) | `ElemGreaterEqualScalarStrided[T](dst, src []T, scalar T, shape []int, stridesDst, stridesSrc []int)` | ✅ | None/Minimal | None | 🔒 | New |

**Note**: Comparison operations return numeric type (0 or 1) which works for all numeric types. They roll out their own loops for optimization (not using `ElemApplyBinary`).

---

## BLAS Operations

BLAS Level 1 vector operations. These follow the same contiguous/strided pattern as tensor operations for consistency.

| Operation | Generic Function | Status | Performance | Tests | Threading | Source |
|-----------|------------------|--------|-------------|-------|-----------|--------|
| Copy (Contiguous) | `Copy[T](y, x []T, n int)` | ✅ | None/Minimal | None | 🔒 | `fp32/level1.go:103` |
| Copy (Strided) | `CopyStrided[T](y, x []T, strideY, strideX, n int)` | ✅ | None/Minimal | None | 🔒 | `fp32/level1.go:103` |
| Swap (Contiguous) | `Swap[T](x, y []T, n int)` | ✅ | None/Minimal | None | 🔒 | `fp32/level1.go:120` |
| Swap (Strided) | `SwapStrided[T](x, y []T, strideX, strideY, n int)` | ✅ | None/Minimal | None | 🔒 | `fp32/level1.go:120` |

**Note**: Contiguous versions are optimized for the common case (stride == 1). Strided versions handle arbitrary strides. `Swap` uses tuple assignment which works for all types.

---

## Helper Operations

Utility functions for stride iteration and shape manipulation. These are already generic (work with any numeric type via slice indexing).

### Shape and Stride Utilities

| Operation | Generic Function | Status | Performance | Tests | Source |
|-----------|------------------|--------|-------------|-------|--------|
| Compute Strides Rank | `ComputeStridesRank(shape []int) int` | ✅ | None/Minimal | Complete | New |
| Compute Strides | `ComputeStrides(shape []int) []int` | ✅ | None/Minimal | Complete | `fp32/tensor_helpers.go:11` |
| Size From Shape | `SizeFromShape(shape []int) int` | ✅ | None/Minimal | Complete | `fp32/tensor_helpers.go:26` |
| Ensure Strides | `EnsureStrides(strides []int, shape []int) []int` | ✅ | None/Minimal | Complete | `fp32/tensor_helpers.go:41` |
| Is Contiguous | `IsContiguous(strides []int, shape []int) bool` | ✅ | None/Minimal | Complete | `fp32/tensor_helpers.go:52` |
| Advance Offsets | `AdvanceOffsets(shape []int, indices []int, offsets []int, strides [][]int) bool` | ✅ | None/Minimal | Complete | `fp32/tensor_helpers.go:111` |
| Iterate Offsets | `IterateOffsets(shape []int, strides [][]int, callback func(offsets []int))` | ✅ | None/Minimal | None | `fp32/tensor_helpers.go:138` |
| Iterate Offsets With Indices | `IterateOffsetsWithIndices(shape []int, strides [][]int, callback func(indices []int, offsets []int))` | ✅ | None/Minimal | None | `fp32/tensor_helpers.go:157` |
| Compute Stride Offset | `ComputeStrideOffset(indices []int, strides []int) int` | ✅ | None/Minimal | None | `primitive/copy.go:823` |

### Iterator Functions

**⚠️ PERFORMANCE WARNING**: Generic iterator functions (`Elements`, `ElementsStrided`, `ElementsVec`, `ElementsMat`, `ElementsVecStrided`, `ElementsMatStrided`) are currently **too slow for production use** in performance-critical code paths. These iterators have significant overhead compared to direct nested loops. **Do NOT use these iterators** in hot paths such as:
- Convolution operations (Im2Col, Col2Im)
- Window-based operations (pooling, windowed convolutions)
- Any operation requiring iteration over multi-dimensional indices in performance-critical code

**Use direct nested loops instead** for these operations until iterator performance is improved.

Iterator functions that can be used with Go's `range` keyword. These return iterator functions compatible with `iter.Seq` (Go 1.23+).

| Operation | Generic Function | Status | Performance | Tests | Threading | Source |
|-----------|------------------|--------|-------------|-------|-----------|--------|
| Elements | `Elements(shape []int) func(func([]int) bool)` | ✅ | ⚠️ **TOO SLOW** | Complete | 🔀 | New |
| Elements Strided | `ElementsStrided(shape []int, strides []int) func(func([]int) bool)` | ✅ | ⚠️ **TOO SLOW** | Complete | 🔀 | New |
| Elements Indices | `ElementsIndices(shape []int, dims ...int) func(func([]int) bool)` | ✅ | ⚠️ **TOO SLOW** | Complete | 🔀 | New |
| Elements Indices Strided | `ElementsIndicesStrided(shape []int, strides []int, dims ...int) func(func([]int) bool)` | ✅ | ⚠️ **TOO SLOW** | Complete | 🔀 | New |
| Elements Vec | `ElementsVec(n int) func(func(int) bool)` | ✅ | ⚠️ **TOO SLOW** | Complete | 🔀 | New |
| Elements Vec Strided | `ElementsVecStrided(n int, stride int) func(func(int) bool)` | ✅ | ⚠️ **TOO SLOW** | Complete | 🔀 | New |
| Elements Mat | `ElementsMat(rows, cols int) func(func([2]int) bool)` | ✅ | ⚠️ **TOO SLOW** | Complete | 🔀 | New |
| Elements Mat Strided | `ElementsMatStrided(rows, cols int, ld int) func(func([2]int) bool)` | ✅ | ⚠️ **TOO SLOW** | Complete | 🔀 | New |

**Usage Examples:**
```go
// Tensor: iterate over multi-dimensional indices
for indices := range Elements(shape) {
    // indices is []int
}

// Tensor: iterate over selected dimensions only
for indices := range ElementsIndices(shape, 0, 2) {
    // indices is []int for dimensions 0 and 2 only
}

// Tensor: iterate over all dimensions (equivalent to Elements)
for indices := range ElementsIndices(shape) {
    // indices is []int for all dimensions
}

// Tensor: iterate over selected dimensions with strides
for indices := range ElementsIndicesStrided(shape, strides, 0, 2) {
    // indices is []int for dimensions 0 and 2 only
}

// Tensor: iterate over all dimensions with strides (equivalent to ElementsStrided)
for indices := range ElementsIndicesStrided(shape, strides) {
    // indices is []int for all dimensions
}

// Vector: iterate over linear indices
for idx := range ElementsVec(n) {
    // idx is int
}

// Matrix: iterate over (row, col) tuples
for idx := range ElementsMat(rows, cols) {
    // idx is [2]int (row, col)
}
```

**Note**: 
- ⚠️ **These iterators are currently too slow for production use** - use direct nested loops in performance-critical code paths
- `Elements` and `ElementsStrided` yield `[]int` representing multi-dimensional indices
- `ElementsIndices` yields `[]int` representing indices for selected dimensions only. If `dims` is empty or nil, iterates over all dimensions (equivalent to `Elements`)
- `ElementsIndicesStrided` yields `[]int` representing indices for selected dimensions only with stride support. If `dims` is empty or nil, iterates over all dimensions (equivalent to `ElementsStrided`)
- `ElementsVec` and `ElementsVecStrided` yield `int` representing linear indices (scalar for vectors)
- `ElementsMat` and `ElementsMatStrided` yield `[2]int` representing (row, col) tuples
- All iterators support early exit when `yield` returns `false` (note: early exit behavior may differ in multi-threaded mode)
- `IterateOffsets` and `IterateOffsetsWithIndices` are callback-based convenience wrappers (these are fine to use, they're not iterator-based)

---

## Implementation Status Summary

### ✅ Completed Operations

**Tensor Operations:**
- Copy: `ElemCopy`, `ElemCopyStrided`, `ElemSwap`, `ElemSwapStrided`
- Convert: `ElemConvert`, `ElemConvertStrided`
- Unary: `ElemSign`, `ElemSignStrided`, `ElemNegative`, `ElemNegativeStrided`
- Ternary: `ElemWhere`
- Apply: All contiguous and strided versions (Binary, Unary, Ternary, Scalar variants)
- Boolean: All comparison operations (GreaterThan, Equal, Less, NotEqual, LessEqual, GreaterEqual) with contiguous and strided versions

**Vector Operations:**
- Copy: `ElemVecCopyStrided`, `ElemVecConvertStrided`
- Unary: `ElemVecSignStrided`, `ElemVecNegativeStrided`
- Apply: All strided versions (Unary, Binary, Ternary, Scalar variants)
- Boolean: All comparison operations (strided versions)

**Matrix Operations:**
- Copy: `ElemMatCopyStrided`, `ElemMatConvertStrided`
- Unary: `ElemMatSignStrided`, `ElemMatNegativeStrided`
- Apply: All strided versions (Unary, Binary, Ternary, Scalar variants)
- Boolean: All comparison operations (strided versions)

**Scalar Operations:**
- Fill: `ElemFill`, `ElemFillStrided`
- Scalar comparisons: All 6 operations with contiguous and strided versions (`ElemEqualScalar`, `ElemGreaterScalar`, `ElemLessScalar`, `ElemNotEqualScalar`, `ElemLessEqualScalar`, `ElemGreaterEqualScalar`)

**BLAS Operations:**
- `Copy`, `CopyStrided`, `Swap`, `SwapStrided`

**Helper Operations:**
- Shape/Stride utilities: `ComputeStridesRank`, `ComputeStrides`, `SizeFromShape`, `EnsureStrides`, `IsContiguous`, `AdvanceOffsets`, `IterateOffsets`, `IterateOffsetsWithIndices`, `ComputeStrideOffset`
- Iterator functions: `Elements`, `ElementsStrided`, `ElementsIndices`, `ElementsIndicesStrided`, `ElementsVec`, `ElementsVecStrided`, `ElementsMat`, `ElementsMatStrided`

### ⏳ Pending Operations

None - All specified operations are implemented!

### Missing Test Coverage

- `ElemConvert`, `ElemConvertStrided` - Conversion operations (implemented, tests pending)
- `ElemVecConvertStrided`, `ElemMatConvertStrided` - Vector/matrix conversion (implemented, tests pending)
- All vector/matrix optimized unary and comparison operations (18 operations) (implemented, tests pending)
- `ElemWhere` - Ternary operation (implemented, tests pending)
- `ElemFillStrided` - Scalar fill strided (implemented, tests pending)
- Scalar comparison strided versions (6 operations) (implemented, tests pending)
- Helper functions: `IterateOffsets`, `IterateOffsetsWithIndices`, `ComputeStrideOffset` (implemented, tests pending)

---

## Notes

- ✅ Generic operations are implemented in `primitive/generics` package
- ✅ Type constraint `Numeric` supports: `~float64 | ~int64 | ~float32 | ~int | ~int32 | ~int16 | ~int8`
- ✅ **Implementation Requirements**: See [Implementation Requirements](#implementation-requirements) for requirements on specialized implementations and optimization patterns
- ✅ Performance optimizations: Split contiguous/strided versions, `ComputeStridesRank` for rank-only checks, boundary check elimination patterns
- Type-specific optimizations (like SIMD) may still be needed for float32/float64
- Benchmarks compare generic, non-generic, and direct loop implementations
- Vector and matrix operations are optimized for 1D and 2D data respectively, providing better performance than general tensor operations for these common cases
