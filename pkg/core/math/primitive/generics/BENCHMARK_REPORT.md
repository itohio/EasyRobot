# Generic Operations Benchmark Report

**Generated:** November 6, 2025  
**Platform:** Linux amd64  
**CPU:** Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz  
**Package:** `github.com/itohio/EasyRobot/pkg/core/math/primitive/generics`

**⚠️ IMPORTANT:** This report tracks benchmark results over time. When updating this report:
- Preserve all historical comparison columns (shift left when adding new run)
- Keep only the 5 most recent historical comparisons
- Calculate percentage as: `(implementation_ns/op / baseline_ns/op - 1) * 100`
- Negative percentages mean faster than baseline, positive means slower
- Always include allocations for Generic (G) column when applicable

## Summary

This report compares the performance of generic implementations against non-generic and direct loop implementations. All benchmarks use 10,000 elements unless otherwise specified.

**Latest Run:** November 6, 2025

**Legend:**
- **G** = Generic implementation (ns/op, allocations)
- **NG** = Non-Generic implementation (ns/op, allocations)
- **D** = Direct Loop baseline (ns/op)
- **H1-H5** = Historical comparisons (% vs baseline, most recent first)

---

## Performance Comparison Table

| Operation | G (ns/op, allocs) | NG (ns/op, allocs) | D (ns/op) | H1 (% vs D) | H2 (% vs D) | H3 (% vs D) | H4 (% vs D) | H5 (% vs D) |
|-----------|-------------------|---------------------|-----------|-------------|-------------|-------------|-------------|-------------|
| **Copy Operations** |
| ElemCopy | 1,674 (0 B, 0 allocs) | 1,590 (0 B, 0 allocs) | 12,548 | -86.7% | -86.1% | - | - | - |
| ElemCopyStrided | 1,724 (32 B, 2 allocs) | 66,170 (16 B, 1 alloc) | 13,582 | -87.3% | -88.6% | - | - | - |
| BLAS Copy | 1,543 (0 B, 0 allocs) | 1,507 (0 B, 0 allocs) | 1,437 | +7.4% | - | - | - | - |
| BLAS CopyStrided | 1,452 (0 B, 0 allocs) | 1,634 (0 B, 0 allocs) | 1,437 | +1.0% | - | - | - | - |
| **Swap Operations** |
| ElemSwap | 11,270 (0 B, 0 allocs) | 9,383 (0 B, 0 allocs) | 16,721 | -32.6% | -38.6% | - | - | - |
| ElemSwapStrided | 10,117 (32 B, 2 allocs) | 63,097 (16 B, 1 alloc) | 21,882 | -53.8% | -39.0% | - | - | - |
| BLAS Swap | 10,209 (0 B, 0 allocs) | 10,195 (0 B, 0 allocs) | 16,323 | -37.5% | - | - | - | - |
| BLAS SwapStrided | 9,891 (0 B, 0 allocs) | 9,984 (0 B, 0 allocs) | 16,323 | -39.4% | - | - | - | - |
| **Apply Operations** |
| ElemApplyUnary | 10,690 (0 B, 0 allocs) | 9,113 (0 B, 0 allocs) | 12,884 | -17.0% | -25.9% | - | - | - |
| ElemApplyBinary | 10,010 (0 B, 0 allocs) | 9,738 (0 B, 0 allocs) | 21,488 | -53.4% | -43.4% | - | - | - |
| ElemApplyBinaryStrided | 32,459 (48 B, 3 allocs) | 30,321 (48 B, 3 allocs) | 20,361 | +59.4% | +40.1% | - | - | - |
| ElemApplyUnaryScalar | 12,836 (0 B, 0 allocs) | 12,437 (0 B, 0 allocs) | 40,874 | -68.6% | -34.0% | - | - | - |
| ElemApplyUnaryScalarStrided | 40,485 (32 B, 2 allocs) | 30,117 (32 B, 2 allocs) | 24,855 | +62.9% | +56.8% | - | - | - |
| **Comparison Operations** |
| ElemGreaterThan | 19,296 (0 B, 0 allocs) | 21,180 (0 B, 0 allocs) | 20,622 | -6.4% | -31.7% | - | - | - |
| ElemGreaterThanStrided | 14,884 (48 B, 3 allocs) | 15,287 (48 B, 3 allocs) | 23,495 | -36.7% | -32.7% | - | - | - |
| ElemEqual | 16,008 (0 B, 0 allocs) | 15,456 (0 B, 0 allocs) | 22,848 | -29.9% | - | - | - | - |
| ElemLess | 16,774 (0 B, 0 allocs) | 16,684 (0 B, 0 allocs) | 19,648 | -14.6% | - | - | - | - |
| **Unary Operations** |
| ElemSign | 18,054 (0 B, 0 allocs) | 18,361 (0 B, 0 allocs) | 18,675 | -3.3% | +1.7% | - | - | - |
| ElemSignStrided | 14,877 (32 B, 2 allocs) | 15,301 (32 B, 2 allocs) | 18,758 | -20.7% | -26.8% | - | - | - |
| ElemNegative | 9,208 (0 B, 0 allocs) | 9,788 (0 B, 0 allocs) | 15,629 | -41.1% | -45.7% | - | - | - |
| ElemNegativeStrided | 11,181 (32 B, 2 allocs) | 9,487 (32 B, 2 allocs) | 14,721 | -24.0% | -29.2% | - | - | - |
| **Scalar Operations** |
| ElemFill | 9,706 (0 B, 0 allocs) | 6,117 (0 B, 0 allocs) | 27,238 | -64.4% | - | - | - | - |
| ElemEqualScalar | 23,024 (0 B, 0 allocs) | 22,238 (0 B, 0 allocs) | 30,751 | -25.1% | - | - | - | - |
| ElemGreaterScalar | 23,017 (0 B, 0 allocs) | 13,472 (0 B, 0 allocs) | 17,658 | +30.3% | - | - | - | - |
| **Vector/Matrix Apply Operations** |
| ElemVecApplyStrided | 25,536 (0 B, 0 allocs) | 30,210 (0 B, 0 allocs) | 16,016 | +59.5% | +99.1% | - | - | - |
| ElemMatApplyStrided | 34,117 (0 B, 0 allocs) | 25,422 (0 B, 0 allocs) | 14,464 | +135.8% | +94.8% | - | - | - |
| **Iterator Operations** |
| Elements | 94,316 (88 B, 5 allocs) | 427,740 (160,056 B, 10,004 allocs) | 6,089 | +1448.9% | +7566.0% | - | - | - |
| ElementsVec | 3,954 (0 B, 0 allocs) | 7,931 (0 B, 0 allocs) | 4,077 | -3.0% | - | - | - | - |
| ElementsVecStrided | 8,390 (0 B, 0 allocs) | 4,028 (0 B, 0 allocs) | 4,077 | +105.8% | - | - | - | - |
| ElementsMat | 5,051 (0 B, 0 allocs) | 4,959 (0 B, 0 allocs) | 5,179 | -2.5% | - | - | - | - |
| ElementsMatStrided | 5,217 (0 B, 0 allocs) | 5,691 (0 B, 0 allocs) | 5,179 | +0.7% | - | - | - | - |
| **Conversion Operations** |
| ElemConvert | 18,586 (0 B, 0 allocs) | 11,046 (0 B, 0 allocs) | 27,248 | -31.8% | +5.9% | - | - | - |
| ElemConvert (Clamping) | 13,651 (0 B, 0 allocs) | 10,480 (0 B, 0 allocs) | 5,071 | +169.2% | +138.8% | - | - | - |
| ElemConvertStrided | 118,101 (48 B, 3 allocs) | 116,611 (16 B, 1 alloc) | 16,689 | +607.7% | +612.4% | - | - | - |
| ValueConvert | 2.762 (0 B, 0 allocs) | 0.4256 (0 B, 0 allocs) | N/A | N/A | - | - | - | - |
| ValueConvert (Clamping) | 5.178 (0 B, 0 allocs) | 0.5561 (0 B, 0 allocs) | N/A | N/A | - | - | - | - |

**Note:** ValueConvert operations don't have a Direct Loop baseline as they operate on single values, not arrays.

---

## Historical Run Dates

- **H1 (Current):** November 6, 2025
- **H2 (Previous):** November 6, 2025 (initial run)

---

## Key Findings

### Strengths

1. **Copy Operations:** Generic `ElemCopy` and `ElemCopyStrided` are highly optimized, often faster than direct loops due to builtin `copy()` optimization. ElemCopy is 86.7% faster than baseline.

2. **BLAS Operations:** BLAS `Copy` and `Swap` operations perform excellently, with generic versions competitive with non-generic and direct loops.

3. **Contiguous Operations:** Generic contiguous operations (Unary, Binary, Scalar) perform comparably to non-generic versions, typically 14-68% faster than direct loops.

4. **Swap Operations:** Generic swap operations perform well, competitive with non-generic versions. ElemSwap is 32.6% faster than baseline.

5. **Vector/Matrix Iterators:** `ElementsVec` and `ElementsMat` iterators perform excellently, comparable to direct loops with minimal overhead.

6. **Tensor Iterator Optimization:** `Elements` iterator optimized to eliminate allocations (99.95% reduction: from 10,005 to 5 allocations) by reusing the indices slice, resulting in 79% performance improvement from previous run.

7. **Memory Efficiency:** Most operations have zero allocations for contiguous paths.

### Areas for Improvement

1. **Strided Operations:** Some strided operations have overhead from stride iteration logic, but this is expected and necessary for correctness.

2. **Type Conversion:** Generic type conversion has overhead from type dispatch, but provides type safety and flexibility.

3. **Vector/Matrix Apply Operations:** Some overhead from stride/leading dimension checks, but still reasonable for the flexibility provided.

4. **Elements Iterator:** Still slower than direct loop due to iterator overhead, but massive improvement from previous run (79% faster, 99.95% fewer allocations).

---

## Recommendations

1. **Use contiguous versions when possible:** The contiguous versions (`ElemCopy`, `ElemApplyUnary`, etc.) are highly optimized and should be preferred when arrays are known to be contiguous.

2. **Use BLAS operations for simple cases:** BLAS `Copy` and `Swap` are highly optimized and perform excellently for simple vector operations.

3. **Strided versions for flexibility:** Use strided versions when dealing with non-contiguous memory layouts, accepting the reasonable overhead for correctness.

4. **Use vector/matrix iterators:** `ElementsVec` and `ElementsMat` iterators perform excellently with minimal overhead.

5. **Use tensor iterator efficiently:** `Elements` iterator is now optimized with zero allocations per iteration. The indices slice is reused, so copy it if you need to store indices. For maximum performance, use direct loops in the hottest paths.

6. **Type conversion:** The generic conversion functions provide type safety at a reasonable cost. For hot paths with known types, consider type-specific fast paths.

7. **Benchmark your use case:** These benchmarks use 10,000 elements. Performance characteristics may differ for different sizes and access patterns.

---

## Conclusion

The generic implementations provide excellent performance for contiguous operations, competitive with non-generic versions while providing type safety and code reuse. Many operations showed significant improvements from the previous benchmark run. Strided operations have expected overhead but are necessary for handling non-contiguous memory layouts. The generic approach successfully balances performance, type safety, and code maintainability.
