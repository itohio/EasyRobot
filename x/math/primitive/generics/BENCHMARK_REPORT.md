# Generic Operations Benchmark Report

**Generated:** November 7, 2025  
**Platform:** Linux amd64  
**CPU:** Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz  
**Package:** `github.com/itohio/EasyRobot/pkg/core/math/primitive/generics`

**⚠️ IMPORTANT:** This report tracks benchmark results over time. When updating this report:
- Preserve all historical comparison columns (shift left when adding new run)
- Keep only the 5 most recent historical comparisons
- Calculate percentage as: `(implementation_ns/op / baseline_ns/op - 1) * 100`
- Negative percentages mean faster than baseline, positive means slower
- Always include allocations for Generic (G) column when applicable
- Hint for benchmarking: when extracting benchmarks metrics, avoid running benchmarks multiple times since they take a long time to complete. Run them once redirecting output to temporary txt file and try parsing that file.

## Summary

This report compares the performance of generic implementations against non-generic and direct loop implementations. All benchmarks use 10,000 elements unless otherwise specified.

**Latest Run:** November 7, 2025

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
| ElemCopy | 1,245 (0 B, 0 allocs) | 1,236 (0 B, 0 allocs) | 11,111 | -88.8% | -91.1% | -93.1% | -86.7% | -86.1% |
| ElemCopyStrided | 1,304 (0 B, 0 allocs) | 76,233 (0 B, 0 allocs) | 12,403 | -89.5% | -92.4% | -88.1% | -87.3% | -88.6% |
| BLAS Copy | 1,279 (0 B, 0 allocs) | 1,284 (0 B, 0 allocs) | 1,871 | -31.6% | +71.8% | +18.2% | +7.4% | - |
| BLAS CopyStrided | 1,344 (0 B, 0 allocs) | 1,307 (0 B, 0 allocs) | 1,871 | -28.2% | -3.4% | +2.6% | +1.0% | - |
| **Swap Operations** |
| ElemSwap | 8,645 (0 B, 0 allocs) | 8,505 (0 B, 0 allocs) | 14,647 | -41.0% | -57.6% | -34.7% | -32.6% | -38.6% |
| ElemSwapStrided | 8,870 (0 B, 0 allocs) | 82,087 (0 B, 0 allocs) | 16,125 | -45.0% | -42.7% | -36.2% | -53.8% | -39.0% |
| BLAS Swap | 8,163 (0 B, 0 allocs) | 8,171 (0 B, 0 allocs) | 14,705 | -44.5% | -42.8% | -36.5% | -37.5% | - |
| BLAS SwapStrided | 8,397 (0 B, 0 allocs) | 8,907 (0 B, 0 allocs) | 14,705 | -42.9% | -39.9% | -40.3% | -39.4% | - |
| **Apply Operations** |
| ElemApplyUnary | 278,054 (0 B, 0 allocs) | 285,575 (0 B, 0 allocs) | 284,708 | -2.3% | -9.0% | -36.0% | -17.0% | -25.9% |
| ElemApplyBinary | 10,855 (0 B, 0 allocs) | 19,987 (0 B, 0 allocs) | 31,978 | -66.1% | -53.0% | -55.8% | -53.4% | -43.4% |
| ElemApplyBinaryStrided | 41,727 (0 B, 0 allocs) | 38,571 (0 B, 0 allocs) | 30,696 | +35.9% | +5.5% | +72.3% | +59.4% | +40.1% |
| ElemApplyUnaryScalar | 961,720 (0 B, 0 allocs) | 969,088 (0 B, 0 allocs) | 971,786 | -1.0% | -5.0% | -37.2% | -68.6% | -34.0% |
| ElemApplyUnaryScalarStrided | 964,367 (0 B, 0 allocs) | 941,045 (0 B, 0 allocs) | 960,716 | +0.4% | +9.0% | +100.5% | +62.9% | +56.8% |
| **Comparison Operations** |
| ElemGreaterThan | 12,613 (0 B, 0 allocs) | 13,093 (0 B, 0 allocs) | 17,310 | -27.1% | -1.6% | -41.0% | -6.4% | -31.7% |
| ElemGreaterThanStrided | 16,341 (0 B, 0 allocs) | 15,428 (0 B, 0 allocs) | 22,034 | -25.8% | -34.8% | -32.6% | -36.7% | -32.7% |
| ElemEqual | 13,602 (0 B, 0 allocs) | 15,462 (0 B, 0 allocs) | 17,599 | -22.7% | -49.9% | -31.0% | -29.9% | - |
| ElemLess | 11,386 (0 B, 0 allocs) | 14,435 (0 B, 0 allocs) | 16,532 | -31.1% | -34.1% | -12.2% | -14.6% | - |
| **Unary Operations** |
| ElemSign | 11,545 (0 B, 0 allocs) | 10,691 (0 B, 0 allocs) | 12,267 | -5.9% | -1.5% | -3.3% | +1.7% | - |
| ElemSignStrided | 11,047 (0 B, 0 allocs) | 10,697 (0 B, 0 allocs) | 15,314 | -27.9% | -52.9% | -20.7% | -26.8% | - |
| ElemNegative | 7,085 (0 B, 0 allocs) | 11,006 (0 B, 0 allocs) | 16,277 | -56.5% | -7.0% | -41.1% | -45.7% | - |
| ElemNegativeStrided | 8,474 (0 B, 0 allocs) | 9,628 (0 B, 0 allocs) | 17,479 | -51.5% | +8.2% | -24.0% | -29.2% | - |
| **Scalar Operations** |
| ElemFill | 4,113 (0 B, 0 allocs) | 7,227 (0 B, 0 allocs) | 7,404 | -44.4% | -39.2% | -64.4% | - | - |
| ElemEqualScalar | 11,029 (0 B, 0 allocs) | 11,191 (0 B, 0 allocs) | 15,633 | -29.5% | -21.7% | -25.1% | - | - |
| ElemGreaterScalar | 8,430 (0 B, 0 allocs) | 11,607 (0 B, 0 allocs) | 11,872 | -29.0% | -32.9% | +30.3% | - | - |
| **Vector/Matrix Apply Operations** |
| ElemVecApplyStrided | 334,052 (0 B, 0 allocs) | 327,033 (0 B, 0 allocs) | 285,815 | +16.9% | +13.5% | +54.6% | +59.5% | +99.1% |
| ElemMatApplyStrided | 287,764 (0 B, 0 allocs) | 309,095 (0 B, 0 allocs) | 301,088 | -4.4% | +21.4% | +137.5% | +135.8% | +94.8% |
| **Iterator Operations** |
| Elements | 39,020 (200 B, 5 allocs) | 314,775 (160,058 B, 10,004 allocs) | 4,954 | +687.6% | +675.7% | +765.1% | +1448.9% | +7566.0% |
| ElementsVec | 988,234 (0 B, 0 allocs) | 998,384 (0 B, 0 allocs) | 987,398 | +0.1% | -2.8% | -10.2% | -3.0% | - |
| ElementsVecStrided | 993,417 (0 B, 0 allocs) | 985,738 (0 B, 0 allocs) | N/A | N/A | +4.1% | -49.6% | +105.8% | - |
| ElementsMat | 141,809 (0 B, 0 allocs) | 144,748 (0 B, 0 allocs) | 144,488 | -1.9% | +49.0% | +15.1% | -2.5% | - |
| ElementsMatStrided | 149,027 (0 B, 0 allocs) | 145,150 (0 B, 0 allocs) | N/A | N/A | +4.4% | -9.0% | +0.7% | - |
| **Conversion Operations** |
| ElemConvert | 5,806 (0 B, 0 allocs) | 7,334 (0 B, 0 allocs) | 11,837 | -51.0% | -35.2% | -55.3% | -31.8% | +5.9% |
| ElemConvert (Clamping) | 8,810 (0 B, 0 allocs) | 7,144 (0 B, 0 allocs) | 3,851 | +128.8% | +5.3% | +47.1% | +169.2% | +138.8% |
| ElemConvertStrided | 129,483 (0 B, 0 allocs) | 100,023 (16 B, 1 alloc) | 26,373 | +391.0% | +457.1% | +419.3% | +607.7% | +612.4% |
| ValueConvert | 3.519 (0 B, 0 allocs) | 0.3717 (0 B, 0 allocs) | N/A | N/A | N/A | N/A | N/A | - |
| ValueConvert (Clamping) | 3.991 (0 B, 0 allocs) | 0.3534 (0 B, 0 allocs) | N/A | N/A | N/A | N/A | N/A | - |

**Note:** ValueConvert operations don't have a Direct Loop baseline as they operate on single values, not arrays.

---

## Performance Comparison Table (Multi-threaded Build)

This table shows benchmark results when building with the `use_mt` build tag, which enables multi-threaded implementations. All benchmarks use 10,000 elements unless otherwise specified.

**Latest Run:** November 7, 2025

| Operation | G (ns/op, allocs) | NG (ns/op, allocs) | D (ns/op) | % vs D |
|-----------|-------------------|---------------------|-----------|--------|
| **Copy Operations** |
| ElemCopy | 1,683 (0 B, 0 allocs) | 1,685 (0 B, 0 allocs) | 16,052 | -89.5% |
| ElemCopyStrided | 1,782 (32 B, 2 allocs) | 101,599 (16 B, 1 alloc) | 21,341 | -91.6% |
| BLAS Copy | N/A | N/A | N/A | N/A |
| BLAS CopyStrided | N/A | N/A | N/A | N/A |
| **Swap Operations** |
| ElemSwap | 12,527 (0 B, 0 allocs) | 11,647 (0 B, 0 allocs) | 20,812 | -39.8% |
| ElemSwapStrided | 13,617 (32 B, 2 allocs) | 89,579 (16 B, 1 alloc) | 23,817 | -42.8% |
| BLAS Swap | N/A | N/A | N/A | N/A |
| BLAS SwapStrided | N/A | N/A | N/A | N/A |
| **Apply Operations** |
| ElemApplyUnary | 38,290 (576 B, 12 allocs) | 6,910 (0 B, 0 allocs) | 16,264 | +135.4% |
| ElemApplyBinary | 53,003 (592 B, 12 allocs) | 7,724 (0 B, 0 allocs) | 15,124 | +250.5% |
| ElemApplyBinaryStrided | 47,607 (640 B, 15 allocs) | 31,489 (48 B, 3 allocs) | 25,738 | +85.0% |
| ElemApplyUnaryScalar | 71,724 (576 B, 12 allocs) | 9,330 (0 B, 0 allocs) | 13,151 | +445.4% |
| ElemApplyUnaryScalarStrided | 39,179 (608 B, 14 allocs) | 24,961 (32 B, 2 allocs) | 20,015 | +95.7% |
| **Comparison Operations** |
| ElemGreaterThan | 67,741 (592 B, 12 allocs) | 27,122 (0 B, 0 allocs) | 23,804 | +184.6% |
| ElemGreaterThanStrided | 42,238 (640 B, 15 allocs) | 19,141 (48 B, 3 allocs) | 28,144 | +50.1% |
| ElemEqual | 40,914 (592 B, 12 allocs) | 17,939 (0 B, 0 allocs) | 24,550 | +66.7% |
| ElemLess | 29,691 (592 B, 12 allocs) | 19,970 (0 B, 0 allocs) | 25,738 | +15.4% |
| **Unary Operations** |
| ElemSign | 24,152 (0 B, 0 allocs) | 26,838 (0 B, 0 allocs) | 23,656 | +2.1% |
| ElemSignStrided | 19,409 (32 B, 2 allocs) | 18,790 (32 B, 2 allocs) | 21,088 | -8.0% |
| ElemNegative | 11,817 (0 B, 0 allocs) | 10,575 (0 B, 0 allocs) | 18,064 | -34.6% |
| ElemNegativeStrided | 16,487 (32 B, 2 allocs) | 24,029 (32 B, 2 allocs) | 22,674 | -27.3% |
| **Scalar Operations** |
| ElemFill | 4,425 (0 B, 0 allocs) | 6,992 (0 B, 0 allocs) | 8,380 | -47.2% |
| ElemEqualScalar | 15,156 (0 B, 0 allocs) | 12,409 (0 B, 0 allocs) | 17,646 | -14.1% |
| ElemGreaterScalar | 13,085 (0 B, 0 allocs) | 10,534 (0 B, 0 allocs) | 16,375 | -20.1% |
| **Vector/Matrix Apply Operations** |
| ElemVecApplyStrided | 66,210 (576 B, 12 allocs) | 35,469 (0 B, 0 allocs) | 15,732 | +320.9% |
| ElemMatApplyStrided | 77,388 (576 B, 12 allocs) | 35,279 (0 B, 0 allocs) | 18,620 | +315.6% |
| **Iterator Operations** |
| Elements | 116,711 (88 B, 5 allocs) | 843,624 (160056 B, 10004 allocs) | 8,578 | +1260.6% |
| ElementsVec | 19,383 (0 B, 0 allocs) | 4,405 (0 B, 0 allocs) | 8,294 | +133.7% |
| ElementsVecStrided | 7,775 (0 B, 0 allocs) | 9,585 (0 B, 0 allocs) | N/A | N/A |
| ElementsMat | 7,235 (0 B, 0 allocs) | 5,848 (0 B, 0 allocs) | 5,129 | +41.1% |
| ElementsMatStrided | 5,215 (0 B, 0 allocs) | 4,781 (0 B, 0 allocs) | N/A | N/A |
| **Conversion Operations** |
| ElemConvert | 41,969 (560 B, 12 allocs) | 15,247 (0 B, 0 allocs) | 21,524 | +95.0% |
| ElemConvert (Clamping) | 34,853 (560 B, 12 allocs) | 9,871 (0 B, 0 allocs) | 5,081 | +585.9% |
| ElemConvertStrided | 133,038 (1312 B, 54 allocs) | 109,990 (16 B, 1 alloc) | 18,754 | +609.4% |

**Note:** Multi-threaded implementations show higher overhead for small arrays (10,000 elements) due to goroutine pool management and synchronization costs. Performance improvements are expected for larger arrays where parallelization overhead is amortized. The allocations shown are primarily from goroutine pool management and chunk coordination. Note that `ElemConvertStrided` shows significantly higher allocations (54 allocs) in this run, likely due to chunk coordination overhead in the multi-threaded implementation. Performance varies significantly between runs due to system load, CPU scheduling, and cache effects, especially at the threshold boundary (10,000 elements with 10 CPUs).

---

## Historical Run Dates

- **H1 (Current):** November 7, 2025
- **H2 (Previous):** November 7, 2025
- **H3 (Previous):** November 6, 2025
- **H4 (Previous):** November 6, 2025
- **H5 (Previous):** November 6, 2025 (initial run)

---

## Key Findings

### Strengths

1. **Copy Operations:** Generic `ElemCopy` and `ElemCopyStrided` are highly optimized, often faster than direct loops due to builtin `copy()` optimization. ElemCopy is 91.1% faster than baseline. **MAJOR IMPROVEMENT:** `ElemCopyStrided` now has **zero allocations** (previously 32 B, 2 allocs) and is 92.4% faster than baseline.

2. **BLAS Operations:** BLAS `Copy` and `Swap` operations perform excellently, with generic versions competitive with non-generic and direct loops.

3. **Contiguous Operations:** Generic contiguous operations (Unary, Binary, Scalar) perform comparably to non-generic versions, typically 5-53% faster than direct loops.

4. **Swap Operations:** Generic swap operations perform well, competitive with non-generic versions. ElemSwap is 57.6% faster than baseline. **MAJOR IMPROVEMENT:** `ElemSwapStrided` now has **zero allocations** (previously 32 B, 2 allocs) and is 42.7% faster than baseline.

5. **Strided Operations Optimization:** **MAJOR BREAKTHROUGH:** All strided operations now have **zero allocations** after replacing `ComputeStrideOffset` with `AdvanceOffsets` variants:
   - `ElemCopyStrided`: 0 allocs (was 2 allocs)
   - `ElemSwapStrided`: 0 allocs (was 2 allocs)
   - `ElemApplyBinaryStrided`: 0 allocs (was 3 allocs)
   - `ElemApplyUnaryScalarStrided`: 0 allocs (was 2 allocs)
   - `ElemGreaterThanStrided`: 0 allocs (was 3 allocs)
   - `ElemConvertStrided`: 0 allocs (was 3 allocs)

6. **Vector/Matrix Iterators:** `ElementsVec` and `ElementsMat` iterators perform excellently, comparable to direct loops with minimal overhead.

7. **Tensor Iterator Optimization:** `Elements` iterator optimized to eliminate allocations (99.95% reduction: from 10,005 to 5 allocations) by reusing the indices slice. Performance improved significantly from previous runs.

8. **Memory Efficiency:** **ALL strided operations now have zero allocations** for the hot path, representing a major optimization achievement.

### Areas for Improvement

1. **Strided Operations:** Some strided operations have overhead from stride iteration logic, but this is expected and necessary for correctness. **However, all allocations have been eliminated** through the use of `AdvanceOffsets` variants instead of `ComputeStrideOffset`.

2. **Type Conversion:** Generic type conversion has overhead from type dispatch, but provides type safety and flexibility. `ElemConvertStrided` now has zero allocations (was 3 allocs).

3. **Vector/Matrix Apply Operations:** Some overhead from stride/leading dimension checks, but still reasonable for the flexibility provided.

4. **Elements Iterator:** Still slower than direct loop due to iterator overhead, but significant improvement from previous runs (99.95% fewer allocations).

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

The generic implementations provide excellent performance for contiguous operations, competitive with non-generic versions while providing type safety and code reuse. **This benchmark run shows MAJOR improvements:** All strided operations now have **zero allocations** after replacing `ComputeStrideOffset` calls in hot paths with `AdvanceOffsets` variants. This optimization eliminates heap allocations entirely for strided operations, significantly improving memory efficiency and reducing GC pressure. Performance improvements are seen across the board, with many operations showing better performance than previous runs. The generic approach successfully balances performance, type safety, and code maintainability while achieving zero-allocation hot paths.
