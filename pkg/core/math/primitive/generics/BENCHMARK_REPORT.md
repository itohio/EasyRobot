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
- Hint for benchmarking: when extracting benchmarks metrics, avoid running benchmarks multiple times since they take a long time to complete. Run them once redirecting output to temporary txt file and try parsing that file.

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
| ElemCopy | 1,780 (0 B, 0 allocs) | 1,901 (0 B, 0 allocs) | 25,654 | -93.1% | -86.7% | -86.1% | - | - |
| ElemCopyStrided | 1,855 (32 B, 2 allocs) | 81,367 (16 B, 1 alloc) | 15,644 | -88.1% | -87.3% | -88.6% | - | - |
| BLAS Copy | 1,840 (0 B, 0 allocs) | 1,593 (0 B, 0 allocs) | 1,557 | +18.2% | +7.4% | - | - | - |
| BLAS CopyStrided | 1,598 (0 B, 0 allocs) | 1,540 (0 B, 0 allocs) | 1,557 | +2.6% | +1.0% | - | - | - |
| **Swap Operations** |
| ElemSwap | 11,629 (0 B, 0 allocs) | 11,114 (0 B, 0 allocs) | 17,819 | -34.7% | -32.6% | -38.6% | - | - |
| ElemSwapStrided | 12,619 (32 B, 2 allocs) | 103,386 (16 B, 1 alloc) | 19,782 | -36.2% | -53.8% | -39.0% | - | - |
| BLAS Swap | 13,966 (0 B, 0 allocs) | 16,008 (0 B, 0 allocs) | 22,006 | -36.5% | -37.5% | - | - | - |
| BLAS SwapStrided | 13,136 (0 B, 0 allocs) | 11,891 (0 B, 0 allocs) | 22,006 | -40.3% | -39.4% | - | - | - |
| **Apply Operations** |
| ElemApplyUnary | 10,000 (0 B, 0 allocs) | 10,165 (0 B, 0 allocs) | 15,626 | -36.0% | -17.0% | -25.9% | - | - |
| ElemApplyBinary | 11,801 (0 B, 0 allocs) | 10,326 (0 B, 0 allocs) | 26,680 | -55.8% | -53.4% | -43.4% | - | - |
| ElemApplyBinaryStrided | 43,207 (48 B, 3 allocs) | 34,839 (48 B, 3 allocs) | 25,082 | +72.3% | +59.4% | +40.1% | - | - |
| ElemApplyUnaryScalar | 10,519 (0 B, 0 allocs) | 10,411 (0 B, 0 allocs) | 16,757 | -37.2% | -68.6% | -34.0% | - | - |
| ElemApplyUnaryScalarStrided | 34,370 (32 B, 2 allocs) | 32,260 (32 B, 2 allocs) | 17,142 | +100.5% | +62.9% | +56.8% | - | - |
| **Comparison Operations** |
| ElemGreaterThan | 16,207 (0 B, 0 allocs) | 19,516 (0 B, 0 allocs) | 27,482 | -41.0% | -6.4% | -31.7% | - | - |
| ElemGreaterThanStrided | 18,556 (48 B, 3 allocs) | 18,448 (48 B, 3 allocs) | 27,524 | -32.6% | -36.7% | -32.7% | - | - |
| ElemEqual | 15,937 (0 B, 0 allocs) | 16,467 (0 B, 0 allocs) | 23,092 | -31.0% | -29.9% | - | - | - |
| ElemLess | 19,143 (0 B, 0 allocs) | 18,784 (0 B, 0 allocs) | 21,801 | -12.2% | -14.6% | - | - | - |
| **Unary Operations** |
| ElemSign | 17,589 (0 B, 0 allocs) | 18,454 (0 B, 0 allocs) | 17,860 | -1.5% | -3.3% | +1.7% | - | - |
| ElemSignStrided | 15,383 (32 B, 2 allocs) | 23,747 (32 B, 2 allocs) | 32,630 | -52.9% | -20.7% | -26.8% | - | - |
| ElemNegative | 17,256 (0 B, 0 allocs) | 13,339 (0 B, 0 allocs) | 18,556 | -7.0% | -41.1% | -45.7% | - | - |
| ElemNegativeStrided | 21,652 (32 B, 2 allocs) | 14,938 (32 B, 2 allocs) | 20,012 | +8.2% | -24.0% | -29.2% | - | - |
| **Scalar Operations** |
| ElemFill | 5,468 (0 B, 0 allocs) | 8,502 (0 B, 0 allocs) | 8,997 | -39.2% | -64.4% | - | - | - |
| ElemEqualScalar | 13,085 (0 B, 0 allocs) | 13,134 (0 B, 0 allocs) | 16,714 | -21.7% | -25.1% | - | - | - |
| ElemGreaterScalar | 12,526 (0 B, 0 allocs) | 14,906 (0 B, 0 allocs) | 18,661 | -32.9% | +30.3% | - | - | - |
| **Vector/Matrix Apply Operations** |
| ElemVecApplyStrided | 28,237 (0 B, 0 allocs) | 67,655 (0 B, 0 allocs) | 18,270 | +54.6% | +59.5% | +99.1% | - | - |
| ElemMatApplyStrided | 49,160 (0 B, 0 allocs) | 58,466 (0 B, 0 allocs) | 20,699 | +137.5% | +135.8% | +94.8% | - | - |
| **Iterator Operations** |
| Elements | 46,783 (88 B, 5 allocs) | 418,696 (160,056 B, 10,004 allocs) | 5,408 | +765.1% | +1448.9% | +7566.0% | - | - |
| ElementsVec | 8,425 (0 B, 0 allocs) | 4,587 (0 B, 0 allocs) | 9,384 | -10.2% | -3.0% | - | - | - |
| ElementsVecStrided | 4,724 (0 B, 0 allocs) | 9,815 (0 B, 0 allocs) | 9,384 | -49.6% | +105.8% | - | - | - |
| ElementsMat | 7,102 (0 B, 0 allocs) | 5,749 (0 B, 0 allocs) | 6,171 | +15.1% | -2.5% | - | - | - |
| ElementsMatStrided | 5,618 (0 B, 0 allocs) | 6,303 (0 B, 0 allocs) | 6,171 | -9.0% | +0.7% | - | - | - |
| **Conversion Operations** |
| ElemConvert | 7,065 (0 B, 0 allocs) | 7,519 (0 B, 0 allocs) | 15,815 | -55.3% | -31.8% | +5.9% | - | - |
| ElemConvert (Clamping) | 13,067 (0 B, 0 allocs) | 4,970 (0 B, 0 allocs) | 8,883 | +47.1% | +169.2% | +138.8% | - | - |
| ElemConvertStrided | 101,101 (48 B, 3 allocs) | 106,885 (16 B, 1 alloc) | 19,470 | +419.3% | +607.7% | +612.4% | - | - |
| ValueConvert | 4.395 (0 B, 0 allocs) | 0.5562 (0 B, 0 allocs) | N/A | N/A | N/A | - | - | - |
| ValueConvert (Clamping) | 11.11 (0 B, 0 allocs) | 0.5387 (0 B, 0 allocs) | N/A | N/A | N/A | - | - | - |

**Note:** ValueConvert operations don't have a Direct Loop baseline as they operate on single values, not arrays.

---

## Performance Comparison Table (Multi-threaded Build)

This table shows benchmark results when building with the `use_mt` build tag, which enables multi-threaded implementations. All benchmarks use 10,000 elements unless otherwise specified.

**Latest Run:** November 6, 2025

| Operation | G (ns/op, allocs) | NG (ns/op, allocs) | D (ns/op) | % vs D |
|-----------|-------------------|---------------------|-----------|--------|
| **Copy Operations** |
| ElemCopy | 1,685 (0 B, 0 allocs) | 1,588 (0 B, 0 allocs) | 12,928 | -87.0% |
| ElemCopyStrided | 2,044 (32 B, 2 allocs) | 71,846 (16 B, 1 alloc) | 17,337 | -88.2% |
| BLAS Copy | N/A | N/A | N/A | N/A |
| BLAS CopyStrided | N/A | N/A | N/A | N/A |
| **Swap Operations** |
| ElemSwap | 12,582 (0 B, 0 allocs) | 9,584 (0 B, 0 allocs) | 21,695 | -42.0% |
| ElemSwapStrided | 10,608 (32 B, 2 allocs) | 89,949 (16 B, 1 alloc) | 22,138 | -52.1% |
| BLAS Swap | N/A | N/A | N/A | N/A |
| BLAS SwapStrided | N/A | N/A | N/A | N/A |
| **Apply Operations** |
| ElemApplyUnary | 66,855 (576 B, 12 allocs) | 8,540 (0 B, 0 allocs) | 13,470 | +396.3% |
| ElemApplyBinary | 45,895 (592 B, 12 allocs) | 8,929 (0 B, 0 allocs) | 21,014 | +118.4% |
| ElemApplyBinaryStrided | 44,981 (640 B, 15 allocs) | 32,169 (48 B, 3 allocs) | 22,091 | +103.6% |
| ElemApplyUnaryScalar | 58,360 (576 B, 12 allocs) | 10,020 (0 B, 0 allocs) | 24,026 | +142.9% |
| ElemApplyUnaryScalarStrided | 53,633 (608 B, 14 allocs) | 29,898 (32 B, 2 allocs) | 19,163 | +179.9% |
| **Comparison Operations** |
| ElemGreaterThan | 29,346 (592 B, 12 allocs) | 14,757 (0 B, 0 allocs) | 19,699 | +49.0% |
| ElemGreaterThanStrided | 44,083 (640 B, 15 allocs) | 16,303 (48 B, 3 allocs) | 33,458 | +31.8% |
| ElemEqual | 74,120 (592 B, 12 allocs) | 18,547 (0 B, 0 allocs) | 22,800 | +225.1% |
| ElemLess | 32,324 (592 B, 12 allocs) | 18,315 (0 B, 0 allocs) | 23,334 | +38.5% |
| **Unary Operations** |
| ElemSign | 16,097 (0 B, 0 allocs) | 18,782 (0 B, 0 allocs) | 25,234 | -36.2% |
| ElemSignStrided | 19,350 (32 B, 2 allocs) | 24,499 (32 B, 2 allocs) | 30,701 | -37.0% |
| ElemNegative | 12,273 (0 B, 0 allocs) | 9,271 (0 B, 0 allocs) | 15,322 | -19.9% |
| ElemNegativeStrided | 12,520 (32 B, 2 allocs) | 10,127 (32 B, 2 allocs) | 19,134 | -34.6% |
| **Scalar Operations** |
| ElemFill | 9,498 (0 B, 0 allocs) | 8,082 (0 B, 0 allocs) | 8,457 | +12.3% |
| ElemEqualScalar | 19,063 (0 B, 0 allocs) | 14,859 (0 B, 0 allocs) | 23,091 | -17.4% |
| ElemGreaterScalar | 14,511 (0 B, 0 allocs) | 17,192 (0 B, 0 allocs) | 22,443 | -35.3% |
| **Vector/Matrix Apply Operations** |
| ElemVecApplyStrided | 49,551 (576 B, 12 allocs) | 28,387 (0 B, 0 allocs) | 11,202 | +342.3% |
| ElemMatApplyStrided | 48,864 (576 B, 12 allocs) | 38,903 (0 B, 0 allocs) | 16,584 | +194.6% |
| **Iterator Operations** |
| Elements | 55,832 (88 B, 5 allocs) | 398,002 (160056 B, 10004 allocs) | 5,897 | +846.8% |
| ElementsVec | 4,227 (0 B, 0 allocs) | 8,216 (0 B, 0 allocs) | 4,159 | +1.6% |
| ElementsVecStrided | 8,446 (0 B, 0 allocs) | 5,037 (0 B, 0 allocs) | N/A | N/A |
| ElementsMat | 6,223 (0 B, 0 allocs) | 5,295 (0 B, 0 allocs) | 5,365 | +16.0% |
| ElementsMatStrided | 6,592 (0 B, 0 allocs) | 7,415 (0 B, 0 allocs) | N/A | N/A |
| **Conversion Operations** |
| ElemConvert | 29,808 (560 B, 12 allocs) | 12,305 (0 B, 0 allocs) | 15,634 | +90.7% |
| ElemConvert (Clamping) | 37,891 (560 B, 12 allocs) | 7,164 (0 B, 0 allocs) | 10,077 | +276.0% |
| ElemConvertStrided | 126,438 (240 B, 7 allocs) | 140,484 (16 B, 1 alloc) | 20,851 | +506.4% |

**Note:** Multi-threaded implementations show higher overhead for small arrays (10,000 elements) due to goroutine pool management and synchronization costs. Performance improvements are expected for larger arrays where parallelization overhead is amortized. The allocations shown are primarily from goroutine pool management and chunk coordination.

---

## Historical Run Dates

- **H1 (Current):** November 6, 2025
- **H2 (Previous):** November 6, 2025
- **H3 (Previous):** November 6, 2025 (initial run)

---

## Key Findings

### Strengths

1. **Copy Operations:** Generic `ElemCopy` and `ElemCopyStrided` are highly optimized, often faster than direct loops due to builtin `copy()` optimization. ElemCopy is 93.1% faster than baseline.

2. **BLAS Operations:** BLAS `Copy` and `Swap` operations perform excellently, with generic versions competitive with non-generic and direct loops.

3. **Contiguous Operations:** Generic contiguous operations (Unary, Binary, Scalar) perform comparably to non-generic versions, typically 14-68% faster than direct loops.

4. **Swap Operations:** Generic swap operations perform well, competitive with non-generic versions. ElemSwap is 34.7% faster than baseline.

5. **Vector/Matrix Iterators:** `ElementsVec` and `ElementsMat` iterators perform excellently, comparable to direct loops with minimal overhead.

6. **Tensor Iterator Optimization:** `Elements` iterator optimized to eliminate allocations (99.95% reduction: from 10,005 to 5 allocations) by reusing the indices slice. Performance improved significantly from previous runs.

7. **Memory Efficiency:** Most operations have zero allocations for contiguous paths.

### Areas for Improvement

1. **Strided Operations:** Some strided operations have overhead from stride iteration logic, but this is expected and necessary for correctness.

2. **Type Conversion:** Generic type conversion has overhead from type dispatch, but provides type safety and flexibility.

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

The generic implementations provide excellent performance for contiguous operations, competitive with non-generic versions while providing type safety and code reuse. Many operations showed significant improvements from the previous benchmark run. Strided operations have expected overhead but are necessary for handling non-contiguous memory layouts. The generic approach successfully balances performance, type safety, and code maintainability.
