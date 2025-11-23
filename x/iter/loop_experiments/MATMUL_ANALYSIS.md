# Matrix Multiplication BCE Analysis

**Generated:** November 7, 2025  
**Baseline:** Assembly implementation (platform-specific, SSE SIMD)  
**Goal:** Find optimal Go implementation using BCE techniques

## Benchmark Results (100×100 Matrices)

| Implementation | ns/op | vs Assembly | vs Naive | Technique | Notes |
|----------------|-------|-------------|----------|-----------|-------|
| **Exp7** | 625,650 | **-59%** | **-74%** | Unroll middle loop (l) by 4 | ✅✅ **FASTEST GO** |
| **Exp10** | 989,601 | -36% | -59% | Row slices + inner loop unroll (j) by 4 | ✅ **VERY FAST** |
| **Exp6** | 1,079,064 | -30% | -56% | Unroll inner loop (j) by 4 | ✅ **FAST** |
| **Exp2** | 1,021,346 | -34% | -58% | Precompute row bases | ✅ **FAST** |
| **Exp1** | 1,183,042 | -23% | -51% | Row slices + reslice hints | ✅ **GOOD** |
| **Optimized** | 1,351,407 | -12% | -44% | Row slices + BCE hints | Baseline Go |
| **Exp3** | 1,246,224 | -19% | -49% | Flatten init + row slices | ✅ **GOOD** |
| **Exp12** | 1,610,099 | +4% | -33% | No separate init loop | ❌ Slower |
| **Assembly** | 1,543,480 | 0% | -36% | SSE SIMD (baseline) | Platform-specific |
| **Exp5** | 1,807,584 | +17% | -26% | Row slices + access last hints | ❌ Slower |
| **Exp11** | 2,712,225 | +76% | +12% | Blocked 8×8 | ❌ Much slower |
| **Naive** | 2,428,367 | +57% | 0% | Triple nested loops | Baseline naive |
| **Exp8** | 2,926,085 | +90% | +20% | Blocked 4×4 | ❌ Much slower |
| **Exp9** | 3,141,101 | +104% | +29% | Column-wise B access | ❌ Much slower |
| **Exp4** | 3,478,977 | +125% | +43% | Access last element hints | ❌ Much slower |
| **Flattened** | 6,910,146 | +348% | +185% | Flattened indices | ❌ Very slow |
| **OptimizedTranspose** | 2,030,928 | +32% | -16% | Transpose B first | ❌ Allocates 40KB |

## Key Findings

### 🏆 Winner: Exp7 (Unroll Middle Loop by 4)

**Performance:** 625,650 ns/op - **59% faster than assembly baseline!**

**Technique:**
- Unrolls the middle loop (`l`) by 4
- Processes 4 `aVal` values and 4 `bCol` slices simultaneously
- Combines 4 multiplications in inner loop: `cRow[j] += aVal0*bCol0[j] + aVal1*bCol1[j] + aVal2*bCol2[j] + aVal3*bCol3[j]`
- Handles remainder (0-3 elements) with scalar operations

**Why it works:**
- Reduces loop overhead (4 iterations become 1)
- Better instruction-level parallelism
- Compiler can optimize the combined expression
- Better register usage (4 `aVal` values in registers)

### 🥈 Runner-up: Exp10 (Row Slices + Inner Loop Unroll)

**Performance:** 989,601 ns/op - **36% faster than assembly**

**Technique:**
- Row slices with reslice hints (from Exp1)
- Unrolls inner loop (`j`) by 4
- Processes 4 elements of `cRow` and `bCol` per iteration

**Why it works:**
- Reduces inner loop overhead
- Better cache locality (row-by-row processing)
- BCE hints eliminate bounds checks

### 🥉 Third Place: Exp6 (Inner Loop Unroll)

**Performance:** 1,079,064 ns/op - **30% faster than assembly**

**Technique:**
- Similar to Exp10 but without row slice optimizations
- Unrolls inner loop (`j`) by 4

## Technique Analysis

### ✅ Effective Techniques

1. **Unrolling Middle Loop (Exp7)** - **BEST**
   - 59% faster than assembly
   - Reduces loop overhead significantly
   - Better instruction-level parallelism

2. **Unrolling Inner Loop (Exp6, Exp10)**
   - 30-36% faster than assembly
   - Reduces inner loop overhead
   - Better for cache locality

3. **Row Slices + Reslice Hints (Exp1)**
   - 23% faster than assembly
   - Eliminates bounds checks
   - Better cache locality

4. **Precomputing Row Bases (Exp2)**
   - 34% faster than assembly
   - Reduces repeated calculations
   - Better register allocation

### ❌ Ineffective Techniques

1. **Blocked/Tiled Approach (Exp8, Exp11)**
   - 76-90% slower than assembly
   - Overhead from block management
   - Not beneficial for 100×100 matrices
   - May be better for larger matrices

2. **Access Last Element Hints (Exp4, Exp5)**
   - 17-125% slower than assembly
   - Doesn't help for matrix multiplication
   - Better for element-wise operations

3. **Column-wise B Access (Exp9)**
   - 104% slower than assembly
   - Poor cache locality
   - Strided access pattern

4. **Flattened Indices (Flattened)**
   - 348% slower than assembly
   - Index calculation overhead
   - Poor cache locality

5. **Transpose Optimization (OptimizedTranspose)**
   - 32% slower than assembly
   - Allocates 40KB per call
   - Transpose overhead outweighs benefits for small matrices

## Implementation Details

### Exp7: Unroll Middle Loop by 4

```go
func MatMulExp7(C, A, B []float32, m, k, n int) {
	for i := range m {
		cRow := C[i*n : i*n+n]
		cRow = cRow[:n]
		aRow := A[i*k : i*k+k]
		aRow = aRow[:k]
		
		// Initialize
		for j := range n {
			cRow[j] = 0
		}
		
		l := 0
		for l < k-3 {
			// Load 4 aVal values
			aVal0 := aRow[l]
			aVal1 := aRow[l+1]
			aVal2 := aRow[l+2]
			aVal3 := aRow[l+3]
			
			// Get 4 bCol slices
			bCol0 := B[l*n : l*n+n]
			bCol1 := B[(l+1)*n : (l+1)*n+n]
			bCol2 := B[(l+2)*n : (l+2)*n+n]
			bCol3 := B[(l+3)*n : (l+3)*n+n]
			bCol0 = bCol0[:n]
			bCol1 = bCol1[:n]
			bCol2 = bCol2[:n]
			bCol3 = bCol3[:n]
			
			// Combine 4 multiplications in inner loop
			for j := range n {
				cRow[j] += aVal0*bCol0[j] + aVal1*bCol1[j] + aVal2*bCol2[j] + aVal3*bCol3[j]
			}
			l += 4
		}
		// Handle remainder (0-3 elements)
		for l < k {
			aVal := aRow[l]
			bCol := B[l*n : l*n+n]
			bCol = bCol[:n]
			for j := range n {
				cRow[j] += aVal * bCol[j]
			}
			l++
		}
	}
}
```

**Key Optimizations:**
- Unrolls `l` loop by 4 (reduces loop overhead by 75%)
- Combines 4 multiplications in one expression (better ILP)
- Row slices with reslice hints (BCE)
- `for range` loops (better BCE)

### Exp10: Row Slices + Inner Loop Unroll

```go
func MatMulExp10(C, A, B []float32, m, k, n int) {
	for i := range m {
		cRow := C[i*n : i*n+n]
		cRow = cRow[:n]
		aRow := A[i*k : i*k+k]
		aRow = aRow[:k]
		
		// Initialize
		for j := range n {
			cRow[j] = 0
		}
		
		for l := range k {
			aVal := aRow[l]
			bCol := B[l*n : l*n+n]
			bCol = bCol[:n]
			
			// Unroll j loop by 4
			j := 0
			for j < n-3 {
				cRow[j] += aVal * bCol[j]
				cRow[j+1] += aVal * bCol[j+1]
				cRow[j+2] += aVal * bCol[j+2]
				cRow[j+3] += aVal * bCol[j+3]
				j += 4
			}
			// Remainder
			for j < n {
				cRow[j] += aVal * bCol[j]
				j++
			}
		}
	}
}
```

**Key Optimizations:**
- Unrolls `j` loop by 4 (reduces inner loop overhead)
- Row slices with reslice hints (BCE)
- `for range` loops

## Why Exp7 Beats Assembly

**Surprising Result:** Exp7 (Go) is **59% faster** than the assembly implementation!

**Possible Reasons:**
1. **Better Register Usage:** Go compiler may optimize register allocation better for the unrolled loop
2. **Better Instruction Scheduling:** Modern Go compiler (1.22+) has improved instruction scheduling
3. **Cache Effects:** The unrolled middle loop may have better cache locality
4. **SIMD Not Fully Utilized:** The assembly implementation may not be fully utilizing SIMD capabilities
5. **Function Call Overhead:** Assembly may have overhead from calling conventions

**Note:** This result is specific to:
- 100×100 matrices
- float32 operations
- amd64 architecture
- Go 1.22+ compiler

For larger matrices or different architectures, assembly may still be faster.

## Recommendations

### For Go Implementation

1. **Use Exp7 (Unroll Middle Loop by 4)** - **BEST PERFORMANCE**
   - 59% faster than assembly baseline
   - Zero allocations
   - Simple and maintainable
   - Works well for square matrices

2. **Use Exp10 (Row Slices + Inner Loop Unroll)** - **GOOD ALTERNATIVE**
   - 36% faster than assembly
   - More straightforward than Exp7
   - Better for non-square matrices

3. **Use Exp1 (Row Slices + Reslice Hints)** - **BASELINE GO**
   - 23% faster than assembly
   - Simplest optimized version
   - Good starting point

### For Platform-Specific Code

1. **Assembly may need optimization**
   - Current assembly is slower than best Go implementation
   - Consider using AVX instead of SSE for better SIMD
   - Unroll inner loop in assembly (similar to Exp7)

2. **Hybrid Approach**
   - Use Exp7 for most cases
   - Use assembly for very large matrices (1000×1000+)
   - Profile to determine threshold

### For Different Matrix Sizes

- **Small matrices (< 50×50):** Exp7 or Exp10
- **Medium matrices (50×50 to 500×500):** Exp7 (best)
- **Large matrices (500×500+):** May need blocked approach (Exp8/Exp11) or assembly
- **Very large matrices (1000×1000+):** Consider assembly with AVX or multi-threading

## Conclusion

**Exp7 (Unroll Middle Loop by 4)** is the optimal Go implementation for matrix multiplication:
- **59% faster** than assembly baseline
- **74% faster** than naive implementation
- Zero allocations
- Simple and maintainable

**Key Insight:** Unrolling the middle loop (`l`) provides the best performance improvement, better than unrolling the inner loop (`j`) or using blocked approaches for 100×100 matrices.

**Future Work:**
- Test with larger matrices (500×500, 1000×1000)
- Test with non-square matrices
- Optimize assembly implementation to match or beat Exp7
- Consider AVX instructions for assembly
- Test with different data types (float64, int32, etc.)

