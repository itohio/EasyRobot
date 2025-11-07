package loop_experiments

// Operation: square element and add 1
// Non-inlined version for testing function call overhead
//
//go:noinline
func op(x float32) float32 {
	return x*x + 1.0
}

// Inline operation: x*x + 1.0 (for comparison with function call overhead)
// This version may be inlined by the compiler
func opInline(x float32) float32 {
	return x*x + 1.0
}

// Baseline: nested loops without any optimizations
func BaselineNestedLoops(dst, src []float32, rows, cols int) {
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			idx := i*cols + j
			dst[idx] = op(src[idx])
		}
	}
}

// Variant 1: BCE hint with _ = slice[n-1] before loops
func BCE_Hint_AccessLast(dst, src []float32, rows, cols int) {
	size := rows * cols
	if size > 0 {
		_ = dst[size-1]
		_ = src[size-1]
	}
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			idx := i*cols + j
			dst[idx] = op(src[idx])
		}
	}
}

// Variant 2: Reslice to exact size before loops
func BCE_Reslice_ExactSize(dst, src []float32, rows, cols int) {
	size := rows * cols
	dst = dst[:size]
	src = src[:size]
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			idx := i*cols + j
			dst[idx] = op(src[idx])
		}
	}
}

// Variant 3: Create row slices (like matrix operations)
func BCE_RowSlices(dst, src []float32, rows, cols int) {
	for i := 0; i < rows; i++ {
		dstRow := dst[i*cols : i*cols+cols]
		srcRow := src[i*cols : i*cols+cols]
		for j := 0; j < cols; j++ {
			dstRow[j] = op(srcRow[j])
		}
	}
}

// Variant 4: Create row slices with reslicing hint
func BCE_RowSlices_Reslice(dst, src []float32, rows, cols int) {
	for i := 0; i < rows; i++ {
		dstRow := dst[i*cols : i*cols+cols]
		srcRow := src[i*cols : i*cols+cols]
		dstRow = dstRow[:cols] // NOOP but hints compiler
		srcRow = srcRow[:cols]
		for j := 0; j < cols; j++ {
			dstRow[j] = op(srcRow[j])
		}
	}
}

// Variant 5: Use for range on rows, then nested loop
func BCE_RangeRows(dst, src []float32, rows, cols int) {
	for i := range rows {
		for j := 0; j < cols; j++ {
			idx := i*cols + j
			dst[idx] = op(src[idx])
		}
	}
}

// Variant 6: Use for range on both dimensions
func BCE_RangeBoth(dst, src []float32, rows, cols int) {
	for i := range rows {
		for j := range cols {
			idx := i*cols + j
			dst[idx] = op(src[idx])
		}
	}
}

// Variant 7: Reslice + range on both
func BCE_Reslice_RangeBoth(dst, src []float32, rows, cols int) {
	size := rows * cols
	dst = dst[:size]
	src = src[:size]
	for i := range rows {
		for j := range cols {
			idx := i*cols + j
			dst[idx] = op(src[idx])
		}
	}
}

// Variant 8: Row slices + range on cols
func BCE_RowSlices_RangeCols(dst, src []float32, rows, cols int) {
	for i := range rows {
		dstRow := dst[i*cols : i*cols+cols]
		srcRow := src[i*cols : i*cols+cols]
		for j := range cols {
			dstRow[j] = op(srcRow[j])
		}
	}
}

// Variant 9: Row slices + reslice + range
func BCE_RowSlices_Reslice_Range(dst, src []float32, rows, cols int) {
	for i := range rows {
		dstRow := dst[i*cols : i*cols+cols]
		srcRow := src[i*cols : i*cols+cols]
		dstRow = dstRow[:cols]
		srcRow = srcRow[:cols]
		for j := range cols {
			dstRow[j] = op(srcRow[j])
		}
	}
}

// Variant 10: Flatten to 1D and use single loop with reslice
func BCE_Flatten_Reslice(dst, src []float32, rows, cols int) {
	size := rows * cols
	dst = dst[:size]
	src = src[:size]
	for i := range size {
		dst[i] = op(src[i])
	}
}

// Variant 11: Access last element of each row before inner loop
func BCE_AccessLastPerRow(dst, src []float32, rows, cols int) {
	for i := 0; i < rows; i++ {
		if cols > 0 {
			_ = dst[i*cols+cols-1]
			_ = src[i*cols+cols-1]
		}
		for j := 0; j < cols; j++ {
			idx := i*cols + j
			dst[idx] = op(src[idx])
		}
	}
}

// Variant 12: Pre-compute row base offsets
func BCE_PrecomputeOffsets(dst, src []float32, rows, cols int) {
	for i := 0; i < rows; i++ {
		base := i * cols
		for j := 0; j < cols; j++ {
			idx := base + j
			dst[idx] = op(src[idx])
		}
	}
}

// Variant 13: Pre-compute offsets + range
func BCE_PrecomputeOffsets_Range(dst, src []float32, rows, cols int) {
	for i := range rows {
		base := i * cols
		for j := range cols {
			idx := base + j
			dst[idx] = op(src[idx])
		}
	}
}

// Variant 14: Row slices + access last element hint
func BCE_RowSlices_AccessLast(dst, src []float32, rows, cols int) {
	for i := 0; i < rows; i++ {
		dstRow := dst[i*cols : i*cols+cols]
		srcRow := src[i*cols : i*cols+cols]
		if cols > 0 {
			_ = dstRow[cols-1]
			_ = srcRow[cols-1]
		}
		for j := 0; j < cols; j++ {
			dstRow[j] = op(srcRow[j])
		}
	}
}

// ========== STRIDED VARIANTS ==========

// Strided Baseline: nested loops with strides
func StridedBaseline(dst, src []float32, rows, cols int, ldDst, ldSrc int) {
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			dIdx := i*ldDst + j
			sIdx := i*ldSrc + j
			dst[dIdx] = op(src[sIdx])
		}
	}
}

// Strided Variant 1: Row slices with strides
func Strided_RowSlices(dst, src []float32, rows, cols int, ldDst, ldSrc int) {
	for i := 0; i < rows; i++ {
		dstRow := dst[i*ldDst : i*ldDst+cols]
		srcRow := src[i*ldSrc : i*ldSrc+cols]
		for j := 0; j < cols; j++ {
			dstRow[j] = op(srcRow[j])
		}
	}
}

// Strided Variant 2: Row slices with reslice hint
func Strided_RowSlices_Reslice(dst, src []float32, rows, cols int, ldDst, ldSrc int) {
	for i := 0; i < rows; i++ {
		dstRow := dst[i*ldDst : i*ldDst+cols]
		srcRow := src[i*ldSrc : i*ldSrc+cols]
		dstRow = dstRow[:cols]
		srcRow = srcRow[:cols]
		for j := 0; j < cols; j++ {
			dstRow[j] = op(srcRow[j])
		}
	}
}

// Strided Variant 3: Row slices + range
func Strided_RowSlices_Range(dst, src []float32, rows, cols int, ldDst, ldSrc int) {
	for i := range rows {
		dstRow := dst[i*ldDst : i*ldDst+cols]
		srcRow := src[i*ldSrc : i*ldSrc+cols]
		for j := range cols {
			dstRow[j] = op(srcRow[j])
		}
	}
}

// Strided Variant 4: Row slices + reslice + range
func Strided_RowSlices_Reslice_Range(dst, src []float32, rows, cols int, ldDst, ldSrc int) {
	for i := range rows {
		dstRow := dst[i*ldDst : i*ldDst+cols]
		srcRow := src[i*ldSrc : i*ldSrc+cols]
		dstRow = dstRow[:cols]
		srcRow = srcRow[:cols]
		for j := range cols {
			dstRow[j] = op(srcRow[j])
		}
	}
}

// Strided Variant 5: Pre-compute base offsets
func Strided_PrecomputeOffsets(dst, src []float32, rows, cols int, ldDst, ldSrc int) {
	for i := 0; i < rows; i++ {
		dBase := i * ldDst
		sBase := i * ldSrc
		for j := 0; j < cols; j++ {
			dst[dBase+j] = op(src[sBase+j])
		}
	}
}

// Strided Variant 6: Pre-compute offsets + range
func Strided_PrecomputeOffsets_Range(dst, src []float32, rows, cols int, ldDst, ldSrc int) {
	for i := range rows {
		dBase := i * ldDst
		sBase := i * ldSrc
		for j := range cols {
			dst[dBase+j] = op(src[sBase+j])
		}
	}
}

// Helper to initialize test data
func InitTestData(size int) []float32 {
	data := make([]float32, size)
	for i := range data {
		data[i] = float32(i) * 0.001
	}
	return data
}

// Assembly implementations (amd64 only)
// These are declared in asm_amd64.s
// op is a function pointer that will be called from assembly
func asmOpLoop(dst, src *float32, n int, op func(float32) float32)
func asmOpLoopUnrolled(dst, src *float32, n int, op func(float32) float32)

// Assembly-based implementation: direct assembly loop (calls op function)
func BCE_Assembly(dst, src []float32, rows, cols int) {
	size := rows * cols
	if size == 0 {
		return
	}
	// Reslice for BCE
	dst = dst[:size]
	src = src[:size]
	// Call assembly implementation with op function
	asmOpLoop(&dst[0], &src[0], size, op)
}

// Assembly-based implementation: unrolled assembly loop (4 elements at a time, calls op function)
func BCE_Assembly_Unrolled(dst, src []float32, rows, cols int) {
	size := rows * cols
	if size == 0 {
		return
	}
	// Reslice for BCE
	dst = dst[:size]
	src = src[:size]
	// Call unrolled assembly implementation with op function
	asmOpLoopUnrolled(&dst[0], &src[0], size, op)
}

// Assembly-based implementation with inline operation (for testing inlining effects)
func BCE_Assembly_Inline(dst, src []float32, rows, cols int) {
	size := rows * cols
	if size == 0 {
		return
	}
	// Reslice for BCE
	dst = dst[:size]
	src = src[:size]
	// Call assembly implementation with inline op
	asmOpLoop(&dst[0], &src[0], size, opInline)
}

// Assembly-based implementation unrolled with inline operation
func BCE_Assembly_Unrolled_Inline(dst, src []float32, rows, cols int) {
	size := rows * cols
	if size == 0 {
		return
	}
	// Reslice for BCE
	dst = dst[:size]
	src = src[:size]
	// Call unrolled assembly implementation with inline op
	asmOpLoopUnrolled(&dst[0], &src[0], size, opInline)
}

// Platform-specific assembly implementations (amd64 only)
// These are declared in asm_op_amd64.s
func asmOpDirect(dst, src *float32, n int)
func asmOpUnrolled(dst, src *float32, n int)

// Assembly-based implementation: direct assembly loop (no function calls)
func BCE_Assembly_Direct(dst, src []float32, rows, cols int) {
	size := rows * cols
	if size == 0 {
		return
	}
	// Reslice for BCE
	dst = dst[:size]
	src = src[:size]
	// Call assembly implementation
	asmOpDirect(&dst[0], &src[0], size)
}

// Assembly-based implementation: unrolled assembly loop (4 elements at a time, no function calls)
func BCE_Assembly_Unrolled_Direct(dst, src []float32, rows, cols int) {
	size := rows * cols
	if size == 0 {
		return
	}
	// Reslice for BCE
	dst = dst[:size]
	src = src[:size]
	// Call unrolled assembly implementation
	asmOpUnrolled(&dst[0], &src[0], size)
}

// Helper to get offset for cache elimination (200MB array, different offset each time)
func GetCacheOffset(iteration int, arraySize int) int {
	// 200MB = 200 * 1024 * 1024 / 4 (float32) = 52,428,800 elements
	// Use different offsets to avoid cache effects
	offset := (iteration * 1000000) % (arraySize - 1000000) // Ensure we have enough space
	return offset
}
