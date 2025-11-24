package fp32

// CrossCorrelate1D computes cross-correlation between two 1D signals into dst according to information theory.
// Cross-correlation formula: R_xy(τ) = Σ x[n]y[n+τ]
// where signal1 = x, signal2 = y, and τ is the lag
func CrossCorrelate1D(dst, signal1, signal2 []float32) {
	n1 := len(signal1)
	n2 := len(signal2)
	if n1 == 0 || n2 == 0 {
		return
	}
	resultLen := n1 + n2 - 1
	if len(dst) < resultLen {
		resultLen = len(dst)
	}

	// BCE hint: reslice to exact size
	dst = dst[:resultLen]
	signal1 = signal1[:n1]
	signal2 = signal2[:n2]

	// Compute cross-correlation: R_xy(τ) = Σ x[n]y[n+τ]
	// where signal1 = x, signal2 = y, and output index i = τ
	// dst[i] = Σ signal1[n] * signal2[i+n] for all valid n
	// Optimized: split into regions to eliminate bounds checks in hot path
	for i := range resultLen {
		var sum float32
		if i < n2 {
			// Hot path: i < n2, valid range for n is [0, min(n1, n2-i))
			maxN := n1
			if n2-i < n1 {
				maxN = n2 - i
			}
			// BCE: n < maxN <= n1, and i+n < i+maxN <= i+(n2-i) = n2
			for n := range maxN {
				sum += signal1[n] * signal2[i+n]
			}
		}
		// For i >= n2, all n result in idx2 >= n2 (invalid), so sum = 0
		dst[i] = sum
	}
}

// CrossCorrelate2D computes 2D cross-correlation between two matrices into dst according to information theory.
// Cross-correlation formula: R_xy(τ_r, τ_c) = Σ signal1[k,l] * signal2[k+τ_r, l+τ_c]
// signal1: [rows1*cols1] flattened matrix, cols1 = len(signal1) / rows1
// signal2: [rows2*cols2] flattened matrix, cols2 = len(signal2) / rows2
// dst: [dstRows*(cols1+cols2-1)] flattened matrix
// Panics if dimensions don't match or dst doesn't have enough space.
func CrossCorrelate2D(dst []float32, dstRows int, signal1 []float32, rows1 int, signal2 []float32, rows2 int) {
	if rows1 == 0 || rows2 == 0 {
		panic("CrossCorrelate2D: rows must be > 0")
	}
	cols1 := len(signal1) / rows1
	cols2 := len(signal2) / rows2
	if len(signal1) != rows1*cols1 || len(signal2) != rows2*cols2 {
		panic("CrossCorrelate2D: signal dimensions don't match")
	}

	dstCols := cols1 + cols2 - 1
	resultRows := rows1 + rows2 - 1
	resultCols := dstCols

	if dstRows < resultRows {
		resultRows = dstRows
	}

	// Panic if not enough space
	if len(dst) < dstRows*dstCols {
		panic("CrossCorrelate2D: dst too small")
	}

	// BCE hints
	dst = dst[:dstRows*dstCols]
	signal1 = signal1[:rows1*cols1]
	signal2 = signal2[:rows2*cols2]

	// Compute 2D cross-correlation: R_xy(τ_r, τ_c) = Σ signal1[k,l] * signal2[k+τ_r, l+τ_c]
	// where output position (i,j) = (τ_r, τ_c)
	// dst[i*dstCols+j] = Σ signal1[k,l] * signal2[i+k, j+l] for all valid (k,l)
	// Optimized: split into regions to eliminate bounds checks in hot path
	for i := range resultRows {
		for j := range resultCols {
			var sum float32
			if i < rows2 && j < cols2 {
				// Hot path: compute valid ranges for k and l
				maxK := rows1
				if rows2-i < rows1 {
					maxK = rows2 - i
				}
				maxL := cols1
				if cols2-j < cols1 {
					maxL = cols2 - j
				}
				// BCE: k < maxK <= rows1, l < maxL <= cols1
				//      i+k < i+maxK <= i+(rows2-i) = rows2
				//      j+l < j+maxL <= j+(cols2-j) = cols2
				for k := range maxK {
					rowIdx := i + k
					baseIdx := k * cols1
					rowBase := rowIdx * cols2
					for l := range maxL {
						colIdx := j + l
						sum += signal1[baseIdx+l] * signal2[rowBase+colIdx]
					}
				}
			}
			// For i >= rows2 or j >= cols2, all (k,l) result in invalid indices, so sum = 0
			dst[i*dstCols+j] = sum
		}
	}
}

// CrossCorrelate1DAdd computes cross-correlation between two 1D signals and adds the result to dst.
// Cross-correlation formula: R_xy(τ) = Σ x[n]y[n+τ]
// where signal1 = x, signal2 = y, and output index i = τ
// dst[i] += Σ signal1[n] * signal2[i+n] for all valid n
func CrossCorrelate1DAdd(dst, signal1, signal2 []float32) {
	n1 := len(signal1)
	n2 := len(signal2)
	if n1 == 0 || n2 == 0 {
		return
	}
	resultLen := n1 + n2 - 1
	if len(dst) < resultLen {
		resultLen = len(dst)
	}

	// BCE hint: reslice to exact size
	dst = dst[:resultLen]
	signal1 = signal1[:n1]
	signal2 = signal2[:n2]

	for i := range resultLen {
		var sum float32
		if i < n2 {
			// Hot path: i < n2, valid range for n is [0, min(n1, n2-i))
			maxN := n1
			if n2-i < n1 {
				maxN = n2 - i
			}
			// BCE: n < maxN <= n1, and i+n < i+maxN <= i+(n2-i) = n2
			for n := range maxN {
				sum += signal1[n] * signal2[i+n]
			}
		}
		// For i >= n2, all n result in idx2 >= n2 (invalid), so sum = 0
		dst[i] += sum
	}
}

// CrossCorrelate2DAdd computes 2D cross-correlation between two matrices and adds the result to dst.
// Cross-correlation formula: R_xy(τ_r, τ_c) = Σ signal1[k,l] * signal2[k+τ_r, l+τ_c]
// signal1: [rows1*cols1] flattened matrix, cols1 = len(signal1) / rows1
// signal2: [rows2*cols2] flattened matrix, cols2 = len(signal2) / rows2
// dst: [dstRows*(cols1+cols2-1)] flattened matrix
// Panics if dimensions don't match or dst doesn't have enough space.
func CrossCorrelate2DAdd(dst []float32, dstRows int, signal1 []float32, rows1 int, signal2 []float32, rows2 int) {
	if rows1 == 0 || rows2 == 0 {
		panic("CrossCorrelate2DAdd: rows must be > 0")
	}
	cols1 := len(signal1) / rows1
	cols2 := len(signal2) / rows2
	if len(signal1) != rows1*cols1 || len(signal2) != rows2*cols2 {
		panic("CrossCorrelate2DAdd: signal dimensions don't match")
	}

	dstCols := cols1 + cols2 - 1
	resultRows := rows1 + rows2 - 1
	resultCols := dstCols

	if dstRows < resultRows {
		resultRows = dstRows
	}

	// Panic if not enough space
	if len(dst) < dstRows*dstCols {
		panic("CrossCorrelate2DAdd: dst too small")
	}

	// BCE hints
	dst = dst[:dstRows*dstCols]
	signal1 = signal1[:rows1*cols1]
	signal2 = signal2[:rows2*cols2]

	for i := range resultRows {
		for j := range resultCols {
			var sum float32
			if i < rows2 && j < cols2 {
				// Hot path: compute valid ranges for k and l
				maxK := rows1
				if rows2-i < rows1 {
					maxK = rows2 - i
				}
				maxL := cols1
				if cols2-j < cols1 {
					maxL = cols2 - j
				}
				// BCE: k < maxK <= rows1, l < maxL <= cols1
				//      i+k < i+maxK <= i+(rows2-i) = rows2
				//      j+l < j+maxL <= j+(cols2-j) = cols2
				for k := range maxK {
					rowIdx := i + k
					baseIdx := k * cols1
					rowBase := rowIdx * cols2
					for l := range maxL {
						colIdx := j + l
						sum += signal1[baseIdx+l] * signal2[rowBase+colIdx]
					}
				}
			}
			// For i >= rows2 or j >= cols2, all (k,l) result in invalid indices, so sum = 0
			dst[i*dstCols+j] += sum
		}
	}
}
