package fp32

// CrossCorrelate1D computes cross-correlation between two 1D signals into dst.
// Returns the number of samples filled in dst.
// dst must have at least len(signal1) + len(signal2) - 1 elements.
// corr[i] = sum(signal1[j] * signal2[j+i]) for all valid j
func CrossCorrelate1D(dst, signal1, signal2 []float32) int {
	n1 := len(signal1)
	n2 := len(signal2)
	resultLen := n1 + n2 - 1
	if len(dst) < resultLen {
		resultLen = len(dst)
	}

	// BCE hint: reslice to exact size
	dst = dst[:resultLen]
	signal1 = signal1[:n1]
	signal2 = signal2[:n2]

	// Compute cross-correlation: corr[i] = sum(signal1[j] * signal2[j+i])
	for i := range resultLen {
		sum := float32(0)
		for j := range n1 {
			k := j + i
			if k >= 0 && k < n2 {
				sum += signal1[j] * signal2[k]
			}
		}
		dst[i] = sum
	}

	return resultLen
}

// CrossCorrelate1DByConvolution computes cross-correlation using reversed signal2.
// Returns the number of samples filled in dst.
// This is an alternative implementation that reverses signal2 first.
func CrossCorrelate1DByConvolution(dst, signal1, signal2 []float32) int {
	n1 := len(signal1)
	n2 := len(signal2)

	if n1 == 0 || n2 == 0 {
		return 0
	}

	resultLen := n1 + n2 - 1
	if len(dst) < resultLen {
		resultLen = len(dst)
	}

	// Use stack array for small buffers, heap for large
	var reversed []float32
	needsAlloc := len(dst) < resultLen+n2
	if !needsAlloc {
		// Use end of dst as temporary buffer
		reversed = dst[resultLen : resultLen+n2]
	} else {
		// Stack array for small buffers, heap for large
		if n2 <= 64 {
			var buf [64]float32
			reversed = buf[:n2]
		} else {
			reversed = make([]float32, n2)
		}
	}

	// Reverse signal2: reversed[i] = signal2[n2-1-i]
	for i := range n2 {
		reversed[i] = signal2[n2-1-i]
	}

	// BCE hints
	dst = dst[:resultLen]
	signal1 = signal1[:n1]
	reversed = reversed[:n2]

	// Compute correlation using reversed signal
	for i := range resultLen {
		sum := float32(0)
		for j := range n1 {
			idx2 := j + i
			if idx2 >= 0 && idx2 < n2 {
				// signal2[idx2] = reversed[n2-1-idx2]
				sum += signal1[j] * reversed[n2-1-idx2]
			}
		}
		dst[i] = sum
	}

	return resultLen
}

// CrossCorrelate2D computes 2D cross-correlation between two matrices into dst.
// Returns the number of rows filled.
// signal1: [rows1*cols1] flattened matrix
// signal2: [rows2*cols2] flattened matrix
// dst: [dstRows*dstCols] flattened matrix
func CrossCorrelate2D(dst []float32, dstRows, dstCols int, signal1 []float32, rows1, cols1 int, signal2 []float32, rows2, cols2 int) int {
	resultRows := rows1 + rows2 - 1
	resultCols := cols1 + cols2 - 1

	if dstRows < resultRows {
		resultRows = dstRows
	}
	if dstCols < resultCols {
		resultCols = dstCols
	}

	// BCE hints
	dst = dst[:dstRows*dstCols]
	signal1 = signal1[:rows1*cols1]
	signal2 = signal2[:rows2*cols2]

	// Compute 2D cross-correlation
	for i := range resultRows {
		for j := range resultCols {
			sum := float32(0)
			for k := range rows1 {
				for l := range cols1 {
					rowIdx := i - k
					colIdx := j - l
					if rowIdx >= 0 && rowIdx < rows2 && colIdx >= 0 && colIdx < cols2 {
						sum += signal1[k*cols1+l] * signal2[rowIdx*cols2+colIdx]
					}
				}
			}
			dst[i*dstCols+j] = sum
		}
	}

	return resultRows
}

// FindPeakCorrelation finds the peak correlation and its position.
// Returns the peak value and its index.
func FindPeakCorrelation(correlation []float32) (maxCorr float32, maxIdx int) {
	n := len(correlation)

	if n == 0 {
		return 0, 0
	}

	maxCorr = float32(-1e9)
	maxIdx = 0

	correlation = correlation[:n]
	for i, v := range correlation {
		if v > maxCorr {
			maxCorr = v
			maxIdx = i
		}
	}

	return maxCorr, maxIdx
}

// FindPeakCorrelation2D finds the peak correlation in 2D and its position.
// Returns the peak value and its (row, col) position.
func FindPeakCorrelation2D(correlation []float32, rows, cols int) (maxCorr float32, maxRow, maxCol int) {
	maxCorr = float32(-1e9)
	maxRow, maxCol = 0, 0

	correlation = correlation[:rows*cols]
	for i := range rows {
		for j := range cols {
			idx := i*cols + j
			if correlation[idx] > maxCorr {
				maxCorr = correlation[idx]
				maxRow = i
				maxCol = j
			}
		}
	}

	return maxCorr, maxRow, maxCol
}

