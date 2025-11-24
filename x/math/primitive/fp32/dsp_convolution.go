package fp32

// Convolve1D computes 1D convolution between two signals into dst according to information theory.
// Convolution formula: y[n] = Σ x[k]h[n-k]
// where signal1 = x, signal2 = h
// dst[i] = Σ signal1[k] * signal2[i-k] for all valid k
func Convolve1D(dst, signal1, signal2 []float32) {
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

	// Compute convolution: y[n] = Σ x[k]h[n-k]
	// dst[i] = Σ signal1[k] * signal2[i-k] for all valid k
	// Optimized: split into regions to eliminate bounds checks in hot path
	for i := range resultLen {
		var sum float32
		// idx2 = i - k must be in [0, n2), so k in (i-n2, i] and k in [0, n1)
		// Valid range: k in [max(0, i-n2+1), min(n1, i+1))
		minK := 0
		if i >= n2 {
			minK = i - n2 + 1
		}
		maxK := i + 1
		if n1 < maxK {
			maxK = n1
		}
		if minK < maxK {
			// Hot path: valid k range, no bounds checks needed
			// BCE: minK >= 0, maxK <= n1, and i-k in [i-maxK+1, i-minK] = [i-(i+1)+1, i-0] = [0, i] ⊆ [0, n2) when i < n2
			//      When i >= n2, minK = i-n2+1 ensures i-k <= n2-1
			for k := minK; k < maxK; k++ {
				sum += signal1[k] * signal2[i-k]
			}
		}
		// For minK >= maxK, no valid k, so sum = 0
		dst[i] = sum
	}
}

// Convolve1DAdd computes 1D convolution between two signals and adds the result to dst according to information theory.
// Convolution formula: y[n] = Σ x[k]h[n-k]
// where signal1 = x, signal2 = h
// dst[i] += Σ signal1[k] * signal2[i-k] for all valid k
func Convolve1DAdd(dst, signal1, signal2 []float32) {
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
		// idx2 = i - k must be in [0, n2), so k in (i-n2, i] and k in [0, n1)
		// Valid range: k in [max(0, i-n2+1), min(n1, i+1))
		minK := 0
		if i >= n2 {
			minK = i - n2 + 1
		}
		maxK := i + 1
		if n1 < maxK {
			maxK = n1
		}
		if minK < maxK {
			// Hot path: valid k range, no bounds checks needed
			// BCE: minK >= 0, maxK <= n1, and i-k in [i-maxK+1, i-minK] = [i-(i+1)+1, i-0] = [0, i] ⊆ [0, n2) when i < n2
			//      When i >= n2, minK = i-n2+1 ensures i-k <= n2-1
			for k := minK; k < maxK; k++ {
				sum += signal1[k] * signal2[i-k]
			}
		}
		// For minK >= maxK, no valid k, so sum = 0
		dst[i] += sum
	}
}

// Convolve2D computes 2D convolution between two matrices into dst according to information theory.
// Convolution formula: y[i,j] = Σ signal1[k,l] * signal2[i-k, j-l]
// signal1: [rows1*cols1] flattened matrix, cols1 = len(signal1) / rows1
// signal2: [rows2*cols2] flattened matrix, cols2 = len(signal2) / rows2
// dst: [dstRows*max(cols1,cols2)] flattened matrix
// Panics if dimensions don't match or dst doesn't have enough space.
func Convolve2D(dst []float32, dstRows int, signal1 []float32, rows1 int, signal2 []float32, rows2 int) {
	if rows1 == 0 || rows2 == 0 {
		panic("Convolve2D: rows must be > 0")
	}
	cols1 := len(signal1) / rows1
	cols2 := len(signal2) / rows2
	if len(signal1) != rows1*cols1 || len(signal2) != rows2*cols2 {
		panic("Convolve2D: signal dimensions don't match")
	}

	dstCols := cols1
	if cols2 > cols1 {
		dstCols = cols2
	}

	// Panic if not enough space
	if len(dst) < dstRows*dstCols {
		panic("Convolve2D: dst too small")
	}

	// BCE hints
	dst = dst[:dstRows*dstCols]
	signal1 = signal1[:rows1*cols1]
	signal2 = signal2[:rows2*cols2]

	// Compute 2D convolution: y[i,j] = Σ signal1[k,l] * signal2[i-k, j-l]
	// Optimized: split into regions to eliminate bounds checks in hot path
	for i := range dstRows {
		for j := range dstCols {
			var sum float32
			// rowIdx = i - k must be in [0, rows2), so k in (i-rows2, i] and k in [0, rows1)
			// colIdx = j - l must be in [0, cols2), so l in (j-cols2, j] and l in [0, cols1)
			// Valid ranges: k in [max(0, i-rows2+1), min(rows1, i+1))
			//               l in [max(0, j-cols2+1), min(cols1, j+1))
			minK := 0
			if i >= rows2 {
				minK = i - rows2 + 1
			}
			maxK := i + 1
			if rows1 < maxK {
				maxK = rows1
			}
			minL := 0
			if j >= cols2 {
				minL = j - cols2 + 1
			}
			maxL := j + 1
			if cols1 < maxL {
				maxL = cols1
			}
			if minK < maxK && minL < maxL {
				// Hot path: valid k and l ranges, no bounds checks needed
				// BCE: minK >= 0, maxK <= rows1, minL >= 0, maxL <= cols1
				//      i-k in [i-maxK+1, i-minK] ⊆ [0, rows2)
				//      j-l in [j-maxL+1, j-minL] ⊆ [0, cols2)
				for k := minK; k < maxK; k++ {
					rowIdx := i - k
					baseIdx := k * cols1
					rowBase := rowIdx * cols2
					for l := minL; l < maxL; l++ {
						colIdx := j - l
						sum += signal1[baseIdx+l] * signal2[rowBase+colIdx]
					}
				}
			}
			// For minK >= maxK or minL >= maxL, no valid (k,l), so sum = 0
			dst[i*dstCols+j] = sum
		}
	}
}

// Convolve2DAdd computes 2D convolution between two matrices and adds the result to dst according to information theory.
// Convolution formula: y[i,j] = Σ signal1[k,l] * signal2[i-k, j-l]
// signal1: [rows1*cols1] flattened matrix, cols1 = len(signal1) / rows1
// signal2: [rows2*cols2] flattened matrix, cols2 = len(signal2) / rows2
// dst: [dstRows*max(cols1,cols2)] flattened matrix
// Panics if dimensions don't match or dst doesn't have enough space.
func Convolve2DAdd(dst []float32, dstRows int, signal1 []float32, rows1 int, signal2 []float32, rows2 int) {
	if rows1 == 0 || rows2 == 0 {
		panic("Convolve2DAdd: rows must be > 0")
	}
	cols1 := len(signal1) / rows1
	cols2 := len(signal2) / rows2
	if len(signal1) != rows1*cols1 || len(signal2) != rows2*cols2 {
		panic("Convolve2DAdd: signal dimensions don't match")
	}

	dstCols := cols1
	if cols2 > cols1 {
		dstCols = cols2
	}

	// Panic if not enough space
	if len(dst) < dstRows*dstCols {
		panic("Convolve2DAdd: dst too small")
	}

	// BCE hints
	dst = dst[:dstRows*dstCols]
	signal1 = signal1[:rows1*cols1]
	signal2 = signal2[:rows2*cols2]

	for i := range dstRows {
		for j := range dstCols {
			var sum float32
			// rowIdx = i - k must be in [0, rows2), so k in (i-rows2, i] and k in [0, rows1)
			// colIdx = j - l must be in [0, cols2), so l in (j-cols2, j] and l in [0, cols1)
			// Valid ranges: k in [max(0, i-rows2+1), min(rows1, i+1))
			//               l in [max(0, j-cols2+1), min(cols1, j+1))
			minK := 0
			if i >= rows2 {
				minK = i - rows2 + 1
			}
			maxK := i + 1
			if rows1 < maxK {
				maxK = rows1
			}
			minL := 0
			if j >= cols2 {
				minL = j - cols2 + 1
			}
			maxL := j + 1
			if cols1 < maxL {
				maxL = cols1
			}
			if minK < maxK && minL < maxL {
				// Hot path: valid k and l ranges, no bounds checks needed
				// BCE: minK >= 0, maxK <= rows1, minL >= 0, maxL <= cols1
				//      i-k in [i-maxK+1, i-minK] ⊆ [0, rows2)
				//      j-l in [j-maxL+1, j-minL] ⊆ [0, cols2)
				for k := minK; k < maxK; k++ {
					rowIdx := i - k
					baseIdx := k * cols1
					rowBase := rowIdx * cols2
					for l := minL; l < maxL; l++ {
						colIdx := j - l
						sum += signal1[baseIdx+l] * signal2[rowBase+colIdx]
					}
				}
			}
			// For minK >= maxK or minL >= maxL, no valid (k,l), so sum = 0
			dst[i*dstCols+j] += sum
		}
	}
}
