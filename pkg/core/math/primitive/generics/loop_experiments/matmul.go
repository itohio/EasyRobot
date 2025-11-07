package loop_experiments

// Assembly matrix multiplication (amd64 only)
// Declared in asm_matmul_amd64.s
func asmMatMul(C, A, B *float32, m, k, n int)

// Naive matrix multiplication: C = A * B
// A is m×k, B is k×n, C is m×n
func MatMulNaive(C, A, B []float32, m, k, n int) {
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var sum float32
			for l := 0; l < k; l++ {
				sum += A[i*k+l] * B[l*n+j]
			}
			C[i*n+j] = sum
		}
	}
}

// Optimized matrix multiplication using BCE techniques
// Uses row slices and reslicing for better performance
func MatMulOptimized(C, A, B []float32, m, k, n int) {
	// Process row by row
	for i := range m {
		cRow := C[i*n : i*n+n]
		cRow = cRow[:n] // BCE hint
		aRow := A[i*k : i*k+k]
		aRow = aRow[:k] // BCE hint

		// Initialize row to zero
		for j := range n {
			cRow[j] = 0
		}

		// Multiply row of A with columns of B
		for l := range k {
			aVal := aRow[l]
			bCol := B[l*n : l*n+n]
			bCol = bCol[:n] // BCE hint

			// Add aVal * bCol to cRow
			for j := range n {
				cRow[j] += aVal * bCol[j]
			}
		}
	}
}

// Optimized matrix multiplication with better cache locality
// Transpose B first for better memory access pattern
func MatMulOptimizedTranspose(C, A, B []float32, m, k, n int) {
	// Allocate temporary storage for B^T
	BT := make([]float32, k*n)

	// Transpose B into BT
	for i := range k {
		btRow := BT[i*n : i*n+n]
		btRow = btRow[:n]
		for j := range n {
			btRow[j] = B[j*k+i]
		}
	}

	// Now multiply A * BT^T (which is A * B)
	for i := range m {
		cRow := C[i*n : i*n+n]
		cRow = cRow[:n]
		aRow := A[i*k : i*k+k]
		aRow = aRow[:k]

		for j := range n {
			var sum float32
			btCol := BT[j*k : j*k+k]
			btCol = btCol[:k]

			for l := range k {
				sum += aRow[l] * btCol[l]
			}
			cRow[j] = sum
		}
	}
}

// Flattened matrix multiplication (best for contiguous)
func MatMulFlattened(C, A, B []float32, m, k, n int) {
	sizeC := m * n
	sizeA := m * k
	sizeB := k * n
	C = C[:sizeC]
	A = A[:sizeA]
	B = B[:sizeB]
	
	// Initialize C to zero
	for i := range sizeC {
		C[i] = 0
	}
	
	// Multiply using flattened indices
	for i := range m {
		for l := range k {
			aVal := A[i*k+l]
			baseB := l * n
			baseC := i * n
			for j := range n {
				C[baseC+j] += aVal * B[baseB+j]
			}
		}
	}
}

// Experimental implementations - trying different BCE techniques

// MatMulExp1: Row slices with reslice hints
func MatMulExp1(C, A, B []float32, m, k, n int) {
	for i := range m {
		cRow := C[i*n : i*n+n]
		cRow = cRow[:n] // BCE hint
		aRow := A[i*k : i*k+k]
		aRow = aRow[:k] // BCE hint
		
		// Initialize
		for j := range n {
			cRow[j] = 0
		}
		
		for l := range k {
			aVal := aRow[l]
			bCol := B[l*n : l*n+n]
			bCol = bCol[:n] // BCE hint
			for j := range n {
				cRow[j] += aVal * bCol[j]
			}
		}
	}
}

// MatMulExp2: Precompute row bases
func MatMulExp2(C, A, B []float32, m, k, n int) {
	for i := range m {
		baseC := i * n
		baseA := i * k
		cRow := C[baseC : baseC+n]
		cRow = cRow[:n]
		aRow := A[baseA : baseA+k]
		aRow = aRow[:k]
		
		// Initialize
		for j := range n {
			cRow[j] = 0
		}
		
		for l := range k {
			aVal := aRow[l]
			baseB := l * n
			bCol := B[baseB : baseB+n]
			bCol = bCol[:n]
			for j := range n {
				cRow[j] += aVal * bCol[j]
			}
		}
	}
}

// MatMulExp3: Flatten C initialization, then row slices
func MatMulExp3(C, A, B []float32, m, k, n int) {
	// Initialize C to zero (flattened)
	sizeC := m * n
	C = C[:sizeC]
	for i := range sizeC {
		C[i] = 0
	}
	
	// Multiply using row slices
	for i := range m {
		cRow := C[i*n : i*n+n]
		cRow = cRow[:n]
		aRow := A[i*k : i*k+k]
		aRow = aRow[:k]
		
		for l := range k {
			aVal := aRow[l]
			bCol := B[l*n : l*n+n]
			bCol = bCol[:n]
			for j := range n {
				cRow[j] += aVal * bCol[j]
			}
		}
	}
}

// MatMulExp4: Access last element hints
func MatMulExp4(C, A, B []float32, m, k, n int) {
	if m > 0 && n > 0 && k > 0 {
		_ = C[(m-1)*n+(n-1)]
		_ = A[(m-1)*k+(k-1)]
		_ = B[(k-1)*n+(n-1)]
	}
	
	for i := range m {
		for j := range n {
			var sum float32
			for l := range k {
				sum += A[i*k+l] * B[l*n+j]
			}
			C[i*n+j] = sum
		}
	}
}

// MatMulExp5: Row slices with access last hints
func MatMulExp5(C, A, B []float32, m, k, n int) {
	for i := range m {
		cRow := C[i*n : i*n+n]
		aRow := A[i*k : i*k+k]
		if n > 0 {
			_ = cRow[n-1]
		}
		if k > 0 {
			_ = aRow[k-1]
		}
		
		// Initialize
		for j := range n {
			cRow[j] = 0
		}
		
		for l := range k {
			aVal := aRow[l]
			bCol := B[l*n : l*n+n]
			if n > 0 {
				_ = bCol[n-1]
			}
			for j := range n {
				cRow[j] += aVal * bCol[j]
			}
		}
	}
}

// MatMulExp6: Unroll inner loop (j) by 4
func MatMulExp6(C, A, B []float32, m, k, n int) {
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

// MatMulExp7: Unroll middle loop (l) by 4
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
			aVal0 := aRow[l]
			aVal1 := aRow[l+1]
			aVal2 := aRow[l+2]
			aVal3 := aRow[l+3]
			bCol0 := B[l*n : l*n+n]
			bCol1 := B[(l+1)*n : (l+1)*n+n]
			bCol2 := B[(l+2)*n : (l+2)*n+n]
			bCol3 := B[(l+3)*n : (l+3)*n+n]
			bCol0 = bCol0[:n]
			bCol1 = bCol1[:n]
			bCol2 = bCol2[:n]
			bCol3 = bCol3[:n]
			
			for j := range n {
				cRow[j] += aVal0*bCol0[j] + aVal1*bCol1[j] + aVal2*bCol2[j] + aVal3*bCol3[j]
			}
			l += 4
		}
		// Remainder
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

// MatMulExp8: Blocked/tiled approach (4x4 blocks)
func MatMulExp8(C, A, B []float32, m, k, n int) {
	blockSize := 4
	for i := 0; i < m; i += blockSize {
		iEnd := i + blockSize
		if iEnd > m {
			iEnd = m
		}
		for j := 0; j < n; j += blockSize {
			jEnd := j + blockSize
			if jEnd > n {
				jEnd = n
			}
			// Initialize block of C
			for ii := i; ii < iEnd; ii++ {
				cRow := C[ii*n+j : ii*n+jEnd]
				cRow = cRow[:jEnd-j]
				for jj := range cRow {
					cRow[jj] = 0
				}
			}
			// Multiply block
			for l := range k {
				for ii := i; ii < iEnd; ii++ {
					aVal := A[ii*k+l]
					cRow := C[ii*n+j : ii*n+jEnd]
					cRow = cRow[:jEnd-j]
					bCol := B[l*n+j : l*n+jEnd]
					bCol = bCol[:jEnd-j]
					for jj := range cRow {
						cRow[jj] += aVal * bCol[jj]
					}
				}
			}
		}
	}
}

// MatMulExp9: Transpose B in-place (no allocation)
func MatMulExp9(C, A, B []float32, m, k, n int) {
	// Use BT stored in a temporary slice (but try to minimize allocation)
	// Actually, let's try accessing B column-wise more efficiently
	for i := range m {
		cRow := C[i*n : i*n+n]
		cRow = cRow[:n]
		aRow := A[i*k : i*k+k]
		aRow = aRow[:k]
		
		// Initialize
		for j := range n {
			cRow[j] = 0
		}
		
		// Access B column-wise (better cache locality for B)
		for j := range n {
			sum := cRow[j]
			for l := range k {
				sum += aRow[l] * B[l*n+j]
			}
			cRow[j] = sum
		}
	}
}

// MatMulExp10: Combine Exp1 with inner loop unrolling
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
			
			// Process 4 elements at a time
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

// MatMulExp11: Blocked with larger blocks (8x8)
func MatMulExp11(C, A, B []float32, m, k, n int) {
	blockSize := 8
	for i := 0; i < m; i += blockSize {
		iEnd := i + blockSize
		if iEnd > m {
			iEnd = m
		}
		for j := 0; j < n; j += blockSize {
			jEnd := j + blockSize
			if jEnd > n {
				jEnd = n
			}
			// Initialize block
			for ii := i; ii < iEnd; ii++ {
				for jj := j; jj < jEnd; jj++ {
					C[ii*n+jj] = 0
				}
			}
			// Multiply block
			for l := range k {
				for ii := i; ii < iEnd; ii++ {
					aVal := A[ii*k+l]
					for jj := j; jj < jEnd; jj++ {
						C[ii*n+jj] += aVal * B[l*n+jj]
					}
				}
			}
		}
	}
}

// MatMulExp12: Row slices with range, no initialization loop
func MatMulExp12(C, A, B []float32, m, k, n int) {
	for i := range m {
		cRow := C[i*n : i*n+n]
		cRow = cRow[:n]
		aRow := A[i*k : i*k+k]
		aRow = aRow[:k]
		
		// Initialize and first multiply in one pass
		l := 0
		if l < k {
			aVal := aRow[l]
			bCol := B[l*n : l*n+n]
			bCol = bCol[:n]
			for j := range n {
				cRow[j] = aVal * bCol[j]
			}
			l++
		}
		// Remaining l values
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
