package fp32

// Gemv_N computes: y = alpha*A*x + beta*y (no transpose)
// This is BLAS GEMV_N operation
// A: M × N matrix (row-major, ldA ≥ N)
// x: N × 1 vector
// y: M × 1 vector
// Result: y = alpha*A*x + beta*y
func Gemv_N(y []float32, a, x []float32, ldA, M, N int, alpha, beta float32) {
	if M == 0 || N == 0 {
		return
	}

	// Scale y by beta first
	if beta != 1.0 {
		if beta == 0.0 {
			for i := 0; i < M; i++ {
				y[i] = 0
			}
		} else {
			for i := 0; i < M; i++ {
				y[i] *= beta
			}
		}
	}

	// If alpha is zero, we're done
	if alpha == 0.0 {
		return
	}

	// Compute y = alpha*A*x + beta*y
	// For each row i of A: y[i] += alpha * dot(row i of A, x)
	pa := 0
	for i := 0; i < M; i++ {
		dot := float32(0.0)
		px := 0

		// Compute dot product of row i of A with x
		for j := 0; j < N; j++ {
			dot += a[pa+j] * x[px]
			px++
		}

		y[i] += alpha * dot
		pa += ldA
	}
}

// Gemv_T computes: y = alpha*A^T*x + beta*y (transpose)
// This is BLAS GEMV_T operation
// A: M × N matrix (row-major, ldA ≥ N)
// A^T: N × M matrix (logical transpose)
// x: M × 1 vector
// y: N × 1 vector
// Result: y = alpha*A^T*x + beta*y
func Gemv_T(y []float32, a, x []float32, ldA, M, N int, alpha, beta float32) {
	if M == 0 || N == 0 {
		return
	}

	// Scale y by beta first
	if beta != 1.0 {
		if beta == 0.0 {
			for j := 0; j < N; j++ {
				y[j] = 0
			}
		} else {
			for j := 0; j < N; j++ {
				y[j] *= beta
			}
		}
	}

	// If alpha is zero, we're done
	if alpha == 0.0 {
		return
	}

	// Compute y = alpha*A^T*x + beta*y
	// For each column j of A (which is row j of A^T): y[j] += alpha * dot(column j of A, x)
	for j := 0; j < N; j++ {
		dot := float32(0.0)
		px := 0

		// Compute dot product of column j of A with x
		// Column j has elements: a[0*ldA + j], a[1*ldA + j], ..., a[(M-1)*ldA + j]
		pa := j
		for i := 0; i < M; i++ {
			dot += a[pa] * x[px]
			pa += ldA
			px++
		}

		y[j] += alpha * dot
	}
}

// Ger computes: A = alpha*x*y^T + A (rank-1 update)
// This is BLAS GER operation
// A: M × N matrix (row-major, ldA ≥ N, updated in-place)
// x: M × 1 vector
// y: N × 1 vector
// alpha: scalar
// Operation: A += alpha * x * y^T
func Ger(a []float32, x, y []float32, ldA, M, N int, alpha float32) {
	if M == 0 || N == 0 || alpha == 0.0 {
		return
	}

	// A[i][j] += alpha * x[i] * y[j]
	// Access A[i][j] = a[i*ldA + j]
	pa := 0
	for i := 0; i < M; i++ {
		xiAlpha := alpha * x[i]

		for j := 0; j < N; j++ {
			a[pa+j] += xiAlpha * y[j]
		}

		pa += ldA
	}
}

// Symv computes: y = alpha*A*x + beta*y (symmetric matrix-vector multiply)
// This is BLAS SYMV operation
// A: N × N symmetric matrix (row-major, ldA ≥ N)
// x: N × 1 vector
// y: N × 1 vector
// uplo: 'U' for upper triangle, 'L' for lower triangle
// Result: y = alpha*A*x + beta*y
// Note: A is symmetric, so we only need to access one triangle but multiply by both
func Symv(y []float32, a, x []float32, ldA, N int, alpha, beta float32, uplo byte) {
	if N == 0 {
		return
	}

	// Scale y by beta first
	if beta != 1.0 {
		if beta == 0.0 {
			for i := 0; i < N; i++ {
				y[i] = 0
			}
		} else {
			for i := 0; i < N; i++ {
				y[i] *= beta
			}
		}
	}

	// If alpha is zero, we're done
	if alpha == 0.0 {
		return
	}

	if uplo == 'U' || uplo == 'u' {
		// Upper triangle stored: access A[i][j] where j >= i
		for i := 0; i < N; i++ {
			pa := i * ldA
			sum := float32(0.0)

			// Diagonal and upper triangle: A[i][j] where j >= i
			for j := i; j < N; j++ {
				sum += a[pa+j] * x[j]
			}

			// Lower triangle (using symmetry): A[j][i] = A[i][j] where j < i
			for j := 0; j < i; j++ {
				sum += a[j*ldA+i] * x[j]
			}

			y[i] += alpha * sum
		}
	} else {
		// Lower triangle stored: access A[i][j] where j <= i
		for i := 0; i < N; i++ {
			pa := i * ldA
			sum := float32(0.0)

			// Diagonal and lower triangle: A[i][j] where j <= i
			for j := 0; j <= i; j++ {
				sum += a[pa+j] * x[j]
			}

			// Upper triangle (using symmetry): A[j][i] = A[i][j] where j > i
			for j := i + 1; j < N; j++ {
				sum += a[j*ldA+i] * x[j]
			}

			y[i] += alpha * sum
		}
	}
}

// Trmv computes: y = A*x (triangular matrix-vector multiply)
// This is BLAS TRMV operation
// A: N × N triangular matrix (row-major, ldA ≥ N)
// x: N × 1 vector (input)
// y: N × 1 vector (output, can be same as x for in-place)
// uplo: 'U' for upper triangle, 'L' for lower triangle
// trans: 'N' for no transpose, 'T' for transpose
// diag: 'N' for non-unit diagonal, 'U' for unit diagonal
func Trmv(y, a, x []float32, ldA, N int, uplo, trans, diag byte) {
	if N == 0 {
		return
	}

	if uplo == 'U' || uplo == 'u' {
		// Upper triangular matrix
		if trans == 'T' || trans == 't' {
			// A^T * x (lower triangular after transpose)
			if diag == 'U' || diag == 'u' {
				// Unit diagonal
				for i := N - 1; i >= 0; i-- {
					sum := x[i]
					pa := i * ldA
					for j := i + 1; j < N; j++ {
						sum += a[pa+j] * x[j]
					}
					y[i] = sum
				}
			} else {
				// Non-unit diagonal
				for i := N - 1; i >= 0; i-- {
					sum := float32(0.0)
					pa := i * ldA
					for j := i; j < N; j++ {
						sum += a[pa+j] * x[j]
					}
					y[i] = sum
				}
			}
		} else {
			// A * x (upper triangular)
			if diag == 'U' || diag == 'u' {
				// Unit diagonal
				for i := 0; i < N; i++ {
					sum := x[i]
					pa := i * ldA
					for j := i + 1; j < N; j++ {
						sum += a[pa+j] * x[j]
					}
					y[i] = sum
				}
			} else {
				// Non-unit diagonal
				for i := 0; i < N; i++ {
					sum := float32(0.0)
					pa := i * ldA
					for j := i; j < N; j++ {
						sum += a[pa+j] * x[j]
					}
					y[i] = sum
				}
			}
		}
	} else {
		// Lower triangular matrix
		if trans == 'T' || trans == 't' {
			// A^T * x (upper triangular after transpose)
			if diag == 'U' || diag == 'u' {
				// Unit diagonal
				for i := 0; i < N; i++ {
					sum := x[i]
					for j := 0; j < i; j++ {
						sum += a[j*ldA+i] * x[j]
					}
					y[i] = sum
				}
			} else {
				// Non-unit diagonal
				for i := 0; i < N; i++ {
					sum := float32(0.0)
					for j := 0; j <= i; j++ {
						sum += a[j*ldA+i] * x[j]
					}
					y[i] = sum
				}
			}
		} else {
			// A * x (lower triangular)
			if diag == 'U' || diag == 'u' {
				// Unit diagonal
				for i := 0; i < N; i++ {
					sum := x[i]
					pa := i * ldA
					for j := 0; j < i; j++ {
						sum += a[pa+j] * x[j]
					}
					y[i] = sum
				}
			} else {
				// Non-unit diagonal
				for i := 0; i < N; i++ {
					sum := float32(0.0)
					pa := i * ldA
					for j := 0; j <= i; j++ {
						sum += a[pa+j] * x[j]
					}
					y[i] = sum
				}
			}
		}
	}
}
