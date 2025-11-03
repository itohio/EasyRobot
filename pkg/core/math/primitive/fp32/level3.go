package fp32

// Gemm_NN computes: C = alpha*A*B + beta*C (neither transposed)
// This is BLAS GEMM_NN operation
// A: M × K matrix (row-major, ldA ≥ K)
// B: K × N matrix (row-major, ldB ≥ N)
// C: M × N matrix (row-major, ldC ≥ N)
// Result: C = alpha*A*B + beta*C
func Gemm_NN(c, a, b []float32, ldC, ldA, ldB, M, N, K int, alpha, beta float32) {
	if M == 0 || N == 0 || K == 0 {
		return
	}

	// Scale C by beta first
	if beta != 1.0 {
		if beta == 0.0 {
			for i := 0; i < M; i++ {
				pc := i * ldC
				for j := 0; j < N; j++ {
					c[pc+j] = 0
				}
			}
		} else {
			for i := 0; i < M; i++ {
				pc := i * ldC
				for j := 0; j < N; j++ {
					c[pc+j] *= beta
				}
			}
		}
	}

	// If alpha is zero, we're done
	if alpha == 0.0 {
		return
	}

	// Compute C = alpha*A*B + beta*C
	// C[i][j] = alpha * sum_k(A[i][k] * B[k][j]) + beta * C[i][j]
	pa := 0
	pc := 0
	for i := 0; i < M; i++ {
		for j := 0; j < N; j++ {
			sum := float32(0.0)
			pb := 0

			// Compute dot product of row i of A with column j of B
			for k := 0; k < K; k++ {
				sum += a[pa+k] * b[pb+j]
				pb += ldB
			}

			c[pc+j] += alpha * sum
		}
		pa += ldA
		pc += ldC
	}
}

// Gemm_NT computes: C = alpha*A*B^T + beta*C (B transposed)
// This is BLAS GEMM_NT operation
// A: M × K matrix (row-major, ldA ≥ K)
// B: N × K matrix (row-major, ldB ≥ K), but we treat it as B^T which is K × N
// C: M × N matrix (row-major, ldC ≥ N)
// Result: C = alpha*A*B^T + beta*C
func Gemm_NT(c, a, b []float32, ldC, ldA, ldB, M, N, K int, alpha, beta float32) {
	if M == 0 || N == 0 || K == 0 {
		return
	}

	// Scale C by beta first
	if beta != 1.0 {
		if beta == 0.0 {
			for i := 0; i < M; i++ {
				pc := i * ldC
				for j := 0; j < N; j++ {
					c[pc+j] = 0
				}
			}
		} else {
			for i := 0; i < M; i++ {
				pc := i * ldC
				for j := 0; j < N; j++ {
					c[pc+j] *= beta
				}
			}
		}
	}

	// If alpha is zero, we're done
	if alpha == 0.0 {
		return
	}

	// Compute C = alpha*A*B^T + beta*C
	// C[i][j] = alpha * sum_k(A[i][k] * B^T[k][j]) + beta * C[i][j]
	// B^T[k][j] = B[j][k] = b[j*ldB + k]
	pa := 0
	pc := 0
	for i := 0; i < M; i++ {
		for j := 0; j < N; j++ {
			sum := float32(0.0)
			pbRow := j * ldB

			// Compute dot product of row i of A with row j of B (which is column j of B^T)
			for k := 0; k < K; k++ {
				sum += a[pa+k] * b[pbRow+k]
			}

			c[pc+j] += alpha * sum
		}
		pa += ldA
		pc += ldC
	}
}

// Gemm_TN computes: C = alpha*A^T*B + beta*C (A transposed)
// This is BLAS GEMM_TN operation
// A: K × M matrix (row-major, ldA ≥ M), but we treat it as A^T which is M × K
// B: K × N matrix (row-major, ldB ≥ N)
// C: M × N matrix (row-major, ldC ≥ N)
// Result: C = alpha*A^T*B + beta*C
func Gemm_TN(c, a, b []float32, ldC, ldA, ldB, M, N, K int, alpha, beta float32) {
	if M == 0 || N == 0 || K == 0 {
		return
	}

	// Scale C by beta first
	if beta != 1.0 {
		if beta == 0.0 {
			for i := 0; i < M; i++ {
				pc := i * ldC
				for j := 0; j < N; j++ {
					c[pc+j] = 0
				}
			}
		} else {
			for i := 0; i < M; i++ {
				pc := i * ldC
				for j := 0; j < N; j++ {
					c[pc+j] *= beta
				}
			}
		}
	}

	// If alpha is zero, we're done
	if alpha == 0.0 {
		return
	}

	// Compute C = alpha*A^T*B + beta*C
	// C[i][j] = alpha * sum_k(A^T[i][k] * B[k][j]) + beta * C[i][j]
	// A^T[i][k] = A[k][i] = a[k*ldA + i]
	pc := 0
	for i := 0; i < M; i++ {
		for j := 0; j < N; j++ {
			sum := float32(0.0)
			pb := 0

			// Compute dot product of column i of A (which is row i of A^T) with column j of B
			for k := 0; k < K; k++ {
				paElem := k*ldA + i // A[k][i] = a[k*ldA + i]
				sum += a[paElem] * b[pb+j]
				pb += ldB
			}

			c[pc+j] += alpha * sum
		}
		pc += ldC
	}
}

// Gemm_TT computes: C = alpha*A^T*B^T + beta*C (both transposed)
// This is BLAS GEMM_TT operation
// A: K × M matrix (row-major, ldA ≥ M), but we treat it as A^T which is M × K
// B: N × K matrix (row-major, ldB ≥ K), but we treat it as B^T which is K × N
// C: M × N matrix (row-major, ldC ≥ N)
// Result: C = alpha*A^T*B^T + beta*C
func Gemm_TT(c, a, b []float32, ldC, ldA, ldB, M, N, K int, alpha, beta float32) {
	if M == 0 || N == 0 || K == 0 {
		return
	}

	// Scale C by beta first
	if beta != 1.0 {
		if beta == 0.0 {
			for i := 0; i < M; i++ {
				pc := i * ldC
				for j := 0; j < N; j++ {
					c[pc+j] = 0
				}
			}
		} else {
			for i := 0; i < M; i++ {
				pc := i * ldC
				for j := 0; j < N; j++ {
					c[pc+j] *= beta
				}
			}
		}
	}

	// If alpha is zero, we're done
	if alpha == 0.0 {
		return
	}

	// Compute C = alpha*A^T*B^T + beta*C
	// C[i][j] = alpha * sum_k(A^T[i][k] * B^T[k][j]) + beta * C[i][j]
	// A^T[i][k] = A[k][i] = a[k*ldA + i]
	// B^T[k][j] = B[j][k] = b[j*ldB + k]
	pc := 0
	for i := 0; i < M; i++ {
		for j := 0; j < N; j++ {
			sum := float32(0.0)
			pbRow := j * ldB

			// Compute dot product of column i of A (row i of A^T) with row j of B (column j of B^T)
			for k := 0; k < K; k++ {
				paElem := k*ldA + i // A[k][i] = a[k*ldA + i]
				sum += a[paElem] * b[pbRow+k]
			}

			c[pc+j] += alpha * sum
		}
		pc += ldC
	}
}

// Syrk computes: C = alpha*A*A^T + beta*C (symmetric rank-k update)
// This is BLAS SYRK operation
// A: N × K matrix (row-major, ldA ≥ K)
// C: N × N symmetric matrix (row-major, ldC ≥ N, updated in-place)
// uplo: 'U' for upper triangle, 'L' for lower triangle
// Result: C = alpha*A*A^T + beta*C
func Syrk(c, a []float32, ldC, ldA, N, K int, alpha, beta float32, uplo byte) {
	if N == 0 || K == 0 {
		return
	}

	// Scale C by beta first
	if beta != 1.0 {
		if beta == 0.0 {
			if uplo == 'U' || uplo == 'u' {
				for i := 0; i < N; i++ {
					pc := i * ldC
					for j := i; j < N; j++ {
						c[pc+j] = 0
					}
				}
			} else {
				for i := 0; i < N; i++ {
					pc := i * ldC
					for j := 0; j <= i; j++ {
						c[pc+j] = 0
					}
				}
			}
		} else {
			if uplo == 'U' || uplo == 'u' {
				for i := 0; i < N; i++ {
					pc := i * ldC
					for j := i; j < N; j++ {
						c[pc+j] *= beta
					}
				}
			} else {
				for i := 0; i < N; i++ {
					pc := i * ldC
					for j := 0; j <= i; j++ {
						c[pc+j] *= beta
					}
				}
			}
		}
	}

	// If alpha is zero, we're done
	if alpha == 0.0 {
		return
	}

	// Compute C = alpha*A*A^T + beta*C
	// C[i][j] = alpha * sum_k(A[i][k] * A[j][k]) + beta * C[i][j]
	if uplo == 'U' || uplo == 'u' {
		// Upper triangle
		for i := 0; i < N; i++ {
			pc := i * ldC
			pa_i := i * ldA
			for j := i; j < N; j++ {
				pa_j := j * ldA
				sum := float32(0.0)
				for k := 0; k < K; k++ {
					sum += a[pa_i+k] * a[pa_j+k]
				}
				c[pc+j] += alpha * sum
			}
		}
	} else {
		// Lower triangle
		for i := 0; i < N; i++ {
			pc := i * ldC
			pa_i := i * ldA
			for j := 0; j <= i; j++ {
				pa_j := j * ldA
				sum := float32(0.0)
				for k := 0; k < K; k++ {
					sum += a[pa_i+k] * a[pa_j+k]
				}
				c[pc+j] += alpha * sum
			}
		}
	}
}

// Trmm computes: C = alpha*A*B + beta*C (triangular matrix-matrix multiply)
// This is BLAS TRMM operation
// A: M × M triangular matrix (row-major, ldA ≥ M)
// B: M × N matrix (row-major, ldB ≥ N)
// C: M × N matrix (row-major, ldC ≥ N)
// side: 'L' for left (A*B), 'R' for right (B*A)
// uplo: 'U' for upper triangle, 'L' for lower triangle
// trans: 'N' for no transpose, 'T' for transpose
// diag: 'N' for non-unit diagonal, 'U' for unit diagonal
func Trmm(c, a, b []float32, ldC, ldA, ldB, M, N int, alpha, beta float32, side, uplo, trans, diag byte) {
	if M == 0 || N == 0 {
		return
	}

	// Scale C by beta first
	if beta != 1.0 {
		if beta == 0.0 {
			for i := 0; i < M; i++ {
				pc := i * ldC
				for j := 0; j < N; j++ {
					c[pc+j] = 0
				}
			}
		} else {
			for i := 0; i < M; i++ {
				pc := i * ldC
				for j := 0; j < N; j++ {
					c[pc+j] *= beta
				}
			}
		}
	}

	// If alpha is zero, we're done
	if alpha == 0.0 {
		return
	}

	if side == 'L' || side == 'l' {
		// C = alpha*A*B + beta*C (left side)
		if uplo == 'U' || uplo == 'u' {
			// Upper triangular A
			if trans == 'T' || trans == 't' {
				// A^T * B (lower triangular after transpose)
				if diag == 'U' || diag == 'u' {
					// Unit diagonal
					for j := 0; j < N; j++ {
						for i := M - 1; i >= 0; i-- {
							sum := b[i*ldB+j]
							pa := i * ldA
							for k := i + 1; k < M; k++ {
								sum += a[pa+k] * c[k*ldC+j]
							}
							c[i*ldC+j] = alpha*sum + c[i*ldC+j]
						}
					}
				} else {
					// Non-unit diagonal
					for j := 0; j < N; j++ {
						for i := M - 1; i >= 0; i-- {
							sum := float32(0.0)
							pa := i * ldA
							for k := i; k < M; k++ {
								sum += a[pa+k] * c[k*ldC+j]
							}
							c[i*ldC+j] = alpha*sum + c[i*ldC+j]
						}
					}
				}
			} else {
				// A * B (upper triangular)
				if diag == 'U' || diag == 'u' {
					// Unit diagonal
					for j := 0; j < N; j++ {
						for i := 0; i < M; i++ {
							sum := b[i*ldB+j]
							pa := i * ldA
							for k := i + 1; k < M; k++ {
								sum += a[pa+k] * b[k*ldB+j]
							}
							c[i*ldC+j] = alpha*sum + c[i*ldC+j]
						}
					}
				} else {
					// Non-unit diagonal
					for j := 0; j < N; j++ {
						for i := 0; i < M; i++ {
							sum := float32(0.0)
							pa := i * ldA
							for k := i; k < M; k++ {
								sum += a[pa+k] * b[k*ldB+j]
							}
							c[i*ldC+j] = alpha*sum + c[i*ldC+j]
						}
					}
				}
			}
		} else {
			// Lower triangular A
			if trans == 'T' || trans == 't' {
				// A^T * B (upper triangular after transpose)
				if diag == 'U' || diag == 'u' {
					// Unit diagonal
					for j := 0; j < N; j++ {
						for i := 0; i < M; i++ {
							sum := b[i*ldB+j]
							for k := 0; k < i; k++ {
								sum += a[k*ldA+i] * c[k*ldC+j]
							}
							c[i*ldC+j] = alpha*sum + c[i*ldC+j]
						}
					}
				} else {
					// Non-unit diagonal
					for j := 0; j < N; j++ {
						for i := 0; i < M; i++ {
							sum := float32(0.0)
							for k := 0; k <= i; k++ {
								sum += a[k*ldA+i] * c[k*ldC+j]
							}
							c[i*ldC+j] = alpha*sum + c[i*ldC+j]
						}
					}
				}
			} else {
				// A * B (lower triangular)
				if diag == 'U' || diag == 'u' {
					// Unit diagonal
					for j := 0; j < N; j++ {
						for i := 0; i < M; i++ {
							sum := b[i*ldB+j]
							pa := i * ldA
							for k := 0; k < i; k++ {
								sum += a[pa+k] * b[k*ldB+j]
							}
							c[i*ldC+j] = alpha*sum + c[i*ldC+j]
						}
					}
				} else {
					// Non-unit diagonal
					for j := 0; j < N; j++ {
						for i := 0; i < M; i++ {
							sum := float32(0.0)
							pa := i * ldA
							for k := 0; k <= i; k++ {
								sum += a[pa+k] * b[k*ldB+j]
							}
							c[i*ldC+j] = alpha*sum + c[i*ldC+j]
						}
					}
				}
			}
		}
	} else {
		// C = alpha*B*A + beta*C (right side)
		// For right side, we need to handle B*A where A is N×N triangular
		// This is less common but included for completeness
		// Implementation: treat as (A^T * B^T)^T for computational efficiency
		// For now, we implement a direct version
		if uplo == 'U' || uplo == 'u' {
			// Upper triangular A
			if trans == 'T' || trans == 't' {
				// B * A^T (lower triangular after transpose)
				if diag == 'U' || diag == 'u' {
					// Unit diagonal
					for i := 0; i < M; i++ {
						for j := N - 1; j >= 0; j-- {
							sum := b[i*ldB+j]
							for k := j + 1; k < N; k++ {
								sum += b[i*ldB+k] * a[j*ldA+k]
							}
							c[i*ldC+j] = alpha*sum + c[i*ldC+j]
						}
					}
				} else {
					// Non-unit diagonal
					for i := 0; i < M; i++ {
						for j := N - 1; j >= 0; j-- {
							sum := float32(0.0)
							for k := j; k < N; k++ {
								sum += b[i*ldB+k] * a[j*ldA+k]
							}
							c[i*ldC+j] = alpha*sum + c[i*ldC+j]
						}
					}
				}
			} else {
				// B * A (upper triangular)
				if diag == 'U' || diag == 'u' {
					// Unit diagonal
					for i := 0; i < M; i++ {
						for j := 0; j < N; j++ {
							sum := b[i*ldB+j]
							for k := j + 1; k < N; k++ {
								sum += b[i*ldB+k] * a[j*ldA+k]
							}
							c[i*ldC+j] = alpha*sum + c[i*ldC+j]
						}
					}
				} else {
					// Non-unit diagonal
					for i := 0; i < M; i++ {
						for j := 0; j < N; j++ {
							sum := float32(0.0)
							for k := j; k < N; k++ {
								sum += b[i*ldB+k] * a[j*ldA+k]
							}
							c[i*ldC+j] = alpha*sum + c[i*ldC+j]
						}
					}
				}
			}
		} else {
			// Lower triangular A
			if trans == 'T' || trans == 't' {
				// B * A^T (upper triangular after transpose)
				if diag == 'U' || diag == 'u' {
					// Unit diagonal
					for i := 0; i < M; i++ {
						for j := 0; j < N; j++ {
							sum := b[i*ldB+j]
							for k := 0; k < j; k++ {
								sum += b[i*ldB+k] * a[k*ldA+j]
							}
							c[i*ldC+j] = alpha*sum + c[i*ldC+j]
						}
					}
				} else {
					// Non-unit diagonal
					for i := 0; i < M; i++ {
						for j := 0; j < N; j++ {
							sum := float32(0.0)
							for k := 0; k <= j; k++ {
								sum += b[i*ldB+k] * a[k*ldA+j]
							}
							c[i*ldC+j] = alpha*sum + c[i*ldC+j]
						}
					}
				}
			} else {
				// B * A (lower triangular)
				if diag == 'U' || diag == 'u' {
					// Unit diagonal
					for i := 0; i < M; i++ {
						for j := 0; j < N; j++ {
							sum := b[i*ldB+j]
							for k := 0; k < j; k++ {
								sum += b[i*ldB+k] * a[j*ldA+k]
							}
							c[i*ldC+j] = alpha*sum + c[i*ldC+j]
						}
					}
				} else {
					// Non-unit diagonal
					for i := 0; i < M; i++ {
						for j := 0; j < N; j++ {
							sum := float32(0.0)
							for k := 0; k <= j; k++ {
								sum += b[i*ldB+k] * a[j*ldA+k]
							}
							c[i*ldC+j] = alpha*sum + c[i*ldC+j]
						}
					}
				}
			}
		}
	}
}
