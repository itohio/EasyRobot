package fp32

// GemmBatched computes batched matrix-matrix multiply: C[k] = alpha*A[k]*B[k] + beta*C[k]
// This is BLAS batched GEMM operation with separate arrays
// Each batch k: C[k] = alpha*A[k]*B[k] + beta*C[k]
// A[k]: M × K matrix (row-major, ldA ≥ K), stored at offset k*stridea
// B[k]: K × N matrix (row-major, ldB ≥ N), stored at offset k*strideb
// C[k]: M × N matrix (row-major, ldC ≥ N), stored at offset k*stridec
// batchCount: number of matrices in the batch
func GemmBatched(c, a, b []float32, ldC, ldA, ldB, M, N, K int, alpha, beta float32, batchCount int, stridea, strideb, stridec int) {
	if batchCount == 0 || M == 0 || N == 0 || K == 0 {
		return
	}

	for k := 0; k < batchCount; k++ {
		// Get offsets for batch k
		offsetA := k * stridea
		offsetB := k * strideb
		offsetC := k * stridec

		// Call GEMM_NN for this batch
		Gemm_NN(c[offsetC:], a[offsetA:], b[offsetB:], ldC, ldA, ldB, M, N, K, alpha, beta)
	}
}

// GemmStrided computes strided batched matrix-matrix multiply: C[k] = alpha*A[k]*B[k] + beta*C[k]
// This is BLAS strided batched GEMM operation
// All matrices stored in contiguous arrays with fixed strides
// A[k]: M × K matrix (row-major, ldA ≥ K), stored at offset k*stridea
// B[k]: K × N matrix (row-major, ldB ≥ N), stored at offset k*strideb
// C[k]: M × N matrix (row-major, ldC ≥ N), stored at offset k*stridec
// batchCount: number of matrices in the batch
// Note: This is similar to GemmBatched but optimized for strided access patterns
func GemmStrided(c, a, b []float32, ldC, ldA, ldB, M, N, K int, alpha, beta float32, batchCount int, stridea, strideb, stridec int) {
	if batchCount == 0 || M == 0 || N == 0 || K == 0 {
		return
	}

	for k := 0; k < batchCount; k++ {
		// Get offsets for batch k
		offsetA := k * stridea
		offsetB := k * strideb
		offsetC := k * stridec

		// Call GEMM_NN for this batch
		Gemm_NN(c[offsetC:], a[offsetA:], b[offsetB:], ldC, ldA, ldB, M, N, K, alpha, beta)
	}
}

// GemvBatched computes batched matrix-vector multiply: y[k] = alpha*A[k]*x[k] + beta*y[k]
// This is BLAS batched GEMV operation
// Each batch k: y[k] = alpha*A[k]*x[k] + beta*y[k]
// A[k]: M × N matrix (row-major, ldA ≥ N), stored at offset k*strideA
// x[k]: N × 1 vector, stored at offset k*strideX
// y[k]: M × 1 vector, stored at offset k*strideY
// batchCount: number of operations in the batch
func GemvBatched(y, a, x []float32, ldA, M, N int, alpha, beta float32, batchCount int, strideA, strideX, strideY int) {
	if batchCount == 0 || M == 0 || N == 0 {
		return
	}

	for k := 0; k < batchCount; k++ {
		// Get offsets for batch k
		offsetA := k * strideA
		offsetX := k * strideX
		offsetY := k * strideY

		// Call GEMV_N for this batch
		Gemv_N(y[offsetY:], a[offsetA:], x[offsetX:], ldA, M, N, alpha, beta)
	}
}
