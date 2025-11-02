package primitive

import (
	"github.com/chewxy/math32"
)

// HadamardProduct computes element-wise product: dst[i] = a[i] * b[i]
func HadamardProduct(dst, a, b []float32, num int, strideA, strideB int) {
	if num == 0 {
		return
	}

	pa := 0
	pb := 0
	pd := 0

	for i := 0; i < num; i++ {
		dst[pd] = a[pa] * b[pb]
		pa += strideA
		pb += strideB
		pd++
	}
}

// HadamardProductAdd computes element-wise product and add: dst[i] += a[i] * b[i]
func HadamardProductAdd(dst, a, b []float32, num int, strideA, strideB int) {
	if num == 0 {
		return
	}

	pa := 0
	pb := 0
	pd := 0

	for i := 0; i < num; i++ {
		dst[pd] += a[pa] * b[pb]
		pa += strideA
		pb += strideB
		pd++
	}
}

// DotProduct computes dot product of two vectors
func DotProduct(a, b []float32, num int, strideA, strideB int) float32 {
	if num == 0 {
		return 0
	}

	acc := float32(0.0)
	pa := 0
	pb := 0

	for i := 0; i < num; i++ {
		acc += a[pa] * b[pb]
		pa += strideA
		pb += strideB
	}

	return acc
}

// DotProduct2D computes dot product of KxL submatrix
// a is NxM matrix, b is KxL matrix
// Computes sum of element-wise products of KxL window
func DotProduct2D(a, b []float32, N, M, K, L int) float32 {
	// a is NxM, accessed as a[i + j*N]
	// b is KxL, accessed as b[i + j*K]
	acc := float32(0.0)

	for j := 0; j < L; j++ {
		// Compute dot product of row j in both matrices
		pa := j * N // Start of row j in a
		pb := j * K // Start of row j in b

		for i := 0; i < K; i++ {
			acc += a[pa+i] * b[pb+i]
		}
	}

	return acc
}

// NormalizeVec normalizes vector in-place: dst[i] = dst[i] / ||dst||
func NormalizeVec(dst []float32, num int, stride int) {
	if num == 0 {
		return
	}

	sumSq := SqrSum(dst, num, stride)
	if sumSq == 0 {
		return
	}

	norm := math32.Sqrt(sumSq)
	MulArrInPlace(dst, 1.0/norm, num)
}

// OuterProduct computes outer product: dst = u * v^T
// u is vector of size N, v is vector of size M
// dst is NxM matrix (row-major)
// If bias is true, adds extra column with u values
func OuterProduct(dst, u, v []float32, N, M int, bias bool) {
	if N == 0 || M == 0 {
		return
	}

	pd := 0

	for i := 0; i < N; i++ {
		ui := u[i]
		for j := 0; j < M; j++ {
			dst[pd] = ui * v[j]
			pd++
		}
		if bias {
			dst[pd] = ui
			pd++
		}
	}
}

// OuterProductConst computes scaled outer product: dst = alpha * u * v^T
func OuterProductConst(dst, u, v []float32, N, M int, alpha float32, bias bool) {
	if N == 0 || M == 0 {
		return
	}

	pd := 0

	for i := 0; i < N; i++ {
		ui := u[i] * alpha
		for j := 0; j < M; j++ {
			dst[pd] = ui * v[j]
			pd++
		}
		if bias {
			dst[pd] = ui
			pd++
		}
	}
}

// OuterProductAddConst computes scaled outer product and add: dst += alpha * u * v^T
func OuterProductAddConst(dst, u, v []float32, N, M int, alpha float32, bias bool) {
	if N == 0 || M == 0 {
		return
	}

	pd := 0

	for i := 0; i < N; i++ {
		ui := u[i] * alpha
		for j := 0; j < M; j++ {
			dst[pd] += ui * v[j]
			pd++
		}
		if bias {
			dst[pd] += ui
			pd++
		}
	}
}

