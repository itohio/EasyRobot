package fp32

// HadamardProductAdd computes element-wise product and add: dst[i] += a[i] * b[i]
// Element-wise multiplication and addition for tensor operations
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

// DotProduct2D computes dot product of KxL submatrix
// a is NxM matrix, b is KxL matrix
// Computes sum of element-wise products of KxL window
// Specialized function for 2D matrix dot product (not BLAS)
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

// NormalizeVecInPlace normalizes vector in-place: dst[i] = dst[i] / ||dst||
// Uses Nrm2 from level1.go for norm calculation
func NormalizeVecInPlace(dst []float32, num int, stride int) {
	if num == 0 {
		return
	}

	norm := Nrm2(dst, stride, num)
	if norm == 0 {
		return
	}

	// Use Scal for in-place scaling
	Scal(dst, stride, num, 1.0/norm)
}

// SumArrInPlace computes dst[i] += c for all i (in-place)
// Utility function for scalar addition
// Note: Not directly replaceable with BLAS operations efficiently
func SumArrInPlace(dst []float32, c float32, num int) {
	if num == 0 {
		return
	}

	for i := 0; i < num; i++ {
		dst[i] += c
	}
}

// HadamardProduct computes element-wise product: dst[i] = a[i] * b[i]
// Element-wise multiplication for tensor operations (dst-based version)
func HadamardProduct(dst, a, b []float32, num int, strideDst, strideA, strideB int) {
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
		pd += strideDst
	}
}

// NormalizeVec normalizes vector: dst[i] = src[i] / ||src||
// Uses Nrm2 from level1.go for norm calculation (dst-based version)
func NormalizeVec(dst, src []float32, num int, strideDst, strideSrc int) {
	if num == 0 {
		return
	}

	norm := Nrm2(src, strideSrc, num)
	if norm == 0 {
		// If norm is zero, copy src to dst
		Copy(dst, src, strideDst, strideSrc, num)
		return
	}

	// Copy src to dst first, then scale
	Copy(dst, src, strideDst, strideSrc, num)
	Scal(dst, strideDst, num, 1.0/norm)
}

// SumArrScalar computes dst[i] = src[i] + c (dst-based version)
// Utility function for scalar addition
func SumArrScalar(dst, src []float32, c float32, num int, strideDst, strideSrc int) {
	if num == 0 {
		return
	}

	ps := 0
	pd := 0

	for i := 0; i < num; i++ {
		dst[pd] = src[ps] + c
		ps += strideSrc
		pd += strideDst
	}
}
