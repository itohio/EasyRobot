package fp32

// Convolve1DAdd computes 1D convolution and add: dst += conv(vec, kernel).
// The destination slice is not cleared; callers must zero dst themselves when
// they need only the convolution result without additional accumulation.
// vec is vector of size N
// kernel is vector of size M
// dst is output vector
// stride: stride for convolution
// transposed: if true, use transposed convolution
func Convolve1DAdd(dst, vec, kernel []float32, N, M int, stride int, transposed bool) {
	if N == 0 || M == 0 {
		return
	}

	if !transposed {
		// Forward convolution: dst[j] += sum(vec[i + j*stride] * kernel[i])
		dstSize := (N - M) / stride
		if (N-M)%stride != 0 {
			dstSize++
		}

		pv := 0
		pd := 0

		for i := 0; i < dstSize && pv+M <= N; i++ {
			acc := Dot(vec[pv:], kernel, 1, 1, M)
			dst[pd] += acc
			pv += stride
			pd++
		}
	} else {
		// Transposed convolution (deconvolution)
		for i := 0; i < N; i++ {
			pd := i * stride
			for j := 0; j < M && pd+j < len(dst); j++ {
				dst[pd+j] += vec[i] * kernel[j]
			}
		}
	}
}

// Convolve1D computes 1D convolution: dst = conv(vec, kernel)
// vec is vector of size N
// kernel is vector of size M
// dst is output vector (will be overwritten)
// stride: stride for convolution
// transposed: if true, use transposed convolution
func Convolve1D(dst, vec, kernel []float32, N, M int, stride int, transposed bool) {
	if N == 0 || M == 0 {
		return
	}

	// Calculate output size and initialize dst to zero
	var dstSize int
	if !transposed {
		// Forward convolution: dstSize = (N - M) / stride + 1
		dstSize = (N - M) / stride
		if (N-M)%stride != 0 {
			dstSize++
		}
	} else {
		// Transposed convolution: output size is approximately N * stride
		// For simplicity, we'll use a conservative estimate
		dstSize = N * stride
		if dstSize > len(dst) {
			dstSize = len(dst)
		}
	}

	// Initialize dst to zero
	for i := 0; i < dstSize && i < len(dst); i++ {
		dst[i] = 0
	}

	// Use Convolve1DAdd to compute the convolution
	Convolve1DAdd(dst, vec, kernel, N, M, stride, transposed)
}
