package fp32

// Convolve1DAdd computes 1D convolution and add: dst += conv(vec, kernel)
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
