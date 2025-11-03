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

// Convolve2DAdd computes 2D convolution and add: dst += conv(mat, kernel)
// mat is NxM matrix (row-major)
// kernel is KxL matrix (row-major)
// dst is output matrix
// stride: stride for convolution
// transposed: if true, use transposed convolution
func Convolve2DAdd(dst, mat, kernel []float32, N, M, K, L int, stride int, transposed bool) {
	if N == 0 || M == 0 || K == 0 || L == 0 {
		return
	}

	if !transposed {
		// Forward 2D convolution
		dstHeight := (N - K) / stride
		if (N-K)%stride != 0 {
			dstHeight++
		}
		dstWidth := (M - L) / stride
		if (M-L)%stride != 0 {
			dstWidth++
		}

		pd := 0

		for b := 0; b < dstHeight; b++ {
			matRowStart := b * stride * M
			for a := 0; a < dstWidth; a++ {
				matStart := matRowStart + a*stride
				if matStart+K*M <= N*M && matStart+K <= N*M {
					// Compute dot product of KxL window
					acc := float32(0.0)
					for j := 0; j < L; j++ {
						matOffset := matStart + j*M
						kernelOffset := j * K
						for i := 0; i < K; i++ {
							acc += mat[matOffset+i] * kernel[kernelOffset+i]
						}
					}
					dst[pd] += acc
				}
				pd++
			}
		}
	} else {
		// Transposed 2D convolution (deconvolution)
		// This is a simplified version - full implementation would handle padding
		// Calculate output dimensions for transposed convolution
		dstWidth := M * stride

		for i := 0; i < N; i++ {
			for j := 0; j < M; j++ {
				dstRowStart := i * stride
				dstColStart := j * stride
				matIdx := i*M + j
				for kj := 0; kj < L; kj++ {
					for ki := 0; ki < K; ki++ {
						dstIdx := (dstRowStart+kj)*dstWidth + (dstColStart + ki)
						if dstIdx < len(dst) {
							kernelIdx := kj*K + ki
							dst[dstIdx] += mat[matIdx] * kernel[kernelIdx]
						}
					}
				}
			}
		}
	}
}
