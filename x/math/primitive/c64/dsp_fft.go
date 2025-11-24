package c64

import (
	"github.com/chewxy/math32"
)

// isPowerOfTwo checks if a number is a power of two
func isPowerOfTwo(n int) bool {
	return n > 0 && (n&(n-1)) == 0
}

// nextPowerOfTwo returns the next power of two >= n
func nextPowerOfTwo(n int) int {
	if n == 0 {
		return 1
	}
	n--
	n |= n >> 1
	n |= n >> 2
	n |= n >> 4
	n |= n >> 8
	n |= n >> 16
	return n + 1
}

// FFT performs in-place complex FFT using Cooley-Tukey algorithm.
// Optimized for power-of-2 sizes. For non-power-of-2, use nextPowerOfTwo to pad.
// Optimized with BCE techniques: reslicing and for range loops, math32 for float32 operations.
func FFT(data []complex64) {
	n := len(data)
	if n <= 1 {
		return
	}

	// Optimize for power-of-2
	if !isPowerOfTwo(n) {
		panic("FFT: size must be power of 2, use nextPowerOfTwo to pad")
	}

	// BCE hint: reslice to exact size
	data = data[:n]

	// Bit-reversal permutation
	j := 0
	for i := range n {
		if i < j {
			data[i], data[j] = data[j], data[i]
		}
		m := n >> 1
		for ; m >= 1 && j >= m; m >>= 1 {
			j -= m
		}
		j += m
	}

	// Cooley-Tukey FFT
	for length := 2; length <= n; length <<= 1 {
		lengthF := float32(length)
		angle := -2 * math32.Pi / lengthF
		cosAngle := math32.Cos(angle)
		sinAngle := math32.Sin(angle)

		halfLen := length / 2
		for i := 0; i < n; i += length {
			wReal := float32(1.0)
			wImag := float32(0.0)
			// BCE hint: reslice current segment
			segment := data[i : i+length]
			segment = segment[:length]
			for j := range halfLen {
				u := segment[j]
				uReal := real(u)
				uImag := imag(u)
				// Multiply v by w: (vReal + vImag*i) * (wReal + wImag*i)
				v := segment[j+halfLen]
				vReal := real(v)
				vImag := imag(v)
				vRealNew := vReal*wReal - vImag*wImag
				vImagNew := vReal*wImag + vImag*wReal

				segment[j] = complex64(complex(uReal+vRealNew, uImag+vImagNew))
				segment[j+halfLen] = complex64(complex(uReal-vRealNew, uImag-vImagNew))

				// Update twiddle factor: w *= wlen
				wRealNew := wReal*cosAngle - wImag*sinAngle
				wImagNew := wReal*sinAngle + wImag*cosAngle
				wReal = wRealNew
				wImag = wImagNew
			}
		}
	}
}

// IFFT performs in-place complex inverse FFT.
// Optimized for power-of-2 sizes. For non-power-of-2, use nextPowerOfTwo to pad.
// Optimized with BCE techniques: reslicing and for range loops, math32 for float32 operations.
func IFFT(data []complex64) {
	n := len(data)
	if n <= 1 {
		return
	}

	// Optimize for power-of-2
	if !isPowerOfTwo(n) {
		panic("IFFT: size must be power of 2, use nextPowerOfTwo to pad")
	}

	// BCE hint: reslice to exact size
	data = data[:n]

	// Conjugate the data for IFFT (no type casts in hot path)
	for i := range n {
		realVal := real(data[i])
		imagVal := imag(data[i])
		data[i] = complex64(complex(realVal, -imagVal))
	}

	// Forward FFT on conjugated data
	FFT(data)

	// Conjugate back and scale by 1/N (no type casts in hot path)
	nF := float32(n)
	scale := float32(1.0) / (nF * nF)
	for i := range n {
		realVal := real(data[i])
		imagVal := imag(data[i])
		// Scale and conjugate: result = (1/N) * conj(FFT(conj(x)))
		scaledReal := realVal * scale
		scaledImag := -imagVal * scale
		data[i] = complex64(complex(scaledReal, scaledImag))
	}
}

// ComputeFFTLUT returns the required buffer size for FFT LUT of given size.
// Returns the total number of complex64 elements needed (totalFactors).
// For FFT size N, the buffer size is (N-1) complex64 elements.
// Layout: [complex(cos0,sin0), complex(cos1,sin1), ..., complex(cosM-1,sinM-1)] where M = totalFactors
func ComputeFFTLUT(fftSize int) int {
	if !isPowerOfTwo(fftSize) {
		panic("ComputeFFTLUT: fftSize must be power of 2")
	}

	// Compute all twiddle factors needed for FFT
	// For each stage length (2, 4, 8, ..., fftSize), we need length/2 twiddle factors
	totalFactors := 0
	for length := 2; length <= fftSize; length <<= 1 {
		totalFactors += length / 2
	}

	// Return total number of complex64 elements
	return totalFactors
}

// FillFFTLUT fills the LUT buffer with precomputed twiddle factors.
// dst must be at least ComputeFFTLUT(fftSize) in size.
// Returns the number of twiddle factors (totalFactors).
// Layout: [complex(cos0,sin0), complex(cos1,sin1), ..., complex(cosM-1,sinM-1)] where M = totalFactors
func FillFFTLUT(dst []complex64, fftSize int) int {
	if !isPowerOfTwo(fftSize) {
		panic("FillFFTLUT: fftSize must be power of 2")
	}

	// Compute all twiddle factors needed for FFT
	totalFactors := 0
	for length := 2; length <= fftSize; length <<= 1 {
		totalFactors += length / 2
	}

	if len(dst) < totalFactors {
		panic("FillFFTLUT: destination buffer too small")
	}

	idx := 0
	for length := 2; length <= fftSize; length <<= 1 {
		lengthF := float32(length)
		angle := -2 * math32.Pi / lengthF
		for j := range length / 2 {
			angleJ := angle * float32(j)
			dst[idx] = complex64(complex(math32.Cos(angleJ), math32.Sin(angleJ)))
			idx++
		}
	}

	return totalFactors
}

// FFTWithLUT performs FFT using precomputed LUT.
// data: complex data of length N
// lut: precomputed twiddle factors from FillFFTLUT
//
//	Layout: [complex(cos0,sin0), complex(cos1,sin1), ..., complex(cosM-1,sinM-1)] where M = totalFactors
//
// totalFactors: number of factors returned by FillFFTLUT
func FFTWithLUT(data []complex64, lut []complex64, totalFactors int) {
	n := len(data)
	if n <= 1 {
		return
	}

	if !isPowerOfTwo(n) {
		panic("FFTWithLUT: size must be power of 2")
	}

	// BCE hint: reslice to exact size
	data = data[:n]

	// Bit-reversal permutation
	j := 0
	for i := range n {
		if i < j {
			data[i], data[j] = data[j], data[i]
		}
		m := n >> 1
		for ; m >= 1 && j >= m; m >>= 1 {
			j -= m
		}
		j += m
	}

	// Cooley-Tukey FFT using precomputed LUT
	lutIdx := 0
	for length := 2; length <= n; length <<= 1 {
		halfLen := length / 2
		for i := 0; i < n; i += length {
			segment := data[i : i+length]
			segment = segment[:length]
			for j := range halfLen {
				// Get twiddle factor directly from LUT and multiply v by w
				w := lut[lutIdx+j]
				u := segment[j]
				v := segment[j+halfLen]

				// Multiply v by w using complex multiplication
				vw := v * w

				// Compute FFT butterfly: sum = u + v*w, diff = u - v*w
				segment[j] = u + vw
				segment[j+halfLen] = u - vw
			}
		}
		lutIdx += halfLen
	}
}

// IFFTWithLUT performs inverse FFT using precomputed LUT.
// data: complex data of length N
// lut: precomputed twiddle factors from FillFFTLUT
//
//	Layout: [complex(cos0,sin0), complex(cos1,sin1), ..., complex(cosM-1,sinM-1)] where M = totalFactors
//
// totalFactors: number of factors returned by FillFFTLUT
func IFFTWithLUT(data []complex64, lut []complex64, totalFactors int) {
	n := len(data)
	if n <= 1 {
		return
	}

	// Optimize for power-of-2
	if !isPowerOfTwo(n) {
		panic("IFFTWithLUT: size must be power of 2, use nextPowerOfTwo to pad")
	}

	// BCE hint: reslice to exact size
	data = data[:n]

	// Conjugate the data for IFFT (no type casts in hot path)
	for i := range n {
		realVal := real(data[i])
		imagVal := imag(data[i])
		data[i] = complex64(complex(realVal, -imagVal))
	}

	// Forward FFT on conjugated data using LUT
	FFTWithLUT(data, lut, totalFactors)

	// Conjugate back and scale by 1/N^2
	// Note: Since FFT accumulates by N^2 when applied to frequency domain data in IFFT,
	// we need to scale by 1/N^2 to get the correct result (1/N for IFFT * 1/N for FFT accumulation)
	nF := float32(n)
	scale := float32(1.0) / (nF * nF)
	for i := range n {
		realVal := real(data[i])
		imagVal := imag(data[i])
		// Scale and conjugate: result = (1/N) * conj(FFT(conj(x)))
		scaledReal := realVal * scale
		scaledImag := -imagVal * scale
		data[i] = complex64(complex(scaledReal, scaledImag))
	}
}
