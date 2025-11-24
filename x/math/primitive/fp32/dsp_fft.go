package fp32

import (
	"github.com/chewxy/math32"
)

// isPowerOfTwo checks if a number is a power of two
func isPowerOfTwo(n int) bool {
	return n > 0 && (n&(n-1)) == 0
}

// NextPowerOfTwo returns the next power of two >= n
func NextPowerOfTwo(n int) int {
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

// FFT performs in-place FFT on packed complex data using Cooley-Tukey algorithm.
// Data layout: real parts in data[0:N], imaginary parts in data[N:2*N] where N = len(data)/2.
// The input must have even length (2*N) to accommodate both real and imaginary parts.
// Optimized for power-of-2 sizes. For non-power-of-2, use nextPowerOfTwo to pad.
// Optimized with BCE techniques: reslicing and for range loops, math32 for float32 operations.
func FFT(data []float32) {
	n := len(data)
	if n <= 2 {
		return
	}
	if n%2 != 0 {
		panic("FFT: data length must be even (real and imaginary parts)")
	}

	fftSize := n / 2 // Actual FFT size (number of complex samples)

	// Optimize for power-of-2
	if !isPowerOfTwo(fftSize) {
		panic("FFT: size must be power of 2, use nextPowerOfTwo to pad")
	}

	// BCE hint: reslice to exact size
	data = data[:n]

	// Bit-reversal permutation on complex pairs
	j := 0
	for i := range fftSize {
		if i < j {
			// Swap real parts
			data[i], data[j] = data[j], data[i]
			// Swap imaginary parts
			data[i+fftSize], data[j+fftSize] = data[j+fftSize], data[i+fftSize]
		}
		m := fftSize >> 1
		for ; m >= 1 && j >= m; m >>= 1 {
			j -= m
		}
		j += m
	}

	// Cooley-Tukey FFT on complex data
	for length := 2; length <= fftSize; length <<= 1 {
		lengthF := float32(length)
		angle := -2 * math32.Pi / lengthF
		cosAngle := math32.Cos(angle)
		sinAngle := math32.Sin(angle)

		halfLen := length / 2
		for i := 0; i < fftSize; i += length {
			wReal := float32(1.0)
			wImag := float32(0.0)
			for j := range halfLen {
				idx1 := i + j
				idx2 := i + j + halfLen
				idx1Imag := idx1 + fftSize
				idx2Imag := idx2 + fftSize

				// Get complex values from packed format (no type casts in hot path)
				uReal := data[idx1]
				uImag := data[idx1Imag]
				vReal := data[idx2]
				vImag := data[idx2Imag]

				// Multiply v by w: (vReal + vImag*i) * (wReal + wImag*i)
				vRealNew := vReal*wReal - vImag*wImag
				vImagNew := vReal*wImag + vImag*wReal

				// Compute FFT butterfly: sum = u + v, diff = u - v
				sumReal := uReal + vRealNew
				sumImag := uImag + vImagNew
				diffReal := uReal - vRealNew
				diffImag := uImag - vImagNew

				// Store back in packed format (no type casts)
				data[idx1] = sumReal
				data[idx1Imag] = sumImag
				data[idx2] = diffReal
				data[idx2Imag] = diffImag

				// Update twiddle factor: w *= wlen
				wRealNew := wReal*cosAngle - wImag*sinAngle
				wImagNew := wReal*sinAngle + wImag*cosAngle
				wReal = wRealNew
				wImag = wImagNew
			}
		}
	}
}

// IFFT performs in-place inverse FFT on packed complex data.
// Data layout: real parts in data[0:N], imaginary parts in data[N:2*N] where N = len(data)/2.
// The input must have even length (2*N) to accommodate both real and imaginary parts.
// Optimized for power-of-2 sizes. For non-power-of-2, use nextPowerOfTwo to pad.
// Optimized with BCE techniques: reslicing and for range loops, math32 for float32 operations.
func IFFT(data []float32) {
	n := len(data)
	if n <= 2 {
		return
	}
	if n%2 != 0 {
		panic("IFFT: data length must be even (real and imaginary parts)")
	}

	fftSize := n / 2 // Actual FFT size (number of complex samples)

	// Optimize for power-of-2
	if !isPowerOfTwo(fftSize) {
		panic("IFFT: size must be power of 2, use nextPowerOfTwo to pad")
	}

	// Conjugate the complex data (negate imaginary parts)
	for i := range fftSize {
		data[i+fftSize] = -data[i+fftSize]
	}

	// Forward FFT on conjugated data
	FFT(data)

	// Conjugate back and scale by 1/N
	fftSizeF := float32(fftSize)
	scale := float32(1.0) / fftSizeF
	for i := 0; i < fftSize; i++ {
		// Conjugate back (negate imag) and scale both parts
		imagIdx := i + fftSize
		data[imagIdx] = -data[imagIdx]
		data[i] *= scale
		data[imagIdx] *= scale
	}
}

// FFTLUT holds precomputed twiddle factors for FFT
type FFTLUT struct {
	size   int
	cosLUT []float32
	sinLUT []float32
}

// ComputeFFTLUT returns the required buffer size for FFT LUT of given size.
// Returns the total number of float32 elements needed (2 * totalFactors).
// For FFT size N, the buffer size is 2 * (N-1) elements.
// Layout: [cos0, cos1, ..., cosM-1, sin0, sin1, ..., sinM-1] where M = totalFactors
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

	// Return total buffer size (cos + sin for each factor)
	return totalFactors * 2
}

// FillFFTLUT fills the LUT buffer with precomputed twiddle factors.
// dst must be at least ComputeFFTLUT(fftSize) in size.
// Returns the number of twiddle factors (totalFactors).
// Layout: [cos0, cos1, ..., cosM-1, sin0, sin1, ..., sinM-1] where M = totalFactors
func FillFFTLUT(dst []float32, fftSize int) int {
	if !isPowerOfTwo(fftSize) {
		panic("FillFFTLUT: fftSize must be power of 2")
	}

	// Compute all twiddle factors needed for FFT
	totalFactors := 0
	for length := 2; length <= fftSize; length <<= 1 {
		totalFactors += length / 2
	}

	if len(dst) < totalFactors*2 {
		panic("FillFFTLUT: destination buffer too small")
	}

	idx := 0
	for length := 2; length <= fftSize; length <<= 1 {
		lengthF := float32(length)
		angle := -2 * math32.Pi / lengthF
		for j := range length / 2 {
			angleJ := angle * float32(j)
			dst[idx] = math32.Cos(angleJ)
			dst[idx+totalFactors] = math32.Sin(angleJ)
			idx++
		}
	}

	return totalFactors
}

// FFTWithLUT performs FFT using precomputed LUT.
// data: packed complex data [real, imag] of length 2*N
// lut: precomputed twiddle factors from FillFFTLUT
//
//	Layout: [cos0, cos1, ..., cosM-1, sin0, sin1, ..., sinM-1] where M = totalFactors
//
// totalFactors: number of factors returned by FillFFTLUT
func FFTWithLUT(data []float32, lut []float32, totalFactors int) {
	n := len(data)
	if n <= 2 {
		return
	}
	if n%2 != 0 {
		panic("FFTWithLUT: data length must be even")
	}

	fftSize := n / 2
	if !isPowerOfTwo(fftSize) {
		panic("FFTWithLUT: size must be power of 2")
	}

	// BCE hint: reslice to exact size
	data = data[:n]

	// Bit-reversal permutation on complex pairs
	j := 0
	for i := range fftSize {
		if i < j {
			// Swap real parts
			data[i], data[j] = data[j], data[i]
			// Swap imaginary parts
			data[i+fftSize], data[j+fftSize] = data[j+fftSize], data[i+fftSize]
		}
		m := fftSize >> 1
		for ; m >= 1 && j >= m; m >>= 1 {
			j -= m
		}
		j += m
	}

	// Cooley-Tukey FFT using precomputed LUT
	lutIdx := 0
	for length := 2; length <= fftSize; length <<= 1 {
		halfLen := length / 2
		for i := 0; i < fftSize; i += length {
			for j := range halfLen {
				idx1 := i + j
				idx2 := i + j + halfLen
				idx1Imag := idx1 + fftSize
				idx2Imag := idx2 + fftSize

				// Get twiddle factor directly from LUT
				cosVal := lut[lutIdx+j]
				sinVal := lut[lutIdx+j+totalFactors]

				// Get complex values from packed format
				uReal := data[idx1]
				uImag := data[idx1Imag]
				vReal := data[idx2]
				vImag := data[idx2Imag]

				// Multiply v by w: (vReal + vImag*i) * (cosVal + sinVal*i)
				vRealNew := vReal*cosVal - vImag*sinVal
				vImagNew := vReal*sinVal + vImag*cosVal

				// Compute FFT butterfly: sum = u + v, diff = u - v
				data[idx1] = uReal + vRealNew
				data[idx1Imag] = uImag + vImagNew
				data[idx2] = uReal - vRealNew
				data[idx2Imag] = uImag - vImagNew
			}
		}
		lutIdx += halfLen
	}
}

// IFFTWithLUT performs inverse FFT using precomputed LUT.
// data: packed complex data [real, imag] of length 2*N
// lut: precomputed twiddle factors from FillFFTLUT
//
//	Layout: [cos0, cos1, ..., cosM-1, sin0, sin1, ..., sinM-1] where M = totalFactors
//
// totalFactors: number of factors returned by FillFFTLUT
func IFFTWithLUT(data []float32, lut []float32, totalFactors int) {
	n := len(data)
	if n <= 2 {
		return
	}
	if n%2 != 0 {
		panic("IFFTWithLUT: data length must be even (real and imaginary parts)")
	}

	fftSize := n / 2 // Actual FFT size (number of complex samples)

	// Optimize for power-of-2
	if !isPowerOfTwo(fftSize) {
		panic("IFFTWithLUT: size must be power of 2, use nextPowerOfTwo to pad")
	}

	// Conjugate the complex data (negate imaginary parts)
	for i := range fftSize {
		data[i+fftSize] = -data[i+fftSize]
	}

	// Forward FFT on conjugated data using LUT
	FFTWithLUT(data, lut, totalFactors)

	// Conjugate back and scale by 1/N
	fftSizeF := float32(fftSize)
	scale := float32(1.0) / fftSizeF
	for i := 0; i < fftSize; i++ {
		// Conjugate back (negate imag) and scale both parts
		imagIdx := i + fftSize
		data[imagIdx] = -data[imagIdx]
		data[i] *= scale
		data[imagIdx] *= scale
	}
}
