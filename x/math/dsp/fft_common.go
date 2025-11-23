// Package dsp provides digital signal processing functions for EasyRobot.
package dsp

import (
	"github.com/chewxy/math32"
)

// Helper functions for FFT operations

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

// fftReal performs in-place FFT on packed complex data using Cooley-Tukey algorithm.
// Data layout: real parts in data[0:N], imaginary parts in data[N:2*N] where N = len(data)/2.
// The input must have even length (2*N) to accommodate both real and imaginary parts.
// Optimized with BCE techniques: reslicing and for range loops, math32 for float32 operations.
func fftReal(data []float32) {
	n := len(data)
	if n <= 2 {
		return
	}
	if n%2 != 0 {
		panic("fftReal: data length must be even (real and imaginary parts)")
	}

	fftSize := n / 2 // Actual FFT size (number of complex samples)

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
			w := complex64(1 + 0i)
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
				wReal := real(w)
				wImag := imag(w)
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
				w = complex64(complex(wRealNew, wImagNew))
			}
		}
	}
}

// ifftReal performs in-place inverse FFT on packed complex data.
// Data layout: real parts in data[0:N], imaginary parts in data[N:2*N] where N = len(data)/2.
// The input must have even length (2*N) to accommodate both real and imaginary parts.
// Optimized with BCE techniques: reslicing and for range loops, math32 for float32 operations.
func ifftReal(data []float32) {
	n := len(data)
	if n <= 2 {
		return
	}
	if n%2 != 0 {
		panic("ifftReal: data length must be even (real and imaginary parts)")
	}

	fftSize := n / 2 // Actual FFT size (number of complex samples)

	// BCE hint: reslice to exact size
	data = data[:n]

	// Conjugate the complex data (negate imaginary parts)
	for i := range fftSize {
		data[i+fftSize] = -data[i+fftSize]
	}

	// Forward FFT on conjugated data
	fftReal(data)

	// Conjugate back and scale by 1/N
	fftSizeF := float32(fftSize)
	scale := 1.0 / fftSizeF
	for i := range fftSize {
		// Conjugate back (negate imag) and scale both parts
		imagIdx := i + fftSize
		data[imagIdx] = -data[imagIdx]
		data[i] *= scale
		data[imagIdx] *= scale
	}
}

// fftComplex performs in-place complex FFT using Cooley-Tukey algorithm
// Optimized with BCE techniques: reslicing and for range loops, math32 for float32 operations.
func fftComplex(data []complex64) {
	n := len(data)
	if n <= 1 {
		return
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
			w := complex64(1 + 0i)
			// BCE hint: reslice current segment
			segment := data[i : i+length]
			segment = segment[:length]
			for j := range halfLen {
				u := segment[j]
				// Multiply v by w: (vReal + vImag*i) * (wReal + wImag*i)
				v := segment[j+halfLen]
				vReal := real(v)
				vImag := imag(v)
				wReal := real(w)
				wImag := imag(w)
				vRealNew := vReal*wReal - vImag*wImag
				vImagNew := vReal*wImag + vImag*wReal
				vNew := complex64(complex(vRealNew, vImagNew))

				segment[j] = u + vNew
				segment[j+halfLen] = u - vNew

				// Update twiddle factor: w *= wlen
				wRealNew := wReal*cosAngle - wImag*sinAngle
				wImagNew := wReal*sinAngle + wImag*cosAngle
				w = complex64(complex(wRealNew, wImagNew))
			}
		}
	}
}

// ifftComplex performs in-place complex inverse FFT
// Optimized with BCE techniques: reslicing and for range loops, math32 for float32 operations.
func ifftComplex(data []complex64) {
	n := len(data)
	if n <= 1 {
		return
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
	fftComplex(data)

	// Conjugate back and scale (no type casts in hot path)
	nF := float32(n)
	scale := 1.0 / nF
	for i := range n {
		realVal := real(data[i])
		imagVal := imag(data[i])
		data[i] = complex64(complex(realVal*scale, -imagVal*scale))
	}
}
