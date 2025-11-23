// Package dsp provides digital signal processing functions for EasyRobot.
package dsp

import "math"

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
// Optimized with BCE techniques: reslicing and for range loops
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
		angle := -2 * math.Pi / float64(length)
		wlen := complex64(complex(math.Cos(angle), math.Sin(angle)))

		halfLen := length / 2
		for i := 0; i < fftSize; i += length {
			w := complex64(1 + 0i)
			for j := range halfLen {
				idx1 := i + j
				idx2 := i + j + halfLen

				// Get complex values from packed format
				u := complex64(complex(data[idx1], data[idx1+fftSize]))
				v := complex64(complex(data[idx2], data[idx2+fftSize])) * w

				// Compute FFT butterfly
				sum := u + v
				diff := u - v

				// Store back in packed format
				data[idx1] = float32(real(sum))
				data[idx1+fftSize] = float32(imag(sum))
				data[idx2] = float32(real(diff))
				data[idx2+fftSize] = float32(imag(diff))

				w *= wlen
			}
		}
	}
}

// ifftReal performs in-place inverse FFT on packed complex data.
// Data layout: real parts in data[0:N], imaginary parts in data[N:2*N] where N = len(data)/2.
// The input must have even length (2*N) to accommodate both real and imaginary parts.
// Optimized with BCE techniques: reslicing and for range loops
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
	scale := 1.0 / float32(fftSize)
	for i := range fftSize {
		// Conjugate back (negate imag) and scale both parts
		imagIdx := i + fftSize
		data[imagIdx] = -data[imagIdx]
		data[i] *= scale
		data[imagIdx] *= scale
	}
}

// fftComplex performs in-place complex FFT using Cooley-Tukey algorithm
// Optimized with BCE techniques: reslicing and for range loops
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
		angle := -2 * math.Pi / float64(length)
		wlen := complex64(complex(math.Cos(angle), math.Sin(angle)))

		halfLen := length / 2
		for i := 0; i < n; i += length {
			w := complex64(1 + 0i)
			// BCE hint: reslice current segment
			segment := data[i : i+length]
			segment = segment[:length]
			for j := range halfLen {
				u := segment[j]
				v := segment[j+halfLen] * w
				segment[j] = u + v
				segment[j+halfLen] = u - v
				w *= wlen
			}
		}
	}
}

// ifftComplex performs in-place complex inverse FFT
// Optimized with BCE techniques: reslicing and for range loops
func ifftComplex(data []complex64) {
	n := len(data)
	if n <= 1 {
		return
	}

	// BCE hint: reslice to exact size
	data = data[:n]

	// Conjugate the data for IFFT
	for i := range n {
		data[i] = complex64(complex(real(data[i]), -imag(data[i])))
	}

	// Forward FFT on conjugated data
	fftComplex(data)

	// Conjugate back and scale
	scale := complex64(complex(1.0/float32(n), 0))
	for i := range n {
		data[i] = complex64(complex(real(data[i]), -imag(data[i]))) * scale
	}
}
