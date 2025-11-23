package dsp

import (
	"fmt"
	"testing"

	"github.com/chewxy/math32"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestIsPowerOfTwo(t *testing.T) {
	tests := []struct {
		name     string
		n        int
		expected bool
	}{
		{"zero", 0, false},
		{"one", 1, true},
		{"two", 2, true},
		{"three", 3, false},
		{"four", 4, true},
		{"five", 5, false},
		{"eight", 8, true},
		{"sixteen", 16, true},
		{"thirty_two", 32, true},
		{"sixty_four", 64, true},
		{"one_hundred", 100, false},
		{"two_fifty_six", 256, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := isPowerOfTwo(tt.n)
			assert.Equal(t, tt.expected, result, "isPowerOfTwo(%d)", tt.n)
		})
	}
}

func TestNextPowerOfTwo(t *testing.T) {
	tests := []struct {
		name     string
		n        int
		expected int
	}{
		{"zero", 0, 1},
		{"one", 1, 1},
		{"two", 2, 2},
		{"three", 3, 4},
		{"four", 4, 4},
		{"five", 5, 8},
		{"seven", 7, 8},
		{"eight", 8, 8},
		{"nine", 9, 16},
		{"fifteen", 15, 16},
		{"sixteen", 16, 16},
		{"seventeen", 17, 32},
		{"one_hundred", 100, 128},
		{"two_fifty_five", 255, 256},
		{"two_fifty_six", 256, 256},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := nextPowerOfTwo(tt.n)
			assert.Equal(t, tt.expected, result, "nextPowerOfTwo(%d)", tt.n)
		})
	}
}

func TestFFTReal_Impulse(t *testing.T) {
	// Test FFT of impulse signal [1, 0, 0, 0]
	// Expected: all frequency components should be 1 (DC component)
	data := make([]float32, 8) // 4 complex samples = 8 float32
	data[0] = 1.0              // Real part of first sample
	// Imaginary parts are already zero

	fftReal(data)

	// DC component (index 0) should be 1
	assert.InDelta(t, 1.0, data[0], 1e-5, "DC component real part")
	assert.InDelta(t, 0.0, data[4], 1e-5, "DC component imaginary part")
}

func TestFFTReal_Constant(t *testing.T) {
	// Test FFT of constant signal [1, 1, 1, 1]
	// Expected: DC component = 4, all others = 0
	data := make([]float32, 8) // 4 complex samples
	for i := range 4 {
		data[i] = 1.0
	}

	fftReal(data)

	// DC component should be 4
	assert.InDelta(t, 4.0, data[0], 1e-5, "DC component")
	// Other components should be approximately zero
	for i := 1; i < 4; i++ {
		assert.InDelta(t, 0.0, data[i], 1e-4, "Frequency component %d real", i)
		assert.InDelta(t, 0.0, data[i+4], 1e-4, "Frequency component %d imag", i)
	}
}

func TestIFFTReal_Roundtrip(t *testing.T) {
	// Test forward FFT then inverse FFT should recover original (scaled)
	original := []float32{1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0} // 4 real samples + 4 imag (zero)
	data := make([]float32, 8)
	copy(data, original)

	fftReal(data)
	ifftReal(data)

	// IFFT scales by 1/N, so result should be original / 4
	scale := float32(1.0 / 4.0)
	for i := range 4 {
		expected := original[i] * scale
		assert.InDelta(t, expected, data[i], 1e-4, "Roundtrip index %d", i)
	}
}

func TestFFTReal_IFFTReal_Identity(t *testing.T) {
	// Test that FFT then IFFT gives identity (scaled)
	sizes := []int{4, 8, 16, 32}
	for _, size := range sizes {
		t.Run(fmt.Sprintf("size_%d", size), func(t *testing.T) {
			data := make([]float32, 2*size)
			// Fill with test pattern
			for i := range size {
				data[i] = float32(i + 1)
			}

			original := make([]float32, 2*size)
			copy(original, data)

			fftReal(data)
			ifftReal(data)

			scale := 1.0 / float32(size)
			for i := range size {
				expected := original[i] * scale
				assert.InDelta(t, expected, data[i], 1e-4, "Index %d", i)
			}
		})
	}
}

func TestFFTReal_SineWave(t *testing.T) {
	// Test FFT of sine wave - should have peak at corresponding frequency
	size := 16
	data := make([]float32, 2*size)

	// Generate sine wave: sin(2*pi*k/N) for k=0..N-1
	// Using k=2 (second frequency bin)
	k := 2
	for i := range size {
		angle := 2 * math32.Pi * float32(k) * float32(i) / float32(size)
		data[i] = math32.Sin(angle)
	}

	fftReal(data)

	// Should have peak at frequency bin k (and N-k due to symmetry)
	// Check that bin k has significant magnitude
	magK := math32.Sqrt(data[k]*data[k] + data[k+size]*data[k+size])
	assert.Greater(t, magK, float32(0.1), "Magnitude at frequency bin %d should be significant", k)
}

func TestFFTComplex_Impulse(t *testing.T) {
	// Test FFT of complex impulse [1+0i, 0+0i, 0+0i, 0+0i]
	data := make([]complex64, 4)
	data[0] = complex64(1 + 0i)

	fftComplex(data)

	// All frequency components should be 1
	for i := range 4 {
		assert.InDelta(t, 1.0, real(data[i]), 1e-5, "Real part at index %d", i)
		assert.InDelta(t, 0.0, imag(data[i]), 1e-5, "Imag part at index %d", i)
	}
}

func TestFFTComplex_Constant(t *testing.T) {
	// Test FFT of constant complex signal
	data := make([]complex64, 4)
	for i := range 4 {
		data[i] = complex64(1 + 0i)
	}

	fftComplex(data)

	// DC component should be 4, others should be 0
	assert.InDelta(t, 4.0, real(data[0]), 1e-5, "DC component")
	assert.InDelta(t, 0.0, imag(data[0]), 1e-5, "DC component imag")
	for i := 1; i < 4; i++ {
		assert.InDelta(t, 0.0, real(data[i]), 1e-4, "Frequency %d real", i)
		assert.InDelta(t, 0.0, imag(data[i]), 1e-4, "Frequency %d imag", i)
	}
}

func TestIFFTComplex_Roundtrip(t *testing.T) {
	// Test forward FFT then inverse FFT should recover original (scaled)
	original := make([]complex64, 4)
	for i := range 4 {
		original[i] = complex64(complex(float32(i+1), float32(i+1)*0.5))
	}

	data := make([]complex64, 4)
	copy(data, original)

	fftComplex(data)
	ifftComplex(data)

	// IFFT scales by 1/N, so result should be original / 4
	scale := float32(1.0 / 4.0)
	for i := range 4 {
		expectedReal := real(original[i]) * scale
		expectedImag := imag(original[i]) * scale
		assert.InDelta(t, expectedReal, real(data[i]), 1e-4, "Roundtrip index %d real", i)
		assert.InDelta(t, expectedImag, imag(data[i]), 1e-4, "Roundtrip index %d imag", i)
	}
}

func TestFFTComplex_IFFTComplex_Identity(t *testing.T) {
	// Test that FFT then IFFT gives identity (scaled) for various sizes
	sizes := []int{2, 4, 8, 16, 32}
	for _, size := range sizes {
		t.Run(fmt.Sprintf("size_%d", size), func(t *testing.T) {
			data := make([]complex64, size)
			// Fill with test pattern
			for i := range size {
				data[i] = complex64(complex(float32(i+1), float32(i+1)*0.3))
			}

			original := make([]complex64, size)
			copy(original, data)

			fftComplex(data)
			ifftComplex(data)

			scale := 1.0 / float32(size)
			for i := range size {
				expectedReal := real(original[i]) * scale
				expectedImag := imag(original[i]) * scale
				assert.InDelta(t, expectedReal, real(data[i]), 1e-4, "Index %d real", i)
				assert.InDelta(t, expectedImag, imag(data[i]), 1e-4, "Index %d imag", i)
			}
		})
	}
}

func TestFFTReal_ErrorConditions(t *testing.T) {
	// Test panic for odd length
	t.Run("odd_length", func(t *testing.T) {
		data := make([]float32, 5) // Odd length
		require.Panics(t, func() {
			fftReal(data)
		}, "Should panic for odd length")
	})

	// Test small sizes
	t.Run("size_0", func(t *testing.T) {
		data := make([]float32, 0)
		require.NotPanics(t, func() {
			fftReal(data)
		}, "Should handle size 0")
	})

	t.Run("size_2", func(t *testing.T) {
		data := []float32{1.0, 0.0}
		require.NotPanics(t, func() {
			fftReal(data)
		}, "Should handle size 2")
	})
}

func TestIFFTReal_ErrorConditions(t *testing.T) {
	// Test panic for odd length
	t.Run("odd_length", func(t *testing.T) {
		data := make([]float32, 5) // Odd length
		require.Panics(t, func() {
			ifftReal(data)
		}, "Should panic for odd length")
	})
}

func TestFFTComplex_ErrorConditions(t *testing.T) {
	// Test small sizes
	t.Run("size_0", func(t *testing.T) {
		data := make([]complex64, 0)
		require.NotPanics(t, func() {
			fftComplex(data)
		}, "Should handle size 0")
	})

	t.Run("size_1", func(t *testing.T) {
		data := []complex64{1 + 0i}
		require.NotPanics(t, func() {
			fftComplex(data)
		}, "Should handle size 1")
	})
}

func TestFFTReal_LargeSize(t *testing.T) {
	// Test with larger size to ensure performance
	size := 64
	data := make([]float32, 2*size)
	for i := range size {
		data[i] = float32(i)
	}

	original := make([]float32, 2*size)
	copy(original, data)

	fftReal(data)
	ifftReal(data)

	scale := 1.0 / float32(size)
	for i := range size {
		expected := original[i] * scale
		assert.InDelta(t, expected, data[i], 1e-3, "Large size index %d", i)
	}
}

func TestFFTComplex_LargeSize(t *testing.T) {
	// Test with larger size
	size := 64
	data := make([]complex64, size)
	for i := range size {
		data[i] = complex64(complex(float32(i), float32(i)*0.5))
	}

	original := make([]complex64, size)
	copy(original, data)

	fftComplex(data)
	ifftComplex(data)

	scale := 1.0 / float32(size)
	for i := range size {
		expectedReal := real(original[i]) * scale
		expectedImag := imag(original[i]) * scale
		assert.InDelta(t, expectedReal, real(data[i]), 1e-3, "Large size index %d real", i)
		assert.InDelta(t, expectedImag, imag(data[i]), 1e-3, "Large size index %d imag", i)
	}
}

func TestFFTReal_HermitianSymmetry(t *testing.T) {
	// For real input, FFT output should have Hermitian symmetry
	// X[k] = conj(X[N-k]) for k=1..N/2-1
	size := 8
	data := make([]float32, 2*size)
	for i := range size {
		data[i] = float32(i + 1)
	}

	fftReal(data)

	// Check Hermitian symmetry
	for k := 1; k < size/2; k++ {
		realK := data[k]
		imagK := data[k+size]
		realNK := data[size-k]
		imagNK := data[2*size-k]

		// X[k] should equal conj(X[N-k])
		assert.InDelta(t, realK, realNK, 1e-4, "Hermitian symmetry real k=%d", k)
		assert.InDelta(t, imagK, -imagNK, 1e-4, "Hermitian symmetry imag k=%d", k)
	}
}

func TestFFTReal_IFFTReal_Roundtrip(t *testing.T) {
	// Comprehensive roundtrip test for fftReal/ifftReal with various signal patterns
	tests := []struct {
		name   string
		size   int
		signal func(size int) []float32
	}{
		{
			name: "linear_ramp",
			size: 8,
			signal: func(size int) []float32 {
				data := make([]float32, 2*size)
				for i := range size {
					data[i] = float32(i)
				}
				return data
			},
		},
		{
			name: "sine_wave",
			size: 16,
			signal: func(size int) []float32 {
				data := make([]float32, 2*size)
				for i := range size {
					angle := 2 * math32.Pi * float32(i) / float32(size)
					data[i] = math32.Sin(angle)
				}
				return data
			},
		},
		{
			name: "cosine_wave",
			size: 16,
			signal: func(size int) []float32 {
				data := make([]float32, 2*size)
				for i := range size {
					angle := 2 * math32.Pi * float32(i) / float32(size)
					data[i] = math32.Cos(angle)
				}
				return data
			},
		},
		{
			name: "mixed_frequencies",
			size: 32,
			signal: func(size int) []float32 {
				data := make([]float32, 2*size)
				for i := range size {
					angle1 := 2 * math32.Pi * float32(i) / float32(size)
					angle2 := 4 * math32.Pi * float32(i) / float32(size)
					data[i] = math32.Sin(angle1) + 0.5*math32.Cos(angle2)
				}
				return data
			},
		},
		{
			name: "exponential_decay",
			size: 16,
			signal: func(size int) []float32 {
				data := make([]float32, 2*size)
				for i := range size {
					data[i] = math32.Exp(-float32(i) / float32(size))
				}
				return data
			},
		},
		{
			name: "random_values",
			size: 16,
			signal: func(size int) []float32 {
				data := make([]float32, 2*size)
				// Use deterministic "random" values
				for i := range size {
					val := float32(i*7+13) / 100.0 // Pseudo-random pattern
					data[i] = val - 0.5            // Center around zero
				}
				return data
			},
		},
		{
			name: "impulse_response",
			size: 8,
			signal: func(size int) []float32 {
				data := make([]float32, 2*size)
				data[0] = 1.0
				return data
			},
		},
		{
			name: "step_function",
			size: 16,
			signal: func(size int) []float32 {
				data := make([]float32, 2*size)
				half := size / 2
				for i := range size {
					if i < half {
						data[i] = 0.0
					} else {
						data[i] = 1.0
					}
				}
				return data
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			original := tt.signal(tt.size)
			data := make([]float32, len(original))
			copy(data, original)

			// Forward FFT
			fftReal(data)

			// Inverse FFT
			ifftReal(data)

			// Verify roundtrip: result should be original scaled by 1/N
			scale := 1.0 / float32(tt.size)
			for i := range tt.size {
				expected := original[i] * scale
				actual := data[i]
				assert.InDelta(t, expected, actual, 1e-4,
					"Roundtrip mismatch at index %d: expected %f, got %f", i, expected, actual)
			}

			// Verify imaginary parts are approximately zero (for real input)
			for i := range tt.size {
				imagVal := data[i+tt.size]
				assert.InDelta(t, 0.0, imagVal, 1e-4,
					"Imaginary part should be ~0 at index %d, got %f", i, imagVal)
			}
		})
	}
}

func TestFFTComplex_IFFTComplex_Roundtrip(t *testing.T) {
	// Comprehensive roundtrip test for fftComplex/ifftComplex with various signal patterns
	tests := []struct {
		name   string
		size   int
		signal func(size int) []complex64
	}{
		{
			name: "linear_ramp",
			size: 8,
			signal: func(size int) []complex64 {
				data := make([]complex64, size)
				for i := range size {
					data[i] = complex64(complex(float32(i), float32(i)*0.5))
				}
				return data
			},
		},
		{
			name: "complex_exponential",
			size: 16,
			signal: func(size int) []complex64 {
				data := make([]complex64, size)
				for i := range size {
					angle := 2 * math32.Pi * float32(i) / float32(size)
					realVal := math32.Cos(angle)
					imagVal := math32.Sin(angle)
					data[i] = complex64(complex(realVal, imagVal))
				}
				return data
			},
		},
		{
			name: "mixed_real_imag",
			size: 16,
			signal: func(size int) []complex64 {
				data := make([]complex64, size)
				for i := range size {
					angle1 := 2 * math32.Pi * float32(i) / float32(size)
					angle2 := 4 * math32.Pi * float32(i) / float32(size)
					realVal := math32.Sin(angle1)
					imagVal := math32.Cos(angle2)
					data[i] = complex64(complex(realVal, imagVal))
				}
				return data
			},
		},
		{
			name: "purely_real",
			size: 16,
			signal: func(size int) []complex64 {
				data := make([]complex64, size)
				for i := range size {
					data[i] = complex64(complex(float32(i+1), 0.0))
				}
				return data
			},
		},
		{
			name: "purely_imaginary",
			size: 16,
			signal: func(size int) []complex64 {
				data := make([]complex64, size)
				for i := range size {
					data[i] = complex64(complex(0.0, float32(i+1)))
				}
				return data
			},
		},
		{
			name: "spiral",
			size: 32,
			signal: func(size int) []complex64 {
				data := make([]complex64, size)
				for i := range size {
					angle := 2 * math32.Pi * float32(i) / float32(size)
					radius := float32(i+1) / float32(size)
					realVal := radius * math32.Cos(angle)
					imagVal := radius * math32.Sin(angle)
					data[i] = complex64(complex(realVal, imagVal))
				}
				return data
			},
		},
		{
			name: "complex_impulse",
			size: 8,
			signal: func(size int) []complex64 {
				data := make([]complex64, size)
				data[0] = complex64(complex(1.0, 1.0))
				return data
			},
		},
		{
			name: "quadratic_phase",
			size: 16,
			signal: func(size int) []complex64 {
				data := make([]complex64, size)
				for i := range size {
					phase := float32(i*i) / float32(size*size) * 2 * math32.Pi
					realVal := math32.Cos(phase)
					imagVal := math32.Sin(phase)
					data[i] = complex64(complex(realVal, imagVal))
				}
				return data
			},
		},
		{
			name: "random_complex",
			size: 16,
			signal: func(size int) []complex64 {
				data := make([]complex64, size)
				// Use deterministic "random" values
				for i := range size {
					realVal := (float32(i*7+13) / 100.0) - 0.5
					imagVal := (float32(i*11+17) / 100.0) - 0.5
					data[i] = complex64(complex(realVal, imagVal))
				}
				return data
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			original := tt.signal(tt.size)
			data := make([]complex64, len(original))
			copy(data, original)

			// Forward FFT
			fftComplex(data)

			// Inverse FFT
			ifftComplex(data)

			// Verify roundtrip: result should be original scaled by 1/N
			scale := 1.0 / float32(tt.size)
			for i := range tt.size {
				expectedReal := real(original[i]) * scale
				expectedImag := imag(original[i]) * scale
				actualReal := real(data[i])
				actualImag := imag(data[i])

				assert.InDelta(t, expectedReal, actualReal, 1e-4,
					"Roundtrip real part mismatch at index %d: expected %f, got %f", i, expectedReal, actualReal)
				assert.InDelta(t, expectedImag, actualImag, 1e-4,
					"Roundtrip imag part mismatch at index %d: expected %f, got %f", i, expectedImag, actualImag)
			}
		})
	}
}

func TestFFTReal_IFFTReal_Roundtrip_MultipleSizes(t *testing.T) {
	// Test roundtrip for various sizes to ensure it works across different FFT sizes
	sizes := []int{2, 4, 8, 16, 32, 64, 128}
	for _, size := range sizes {
		t.Run(fmt.Sprintf("size_%d", size), func(t *testing.T) {
			// Create test signal: combination of sine and cosine
			data := make([]float32, 2*size)
			for i := range size {
				angle1 := 2 * math32.Pi * float32(i) / float32(size)
				angle2 := 4 * math32.Pi * float32(i) / float32(size)
				data[i] = math32.Sin(angle1) + 0.3*math32.Cos(angle2)
			}

			original := make([]float32, 2*size)
			copy(original, data)

			// Forward then inverse FFT
			fftReal(data)
			ifftReal(data)

			// Verify roundtrip
			scale := 1.0 / float32(size)
			maxError := float32(0.0)
			for i := range size {
				expected := original[i] * scale
				actual := data[i]
				error := math32.Abs(actual - expected)
				if error > maxError {
					maxError = error
				}
				assert.InDelta(t, expected, actual, 1e-3,
					"Size %d: roundtrip mismatch at index %d", size, i)
			}
			t.Logf("Size %d: max roundtrip error = %e", size, maxError)
		})
	}
}

func TestFFTComplex_IFFTComplex_Roundtrip_MultipleSizes(t *testing.T) {
	// Test roundtrip for various sizes to ensure it works across different FFT sizes
	sizes := []int{2, 4, 8, 16, 32, 64, 128}
	for _, size := range sizes {
		t.Run(fmt.Sprintf("size_%d", size), func(t *testing.T) {
			// Create test signal: complex exponential
			data := make([]complex64, size)
			for i := range size {
				angle := 2 * math32.Pi * float32(i) / float32(size)
				realVal := math32.Cos(angle)
				imagVal := math32.Sin(angle)
				data[i] = complex64(complex(realVal, imagVal))
			}

			original := make([]complex64, size)
			copy(original, data)

			// Forward then inverse FFT
			fftComplex(data)
			ifftComplex(data)

			// Verify roundtrip
			scale := 1.0 / float32(size)
			maxError := float32(0.0)
			for i := range size {
				expectedReal := real(original[i]) * scale
				expectedImag := imag(original[i]) * scale
				actualReal := real(data[i])
				actualImag := imag(data[i])

				errorReal := math32.Abs(actualReal - expectedReal)
				errorImag := math32.Abs(actualImag - expectedImag)
				if errorReal > maxError {
					maxError = errorReal
				}
				if errorImag > maxError {
					maxError = errorImag
				}

				assert.InDelta(t, expectedReal, actualReal, 1e-3,
					"Size %d: roundtrip real mismatch at index %d", size, i)
				assert.InDelta(t, expectedImag, actualImag, 1e-3,
					"Size %d: roundtrip imag mismatch at index %d", size, i)
			}
			t.Logf("Size %d: max roundtrip error = %e", size, maxError)
		})
	}
}

