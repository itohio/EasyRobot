package c64

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestFFT_Impulse(t *testing.T) {
	// Test FFT of complex impulse [1+0i, 0+0i, 0+0i, 0+0i]
	data := make([]complex64, 4)
	data[0] = complex64(1 + 0i)

	FFT(data)

	// All frequency components should be 1
	for i := range len(data) {
		assert.InDelta(t, 1.0, real(data[i]), 1e-6, "Real part at index %d", i)
		assert.InDelta(t, 0.0, imag(data[i]), 1e-6, "Imag part at index %d", i)
	}
}

func TestFFT_Constant(t *testing.T) {
	// Test FFT of constant complex signal
	data := make([]complex64, 4)
	for i := range len(data) {
		data[i] = complex64(1 + 0i)
	}

	FFT(data)

	// DC component should be 4, others should be 0
	assert.InDelta(t, 4.0, real(data[0]), 1e-6, "DC component should be 4")
	assert.InDelta(t, 0.0, imag(data[0]), 1e-6, "DC imag should be 0")
	for i := 1; i < len(data); i++ {
		assert.InDelta(t, 0.0, real(data[i]), 1e-6, "Frequency component %d real should be 0", i)
		assert.InDelta(t, 0.0, imag(data[i]), 1e-6, "Frequency component %d imag should be 0", i)
	}
}

func TestIFFT_Roundtrip(t *testing.T) {
	// Test forward FFT then inverse FFT should recover original (scaled)
	original := []complex64{1 + 0i, 2 + 0i, 3 + 0i, 4 + 0i}
	data := make([]complex64, len(original))
	copy(data, original)

	FFT(data)
	IFFT(data)

	// IFFT scales by 1/N, so result should be original / 4
	scale := float32(1.0 / 4.0)
	for i := range len(original) {
		expectedReal := real(original[i]) * scale
		expectedImag := imag(original[i]) * scale
		assert.InDelta(t, expectedReal, real(data[i]), 1e-5, "Roundtrip real at index %d", i)
		assert.InDelta(t, expectedImag, imag(data[i]), 1e-5, "Roundtrip imag at index %d", i)
	}
}

func TestFFT_IFFT_Identity(t *testing.T) {
	// Test that FFT then IFFT gives identity (scaled) for various sizes
	for size := 2; size <= 64; size *= 2 {
		t.Run(fmt.Sprintf("size_%d", size), func(t *testing.T) {
			data := make([]complex64, size)
			for i := range size {
				data[i] = complex64(complex(float32(i+1), float32(i+1)))
			}

			original := make([]complex64, size)
			copy(original, data)

			FFT(data)
			IFFT(data)

			scale := 1.0 / float32(size)
			for i := range size {
				expectedReal := float32(i+1) * scale
				expectedImag := float32(i+1) * scale
				assert.InDelta(t, expectedReal, real(data[i]), 1e-5, "Identity real at index %d", i)
				assert.InDelta(t, expectedImag, imag(data[i]), 1e-5, "Identity imag at index %d", i)
			}
		})
	}
}

func TestFFT_ErrorConditions(t *testing.T) {
	t.Run("non_power_of_2", func(t *testing.T) {
		data := make([]complex64, 3) // Not power of 2
		require.Panics(t, func() {
			FFT(data)
		}, "Should panic for non-power-of-2 size")
	})

	t.Run("size_0", func(t *testing.T) {
		data := make([]complex64, 0)
		require.NotPanics(t, func() {
			FFT(data)
		}, "Should handle size 0")
	})

	t.Run("size_1", func(t *testing.T) {
		data := []complex64{1 + 0i}
		require.NotPanics(t, func() {
			FFT(data)
		}, "Should handle size 1")
	})
}

func TestIFFT_ErrorConditions(t *testing.T) {
	t.Run("non_power_of_2", func(t *testing.T) {
		data := make([]complex64, 3) // Not power of 2
		require.Panics(t, func() {
			IFFT(data)
		}, "Should panic for non-power-of-2 size")
	})
}

func TestFFT_LargeSize(t *testing.T) {
	size := 1024
	data := make([]complex64, size)
	for i := range size {
		data[i] = complex64(complex(float32(i%10), float32((i+1)%10)))
	}

	original := make([]complex64, size)
	copy(original, data)

	FFT(data)
	IFFT(data)

	scale := 1.0 / float32(size)
	for i := range size {
		expectedReal := real(original[i]) * scale
		expectedImag := imag(original[i]) * scale
		assert.InDelta(t, expectedReal, real(data[i]), 1e-4, "Large size real at index %d", i)
		assert.InDelta(t, expectedImag, imag(data[i]), 1e-4, "Large size imag at index %d", i)
	}
}

func TestComputeFFTLUT(t *testing.T) {
	tests := []struct {
		name     string
		fftSize  int
		expected int
	}{
		{"size_2", 2, 2},
		{"size_4", 4, 6},
		{"size_8", 8, 14},
		{"size_16", 16, 30},
		{"size_32", 32, 62},
		{"size_64", 64, 126},
		{"size_128", 128, 254},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ComputeFFTLUT(tt.fftSize)
			assert.Equal(t, tt.expected, result, "LUT size for FFT size %d", tt.fftSize)
		})
	}

	t.Run("non_power_of_2", func(t *testing.T) {
		require.Panics(t, func() {
			ComputeFFTLUT(10)
		}, "Should panic for non-power-of-2")
	})
}

func TestFillFFTLUT(t *testing.T) {
	fftSize := 8
	lutSize := ComputeFFTLUT(fftSize)
	lut := make([]float32, lutSize)

	totalFactors := FillFFTLUT(lut, fftSize)

	// Verify we got the expected number of factors
	expectedFactors := 0
	for length := 2; length <= fftSize; length <<= 1 {
		expectedFactors += length / 2
	}
	assert.Equal(t, expectedFactors, totalFactors)

	// Verify LUT values are reasonable
	for i := range totalFactors {
		assert.GreaterOrEqual(t, lut[i], float32(-1.1), "cos value at %d", i)
		assert.LessOrEqual(t, lut[i], float32(1.1), "cos value at %d", i)
		assert.GreaterOrEqual(t, lut[i+totalFactors], float32(-1.1), "sin value at %d", i)
		assert.LessOrEqual(t, lut[i+totalFactors], float32(1.1), "sin value at %d", i)
	}
}

func TestNewFFTLUT(t *testing.T) {
	fftSize := 16
	lut := NewFFTLUT(fftSize)

	expectedSize := ComputeFFTLUT(fftSize)
	assert.Equal(t, expectedSize, len(lut), "LUT should have correct size")

	// Verify it's filled
	hasNonZero := false
	for _, v := range lut {
		if v != 0 {
			hasNonZero = true
			break
		}
	}
	assert.True(t, hasNonZero, "LUT should contain non-zero values")
}

func TestFFTWithLUT(t *testing.T) {
	fftSize := 8
	lut := NewFFTLUT(fftSize)
	totalFactors := len(lut) / 2

	data := make([]complex64, fftSize)
	for i := range fftSize {
		data[i] = complex64(complex(float32(i+1), float32(i+1)))
	}

	original := make([]complex64, len(data))
	copy(original, data)

	FFTWithLUT(data, lut, totalFactors)

	// Should produce same result as regular FFT
	expected := make([]complex64, len(original))
	copy(expected, original)
	FFT(expected)

	for i := range len(data) {
		assert.InDelta(t, real(expected[i]), real(data[i]), 1e-5, "FFTWithLUT real should match FFT at index %d", i)
		assert.InDelta(t, imag(expected[i]), imag(data[i]), 1e-5, "FFTWithLUT imag should match FFT at index %d", i)
	}
}

func TestFFTWithLUT_Roundtrip(t *testing.T) {
	fftSize := 16
	lut := NewFFTLUT(fftSize)
	totalFactors := len(lut) / 2

	data := make([]complex64, fftSize)
	for i := range fftSize {
		data[i] = complex64(complex(float32(i+1), float32(i+1)))
	}

	original := make([]complex64, len(data))
	copy(original, data)

	FFTWithLUT(data, lut, totalFactors)
	IFFT(data)

	scale := 1.0 / float32(fftSize)
	for i := range fftSize {
		expectedReal := real(original[i]) * scale
		expectedImag := imag(original[i]) * scale
		assert.InDelta(t, expectedReal, real(data[i]), 1e-5, "Roundtrip with LUT real at index %d", i)
		assert.InDelta(t, expectedImag, imag(data[i]), 1e-5, "Roundtrip with LUT imag at index %d", i)
	}
}

