package fp32

import (
	"fmt"
	"testing"

	"github.com/chewxy/math32"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestFFT_Impulse(t *testing.T) {
	// Test FFT of impulse signal [1, 0, 0, 0]
	// Expected: all frequency components should be 1 (DC component)
	data := make([]float32, 8) // 4 complex samples = 8 float32
	data[0] = 1.0              // Real part of first sample
	// Imaginary parts are already zero

	FFT(data)

	// DC component (index 0) should be 1
	assert.InDelta(t, 1.0, data[0], 1e-6, "DC component should be 1")
	// All other components should be 1 (for impulse)
	for i := 1; i < 4; i++ {
		assert.InDelta(t, 1.0, data[i], 1e-6, "Frequency component %d should be 1", i)
	}
}

func TestFFT_Constant(t *testing.T) {
	// Test FFT of constant signal [1, 1, 1, 1]
	// Expected: DC component should be 4, others should be 0
	data := make([]float32, 8)
	for i := range 4 {
		data[i] = 1.0
	}

	FFT(data)

	// DC component should be 4
	assert.InDelta(t, 4.0, data[0], 1e-6, "DC component should be 4")
	// All other real components should be 0
	for i := 1; i < 4; i++ {
		assert.InDelta(t, 0.0, data[i], 1e-6, "Frequency component %d should be 0", i)
	}
}

func TestIFFT_Roundtrip(t *testing.T) {
	// Test forward FFT then inverse FFT should recover original (scaled)
	original := []float32{1.0, 2.0, 3.0, 4.0, 0, 0, 0, 0}
	data := make([]float32, len(original))
	copy(data, original)

	FFT(data)
	IFFT(data)

	// IFFT scales by 1/N, so result should be original / 4
	scale := float32(1.0 / 4.0)
	for i := range 4 {
		assert.InDelta(t, original[i]*scale, data[i], 1e-5, "Roundtrip failed at index %d", i)
	}
}

func TestFFT_IFFT_Identity(t *testing.T) {
	// Test that FFT then IFFT gives identity (scaled)
	for size := 2; size <= 64; size *= 2 {
		t.Run(fmt.Sprintf("size_%d", size), func(t *testing.T) {
			data := make([]float32, 2*size)
			for i := range size {
				data[i] = float32(i + 1)
			}

			FFT(data)
			IFFT(data)

			scale := 1.0 / float32(size)
			for i := range size {
				expected := float32(i+1) * scale
				require.InDelta(t, expected, data[i], 1e-5, "Identity check failed at index %d", i)
			}
		})
	}
}

func TestFFT_SineWave(t *testing.T) {
	// Test FFT of sine wave - should have peak at corresponding frequency
	size := 16
	data := make([]float32, 2*size)
	freq := 2
	for i := range size {
		angle := 2 * math32.Pi * float32(freq) * float32(i) / float32(size)
		data[i] = math32.Sin(angle)
	}

	FFT(data)

	// Should have peak at frequency bin freq (and size-freq due to symmetry)
	magFreq := math32.Sqrt(data[freq]*data[freq] + data[freq+size]*data[freq+size])
	mag0 := math32.Sqrt(data[0]*data[0] + data[size]*data[size])
	assert.Greater(t, magFreq, mag0, "Peak should be at frequency bin %d", freq)
}

func TestFFT_ErrorConditions(t *testing.T) {
	t.Run("odd_length", func(t *testing.T) {
		data := make([]float32, 5) // Odd length
		require.Panics(t, func() {
			FFT(data)
		}, "Should panic for odd length")
	})

	t.Run("non_power_of_2", func(t *testing.T) {
		data := make([]float32, 6) // 3 complex samples, but 3 is not power of 2
		require.Panics(t, func() {
			FFT(data)
		}, "Should panic for non-power-of-2 size")
	})
}

func TestIFFT_ErrorConditions(t *testing.T) {
	t.Run("odd_length", func(t *testing.T) {
		data := make([]float32, 5) // Odd length
		require.Panics(t, func() {
			IFFT(data)
		}, "Should panic for odd length")
	})
}

func TestFFT_LargeSize(t *testing.T) {
	size := 1024
	data := make([]float32, 2*size)
	for i := range size {
		data[i] = float32(i % 10)
	}

	original := make([]float32, len(data))
	copy(original, data)

	FFT(data)
	IFFT(data)

	scale := 1.0 / float32(size)
	for i := range size {
		expected := original[i] * scale
		require.InDelta(t, expected, data[i], 1e-4, "Large size test failed at index %d", i)
	}
}

func TestComputeFFTLUT(t *testing.T) {
	tests := []struct {
		name     string
		fftSize  int
		expected int
	}{
		{"size_2", 2, 2},       // 1 factor * 2 (cos+sin) = 2
		{"size_4", 4, 6},       // (1+2) factors * 2 = 6
		{"size_8", 8, 14},      // (1+2+4) factors * 2 = 14
		{"size_16", 16, 30},    // (1+2+4+8) factors * 2 = 30
		{"size_32", 32, 62},    // (1+2+4+8+16) factors * 2 = 62
		{"size_64", 64, 126},   // (1+2+4+8+16+32) factors * 2 = 126
		{"size_128", 128, 254}, // (1+2+4+8+16+32+64) factors * 2 = 254
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

	// Verify LUT values are reasonable (cos/sin should be in [-1, 1])
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

	// Verify it's filled (not all zeros)
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

	data := make([]float32, 2*fftSize)
	for i := range fftSize {
		data[i] = float32(i + 1)
	}

	original := make([]float32, len(data))
	copy(original, data)

	FFTWithLUT(data, lut, totalFactors)

	// Should produce same result as regular FFT
	expected := make([]float32, len(original))
	copy(expected, original)
	FFT(expected)

	for i := range len(data) {
		require.InDelta(t, expected[i], data[i], 1e-5, "FFTWithLUT should match FFT at index %d", i)
	}
}

func TestFFTWithLUT_Roundtrip(t *testing.T) {
	fftSize := 16
	lut := NewFFTLUT(fftSize)
	totalFactors := len(lut) / 2

	data := make([]float32, 2*fftSize)
	for i := range fftSize {
		data[i] = float32(i + 1)
	}

	original := make([]float32, len(data))
	copy(original, data)

	FFTWithLUT(data, lut, totalFactors)
	IFFT(data)

	scale := 1.0 / float32(fftSize)
	for i := range fftSize {
		expected := original[i] * scale
		require.InDelta(t, expected, data[i], 1e-5, "Roundtrip with LUT failed at index %d", i)
	}
}

func TestIFFTWithLUT(t *testing.T) {
	fftSize := 8
	lut := NewFFTLUT(fftSize)
	totalFactors := len(lut) / 2

	data := make([]float32, 2*fftSize)
	for i := range fftSize {
		data[i] = float32(i + 1)
	}

	original := make([]float32, len(data))
	copy(original, data)

	FFT(data)

	// Should produce same result as regular IFFT
	expected := make([]float32, len(original))
	copy(expected, original)
	FFT(expected)
	IFFT(expected)

	dataCopy := make([]float32, len(data))
	copy(dataCopy, data)
	IFFTWithLUT(dataCopy, lut, totalFactors)

	for i := range len(dataCopy) {
		require.InDelta(t, expected[i], dataCopy[i], 1e-5, "IFFTWithLUT should match IFFT at index %d", i)
	}
}

func TestIFFTWithLUT_Roundtrip(t *testing.T) {
	fftSize := 16
	lut := NewFFTLUT(fftSize)
	totalFactors := len(lut) / 2

	data := make([]float32, 2*fftSize)
	for i := range fftSize {
		data[i] = float32(i + 1)
	}

	original := make([]float32, len(data))
	copy(original, data)

	FFTWithLUT(data, lut, totalFactors)
	IFFTWithLUT(data, lut, totalFactors)

	scale := 1.0 / float32(fftSize)
	for i := range fftSize {
		expected := original[i] * scale
		require.InDelta(t, expected, data[i], 1e-5, "Roundtrip with IFFTWithLUT failed at index %d", i)
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
			result := NextPowerOfTwo(tt.n)
			assert.Equal(t, tt.expected, result, "nextPowerOfTwo(%d)", tt.n)
		})
	}
}
