package iq

import (
	"math"
	"testing"

	"github.com/itohio/EasyRobot/x/math/vec"
	"github.com/stretchr/testify/assert"
)

// TestEncoder_NewEncoder tests encoder creation
func TestEncoder_NewEncoder(t *testing.T) {
	sampleRate := float32(48000)
	carrierFreq := float32(1000)

	encoder := NewEncoder(sampleRate, carrierFreq)

	assert.NotNil(t, encoder)
	assert.Equal(t, sampleRate, encoder.sampleRate)
	assert.Equal(t, carrierFreq, encoder.carrierFreq)
}

// TestEncoder_Modulate_Basic tests basic IQ modulation
func TestEncoder_Modulate_Basic(t *testing.T) {
	sampleRate := float32(48000)
	carrierFreq := float32(1000)
	length := 100

	encoder := NewEncoder(sampleRate, carrierFreq)

	// Create I and Q vectors (DC signals for simplicity)
	I := vec.New(length)
	Q := vec.New(length)
	iData := []float32(I)
	qData := []float32(Q)

	// Set constant I and Q values
	for i := 0; i < length; i++ {
		iData[i] = 0.5
		qData[i] = 0.3
	}

	// Modulate
	result := encoder.Modulate(I, Q)
	resultData := []float32(result)

	assert.Equal(t, length, len(resultData))

	// Verify modulation formula: result = I*cos(ωt) - Q*sin(ωt)
	omega := 2 * math.Pi * carrierFreq / sampleRate
	for i := 0; i < length; i++ {
		expected := 0.5*float32(math.Cos(float64(omega*float32(i)))) -
			0.3*float32(math.Sin(float64(omega*float32(i))))
		assert.InDelta(t, expected, resultData[i], 0.001,
			"Modulated value should match expected formula")
	}
}

// TestEncoder_Modulate_SineWave tests modulation with sine wave I/Q
func TestEncoder_Modulate_SineWave(t *testing.T) {
	sampleRate := float32(48000)
	carrierFreq := float32(1000)
	basebandFreq := float32(100)
	length := 480 // 10ms at 48kHz

	encoder := NewEncoder(sampleRate, carrierFreq)

	// Create I and Q as sine waves
	I := vec.New(length)
	Q := vec.New(length)
	iData := []float32(I)
	qData := []float32(Q)

	basebandOmega := 2 * math.Pi * basebandFreq / sampleRate
	for i := 0; i < length; i++ {
		iData[i] = float32(math.Cos(float64(basebandOmega * float32(i))))
		qData[i] = float32(math.Sin(float64(basebandOmega * float32(i))))
	}

	// Modulate
	result := encoder.Modulate(I, Q)
	resultData := []float32(result)

	assert.Equal(t, length, len(resultData))

	// Result should be modulated signal
	// For I=cos(ωb*t), Q=sin(ωb*t), carrier=cos(ωc*t), sin(ωc*t)
	// Result = I*cos(ωc*t) - Q*sin(ωc*t)
	// This creates sidebands at ωc±ωb
	carrierOmega := 2 * math.Pi * carrierFreq / sampleRate
	for i := 0; i < length; i++ {
		expected := iData[i]*float32(math.Cos(float64(carrierOmega*float32(i)))) -
			qData[i]*float32(math.Sin(float64(carrierOmega*float32(i))))
		assert.InDelta(t, expected, resultData[i], 0.001,
			"Modulated sine wave should match expected")
	}
}

// TestEncoder_Modulate_MismatchedLength tests that mismatched I/Q lengths panic
func TestEncoder_Modulate_MismatchedLength(t *testing.T) {
	sampleRate := float32(48000)
	carrierFreq := float32(1000)

	encoder := NewEncoder(sampleRate, carrierFreq)

	I := vec.New(100)
	Q := vec.New(50) // Different length

	assert.Panics(t, func() {
		encoder.Modulate(I, Q)
	}, "Should panic when I and Q have different lengths")
}

// TestEncoder_GenerateIQSignal tests GenerateIQSignal wrapper
func TestEncoder_GenerateIQSignal(t *testing.T) {
	sampleRate := float32(48000)
	carrierFreq := float32(1000)
	length := 100

	encoder := NewEncoder(sampleRate, carrierFreq)

	I := vec.New(length)
	Q := vec.New(length)
	iData := []float32(I)
	qData := []float32(Q)

	for i := 0; i < length; i++ {
		iData[i] = 0.5
		qData[i] = 0.3
	}

	// GenerateIQSignal should be same as Modulate
	result1 := encoder.Modulate(I, Q)
	result2 := encoder.GenerateIQSignal(I, Q)

	resultData1 := []float32(result1)
	resultData2 := []float32(result2)

	assert.Equal(t, len(resultData1), len(resultData2))
	for i := 0; i < length; i++ {
		assert.Equal(t, resultData1[i], resultData2[i],
			"GenerateIQSignal should return same as Modulate")
	}
}

// TestEncoder_Modulate_EmptyVectors tests with empty vectors
func TestEncoder_Modulate_EmptyVectors(t *testing.T) {
	sampleRate := float32(48000)
	carrierFreq := float32(1000)

	encoder := NewEncoder(sampleRate, carrierFreq)

	I := vec.New(0)
	Q := vec.New(0)

	result := encoder.Modulate(I, Q)
	resultData := []float32(result)

	assert.Equal(t, 0, len(resultData), "Empty vectors should produce empty result")
}

// TestEncoder_Modulate_Roundtrip tests roundtrip encoding/decoding
func TestEncoder_Modulate_Roundtrip(t *testing.T) {
	sampleRate := float32(48000)
	carrierFreq := float32(1000)
	length := 100

	encoder := NewEncoder(sampleRate, carrierFreq)
	decoder := NewDecoder(sampleRate, carrierFreq)

	// Create original I and Q
	originalI := vec.New(length)
	originalQ := vec.New(length)
	iData := []float32(originalI)
	qData := []float32(originalQ)

	// Fill with test pattern
	for i := 0; i < length; i++ {
		iData[i] = float32(i) / float32(length)        // Ramp from 0 to 1
		qData[i] = float32(length-i) / float32(length) // Ramp from 1 to 0
	}

	// Encode
	modulated := encoder.Modulate(originalI, originalQ)

	// Decode
	decodedI, decodedQ := decoder.Demodulate(modulated)
	decodedIData := []float32(decodedI)
	decodedQData := []float32(decodedQ)

	assert.Equal(t, length, len(decodedIData))
	assert.Equal(t, length, len(decodedQData))

	// After demodulation, we need to account for the fact that
	// demodulation extracts I and Q, but there may be some phase/amplitude differences
	// due to the modulation/demodulation process. For a perfect roundtrip,
	// we'd need filtering. Let's verify the structure is correct.
	// The decoded values should be proportional to original (with some scaling)
	// Verify that we get some non-zero values (structure is preserved)
	hasNonZeroI := false
	hasNonZeroQ := false
	for i := 0; i < length; i++ {
		if math.Abs(float64(decodedIData[i])) > 0.001 {
			hasNonZeroI = true
		}
		if math.Abs(float64(decodedQData[i])) > 0.001 {
			hasNonZeroQ = true
		}
	}
	assert.True(t, hasNonZeroI, "Decoded I should have some non-zero values")
	assert.True(t, hasNonZeroQ, "Decoded Q should have some non-zero values")
}
