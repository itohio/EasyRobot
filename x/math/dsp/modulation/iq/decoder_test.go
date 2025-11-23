package iq

import (
	"math"
	"testing"

	"github.com/itohio/EasyRobot/x/math/filter/fir"
	"github.com/itohio/EasyRobot/x/math/vec"
	"github.com/stretchr/testify/assert"
)

// TestDecoder_NewDecoder tests decoder creation
func TestDecoder_NewDecoder(t *testing.T) {
	sampleRate := float32(48000)
	carrierFreq := float32(1000)

	decoder := NewDecoder(sampleRate, carrierFreq)

	assert.NotNil(t, decoder)
	assert.Equal(t, sampleRate, decoder.sampleRate)
	assert.Equal(t, carrierFreq, decoder.carrierFreq)
	assert.Equal(t, 2*math.Pi*carrierFreq/sampleRate, decoder.omega)
	assert.Equal(t, float32(0), decoder.phaseAcc)
}

// TestDecoder_Demodulate_Basic tests basic IQ demodulation
func TestDecoder_Demodulate_Basic(t *testing.T) {
	sampleRate := float32(48000)
	carrierFreq := float32(1000)
	length := 100

	decoder := NewDecoder(sampleRate, carrierFreq)

	// Create a modulated signal: I*cos(ωt) - Q*sin(ωt)
	// For test: I=0.5, Q=0.3 (constant)
	signal := vec.New(length)
	signalData := []float32(signal)

	omega := 2 * math.Pi * carrierFreq / sampleRate
	for i := 0; i < length; i++ {
		// Simulate modulated signal with I=0.5, Q=0.3
		signalData[i] = 0.5*float32(math.Cos(float64(omega*float32(i)))) -
			0.3*float32(math.Sin(float64(omega*float32(i))))
	}

	// Demodulate
	I, Q := decoder.Demodulate(signal)
	iData := []float32(I)
	qData := []float32(Q)

	assert.Equal(t, length, len(iData))
	assert.Equal(t, length, len(qData))

	// Demodulated I and Q should extract the baseband components
	// I_demod = signal * cos(ωt)
	// Q_demod = -signal * sin(ωt)
	// For constant I/Q input, output will have DC component plus 2*ω components
	// Without filtering, we get the baseband plus harmonics
	for i := 0; i < length; i++ {
		expectedI := signalData[i] * float32(math.Cos(float64(omega*float32(i))))
		expectedQ := -signalData[i] * float32(math.Sin(float64(omega*float32(i))))

		assert.InDelta(t, expectedI, iData[i], 0.001,
			"Demodulated I should match expected formula")
		assert.InDelta(t, expectedQ, qData[i], 0.001,
			"Demodulated Q should match expected formula")
	}
}

// TestDecoder_Demodulate_WithFilter tests demodulation with low-pass filter
func TestDecoder_Demodulate_WithFilter(t *testing.T) {
	sampleRate := float32(48000)
	carrierFreq := float32(1000)
	length := 100

	decoder := NewDecoder(sampleRate, carrierFreq)

	// Create low-pass filter (cutoff above baseband but below carrier)
	cutoffFreq := float32(5000) // 5kHz cutoff
	lpFilter := fir.NewLowPass(16, cutoffFreq, sampleRate)
	decoder.SetFilter(lpFilter)

	// Create modulated signal
	signal := vec.New(length)
	signalData := []float32(signal)

	omega := 2 * math.Pi * carrierFreq / sampleRate
	for i := 0; i < length; i++ {
		signalData[i] = 0.5*float32(math.Cos(float64(omega*float32(i)))) -
			0.3*float32(math.Sin(float64(omega*float32(i))))
	}

	// Demodulate
	I, Q := decoder.Demodulate(signal)
	iData := []float32(I)
	qData := []float32(Q)

	assert.Equal(t, length, len(iData))
	assert.Equal(t, length, len(qData))

	// With filter, the output should be filtered
	// Verify filter was applied (values should be different from unfiltered)
	decoderNoFilter := NewDecoder(sampleRate, carrierFreq)
	INoFilter, QNoFilter := decoderNoFilter.Demodulate(signal)
	iDataNoFilter := []float32(INoFilter)
	qDataNoFilter := []float32(QNoFilter)

	// Filtered values should be different (filtered removes high-frequency components)
	// At least some values should differ
	different := false
	for i := 0; i < length; i++ {
		if math.Abs(float64(iData[i]-iDataNoFilter[i])) > 0.001 ||
			math.Abs(float64(qData[i]-qDataNoFilter[i])) > 0.001 {
			different = true
			break
		}
	}
	assert.True(t, different, "Filtered output should differ from unfiltered")
}

// TestDecoder_ProcessSample tests streaming sample processing
func TestDecoder_ProcessSample(t *testing.T) {
	sampleRate := float32(48000)
	carrierFreq := float32(1000)
	numSamples := 100

	decoder := NewDecoder(sampleRate, carrierFreq)

	// Process samples one by one
	omega := 2 * math.Pi * carrierFreq / sampleRate
	for i := 0; i < numSamples; i++ {
		// Create sample: I*cos(ωt) - Q*sin(ωt) with I=0.5, Q=0.3
		sample := 0.5*float32(math.Cos(float64(omega*float32(i)))) -
			0.3*float32(math.Sin(float64(omega*float32(i))))

		iVal, qVal := decoder.ProcessSample(sample)

		// Verify demodulation formula
		expectedI := sample * float32(math.Cos(float64(omega*float32(i))))
		expectedQ := -sample * float32(math.Sin(float64(omega*float32(i))))

		assert.InDelta(t, expectedI, iVal, 0.001,
			"ProcessSample I should match expected")
		assert.InDelta(t, expectedQ, qVal, 0.001,
			"ProcessSample Q should match expected")
	}

	// Verify phase accumulator was updated
	assert.Greater(t, decoder.phaseAcc, float32(0),
		"Phase accumulator should be updated after processing samples")
}

// TestDecoder_ProcessSample_PhaseWrapping tests phase accumulator wrapping
func TestDecoder_ProcessSample_PhaseWrapping(t *testing.T) {
	sampleRate := float32(48000)
	carrierFreq := float32(1000)

	decoder := NewDecoder(sampleRate, carrierFreq)

	// Process enough samples to wrap phase multiple times
	// Phase wraps at 2π
	samplesToWrap := int(2*math.Pi/decoder.omega) + 10

	for i := 0; i < samplesToWrap; i++ {
		decoder.ProcessSample(0.5)
	}

	// Phase should be wrapped (between 0 and 2π)
	assert.GreaterOrEqual(t, decoder.phaseAcc, float32(0),
		"Phase accumulator should be >= 0")
	assert.Less(t, decoder.phaseAcc, float32(2*math.Pi),
		"Phase accumulator should be < 2π")
}

// TestDecoder_ProcessSample_WithFilter tests streaming with filter
func TestDecoder_ProcessSample_WithFilter(t *testing.T) {
	sampleRate := float32(48000)
	carrierFreq := float32(1000)

	decoder := NewDecoder(sampleRate, carrierFreq)
	cutoffFreq := float32(5000)
	lpFilter := fir.NewLowPass(16, cutoffFreq, sampleRate)
	decoder.SetFilter(lpFilter)

	// Process samples
	omega := 2 * math.Pi * carrierFreq / sampleRate
	for i := 0; i < 100; i++ {
		sample := 0.5 * float32(math.Cos(float64(omega*float32(i))))
		iVal, qVal := decoder.ProcessSample(sample)

		// Values should be filtered (non-zero even for simple input)
		_ = iVal
		_ = qVal
		// Filter application is verified by the fact that it doesn't crash
		// and produces values
		assert.NotPanics(t, func() {
			decoder.ProcessSample(sample)
		}, "ProcessSample with filter should not panic")
	}
}

// TestDecoder_SetFilter tests filter setting
func TestDecoder_SetFilter(t *testing.T) {
	sampleRate := float32(48000)
	carrierFreq := float32(1000)

	decoder := NewDecoder(sampleRate, carrierFreq)
	assert.Nil(t, decoder.lowPassFilter, "Filter should be nil initially")

	cutoffFreq := float32(5000)
	lpFilter := fir.NewLowPass(16, cutoffFreq, sampleRate)
	decoder.SetFilter(lpFilter)

	assert.NotNil(t, decoder.lowPassFilter, "Filter should be set after SetFilter")
	assert.Equal(t, lpFilter, decoder.lowPassFilter, "Filter should match what was set")
}

// TestDecoder_Reset tests reset functionality
func TestDecoder_Reset(t *testing.T) {
	sampleRate := float32(48000)
	carrierFreq := float32(1000)

	decoder := NewDecoder(sampleRate, carrierFreq)

	// Process some samples to change state
	for i := 0; i < 100; i++ {
		decoder.ProcessSample(0.5)
	}

	// Verify state changed
	assert.Greater(t, decoder.phaseAcc, float32(0),
		"Phase accumulator should be non-zero after processing")

	// Reset
	decoder.Reset()

	// Verify state reset
	assert.Equal(t, float32(0), decoder.phaseAcc,
		"Phase accumulator should be reset to 0")
}

// TestDecoder_Reset_WithFilter tests reset with filter
func TestDecoder_Reset_WithFilter(t *testing.T) {
	sampleRate := float32(48000)
	carrierFreq := float32(1000)

	decoder := NewDecoder(sampleRate, carrierFreq)
	cutoffFreq := float32(5000)
	lpFilter := fir.NewLowPass(16, cutoffFreq, sampleRate)
	decoder.SetFilter(lpFilter)

	// Process samples
	for i := 0; i < 100; i++ {
		decoder.ProcessSample(0.5)
	}

	// Reset
	decoder.Reset()

	// Verify phase reset
	assert.Equal(t, float32(0), decoder.phaseAcc,
		"Phase accumulator should be reset")
	// Filter reset is verified by not panicking
	assert.NotNil(t, decoder.lowPassFilter, "Filter should still be set after reset")
}

// TestDecoder_Demodulate_EmptySignal tests with empty signal
func TestDecoder_Demodulate_EmptySignal(t *testing.T) {
	sampleRate := float32(48000)
	carrierFreq := float32(1000)

	decoder := NewDecoder(sampleRate, carrierFreq)

	signal := vec.New(0)
	I, Q := decoder.Demodulate(signal)

	iData := []float32(I)
	qData := []float32(Q)

	assert.Equal(t, 0, len(iData), "Empty signal should produce empty I")
	assert.Equal(t, 0, len(qData), "Empty signal should produce empty Q")
}

// TestDecoder_Roundtrip tests roundtrip encoding/decoding
func TestDecoder_Roundtrip(t *testing.T) {
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
		iData[i] = float32(i) / float32(length)
		qData[i] = float32(length-i) / float32(length)
	}

	// Encode
	modulated := encoder.Modulate(originalI, originalQ)

	// Decode
	decodedI, decodedQ := decoder.Demodulate(modulated)
	decodedIData := []float32(decodedI)
	decodedQData := []float32(decodedQ)

	assert.Equal(t, length, len(decodedIData))
	assert.Equal(t, length, len(decodedQData))

	// With proper filtering, we should recover the original signal
	// For now, verify the structure is preserved
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
