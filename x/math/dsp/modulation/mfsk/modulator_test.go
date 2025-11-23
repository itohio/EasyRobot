package mfsk

import (
	"math"
	"testing"

	"github.com/itohio/EasyRobot/x/math/vec"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const (
	testSampleRate  = float32(48000)
	testFreqLow     = float32(1000)
	testFreqHigh    = float32(2000)
	testSymbolRate  = float32(1000)
	testAmplitude   = float32(1.0)
)

// TestModulator_NewModulator tests modulator creation
func TestModulator_NewModulator(t *testing.T) {
	modulator := NewModulator(testSampleRate, testFreqLow, testFreqHigh, testSymbolRate)

	require.NotNil(t, modulator)
	assert.Equal(t, testSampleRate, modulator.sampleRate)
	assert.Equal(t, testSymbolRate, modulator.symbolRate)
	assert.Equal(t, 2, modulator.modulationOrder)
	assert.Equal(t, testAmplitude, modulator.amplitude)
	assert.Equal(t, int(testSampleRate/testSymbolRate), modulator.samplesPerSymbol)
	assert.Equal(t, 2, len(modulator.frequencies))
	assert.Equal(t, testFreqLow, modulator.frequencies[0])
	assert.Equal(t, testFreqHigh, modulator.frequencies[1])
}

// TestModulator_NewMModulator tests M-FSK modulator creation
func TestModulator_NewMModulator(t *testing.T) {
	freqs := []float32{1000, 1500, 2000, 2500}
	modulator := NewMModulator(testSampleRate, testSymbolRate, freqs...)

	require.NotNil(t, modulator)
	assert.Equal(t, 4, modulator.modulationOrder)
	assert.Equal(t, len(freqs), len(modulator.frequencies))
	for i, freq := range freqs {
		assert.Equal(t, freq, modulator.frequencies[i])
	}
}

// TestModulator_Encode_NilAmplitudes tests Encode with nil amplitudes (size query)
func TestModulator_Encode_NilAmplitudes(t *testing.T) {
	modulator := NewModulator(testSampleRate, testFreqLow, testFreqHigh, testSymbolRate)
	data := []byte{0x55, 0xAA}

	expectedSize := modulator.Encode(nil, data)
	assert.Greater(t, expectedSize, 0, "Expected size should be positive")
	// Encoder doesn't add padding - size is just for the signal
}

// TestModulator_Encode_BFSK tests BFSK encoding
func TestModulator_Encode_BFSK(t *testing.T) {
	modulator := NewModulator(testSampleRate, testFreqLow, testFreqHigh, testSymbolRate)
	data := []byte{0x55} // 01010101

	// Calculate required size (no padding - encoder doesn't add padding)
	expectedSize := modulator.Encode(nil, data)
	amplitudes := vec.New(expectedSize)
	samplesWritten := modulator.Encode(amplitudes, data)

	assert.Greater(t, samplesWritten, 0, "Should write samples")
	assert.LessOrEqual(t, samplesWritten, expectedSize, "Should not exceed buffer size")
	assert.Equal(t, expectedSize, samplesWritten, "Should write all expected samples")

	ampData := []float32(amplitudes)
	
	// Check that signal is present (non-zero values)
	hasNonZero := false
	for i := 0; i < len(ampData); i++ {
		if math.Abs(float64(ampData[i])) > 0.001 {
			hasNonZero = true
			break
		}
	}
	assert.True(t, hasNonZero, "Signal should be present")
}

// TestModulator_Encode_4FSK tests 4-FSK encoding
func TestModulator_Encode_4FSK(t *testing.T) {
	freqs := []float32{1000, 1500, 2000, 2500}
	modulator := NewMModulator(testSampleRate, testSymbolRate, freqs...)
	data := []byte{0x12} // 00010010

	// Calculate required size (no padding - encoder doesn't add padding)
	expectedSize := modulator.Encode(nil, data)
	amplitudes := vec.New(expectedSize)
	samplesWritten := modulator.Encode(amplitudes, data)

	assert.Greater(t, samplesWritten, 0, "Should write samples")
	
	ampData := []float32(amplitudes)
	// Check that we have signal (non-zero values)
	hasNonZero := false
	for i := 0; i < samplesWritten; i++ {
		if math.Abs(float64(ampData[i])) > 0.001 {
			hasNonZero = true
			break
		}
	}
	assert.True(t, hasNonZero, "4-FSK should have non-zero amplitudes")
}

// TestModulator_Encode_8FSK tests 8-FSK encoding
func TestModulator_Encode_8FSK(t *testing.T) {
	freqs := []float32{1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750}
	modulator := NewMModulator(testSampleRate, testSymbolRate, freqs...)
	data := []byte{0x12, 0x34} // Multiple bytes to test symbol encoding

	// Calculate required size (no padding - encoder doesn't add padding)
	expectedSize := modulator.Encode(nil, data)
	amplitudes := vec.New(expectedSize)
	samplesWritten := modulator.Encode(amplitudes, data)

	assert.Greater(t, samplesWritten, 0, "Should write samples")
	assert.Equal(t, expectedSize, samplesWritten, "Should write all expected samples")
	
	ampData := []float32(amplitudes)
	// Check that we have signal (non-zero values)
	hasNonZero := false
	for i := 0; i < len(ampData); i++ {
		if math.Abs(float64(ampData[i])) > 0.001 {
			hasNonZero = true
			break
		}
	}
	assert.True(t, hasNonZero, "8-FSK should have non-zero amplitudes")
	
	// Verify modulation order
	assert.Equal(t, 8, modulator.ModulationOrder())
}

// TestModulator_Encode_16FSK tests 16-FSK encoding
func TestModulator_Encode_16FSK(t *testing.T) {
	freqs := []float32{
		1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700,
		1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500,
	}
	modulator := NewMModulator(testSampleRate, testSymbolRate, freqs...)
	data := []byte{0x12, 0x34, 0x56} // Multiple bytes to test symbol encoding

	// Calculate required size (no padding - encoder doesn't add padding)
	expectedSize := modulator.Encode(nil, data)
	amplitudes := vec.New(expectedSize)
	samplesWritten := modulator.Encode(amplitudes, data)

	assert.Greater(t, samplesWritten, 0, "Should write samples")
	assert.Equal(t, expectedSize, samplesWritten, "Should write all expected samples")
	
	ampData := []float32(amplitudes)
	// Check that we have signal (non-zero values)
	hasNonZero := false
	for i := 0; i < len(ampData); i++ {
		if math.Abs(float64(ampData[i])) > 0.001 {
			hasNonZero = true
			break
		}
	}
	assert.True(t, hasNonZero, "16-FSK should have non-zero amplitudes")
	
	// Verify modulation order
	assert.Equal(t, 16, modulator.ModulationOrder())
	
	// Verify all frequencies are set correctly
	returnedFreqs := modulator.Frequencies()
	assert.Equal(t, 16, len(returnedFreqs))
	for i, freq := range freqs {
		assert.Equal(t, freq, returnedFreqs[i], "Frequency %d should match", i)
	}
}

// TestModulator_Reset tests reset functionality
func TestModulator_Reset(t *testing.T) {
	modulator := NewModulator(testSampleRate, testFreqLow, testFreqHigh, testSymbolRate)
	
	// Reset should not panic (modulator is stateless)
	modulator.Reset()
	
	// Verify we can still encode after reset
	data := []byte{0x55}
	expectedSize := modulator.Encode(nil, data)
	amplitudes := vec.New(expectedSize)
	samplesWritten := modulator.Encode(amplitudes, data)
	assert.Greater(t, samplesWritten, 0, "Should still encode after reset")
}

// TestModulator_SetAmplitude tests amplitude setting
func TestModulator_SetAmplitude(t *testing.T) {
	modulator := NewModulator(testSampleRate, testFreqLow, testFreqHigh, testSymbolRate)
	newAmplitude := float32(0.5)
	
	modulator.SetAmplitude(newAmplitude)
	assert.Equal(t, newAmplitude, modulator.amplitude)
}

// TestModulator_ModulationOrder tests ModulationOrder method
func TestModulator_ModulationOrder(t *testing.T) {
	modulator2 := NewModulator(testSampleRate, testFreqLow, testFreqHigh, testSymbolRate)
	modulator4 := NewMModulator(testSampleRate, testSymbolRate, 1000, 1500, 2000, 2500)
	modulator8 := NewMModulator(testSampleRate, testSymbolRate, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750)
	modulator16 := NewMModulator(testSampleRate, testSymbolRate,
		1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700,
		1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500)

	assert.Equal(t, 2, modulator2.ModulationOrder())
	assert.Equal(t, 4, modulator4.ModulationOrder())
	assert.Equal(t, 8, modulator8.ModulationOrder())
	assert.Equal(t, 16, modulator16.ModulationOrder())
}

// TestModulator_Frequencies tests Frequencies method
func TestModulator_Frequencies(t *testing.T) {
	freqs := []float32{1000, 2000}
	modulator := NewModulator(testSampleRate, freqs[0], freqs[1], testSymbolRate)

	returnedFreqs := modulator.Frequencies()
	assert.Equal(t, len(freqs), len(returnedFreqs))
	for i, freq := range freqs {
		assert.Equal(t, freq, returnedFreqs[i])
	}
}

// TestModulator_GenerateSignal tests GenerateSignal method
func TestModulator_GenerateSignal(t *testing.T) {
	modulator := NewModulator(testSampleRate, testFreqLow, testFreqHigh, testSymbolRate)
	data := []byte{0x55, 0xAA}
	length := 2000

	signal := modulator.GenerateSignal(data, testAmplitude, length)
	signalData := []float32(signal)

	assert.Equal(t, length, len(signalData), "Signal should have requested length")
	assert.Greater(t, len(signalData), 0, "Signal should not be empty")
	
	// Check that signal has non-zero values
	hasNonZero := false
	for i := 0; i < len(signalData); i++ {
		if math.Abs(float64(signalData[i])) > 0.001 {
			hasNonZero = true
			break
		}
	}
	assert.True(t, hasNonZero, "Signal should have non-zero values")
}

