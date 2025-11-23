package mpsk

import (
	"math"
	"testing"

	"github.com/itohio/EasyRobot/x/math/vec"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const (
	testSampleRate     = float32(48000)
	testCarrierFreq    = float32(1000)
	testSymbolRate     = float32(1000)
	testAmplitude      = float32(1.0)
	testModulationOrder = 2 // BPSK
)

// TestModulator_NewModulator tests modulator creation
func TestModulator_NewModulator(t *testing.T) {
	modulator := NewModulator(testSampleRate, testCarrierFreq, testSymbolRate, testModulationOrder)

	require.NotNil(t, modulator)
	assert.Equal(t, testSampleRate, modulator.sampleRate)
	assert.Equal(t, testCarrierFreq, modulator.carrierFreq)
	assert.Equal(t, testSymbolRate, modulator.symbolRate)
	assert.Equal(t, testModulationOrder, modulator.modulationOrder)
	assert.Equal(t, int(testSampleRate/testSymbolRate), modulator.samplesPerSymbol)
	assert.Equal(t, testAmplitude, modulator.amplitude)
}

// TestModulator_Encode_NilAmplitudes tests Encode with nil amplitudes (size query)
func TestModulator_Encode_NilAmplitudes(t *testing.T) {
	modulator := NewModulator(testSampleRate, testCarrierFreq, testSymbolRate, testModulationOrder)
	data := []byte{0x55, 0xAA}

	expectedSize := modulator.Encode(nil, data)
	assert.Greater(t, expectedSize, 0, "Expected size should be positive")
	
	// Encoder doesn't add padding - it only returns the size needed for the signal
	// For 2 bytes with BPSK (1 bit per symbol), we need at least 16 symbols
	// Each symbol needs samplesPerSymbol samples
	samplesPerSymbol := int(testSampleRate / testSymbolRate)
	minExpectedSize := len(data) * 8 * samplesPerSymbol // 2 bytes * 8 bits * samples per symbol
	assert.GreaterOrEqual(t, expectedSize, minExpectedSize, "Should return size for signal samples")
}

// TestModulator_Encode_BPSK tests BPSK encoding
func TestModulator_Encode_BPSK(t *testing.T) {
	modulator := NewModulator(testSampleRate, testCarrierFreq, testSymbolRate, 2)
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

// TestModulator_Encode_QPSK tests QPSK encoding
func TestModulator_Encode_QPSK(t *testing.T) {
	modulator := NewModulator(testSampleRate, testCarrierFreq, testSymbolRate, 4)
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
	assert.True(t, hasNonZero, "QPSK should have non-zero amplitudes")
}

// TestModulator_Reset tests reset functionality
func TestModulator_Reset(t *testing.T) {
	modulator := NewModulator(testSampleRate, testCarrierFreq, testSymbolRate, testModulationOrder)
	
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
	modulator := NewModulator(testSampleRate, testCarrierFreq, testSymbolRate, testModulationOrder)
	newAmplitude := float32(0.5)
	
	modulator.SetAmplitude(newAmplitude)
	assert.Equal(t, newAmplitude, modulator.amplitude)
}

// TestModulator_ModulationOrder tests ModulationOrder method
func TestModulator_ModulationOrder(t *testing.T) {
	modulator2 := NewModulator(testSampleRate, testCarrierFreq, testSymbolRate, 2)
	modulator4 := NewModulator(testSampleRate, testCarrierFreq, testSymbolRate, 4)

	assert.Equal(t, 2, modulator2.ModulationOrder())
	assert.Equal(t, 4, modulator4.ModulationOrder())
}

// TestModulator_GenerateSignal tests GenerateSignal method
func TestModulator_GenerateSignal(t *testing.T) {
	modulator := NewModulator(testSampleRate, testCarrierFreq, testSymbolRate, testModulationOrder)
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

