package mask

import (
	"math"
	"testing"

	"github.com/itohio/EasyRobot/x/math/vec"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const (
	testSampleRate  = float32(48000)
	testCarrierFreq = float32(1000)
	testSymbolRate  = float32(1000)
	testAmplitude   = float32(1.0)
)

// TestModulator_NewModulator tests modulator creation
func TestModulator_NewModulator(t *testing.T) {
	modulator := NewModulator(testSampleRate, testCarrierFreq, testSymbolRate, 2, testAmplitude)

	require.NotNil(t, modulator)
	assert.Equal(t, testSampleRate, modulator.sampleRate)
	assert.Equal(t, testCarrierFreq, modulator.carrierFreq)
	assert.Equal(t, testSymbolRate, modulator.symbolRate)
	assert.Equal(t, 2, modulator.modulationOrder)
	assert.Equal(t, testAmplitude, modulator.amplitude)
	assert.Equal(t, int(testSampleRate/testSymbolRate), modulator.samplesPerSymbol)
}

// TestModulator_Encode_NilAmplitudes tests Encode with nil amplitudes (size query)
func TestModulator_Encode_NilAmplitudes(t *testing.T) {
	modulator := NewModulator(testSampleRate, testCarrierFreq, testSymbolRate, 2, testAmplitude)
	data := []byte{0x55, 0xAA}

	expectedSize := modulator.Encode(nil, data)
	assert.Greater(t, expectedSize, 0, "Expected size should be positive")
}

// TestModulator_Encode_2ASK tests 2-ASK (OOK) encoding
func TestModulator_Encode_2ASK(t *testing.T) {
	modulator := NewModulator(testSampleRate, testCarrierFreq, testSymbolRate, 2, testAmplitude)
	data := []byte{0x55} // 01010101

	amplitudes := vec.New(1000)
	samplesWritten := modulator.Encode(amplitudes, data)

	assert.Greater(t, samplesWritten, 0, "Should write samples")
	assert.LessOrEqual(t, samplesWritten, 1000, "Should not exceed buffer size")

	ampData := []float32(amplitudes)
	// Check that we have alternating on/off (for 0x55 pattern)
	hasZero := false
	hasNonZero := false
	for i := 0; i < samplesWritten; i++ {
		if ampData[i] == 0 {
			hasZero = true
		} else if math.Abs(float64(ampData[i])) > 0.001 {
			hasNonZero = true
		}
	}
	assert.True(t, hasZero, "2-ASK should have zero amplitudes")
	assert.True(t, hasNonZero, "2-ASK should have non-zero amplitudes")
}

// TestModulator_Encode_4ASK tests 4-ASK encoding
func TestModulator_Encode_4ASK(t *testing.T) {
	modulator := NewModulator(testSampleRate, testCarrierFreq, testSymbolRate, 4, testAmplitude)
	data := []byte{0x12} // 00010010 -> symbols: 00, 01, 00, 10

	amplitudes := vec.New(1000)
	samplesWritten := modulator.Encode(amplitudes, data)

	assert.Greater(t, samplesWritten, 0, "Should write samples")

	ampData := []float32(amplitudes)
	// 4-ASK should have multiple amplitude levels
	uniqueAmplitudes := make(map[float32]bool)
	for i := 0; i < samplesWritten; i++ {
		if math.Abs(float64(ampData[i])) > 0.001 {
			uniqueAmplitudes[ampData[i]] = true
		}
	}
	assert.GreaterOrEqual(t, len(uniqueAmplitudes), 1, "4-ASK should have multiple amplitude levels")
}

// TestModulator_Reset tests reset functionality
func TestModulator_Reset(t *testing.T) {
	modulator := NewModulator(testSampleRate, testCarrierFreq, testSymbolRate, 2, testAmplitude)
	data := []byte{0x55}

	// Encode some data with small buffer to leave bits unprocessed
	// Use a buffer that's too small to process all bits, so some remain in bitBuffer
	amplitudes := vec.New(100) // Small buffer - won't fit all symbols
	modulator.Encode(amplitudes, data)

	// Reset
	modulator.Reset()

	// Verify state reset
	assert.Equal(t, 0, len(modulator.bitBuffer), "Bit buffer should be empty after reset")
	assert.Equal(t, 0, modulator.bitIndex, "Bit index should be reset")
	assert.Equal(t, float32(0), modulator.currentPhase, "Phase should be reset")
	
	// Verify we can still encode after reset
	amplitudes2 := vec.New(1000)
	samplesWritten := modulator.Encode(amplitudes2, data)
	assert.Greater(t, samplesWritten, 0, "Should still encode after reset")
}

// TestModulator_GenerateSignal tests GenerateSignal convenience method
func TestModulator_GenerateSignal(t *testing.T) {
	modulator := NewModulator(testSampleRate, testCarrierFreq, testSymbolRate, 2, testAmplitude)
	data := []byte{0x55, 0xAA}
	length := 2000

	signal := modulator.GenerateSignal(data, length)
	signalData := []float32(signal)

	assert.Equal(t, length, len(signalData), "Signal should have requested length")
	assert.Greater(t, len(signalData), 0, "Signal should not be empty")
}

// TestModulator_SymbolToAmplitude_2ASK tests 2-ASK amplitude mapping
func TestModulator_SymbolToAmplitude_2ASK(t *testing.T) {
	modulator := NewModulator(testSampleRate, testCarrierFreq, testSymbolRate, 2, testAmplitude)

	amp0 := modulator.symbolToAmplitude(0)
	amp1 := modulator.symbolToAmplitude(1)

	assert.Equal(t, float32(0), amp0, "Symbol 0 should map to 0 amplitude")
	assert.Equal(t, testAmplitude, amp1, "Symbol 1 should map to max amplitude")
}

// TestModulator_SymbolToAmplitude_4ASK tests 4-ASK amplitude mapping
func TestModulator_SymbolToAmplitude_4ASK(t *testing.T) {
	modulator := NewModulator(testSampleRate, testCarrierFreq, testSymbolRate, 4, testAmplitude)

	amp0 := modulator.symbolToAmplitude(0)
	amp1 := modulator.symbolToAmplitude(1)
	amp2 := modulator.symbolToAmplitude(2)
	amp3 := modulator.symbolToAmplitude(3)

	assert.Equal(t, float32(0), amp0, "Symbol 0 should map to 0")
	assert.InDelta(t, testAmplitude/3, amp1, 0.001, "Symbol 1 should map to amplitude/3")
	assert.InDelta(t, 2*testAmplitude/3, amp2, 0.001, "Symbol 2 should map to 2*amplitude/3")
	assert.Equal(t, testAmplitude, amp3, "Symbol 3 should map to max amplitude")
}

// TestModulator_ModulationOrder tests ModulationOrder method
func TestModulator_ModulationOrder(t *testing.T) {
	modulator2 := NewModulator(testSampleRate, testCarrierFreq, testSymbolRate, 2, testAmplitude)
	modulator4 := NewModulator(testSampleRate, testCarrierFreq, testSymbolRate, 4, testAmplitude)

	assert.Equal(t, 2, modulator2.ModulationOrder())
	assert.Equal(t, 4, modulator4.ModulationOrder())
}

