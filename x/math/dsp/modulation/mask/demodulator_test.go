package mask

import (
	"testing"

	"github.com/itohio/EasyRobot/x/math/vec"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestDemodulator_NewDemodulator tests demodulator creation
func TestDemodulator_NewDemodulator(t *testing.T) {
	demodulator := NewDemodulator(testSampleRate, testCarrierFreq, testSymbolRate, 2)

	require.NotNil(t, demodulator)
	assert.Equal(t, testSampleRate, demodulator.sampleRate)
	assert.Equal(t, testCarrierFreq, demodulator.carrierFreq)
	assert.Equal(t, testSymbolRate, demodulator.symbolRate)
	assert.Equal(t, 2, demodulator.modulationOrder)
	assert.Equal(t, int(testSampleRate/testSymbolRate), demodulator.samplesPerSymbol)
}

// TestDemodulator_Decode_NilData tests Decode with nil data (size query)
func TestDemodulator_Decode_NilData(t *testing.T) {
	demodulator := NewDemodulator(testSampleRate, testCarrierFreq, testSymbolRate, 2)
	amplitudes := vec.New(1000)

	expectedSize := demodulator.Decode(nil, amplitudes)
	assert.GreaterOrEqual(t, expectedSize, 0, "Expected size should be non-negative")
}

// TestDemodulator_Reset tests reset functionality
func TestDemodulator_Reset(t *testing.T) {
	demodulator := NewDemodulator(testSampleRate, testCarrierFreq, testSymbolRate, 2)

	// Process some samples to change state
	amplitudes := vec.New(1000)
	ampData := []float32(amplitudes)
	for i := range ampData {
		ampData[i] = 0.5
	}
	demodulator.ProcessChunk(ampData, make([]byte, 100))

	// Verify state changed
	assert.Greater(t, len(demodulator.sampleBuffer), 0, "Sample buffer should have data")

	// Reset
	demodulator.Reset()

	// Verify state reset
	assert.Equal(t, 0, len(demodulator.sampleBuffer), "Sample buffer should be empty")
	assert.Equal(t, 0, demodulator.bitBuffer.GetWrittenBits(), "Bit buffer should be empty")
	assert.Equal(t, float32(0), demodulator.currentPhase, "Phase should be reset")
	assert.False(t, demodulator.locked, "Should not be locked after reset")
}

// TestDemodulator_OnIsEndOfPacket tests end of packet callback
func TestDemodulator_OnIsEndOfPacket(t *testing.T) {
	demodulator := NewDemodulator(testSampleRate, testCarrierFreq, testSymbolRate, 2)

	callback := func(data []byte) bool {
		return len(data) >= 4
	}

	demodulator.OnIsEndOfPacket(callback)
	assert.NotNil(t, demodulator.endOfPacket, "Callback should be set")
}

// TestDemodulator_QuantizeAmplitude_2ASK tests 2-ASK amplitude quantization
func TestDemodulator_QuantizeAmplitude_2ASK(t *testing.T) {
	demodulator := NewDemodulator(testSampleRate, testCarrierFreq, testSymbolRate, 2)

	symbol0 := demodulator.quantizeAmplitude(0.1)
	symbol1 := demodulator.quantizeAmplitude(0.9)

	assert.Equal(t, 0, symbol0, "Low amplitude should quantize to 0")
	assert.Equal(t, 1, symbol1, "High amplitude should quantize to 1")
}

// TestDemodulator_QuantizeAmplitude_4ASK tests 4-ASK amplitude quantization
func TestDemodulator_QuantizeAmplitude_4ASK(t *testing.T) {
	demodulator := NewDemodulator(testSampleRate, testCarrierFreq, testSymbolRate, 4)

	symbol0 := demodulator.quantizeAmplitude(0.0)
	symbol1 := demodulator.quantizeAmplitude(0.25)
	symbol2 := demodulator.quantizeAmplitude(0.5)
	symbol3 := demodulator.quantizeAmplitude(0.75)

	assert.Equal(t, 0, symbol0, "Very low amplitude should quantize to 0")
	assert.Equal(t, 1, symbol1, "Low amplitude should quantize to 1")
	assert.Equal(t, 2, symbol2, "Medium amplitude should quantize to 2")
	assert.Equal(t, 3, symbol3, "High amplitude should quantize to 3")
}
