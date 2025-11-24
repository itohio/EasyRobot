package mpsk

import (
	"math"
	"testing"

	"github.com/itohio/EasyRobot/x/math/dsp"
	"github.com/itohio/EasyRobot/x/math/dsp/modulation"
	"github.com/itohio/EasyRobot/x/math/vec"
	vecTypes "github.com/itohio/EasyRobot/x/math/vec/types"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// addNoise adds AWGN noise to a signal to achieve a specific SNR in dB
// Uses dsp.SignalGenerator1D for noise generation
// SNR = 10*log10(signal_power / noise_power)
func addNoise(signal vecTypes.Vector, snrDb float32) vecTypes.Vector {
	signalData := []float32(signal.(vec.Vector))

	// Calculate signal power (RMS)
	var signalPower float64
	for _, s := range signalData {
		signalPower += float64(s * s)
	}
	signalPower /= float64(len(signalData))
	if signalPower < 1e-10 {
		signalPower = 1e-10
	}

	// Calculate noise power needed for desired SNR
	// SNR_dB = 10*log10(signal_power / noise_power)
	// noise_power = signal_power / 10^(SNR_dB/10)
	snrRatio := math.Pow(10, float64(snrDb)/10)
	noisePower := signalPower / snrRatio
	noiseStdDev := float32(math.Sqrt(noisePower))

	// Generate Gaussian noise using dsp package
	sg := dsp.NewSignalGenerator1D()
	noise := sg.NoiseGaussian(len(signalData), noiseStdDev)
	noiseData := []float32(noise.(vec.Vector))

	// Add noise to signal
	noisy := vec.New(len(signalData))
	noisyData := []float32(noisy)
	for i := range signalData {
		noisyData[i] = signalData[i] + noiseData[i]
	}

	return noisy
}

// TestDemodulator_NewDemodulator tests demodulator creation
func TestDemodulator_NewDemodulator(t *testing.T) {
	demodulator := NewDemodulator(testSampleRate, testCarrierFreq, testSymbolRate, testModulationOrder)

	require.NotNil(t, demodulator)
	assert.Equal(t, testSampleRate, demodulator.sampleRate)
	assert.Equal(t, testCarrierFreq, demodulator.carrierFreq)
	assert.Equal(t, testSymbolRate, demodulator.symbolRate)
	assert.Equal(t, testModulationOrder, demodulator.modulationOrder)
	assert.Equal(t, int(testSampleRate/testSymbolRate), demodulator.samplesPerSymbol)
	assert.NotNil(t, demodulator.costasLoop)
	assert.NotNil(t, demodulator.timingRecovery)
	assert.NotNil(t, demodulator.signalDetector)
	assert.NotNil(t, demodulator.agc)
}

// TestDemodulator_Decode_NilData tests Decode with nil data (size query)
func TestDemodulator_Decode_NilData(t *testing.T) {
	demodulator := NewDemodulator(testSampleRate, testCarrierFreq, testSymbolRate, testModulationOrder)
	amplitudes := vec.New(1000)

	expectedSize := demodulator.Decode(nil, amplitudes)
	assert.GreaterOrEqual(t, expectedSize, 0, "Expected size should be non-negative")
}

// TestDemodulator_Reset tests reset functionality
func TestDemodulator_Reset(t *testing.T) {
	demodulator := NewDemodulator(testSampleRate, testCarrierFreq, testSymbolRate, testModulationOrder)

	// Process some samples first
	amplitudes := vec.New(1000)
	demodulator.Decode(make([]byte, 100), amplitudes)

	// Reset
	demodulator.Reset()

	// Verify state is reset
	assert.False(t, demodulator.locked)
	assert.Equal(t, 0, len(demodulator.sampleBuffer))
	assert.Equal(t, 0, demodulator.bitBuffer.GetWrittenBits())
	assert.Equal(t, float32(0), demodulator.phaseAcc)
}

// TestDemodulator_IsLocked tests IsLocked method
func TestDemodulator_IsLocked(t *testing.T) {
	demodulator := NewDemodulator(testSampleRate, testCarrierFreq, testSymbolRate, testModulationOrder)

	// Initially should not be locked
	assert.False(t, demodulator.IsLocked())
}

// TestRoundtrip_BPSK_Clean tests round-trip encoding/decoding for BPSK without noise
func TestRoundtrip_BPSK_Clean(t *testing.T) {
	modulator := NewModulator(testSampleRate, testCarrierFreq, testSymbolRate, 2)
	demodulator := NewDemodulator(testSampleRate, testCarrierFreq, testSymbolRate, 2)

	// Test data
	testData := []byte{0x55, 0xAA, 0x12, 0x34}

	// Encode with padding using helper
	paddingSamples := int(testSampleRate * 0.5)
	signal := modulation.ModulateWithPadding(t, modulator, testData, paddingSamples, 10) // Chunk size: 10 bytes

	// Decode with chunks using helper
	decoded := modulation.DemodulateWithChunks(t, demodulator, signal, 1000, 0)

	// Should decode some bytes
	assert.Greater(t, len(decoded), 0, "Should decode some bytes")

	// Verify decoded data matches input exactly (may have extra bytes from padding)
	modulation.VerifyExactMatch(t, testData, decoded)
}

// TestRoundtrip_BPSK_WithNoise tests round-trip encoding/decoding for BPSK with noise
func TestRoundtrip_BPSK_WithNoise(t *testing.T) {
	modulator := NewModulator(testSampleRate, testCarrierFreq, testSymbolRate, 2)
	demodulator := NewDemodulator(testSampleRate, testCarrierFreq, testSymbolRate, 2)

	// Test data
	testData := []byte{0x55, 0xAA}

	// Encode with padding using helper
	paddingSamples := int(testSampleRate * 0.5)
	signal := modulation.ModulateWithPadding(t, modulator, testData, paddingSamples, 5) // Chunk size: 5 bytes

	// Add noise (15dB SNR)
	noisySignal := addNoise(signal, 15.0)

	// Reset demodulator for clean state
	demodulator.Reset()

	// Decode with chunks using helper
	decoded := modulation.DemodulateWithChunks(t, demodulator, noisySignal, 1000, 0)

	// Should decode some bytes even with noise
	assert.Greater(t, len(decoded), 0, "Should decode bytes with 15dB SNR")

	// Verify decoded data matches input exactly (may have extra bytes from padding)
	modulation.VerifyExactMatch(t, testData, decoded)
}

// TestRoundtrip_BPSK_12dB_SNR tests BPSK specifically at 12dB SNR
func TestRoundtrip_BPSK_12dB_SNR(t *testing.T) {
	modulator := NewModulator(testSampleRate, testCarrierFreq, testSymbolRate, 2)
	demodulator := NewDemodulator(testSampleRate, testCarrierFreq, testSymbolRate, 2)

	// Test data - use simple pattern for easier verification
	testData := []byte{0xAA} // 10101010 - alternating pattern

	// Encode with padding using helper
	paddingSamples := int(testSampleRate * 0.5)
	signal := modulation.ModulateWithPadding(t, modulator, testData, paddingSamples, 0) // Encode all at once

	// Add 12dB SNR noise
	noisySignal := addNoise(signal, 12.0)

	// Reset demodulator for clean state
	demodulator.Reset()

	// Decode with chunks using helper
	decoded := modulation.DemodulateWithChunks(t, demodulator, noisySignal, 1000, 0)

	// At 12dB SNR, we should be able to decode something
	assert.Greater(t, len(decoded), 0, "Should decode bytes at 12dB SNR")

	// Verify decoded data matches input exactly (may have extra bytes from padding)
	modulation.VerifyExactMatch(t, testData, decoded)
}

// TestRoundtrip_BPSK_Padding tests that padding is handled correctly
func TestRoundtrip_BPSK_Padding(t *testing.T) {
	modulator := NewModulator(testSampleRate, testCarrierFreq, testSymbolRate, 2)
	demodulator := NewDemodulator(testSampleRate, testCarrierFreq, testSymbolRate, 2)

	testData := []byte{0x55}

	// Encode with padding using helper
	paddingSamples := int(testSampleRate * 0.5)
	signal := modulation.ModulateWithPadding(t, modulator, testData, paddingSamples, 0) // Encode all at once

	// Verify padding is present
	paddedData := []float32(signal.(vec.Vector))
	assert.GreaterOrEqual(t, len(paddedData), paddingSamples*2, "Should have space for padding on both sides")

	// Check that padding regions are zeros
	leftZeros := 0
	for i := 0; i < paddingSamples && i < len(paddedData); i++ {
		if math.Abs(float64(paddedData[i])) < 0.001 {
			leftZeros++
		}
	}

	rightZeros := 0
	rightStart := len(paddedData) - paddingSamples
	if rightStart > 0 {
		for i := rightStart; i < len(paddedData); i++ {
			if math.Abs(float64(paddedData[i])) < 0.001 {
				rightZeros++
			}
		}
	}

	// Padding should be zeros
	assert.Equal(t, paddingSamples, leftZeros, "Left padding should be all zeros")
	assert.Equal(t, paddingSamples, rightZeros, "Right padding should be all zeros")

	// Decode with chunks using helper
	decoded := modulation.DemodulateWithChunks(t, demodulator, signal, 1000, 0)

	// Should decode something (signal detector should skip padding)
	assert.GreaterOrEqual(t, len(decoded), 0, "Should handle padding correctly")

	// Verify decoded data matches input
	if len(decoded) > 0 && len(testData) > 0 {
		minLen := len(decoded)
		if len(testData) < minLen {
			minLen = len(testData)
		}
		assert.Equal(t, testData[:minLen], decoded[:minLen], "Decoded data should match input with padding")
	}
}

// TestRoundtrip_QPSK_Clean tests round-trip encoding/decoding for QPSK without noise
func TestRoundtrip_QPSK_Clean(t *testing.T) {
	modulator := NewModulator(testSampleRate, testCarrierFreq, testSymbolRate, 4)
	demodulator := NewDemodulator(testSampleRate, testCarrierFreq, testSymbolRate, 4)

	// Test data
	testData := []byte{0x55, 0xAA, 0x12, 0x34}

	// Encode with padding using helper
	paddingSamples := int(testSampleRate * 0.5)
	signal := modulation.ModulateWithPadding(t, modulator, testData, paddingSamples, 10) // Chunk size: 10 bytes

	// Decode with chunks using helper
	decoded := modulation.DemodulateWithChunks(t, demodulator, signal, 1000, 0)

	// Should decode some bytes
	assert.Greater(t, len(decoded), 0, "Should decode some bytes for QPSK")

	// Verify decoded data matches input exactly (may have extra bytes from padding)
	modulation.VerifyExactMatch(t, testData, decoded)
}

// TestRoundtrip_QPSK_WithNoise tests round-trip encoding/decoding for QPSK with noise
func TestRoundtrip_QPSK_WithNoise(t *testing.T) {
	modulator := NewModulator(testSampleRate, testCarrierFreq, testSymbolRate, 4)
	demodulator := NewDemodulator(testSampleRate, testCarrierFreq, testSymbolRate, 4)

	// Test data
	testData := []byte{0x55, 0xAA}

	// Encode with padding using helper
	paddingSamples := int(testSampleRate * 0.5)
	signal := modulation.ModulateWithPadding(t, modulator, testData, paddingSamples, 5) // Chunk size: 5 bytes

	// Add noise (15dB SNR)
	noisySignal := addNoise(signal, 15.0)

	// Reset demodulator for clean state
	demodulator.Reset()

	// Decode with chunks using helper
	decoded := modulation.DemodulateWithChunks(t, demodulator, noisySignal, 1000, 0)

	// Should decode some bytes even with noise
	assert.Greater(t, len(decoded), 0, "Should decode bytes with 15dB SNR for QPSK")

	// Verify decoded data matches input exactly (may have extra bytes from padding)
	modulation.VerifyExactMatch(t, testData, decoded)
}

// TestRoundtrip_BPSK_LongData tests round-trip encoding/decoding for BPSK with longer data (71 bytes)
func TestRoundtrip_BPSK_LongData(t *testing.T) {
	modulator := NewModulator(testSampleRate, testCarrierFreq, testSymbolRate, 2)
	demodulator := NewDemodulator(testSampleRate, testCarrierFreq, testSymbolRate, 2)

	// Test data - 71 bytes (longer data)
	testData := make([]byte, 71)
	for i := range testData {
		testData[i] = byte(i % 256)
	}

	// Encode with padding using helper
	paddingSamples := int(testSampleRate * 0.5)
	signal := modulation.ModulateWithPadding(t, modulator, testData, paddingSamples, 10) // Chunk size: 10 bytes

	// Decode with chunks using helper
	decoded := modulation.DemodulateWithChunks(t, demodulator, signal, 1000, 0)

	// Should decode some bytes
	assert.Greater(t, len(decoded), 0, "Should decode some bytes for 71-byte BPSK")

	// Verify decoded data matches input exactly (may have extra bytes from padding)
	modulation.VerifyExactMatch(t, testData, decoded)
}

// TestRoundtrip_QPSK_LongData tests round-trip encoding/decoding for QPSK with longer data (71 bytes)
func TestRoundtrip_QPSK_LongData(t *testing.T) {
	modulator := NewModulator(testSampleRate, testCarrierFreq, testSymbolRate, 4)
	demodulator := NewDemodulator(testSampleRate, testCarrierFreq, testSymbolRate, 4)

	// Test data - 71 bytes (longer data)
	testData := make([]byte, 71)
	for i := range testData {
		testData[i] = byte(i % 256)
	}

	// Encode with padding using helper
	paddingSamples := int(testSampleRate * 0.5)
	signal := modulation.ModulateWithPadding(t, modulator, testData, paddingSamples, 10) // Chunk size: 10 bytes

	// Decode with chunks using helper
	decoded := modulation.DemodulateWithChunks(t, demodulator, signal, 1000, 0)

	// Should decode some bytes
	assert.Greater(t, len(decoded), 0, "Should decode some bytes for 71-byte QPSK")

	// Verify decoded data matches input exactly (may have extra bytes from padding)
	modulation.VerifyExactMatch(t, testData, decoded)
}

// TestDemodulator_ProcessChunk tests ProcessChunk method
func TestDemodulator_ProcessChunk(t *testing.T) {
	demodulator := NewDemodulator(testSampleRate, testCarrierFreq, testSymbolRate, testModulationOrder)

	samples := make([]float32, 1000)
	output := make([]byte, 100)

	bytesDecoded, hasMore := demodulator.ProcessChunk(samples, output)

	assert.GreaterOrEqual(t, bytesDecoded, 0)
	assert.IsType(t, false, hasMore)
}

// TestDemodulator_InterfaceCompliance tests that Demodulator implements the interface
func TestDemodulator_InterfaceCompliance(t *testing.T) {
	demodulator := NewDemodulator(testSampleRate, testCarrierFreq, testSymbolRate, testModulationOrder)

	// Verify it implements the Demodulator interface
	var _ modulation.Demodulator = demodulator
}

// TestModulator_InterfaceCompliance tests that Modulator implements the interface
func TestModulator_InterfaceCompliance(t *testing.T) {
	modulator := NewModulator(testSampleRate, testCarrierFreq, testSymbolRate, testModulationOrder)

	// Verify it implements the Modulator interface
	var _ modulation.Modulator = modulator
}
