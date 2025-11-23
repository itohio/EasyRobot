package qam

import (
	"math"
	"testing"

	"github.com/itohio/EasyRobot/x/math/dsp"
	"github.com/itohio/EasyRobot/x/math/dsp/modulation"
	"github.com/itohio/EasyRobot/x/math/vec"
	vecTypes "github.com/itohio/EasyRobot/x/math/vec/types"
	"github.com/stretchr/testify/assert"
)

const (
	testSampleRate  = float32(48000)
	testCarrierFreq = float32(1000)
	testSymbolRate  = float32(1000)
	testAmplitude   = float32(1.0)
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

// TestRoundtrip_16QAM_Clean tests round-trip encoding/decoding for 16-QAM without noise
func TestRoundtrip_16QAM_Clean(t *testing.T) {
	encoder := NewEncoder(testSampleRate, testCarrierFreq, testSymbolRate, testAmplitude)
	decoder := NewDecoder(testSampleRate, testCarrierFreq, testSymbolRate)

	// Test data
	testData := []byte{0x55, 0xAA, 0x12, 0x34}

	// Encode with padding using helper
	paddingSamples := int(testSampleRate * 0.5)
	signal := modulation.ModulateWithPadding(t, encoder, testData, paddingSamples, 10) // Chunk size: 10 bytes

	// Decode with chunks using helper
	decoded := modulation.DemodulateWithChunks(t, decoder, signal, 1000, 0)

	// Should decode some bytes
	assert.Greater(t, len(decoded), 0, "Should decode bytes without noise")

	// Verify decoded data matches input exactly (may have extra bytes from padding)
	modulation.VerifyExactMatch(t, testData, decoded)
}

// TestRoundtrip_16QAM_WithNoise tests round-trip encoding/decoding for 16-QAM with noise
func TestRoundtrip_16QAM_WithNoise(t *testing.T) {
	encoder := NewEncoder(testSampleRate, testCarrierFreq, testSymbolRate, testAmplitude)
	decoder := NewDecoder(testSampleRate, testCarrierFreq, testSymbolRate)

	// Test data
	testData := []byte{0x55, 0xAA}

	// Encode with padding using helper
	paddingSamples := int(testSampleRate * 0.5)
	signal := modulation.ModulateWithPadding(t, encoder, testData, paddingSamples, 5) // Chunk size: 5 bytes

	// Add noise (18dB SNR - increased by 3dB for M=16 since 16-QAM is more sensitive to noise)
	noisySignal := addNoise(signal, 18.0)

	// Reset decoder for clean state
	decoder.Reset()

	// Decode with chunks using helper
	decoded := modulation.DemodulateWithChunks(t, decoder, noisySignal, 1000, 0)

	// Should decode some bytes even with noise
	assert.Greater(t, len(decoded), 0, "Should decode bytes with 18dB SNR (M=16 requires ~3dB more SNR)")

	// Verify decoded data matches input exactly (may have extra bytes from padding)
	modulation.VerifyExactMatch(t, testData, decoded)
}

// TestRoundtrip_16QAM_12dB_SNR tests 16-QAM at 15dB SNR (12dB + 3dB for M=16 sensitivity)
func TestRoundtrip_16QAM_12dB_SNR(t *testing.T) {
	encoder := NewEncoder(testSampleRate, testCarrierFreq, testSymbolRate, testAmplitude)
	decoder := NewDecoder(testSampleRate, testCarrierFreq, testSymbolRate)

	// Test data - use simple pattern for easier verification
	testData := []byte{0xAA} // 10101010 - alternating pattern

	// Encode with padding using helper
	paddingSamples := int(testSampleRate * 0.5)
	signal := modulation.ModulateWithPadding(t, encoder, testData, paddingSamples, 0) // Encode all at once

	// Add 15dB SNR noise (increased by 3dB for M=16 since 16-QAM is more sensitive to noise)
	noisySignal := addNoise(signal, 15.0)

	// Reset decoder for clean state
	decoder.Reset()

	// Decode with chunks using helper
	decoded := modulation.DemodulateWithChunks(t, decoder, noisySignal, 1000, 0)

	// At 15dB SNR (12dB + 3dB for M=16), we should be able to decode something
	assert.Greater(t, len(decoded), 0, "Should decode bytes at 15dB SNR (M=16 requires ~3dB more SNR)")

	// Verify decoded data matches input exactly (may have extra bytes from padding)
	modulation.VerifyExactMatch(t, testData, decoded)
}

// TestRoundtrip_16QAM_Padding tests that padding is handled correctly
func TestRoundtrip_16QAM_Padding(t *testing.T) {
	encoder := NewEncoder(testSampleRate, testCarrierFreq, testSymbolRate, testAmplitude)
	decoder := NewDecoder(testSampleRate, testCarrierFreq, testSymbolRate)

	testData := []byte{0x55}

	// Encode with padding using helper
	paddingSamples := int(testSampleRate * 0.5)
	signal := modulation.ModulateWithPadding(t, encoder, testData, paddingSamples, 0) // Encode all at once

	// Verify padding exists
	paddedData := []float32(signal.(vec.Vector))
	leftPadZeros := 0
	for i := 0; i < paddingSamples && i < len(paddedData); i++ {
		if math.Abs(float64(paddedData[i])) < 0.001 {
			leftPadZeros++
		}
	}
	assert.Greater(t, leftPadZeros, paddingSamples/2, "Left padding should be mostly zeros")

	// Reset decoder
	decoder.Reset()

	// Decode with chunks using helper
	decoded := modulation.DemodulateWithChunks(t, decoder, signal, 1000, 0)

	// Should decode some bytes even with padding
	assert.Greater(t, len(decoded), 0, "Should decode bytes with padding")

	// Verify decoded data matches input exactly (may have extra bytes from padding)
	modulation.VerifyExactMatch(t, testData, decoded)
}

// TestRoundtrip_16QAM_PaddedWithNoise tests round-trip with padding and 12dB SNR
func TestRoundtrip_16QAM_PaddedWithNoise(t *testing.T) {
	encoder := NewEncoder(testSampleRate, testCarrierFreq, testSymbolRate, testAmplitude)
	decoder := NewDecoder(testSampleRate, testCarrierFreq, testSymbolRate)

	testData := []byte{0x12, 0x34}

	// Encode with padding using helper
	paddingSamples := int(testSampleRate * 0.5)
	signal := modulation.ModulateWithPadding(t, encoder, testData, paddingSamples, 5) // Chunk size: 5 bytes

	// Add 15dB SNR noise (increased by 3dB for M=16 since 16-QAM is more sensitive to noise)
	noisySignal := addNoise(signal, 15.0)

	// Reset decoder
	decoder.Reset()

	// Decode with chunks using helper
	decoded := modulation.DemodulateWithChunks(t, decoder, noisySignal, 1000, 0)

	// Should decode at least some bytes at 15dB SNR (12dB + 3dB for M=16) with padding
	assert.Greater(t, len(decoded), 0, "Should decode bytes at 15dB SNR with padding (M=16 requires ~3dB more SNR)")

	// Verify decoded data matches input (may have some errors due to noise)
	minLen := len(decoded)
	if len(testData) < minLen {
		minLen = len(testData)
	}
	if minLen > 0 {
		matches := 0
		for i := 0; i < minLen; i++ {
			if testData[i] == decoded[i] {
				matches++
			}
		}
		modulation.VerifyExactMatch(t, testData, decoded)
	}
}

// TestRoundtrip_16QAM_MultiplePackets tests multiple packets with padding between them
func TestRoundtrip_16QAM_MultiplePackets(t *testing.T) {
	encoder := NewEncoder(testSampleRate, testCarrierFreq, testSymbolRate, testAmplitude)
	decoder := NewDecoder(testSampleRate, testCarrierFreq, testSymbolRate)

	// First packet
	testData1 := []byte{0x55, 0xAA}
	paddingSamples := int(testSampleRate * 0.5)
	signal1 := modulation.ModulateWithPadding(t, encoder, testData1, paddingSamples, 0)

	// Second packet
	encoder.Reset()
	testData2 := []byte{0x12, 0x34}
	signal2 := modulation.ModulateWithPadding(t, encoder, testData2, paddingSamples, 0)

	// Combine signals
	signal1Data := []float32(signal1.(vec.Vector))
	signal2Data := []float32(signal2.(vec.Vector))
	combined := vec.New(len(signal1Data) + len(signal2Data))
	combinedData := []float32(combined)
	copy(combinedData, signal1Data)
	copy(combinedData[len(signal1Data):], signal2Data)

	// Decode with chunks using helper
	decoded := modulation.DemodulateWithChunks(t, decoder, combined, 1000, 0)

	// Verify we decoded something from packets
	assert.Greater(t, len(decoded), 0, "Should decode bytes from multiple packets")
}
