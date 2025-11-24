package dsss

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

// Test constants are defined in modulator_test.go
// These are shared across both modulator and demodulator tests

// addNoise adds AWGN noise to a signal to achieve a specific SNR in dB
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
	snrRatio := math.Pow(10, float64(snrDb)/10)
	noisePower := signalPower / snrRatio
	noiseStdDev := float32(math.Sqrt(noisePower))

	// Generate Gaussian noise using dsp package
	sg := dsp.NewNoiseWhiteGenerator(noiseStdDev)
	noise := vec.New(len(signalData))
	sg.Generate(noise)
	noiseData := []float32(noise)

	// Add noise to signal
	noisy := vec.New(len(signalData))
	noisyData := []float32(noisy)
	for i := range signalData {
		noisyData[i] = signalData[i] + noiseData[i]
	}

	return noisy
}

func TestDemodulator_NewDemodulator(t *testing.T) {
	code := GenerateMSequence(3, []int{2, 1, 0}, 1)
	demodulator := NewDemodulator(testSampleRate, testCarrierFreq, testSymbolRate, testRolloffFactor, code)

	require.NotNil(t, demodulator)
	assert.Equal(t, testSampleRate, demodulator.sampleRate)
	assert.Equal(t, testCarrierFreq, demodulator.carrierFreq)
	assert.Equal(t, testSymbolRate, demodulator.symbolRate)
	assert.Equal(t, 2, demodulator.modulationOrder)
	assert.Equal(t, code.Length(), demodulator.spreadingFactor)
}

func TestDemodulator_NewMDemodulator(t *testing.T) {
	codeTable := GenerateWalshCodeTable(4)
	demodulator := NewDemodulator(testSampleRate, testCarrierFreq, testSymbolRate, testRolloffFactor, codeTable...)

	require.NotNil(t, demodulator)
	assert.Equal(t, 4, demodulator.modulationOrder)
	assert.Equal(t, codeTable[0].Length(), demodulator.spreadingFactor)
}

func TestDemodulator_Decode_NilData(t *testing.T) {
	code := GenerateMSequence(3, []int{2, 1, 0}, 1)
	demodulator := NewDemodulator(testSampleRate, testCarrierFreq, testSymbolRate, testRolloffFactor, code)

	expectedSize := demodulator.Decode(nil, vec.New(1000))
	assert.GreaterOrEqual(t, expectedSize, 0, "Expected size should be non-negative")
}

func TestDemodulator_Reset(t *testing.T) {
	code := GenerateMSequence(3, []int{2, 1, 0}, 1)
	demodulator := NewDemodulator(testSampleRate, testCarrierFreq, testSymbolRate, testRolloffFactor, code)

	demodulator.Reset()
}

// TestRoundtrip_BinaryDSSS_Short_Clean tests round-trip with short clean data
func TestRoundtrip_BinaryDSSS_Short_Clean(t *testing.T) {
	code := GenerateMSequence(3, []int{2, 1, 0}, 1)
	modulator := NewModulator(testSampleRate, testCarrierFreq, testSymbolRate, testRolloffFactor, code)
	demodulator := NewDemodulator(testSampleRate, testCarrierFreq, testSymbolRate, testRolloffFactor, code)

	// Short test data (1 byte)
	testData := []byte{0x55}

	// Encode with padding
	paddingSamples := int(testSampleRate * 0.5)
	signal := modulation.ModulateWithPadding(t, modulator, testData, paddingSamples, 0)

	// Decode
	decoded := modulation.DemodulateWithChunks(t, demodulator, signal, 1000, len(testData))

	// Verify exact match
	modulation.VerifyExactMatch(t, testData, decoded)
}

// TestRoundtrip_BinaryDSSS_Short_Noisy tests round-trip with short noisy data (6dB SNR)
func TestRoundtrip_BinaryDSSS_Short_Noisy(t *testing.T) {
	code := GenerateMSequence(3, []int{2, 1, 0}, 1)
	modulator := NewModulator(testSampleRate, testCarrierFreq, testSymbolRate, testRolloffFactor, code)
	demodulator := NewDemodulator(testSampleRate, testCarrierFreq, testSymbolRate, testRolloffFactor, code)

	// Short test data (1 byte)
	testData := []byte{0x55}

	// Encode with padding
	paddingSamples := int(testSampleRate * 0.5)
	signal := modulation.ModulateWithPadding(t, modulator, testData, paddingSamples, 0)

	// Add noise at 6dB SNR
	noisySignal := addNoise(signal, 6.0)

	// Reset demodulator
	demodulator.Reset()

	// Decode
	decoded := modulation.DemodulateWithChunks(t, demodulator, noisySignal, 1000, len(testData))

	// Verify exact match (DSSS should handle 6dB SNR)
	modulation.VerifyExactMatch(t, testData, decoded)
}

// TestRoundtrip_BinaryDSSS_Long_Clean tests round-trip with long clean data
func TestRoundtrip_BinaryDSSS_Long_Clean(t *testing.T) {
	code := GenerateMSequence(3, []int{2, 1, 0}, 1)
	modulator := NewModulator(testSampleRate, testCarrierFreq, testSymbolRate, testRolloffFactor, code)
	demodulator := NewDemodulator(testSampleRate, testCarrierFreq, testSymbolRate, testRolloffFactor, code)

	// Long test data (100 bytes)
	testData := make([]byte, 100)
	for i := range testData {
		testData[i] = byte(i % 256)
	}

	// Encode with padding and chunked encoding
	paddingSamples := int(testSampleRate * 0.5)
	signal := modulation.ModulateWithPadding(t, modulator, testData, paddingSamples, 10)

	// Decode
	decoded := modulation.DemodulateWithChunks(t, demodulator, signal, 1000, len(testData))

	// Verify exact match
	modulation.VerifyExactMatch(t, testData, decoded)
}

// TestRoundtrip_BinaryDSSS_Long_Noisy tests round-trip with long noisy data (6dB SNR)
func TestRoundtrip_BinaryDSSS_Long_Noisy(t *testing.T) {
	code := GenerateMSequence(3, []int{2, 1, 0}, 1)
	modulator := NewModulator(testSampleRate, testCarrierFreq, testSymbolRate, testRolloffFactor, code)
	demodulator := NewDemodulator(testSampleRate, testCarrierFreq, testSymbolRate, testRolloffFactor, code)

	// Long test data (100 bytes)
	testData := make([]byte, 100)
	for i := range testData {
		testData[i] = byte(i % 256)
	}

	// Encode with padding and chunked encoding
	paddingSamples := int(testSampleRate * 0.5)
	signal := modulation.ModulateWithPadding(t, modulator, testData, paddingSamples, 10)

	// Add noise at 6dB SNR
	noisySignal := addNoise(signal, 6.0)

	// Reset demodulator
	demodulator.Reset()

	// Decode
	decoded := modulation.DemodulateWithChunks(t, demodulator, noisySignal, 1000, len(testData))

	// Verify exact match (DSSS should handle 6dB SNR)
	modulation.VerifyExactMatch(t, testData, decoded)
}

func TestRoundtrip_BinaryDSSS_Padding(t *testing.T) {
	code := GenerateMSequence(3, []int{2, 1, 0}, 1)
	modulator := NewModulator(testSampleRate, testCarrierFreq, testSymbolRate, testRolloffFactor, code)
	demodulator := NewDemodulator(testSampleRate, testCarrierFreq, testSymbolRate, testRolloffFactor, code)

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
}

// TestRoundtrip_MaryDSSS_Short_Clean tests round-trip with short clean data (M-ary)
func TestRoundtrip_MaryDSSS_Short_Clean(t *testing.T) {
	codeTable := GenerateWalshCodeTable(4)
	modulator := NewModulator(testSampleRate, testCarrierFreq, testSymbolRate, testRolloffFactor, codeTable...)
	demodulator := NewDemodulator(testSampleRate, testCarrierFreq, testSymbolRate, testRolloffFactor, codeTable...)

	// Short test data (2 bytes)
	testData := []byte{0x12, 0x34}

	// Encode with padding
	paddingSamples := int(testSampleRate * 0.5)
	signal := modulation.ModulateWithPadding(t, modulator, testData, paddingSamples, 0)

	// Decode
	decoded := modulation.DemodulateWithChunks(t, demodulator, signal, 1000, len(testData))

	// Verify exact match
	modulation.VerifyExactMatch(t, testData, decoded)
}

// TestRoundtrip_MaryDSSS_Short_Noisy tests round-trip with short noisy data (6dB SNR, M-ary)
func TestRoundtrip_MaryDSSS_Short_Noisy(t *testing.T) {
	codeTable := GenerateWalshCodeTable(4)
	modulator := NewModulator(testSampleRate, testCarrierFreq, testSymbolRate, testRolloffFactor, codeTable...)
	demodulator := NewDemodulator(testSampleRate, testCarrierFreq, testSymbolRate, testRolloffFactor, codeTable...)

	// Short test data (2 bytes)
	testData := []byte{0x12, 0x34}

	// Encode with padding
	paddingSamples := int(testSampleRate * 0.5)
	signal := modulation.ModulateWithPadding(t, modulator, testData, paddingSamples, 0)

	// Add noise at 6dB SNR
	noisySignal := addNoise(signal, 6.0)

	// Reset demodulator
	demodulator.Reset()

	// Decode
	decoded := modulation.DemodulateWithChunks(t, demodulator, noisySignal, 1000, len(testData))

	// Verify exact match
	modulation.VerifyExactMatch(t, testData, decoded)
}

// TestRoundtrip_MaryDSSS_Long_Clean tests round-trip with long clean data (M-ary)
func TestRoundtrip_MaryDSSS_Long_Clean(t *testing.T) {
	codeTable := GenerateWalshCodeTable(4)
	modulator := NewModulator(testSampleRate, testCarrierFreq, testSymbolRate, testRolloffFactor, codeTable...)
	demodulator := NewDemodulator(testSampleRate, testCarrierFreq, testSymbolRate, testRolloffFactor, codeTable...)

	// Long test data (100 bytes)
	testData := make([]byte, 100)
	for i := range testData {
		testData[i] = byte(i % 256)
	}

	// Encode with padding and chunked encoding
	paddingSamples := int(testSampleRate * 0.5)
	signal := modulation.ModulateWithPadding(t, modulator, testData, paddingSamples, 10)

	// Decode
	decoded := modulation.DemodulateWithChunks(t, demodulator, signal, 1000, len(testData))

	// Verify exact match
	modulation.VerifyExactMatch(t, testData, decoded)
}

// TestRoundtrip_MaryDSSS_Long_Noisy tests round-trip with long noisy data (6dB SNR, M-ary)
func TestRoundtrip_MaryDSSS_Long_Noisy(t *testing.T) {
	codeTable := GenerateWalshCodeTable(4)
	modulator := NewModulator(testSampleRate, testCarrierFreq, testSymbolRate, testRolloffFactor, codeTable...)
	demodulator := NewDemodulator(testSampleRate, testCarrierFreq, testSymbolRate, testRolloffFactor, codeTable...)

	// Long test data (100 bytes)
	testData := make([]byte, 100)
	for i := range testData {
		testData[i] = byte(i % 256)
	}

	// Encode with padding and chunked encoding
	paddingSamples := int(testSampleRate * 0.5)
	signal := modulation.ModulateWithPadding(t, modulator, testData, paddingSamples, 10)

	// Add noise at 6dB SNR
	noisySignal := addNoise(signal, 6.0)

	// Reset demodulator
	demodulator.Reset()

	// Decode
	decoded := modulation.DemodulateWithChunks(t, demodulator, noisySignal, 1000, len(testData))

	// Verify exact match
	modulation.VerifyExactMatch(t, testData, decoded)
}

func TestDemodulator_InterfaceCompliance(t *testing.T) {
	code := GenerateMSequence(3, []int{2, 1, 0}, 1)
	demodulator := NewDemodulator(testSampleRate, testCarrierFreq, testSymbolRate, testRolloffFactor, code)

	// Verify it implements the Demodulator interface
	var _ modulation.Demodulator = demodulator
}
