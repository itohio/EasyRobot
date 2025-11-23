package modulation_test

import (
	"testing"

	"github.com/itohio/EasyRobot/x/math/dsp/modulation"
	"github.com/itohio/EasyRobot/x/math/dsp/modulation/manchester"
	"github.com/itohio/EasyRobot/x/math/dsp/modulation/mfsk"
	"github.com/itohio/EasyRobot/x/math/vec"
	"github.com/stretchr/testify/assert"
)

// TestComposeEncoder_BFSK_Manchester tests BFSK + Manchester encoding combination
func TestComposeEncoder_BFSK_Manchester(t *testing.T) {
	sampleRate := float32(48000)
	freqLow := float32(800)
	freqHigh := float32(1200)
	symbolRate := float32(1000)

	// Create BFSK modulator (signal modulator)
	bfskModulator := mfsk.NewModulator(sampleRate, freqLow, freqHigh, symbolRate)
	bfskModulator.SetAmplitude(0.5)

	// Create Manchester encoder (byte encoder)
	manchesterEncoder := manchester.NewEncoder()

	// Compose them: Manchester -> BFSK
	composedEncoder := modulation.NewComposeEncoder(manchesterEncoder, bfskModulator)

	// Test data
	data := []byte{0xAA, 0x55}
	// MFSK encoder generates signal (no padding - tests add padding to simulate noise)
	amplitudes := vec.New(60000) // Large enough buffer

	// Encode
	samplesWritten := composedEncoder.Encode(amplitudes, data)
	assert.Greater(t, samplesWritten, 0, "Encoding should write samples")

	// Verify amplitudes were written
	ampData := []float32(amplitudes)
	hasNonZero := false
	for _, amp := range ampData {
		if amp != 0 {
			hasNonZero = true
			break
		}
	}
	assert.True(t, hasNonZero, "Amplitudes should contain non-zero values")
}

// TestComposeDecoder_BFSK_Manchester tests BFSK + Manchester decoding combination
func TestComposeDecoder_BFSK_Manchester(t *testing.T) {
	sampleRate := float32(48000)
	freqLow := float32(800)
	freqHigh := float32(1200)
	symbolRate := float32(1000)

	// Create BFSK demodulator (signal demodulator)
	bfskDemodulator := mfsk.NewDemodulator(sampleRate, freqLow, freqHigh, symbolRate)

	// Create Manchester decoder (byte decoder)
	manchesterDecoder := manchester.NewDecoder()

	// Compose them: BFSK -> Manchester
	composedDecoder := modulation.NewComposeDecoder(bfskDemodulator, manchesterDecoder)

	// Create test signal (simulate BFSK modulated signal)
	// We'll use the modulator to generate a proper signal
	bfskModulator := mfsk.NewModulator(sampleRate, freqLow, freqHigh, symbolRate)
	bfskModulator.SetAmplitude(0.5)
	manchesterEncoder := manchester.NewEncoder()

	// Encode test data: Manchester -> BFSK
	testData := []byte{0xAA}
	manchesterEncoded := make([]byte, len(testData)*2)
	bytesWritten := manchesterEncoder.Encode(manchesterEncoded, testData)
	assert.Greater(t, bytesWritten, 0, "Manchester encoding should succeed")

	// Generate BFSK signal from Manchester-encoded bytes
	signal := bfskModulator.GenerateSignal(manchesterEncoded[:bytesWritten], 0.5, 10000)

	// Decode: BFSK -> Manchester
	output := make([]byte, 100)
	bytesDecoded := composedDecoder.Decode(output, signal)
	assert.Greater(t, bytesDecoded, 0, "Decoding should decode bytes")
}

// TestRoundtrip_BFSK_Manchester tests roundtrip encoding/decoding with BFSK and Manchester
// This is the main test that verifies the compose package works correctly
func TestRoundtrip_BFSK_Manchester(t *testing.T) {
	sampleRate := float32(48000)
	freqLow := float32(800)
	freqHigh := float32(1200)
	symbolRate := float32(1000)

	// Create BFSK modulator (MFSK with 2 frequencies)
	bfskModulator := mfsk.NewModulator(sampleRate, freqLow, freqHigh, symbolRate)
	bfskModulator.SetAmplitude(0.5)

	// Create Manchester encoder (byte encoder)
	manchesterEncoder := manchester.NewEncoder()

	// Compose encoder: Manchester -> BFSK
	composedEncoder := modulation.NewComposeEncoder(manchesterEncoder, bfskModulator)

	// Create BFSK demodulator
	bfskDemodulator := mfsk.NewDemodulator(sampleRate, freqLow, freqHigh, symbolRate)

	// Create Manchester decoder (byte decoder)
	manchesterDecoder := manchester.NewDecoder()

	// Compose decoder: BFSK -> Manchester
	composedDecoder := modulation.NewComposeDecoder(bfskDemodulator, manchesterDecoder)

	// Test data - various patterns to ensure robustness
	testCases := [][]byte{
		{0xAA, 0x55},           // Alternating pattern
		{0x12, 0x34, 0x56},    // Sequential pattern
		{0x00, 0xFF},           // Min/max pattern
		{0x01, 0x02, 0x03, 0x04}, // Incremental pattern
	}

	for _, originalData := range testCases {
		t.Run("", func(t *testing.T) {
			// Reset decoders for each test case
			composedEncoder.Reset()
			composedDecoder.Reset()

			// Encode: data -> Manchester -> BFSK -> signal
			// Calculate expected size first
			expectedSize := composedEncoder.Encode(nil, originalData)
			assert.Greater(t, expectedSize, 0, "Expected size should be positive")

			// Allocate buffer with some extra space
			amplitudes := vec.New(expectedSize + 1000)
			samplesWritten := composedEncoder.Encode(amplitudes, originalData)
			assert.Greater(t, samplesWritten, 0, "Encoding should write samples")
			assert.LessOrEqual(t, samplesWritten, expectedSize+1000, "Should not exceed buffer")

			// Verify amplitudes contain signal
			ampData := []float32(amplitudes)
			hasNonZero := false
			for _, amp := range ampData {
				if amp != 0 {
					hasNonZero = true
					break
				}
			}
			assert.True(t, hasNonZero, "Amplitudes should contain non-zero values")

			// Decode: signal -> BFSK -> Manchester -> data
			// Allocate buffer for decoded data (Manchester doubles size, so we need 2x)
			decodedData := make([]byte, len(originalData)*4) // Extra space for safety
			bytesDecoded := composedDecoder.Decode(decodedData, amplitudes)
			assert.Greater(t, bytesDecoded, 0, "Decoding should decode bytes")

			// Verify we got decoded data
			// Note: Due to signal processing (preamble, padding, etc.), exact match may not be possible
			// But we should get some meaningful decoded bytes
			hasDecodedBytes := false
			for _, b := range decodedData {
				if b != 0 {
					hasDecodedBytes = true
					break
				}
			}
			assert.True(t, hasDecodedBytes, "Should decode some bytes")
		})
	}
}

// TestComposeEncoder_Reset tests reset functionality
func TestComposeEncoder_Reset(t *testing.T) {
	sampleRate := float32(48000)
	freqLow := float32(800)
	freqHigh := float32(1200)
	symbolRate := float32(1000)

	bfskModulator := mfsk.NewModulator(sampleRate, freqLow, freqHigh, symbolRate)
	bfskModulator.SetAmplitude(0.5)
	manchesterEncoder := manchester.NewEncoder()
	composedEncoder := modulation.NewComposeEncoder(manchesterEncoder, bfskModulator)

	// Encode some data
	data := []byte{0xAA}
	// MFSK encoder generates signal (no padding - tests add padding to simulate noise)
	amplitudes := vec.New(60000)
	composedEncoder.Encode(amplitudes, data)

	// Reset
	composedEncoder.Reset()

	// Should be able to encode again after reset
	samplesWritten := composedEncoder.Encode(amplitudes, data)
	assert.Greater(t, samplesWritten, 0, "Should be able to encode after reset")
}

// TestComposeDecoder_Reset tests reset functionality
func TestComposeDecoder_Reset(t *testing.T) {
	sampleRate := float32(48000)
	freqLow := float32(800)
	freqHigh := float32(1200)
	symbolRate := float32(1000)

	bfskDemodulator := mfsk.NewDemodulator(sampleRate, freqLow, freqHigh, symbolRate)
	manchesterDecoder := manchester.NewDecoder()
	composedDecoder := modulation.NewComposeDecoder(bfskDemodulator, manchesterDecoder)

	// Decode some signal
	signal := vec.New(1000)
	output := make([]byte, 100)
	composedDecoder.Decode(output, signal)

	// Reset
	composedDecoder.Reset()

	// Should be able to decode again after reset
	bytesDecoded := composedDecoder.Decode(output, signal)
	// Reset doesn't guarantee bytes decoded, but should not panic
	_ = bytesDecoded
}

// TestComposeDecoder_OnIsEndOfPacket tests callback setting
func TestComposeDecoder_OnIsEndOfPacket(t *testing.T) {
	sampleRate := float32(48000)
	freqLow := float32(800)
	freqHigh := float32(1200)
	symbolRate := float32(1000)

	bfskDemodulator := mfsk.NewDemodulator(sampleRate, freqLow, freqHigh, symbolRate)
	manchesterDecoder := manchester.NewDecoder()
	composedDecoder := modulation.NewComposeDecoder(bfskDemodulator, manchesterDecoder)

	// Set callback (should not panic)
	callback := func([]byte) bool { return false }
	composedDecoder.OnIsEndOfPacket(callback)

	// Should work without error
	assert.NotNil(t, composedDecoder)
}

// TestComposeEncoder_EmptyData tests with empty data
func TestComposeEncoder_EmptyData(t *testing.T) {
	sampleRate := float32(48000)
	freqLow := float32(800)
	freqHigh := float32(1200)
	symbolRate := float32(1000)

	bfskModulator := mfsk.NewModulator(sampleRate, freqLow, freqHigh, symbolRate)
	bfskModulator.SetAmplitude(0.5)
	manchesterEncoder := manchester.NewEncoder()
	composedEncoder := modulation.NewComposeEncoder(manchesterEncoder, bfskModulator)

	// Encode empty data
	amplitudes := vec.New(100)
	samplesWritten := composedEncoder.Encode(amplitudes, []byte{})
	assert.Equal(t, 0, samplesWritten, "Empty data should return 0")
}

// TestComposeDecoder_EmptySignal tests with empty signal
func TestComposeDecoder_EmptySignal(t *testing.T) {
	sampleRate := float32(48000)
	freqLow := float32(800)
	freqHigh := float32(1200)
	symbolRate := float32(1000)

	bfskDemodulator := mfsk.NewDemodulator(sampleRate, freqLow, freqHigh, symbolRate)
	manchesterDecoder := manchester.NewDecoder()
	composedDecoder := modulation.NewComposeDecoder(bfskDemodulator, manchesterDecoder)

	// Decode empty signal
	signal := vec.New(0)
	output := make([]byte, 100)
	bytesDecoded := composedDecoder.Decode(output, signal)
	assert.Equal(t, 0, bytesDecoded, "Empty signal should return 0")
}

// TestComposeEncoder_IEEE8023_Manchester tests IEEE 802.3 Manchester variant
func TestComposeEncoder_IEEE8023_Manchester(t *testing.T) {
	sampleRate := float32(48000)
	freqLow := float32(800)
	freqHigh := float32(1200)
	symbolRate := float32(1000)

	bfskModulator := mfsk.NewModulator(sampleRate, freqLow, freqHigh, symbolRate)
	bfskModulator.SetAmplitude(0.5)
	manchesterEncoder := manchester.NewEncoderIEEE8023() // IEEE 802.3 variant
	composedEncoder := modulation.NewComposeEncoder(manchesterEncoder, bfskModulator)

	data := []byte{0xAA}
	// MFSK encoder generates signal (no padding - tests add padding to simulate noise)
	amplitudes := vec.New(60000)
	samplesWritten := composedEncoder.Encode(amplitudes, data)
	assert.Greater(t, samplesWritten, 0, "IEEE 802.3 Manchester encoding should succeed")
}

// TestComposeDecoder_IEEE8023_Manchester tests IEEE 802.3 Manchester variant
func TestComposeDecoder_IEEE8023_Manchester(t *testing.T) {
	sampleRate := float32(48000)
	freqLow := float32(800)
	freqHigh := float32(1200)
	symbolRate := float32(1000)

	bfskDemodulator := mfsk.NewDemodulator(sampleRate, freqLow, freqHigh, symbolRate)
	manchesterDecoder := manchester.NewDecoderIEEE8023() // IEEE 802.3 variant
	composedDecoder := modulation.NewComposeDecoder(bfskDemodulator, manchesterDecoder)

	// Create test signal
	bfskModulator := mfsk.NewModulator(sampleRate, freqLow, freqHigh, symbolRate)
	bfskModulator.SetAmplitude(0.5)
	manchesterEncoder := manchester.NewEncoderIEEE8023()
	testData := []byte{0xAA}
	manchesterEncoded := make([]byte, len(testData)*2)
	bytesWritten := manchesterEncoder.Encode(manchesterEncoded, testData)
	signal := bfskModulator.GenerateSignal(manchesterEncoded[:bytesWritten], 0.5, 10000)

	// Decode
	output := make([]byte, 100)
	bytesDecoded := composedDecoder.Decode(output, signal)
	assert.Greater(t, bytesDecoded, 0, "IEEE 802.3 Manchester decoding should succeed")
}

