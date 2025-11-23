package dsss

import (
	"testing"

	"github.com/itohio/EasyRobot/x/math/dsp/modulation"
	"github.com/itohio/EasyRobot/x/math/vec"
	"github.com/stretchr/testify/require"
)

// Test parameters: 16 chips, 500 baud
const (
	roundtripChips        = 16
	roundtripBaud         = float32(500)
	roundtripCarrierFreq  = float32(1000)
	roundtripSampleRate   = float32(64000) // 8 samples/chip: 500 * 16 * 8 = 64000 Hz
)

// TestRoundtrip_1DSSS_16Chips_500Baud tests binary DSSS (1dsss) roundtrip
// Uses 16 chips and 500 baud rate
func TestRoundtrip_1DSSS_16Chips_500Baud(t *testing.T) {
	// Generate a code of length 16
	// For 16 chips, we can use Walsh code or m-sequence
	// Using Walsh code for deterministic orthogonal codes
	codeTable := GenerateWalshCodeTable(16)
	code := codeTable[0] // Use first Walsh code for binary DSSS
	
	require.Equal(t, roundtripChips, code.Length(), "Code length must be 16 chips")
	
	// Verify sample rate calculation
	chipRate := roundtripBaud * float32(roundtripChips)
	samplesPerChip := roundtripSampleRate / chipRate
	t.Logf("Sample rate: %.0f Hz, Chip rate: %.0f chips/sec, Samples per chip: %.2f", 
		roundtripSampleRate, chipRate, samplesPerChip)
	require.Greater(t, samplesPerChip, float32(4), "Must have at least 4 samples per chip")
	
	modulator := NewModulator(roundtripSampleRate, roundtripCarrierFreq, roundtripBaud, code)
	demodulator := NewDemodulator(roundtripSampleRate, roundtripCarrierFreq, roundtripBaud, code)
	
	// Test data: start with smaller data to debug
	testData := []byte{0x55, 0xAA}
	
	// Try without padding first to see if acquisition works
	// Calculate signal size
	expectedSize := modulator.Encode(nil, testData)
	require.Greater(t, expectedSize, 0, "Expected signal size must be positive")
	
	// Encode without padding
	signal := vec.New(expectedSize)
	samplesWritten := modulator.Encode(signal, testData)
	require.Equal(t, expectedSize, samplesWritten, "Should write all expected samples")
	
	t.Logf("Encoded %d samples for %d bytes of data", samplesWritten, len(testData))
	
	// Decode with smaller chunks to allow more acquisition attempts
	decoded := modulation.DemodulateWithChunks(t, demodulator, signal, 500, len(testData))
	
	// Verify exact match
	modulation.VerifyExactMatch(t, testData, decoded)
}

// TestRoundtrip_4DSSS_16Chips_500Baud tests 4-ary DSSS roundtrip
// Uses 16 chips and 500 baud rate
func TestRoundtrip_4DSSS_16Chips_500Baud(t *testing.T) {
	// Generate 4 Walsh codes of length 16
	codeTable := GenerateWalshCodeTable(16)
	// Use first 4 codes for 4-ary DSSS
	codeTable4 := codeTable[:4]
	
	require.Equal(t, 4, len(codeTable4), "Must have 4 codes")
	require.Equal(t, roundtripChips, codeTable4[0].Length(), "Code length must be 16 chips")
	
	modulator := NewModulator(roundtripSampleRate, roundtripCarrierFreq, roundtripBaud, codeTable4...)
	demodulator := NewDemodulator(roundtripSampleRate, roundtripCarrierFreq, roundtripBaud, codeTable4...)
	
	// Test data: 10 bytes (must be multiple of 2 bits per symbol for 4-ary)
	testData := make([]byte, 10)
	for i := range testData {
		testData[i] = byte(i % 256)
	}
	
	// Encode with padding
	paddingSamples := int(roundtripSampleRate * 0.5)
	signal := modulation.ModulateWithPadding(t, modulator, testData, paddingSamples, 0)
	
	// Decode
	decoded := modulation.DemodulateWithChunks(t, demodulator, signal, 1000, len(testData))
	
	// Verify exact match
	modulation.VerifyExactMatch(t, testData, decoded)
}

// TestRoundtrip_8DSSS_16Chips_500Baud tests 8-ary DSSS roundtrip
// Uses 16 chips and 500 baud rate
func TestRoundtrip_8DSSS_16Chips_500Baud(t *testing.T) {
	// Generate 8 Walsh codes of length 16
	codeTable := GenerateWalshCodeTable(16)
	// Use first 8 codes for 8-ary DSSS
	codeTable8 := codeTable[:8]
	
	require.Equal(t, 8, len(codeTable8), "Must have 8 codes")
	require.Equal(t, roundtripChips, codeTable8[0].Length(), "Code length must be 16 chips")
	
	modulator := NewModulator(roundtripSampleRate, roundtripCarrierFreq, roundtripBaud, codeTable8...)
	demodulator := NewDemodulator(roundtripSampleRate, roundtripCarrierFreq, roundtripBaud, codeTable8...)
	
	// Test data: 10 bytes (must be multiple of 3 bits per symbol for 8-ary)
	testData := make([]byte, 10)
	for i := range testData {
		testData[i] = byte(i % 256)
	}
	
	// Encode with padding
	paddingSamples := int(roundtripSampleRate * 0.5)
	signal := modulation.ModulateWithPadding(t, modulator, testData, paddingSamples, 0)
	
	// Decode
	decoded := modulation.DemodulateWithChunks(t, demodulator, signal, 1000, len(testData))
	
	// Verify exact match
	modulation.VerifyExactMatch(t, testData, decoded)
}

// TestRoundtrip_16DSSS_16Chips_500Baud tests 16-ary DSSS roundtrip
// Uses 16 chips and 500 baud rate
func TestRoundtrip_16DSSS_16Chips_500Baud(t *testing.T) {
	// Generate 16 Walsh codes of length 16
	codeTable := GenerateWalshCodeTable(16)
	
	require.Equal(t, 16, len(codeTable), "Must have 16 codes")
	require.Equal(t, roundtripChips, codeTable[0].Length(), "Code length must be 16 chips")
	
	modulator := NewModulator(roundtripSampleRate, roundtripCarrierFreq, roundtripBaud, codeTable...)
	demodulator := NewDemodulator(roundtripSampleRate, roundtripCarrierFreq, roundtripBaud, codeTable...)
	
	// Test data: 10 bytes (must be multiple of 4 bits per symbol for 16-ary)
	testData := make([]byte, 10)
	for i := range testData {
		testData[i] = byte(i % 256)
	}
	
	// Encode with padding
	paddingSamples := int(roundtripSampleRate * 0.5)
	signal := modulation.ModulateWithPadding(t, modulator, testData, paddingSamples, 0)
	
	// Decode
	decoded := modulation.DemodulateWithChunks(t, demodulator, signal, 1000, len(testData))
	
	// Verify exact match
	modulation.VerifyExactMatch(t, testData, decoded)
}

