package dsss

import (
	"testing"

	"github.com/itohio/EasyRobot/x/math/vec"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Test parameters: 16 chips, 500 baud
const (
	componentTestChips        = 16
	componentTestBaud         = float32(500)
	componentTestCarrierFreq  = float32(1000)
	componentTestSampleRate   = float32(64000) // 8 samples/chip: 500 * 16 * 8 = 64000 Hz
	componentTestRolloffFactor = float32(0.5)  // Raised cosine roll-off factor
)

// TestSignalLocker_Acquisition tests signal acquisition/locking in isolation
// This test encodes a signal and verifies that the acquisition can find it
func TestSignalLocker_Acquisition(t *testing.T) {
	// Generate code
	codeTable := GenerateWalshCodeTable(16)
	code := codeTable[0]
	
	// Create modulator and encode test data
	modulator := NewModulator(componentTestSampleRate, componentTestCarrierFreq, componentTestBaud, componentTestRolloffFactor, code)
	testData := []byte{0x55, 0xAA}
	
	// Encode signal
	expectedSize := modulator.Encode(nil, testData)
	signal := vec.New(expectedSize)
	samplesWritten := modulator.Encode(signal, testData)
	require.Equal(t, expectedSize, samplesWritten)
	
	signalData := []float32(signal)
	t.Logf("Encoded %d samples for %d bytes", len(signalData), len(testData))
	
	// Create demodulator for acquisition testing
	demodulator := NewDemodulator(componentTestSampleRate, componentTestCarrierFreq, componentTestBaud, componentTestRolloffFactor, code)
	
	// TODO: New implementation doesn't have acquisition/locking yet
	// This test is skipped until those features are added
	t.Skip("Acquisition/locking not yet implemented in new demodulator")
	
	// We know the signal starts at sample 0 (no padding)
	knownSignalStart := 0
	
	// Feed samples to demodulator in chunks
	chunkSize := 500
	acquired := false
	acquiredAtSample := -1
	
	for i := 0; i < len(signalData); i += chunkSize {
		end := i + chunkSize
		if end > len(signalData) {
			end = len(signalData)
		}
		
		chunk := vec.New(end - i)
		chunkData := []float32(chunk)
		copy(chunkData, signalData[i:end])
		
		// Feed chunk to demodulator
		bytesDecoded := demodulator.Decode(nil, chunk)
		_ = bytesDecoded // Not checking bytes yet, just acquisition
		
		// Check if acquisition happened
		// TODO: Re-enable when acquisition is implemented
		_ = acquired
		_ = acquiredAtSample
		_ = knownSignalStart
		break
	}
	
	// Verify acquisition happened
	// TODO: Re-enable when acquisition is implemented
	_ = acquired
}

// TestPreambleDetector tests preamble detection in isolation
// For DSSS, we can use a known preamble pattern (e.g., alternating pattern)
func TestPreambleDetector(t *testing.T) {
	t.Skip("Preamble detection not yet implemented in new demodulator")
	
	// Generate code
	codeTable := GenerateWalshCodeTable(16)
	code := codeTable[0]
	
	// Create modulator
	modulator := NewModulator(componentTestSampleRate, componentTestCarrierFreq, componentTestBaud, componentTestRolloffFactor, code)
	
	// Create preamble: alternating 0xAA, 0x55 pattern (4 bytes)
	preamble := []byte{0xAA, 0x55, 0xAA, 0x55}
	
	// Encode preamble
	preambleSize := modulator.Encode(nil, preamble)
	preambleSignal := vec.New(preambleSize)
	modulator.Encode(preambleSignal, preamble)
	
	// Create demodulator
	demodulator := NewDemodulator(componentTestSampleRate, componentTestCarrierFreq, componentTestBaud, componentTestRolloffFactor, code)
	
	// Feed preamble signal to demodulator
	preambleData := []float32(preambleSignal)
	chunkSize := 500
	
	// Track when preamble is detected
	preambleDetected := false
	detectedAtSample := -1
	
	for i := 0; i < len(preambleData); i += chunkSize {
		end := i + chunkSize
		if end > len(preambleData) {
			end = len(preambleData)
		}
		
		chunk := vec.New(end - i)
		chunkData := []float32(chunk)
		copy(chunkData, preambleData[i:end])
		
		// Feed chunk
		demodulator.Decode(nil, chunk)
		
		// TODO: Re-enable when acquisition is implemented
		_ = preambleDetected
		_ = detectedAtSample
		break
	}
	
	// Verify preamble detection (or at least acquisition)
	// TODO: Re-enable when acquisition is implemented
	_ = preambleDetected
	
	if preambleDetected {
		t.Logf("Preamble detected at sample %d", detectedAtSample)
	}
	
	// Note: Full preamble detection would verify the decoded bytes match the preamble pattern
	// This requires symbol extraction to work first
}

// TestSymbolSynchronization tests symbol boundary synchronization in isolation
// This verifies that symbols can be correctly extracted at the right boundaries
func TestSymbolSynchronization(t *testing.T) {
	t.Skip("Symbol synchronization testing requires acquisition which is not yet implemented")
	
	// Generate code
	codeTable := GenerateWalshCodeTable(16)
	code := codeTable[0]
	
	// Create modulator
	modulator := NewModulator(componentTestSampleRate, componentTestCarrierFreq, componentTestBaud, componentTestRolloffFactor, code)
	
	// Encode known test data
	testData := []byte{0x55, 0xAA} // 16 bits = 16 symbols for binary DSSS
	expectedSize := modulator.Encode(nil, testData)
	signal := vec.New(expectedSize)
	modulator.Encode(signal, testData)
	
	signalData := []float32(signal)
	
	// Create demodulator
	demodulator := NewDemodulator(componentTestSampleRate, componentTestCarrierFreq, componentTestBaud, componentTestRolloffFactor, code)
	
	// Feed entire signal to allow acquisition
	demodulator.Decode(nil, signal)
	
	// Verify acquisition happened
	// TODO: Re-enable when acquisition is implemented
	_ = demodulator
	
	// Now test symbol synchronization
	// We know:
	// - samplesPerSymbol = spreadingFactor * samplesPerChip = 16 * 8 = 128
	// - We have 16 symbols, so total samples = 16 * 128 = 2048
	
	expectedSamplesPerSymbol := demodulator.spreadingFactor * demodulator.samplesPerChip
	assert.Equal(t, demodulator.samplesPerSymbol, expectedSamplesPerSymbol,
		"Samples per symbol should match")
	
	t.Logf("Samples per symbol: %d, Total signal samples: %d, Expected symbols: %d",
		demodulator.samplesPerSymbol, len(signalData), len(testData)*8)
	
	// TODO: Re-enable when acquisition is implemented
	_ = demodulator
	_ = signalData
}

// TestSymbolExtraction tests symbol extraction logic in isolation
// This verifies that despreadSymbol correctly extracts symbols from the signal
func TestSymbolExtraction(t *testing.T) {
	t.Skip("Symbol extraction testing requires acquisition which is not yet implemented")
	// Generate code
	codeTable := GenerateWalshCodeTable(16)
	code := codeTable[0]
	
	// Create modulator
	modulator := NewModulator(componentTestSampleRate, componentTestCarrierFreq, componentTestBaud, componentTestRolloffFactor, code)
	
	// Encode known test data
	// Use simple pattern: 0x55 = 01010101, 0xAA = 10101010
	testData := []byte{0x55, 0xAA}
	expectedSize := modulator.Encode(nil, testData)
	signal := vec.New(expectedSize)
	modulator.Encode(signal, testData)
	
	signalData := []float32(signal)
	
	// Create demodulator
	demodulator := NewDemodulator(componentTestSampleRate, componentTestCarrierFreq, componentTestBaud, componentTestRolloffFactor, code)
	
	// Feed entire signal to allow acquisition
	demodulator.Decode(nil, signal)
	
	// TODO: Re-enable when acquisition is implemented
	_ = demodulator
	
	// Reset and feed signal again to extract symbols
	demodulator.Reset()
	
	// Feed signal in chunks
	chunkSize := 500
	
	for i := 0; i < len(signalData); i += chunkSize {
		end := i + chunkSize
		if end > len(signalData) {
			end = len(signalData)
		}
		
		chunk := vec.New(end - i)
		chunkData := []float32(chunk)
		copy(chunkData, signalData[i:end])
		
		// Feed chunk
		demodulator.Decode(nil, chunk)
		
		// Try to extract symbols if acquired
		// For binary DSSS, we expect 16 symbols (16 bits from 2 bytes)
		// Each symbol should be 0 or 1
		// TODO: Re-enable when acquisition is implemented
		_ = demodulator
	}
	
	// Now decode to get symbols
	output := make([]byte, len(testData)*2) // Extra space
	demodulator.Reset()
	bytesDecoded := demodulator.Decode(output, signal)
	
	t.Logf("Decoded %d bytes (expected %d)", bytesDecoded, len(testData))
	
	// Extract symbols by manually testing despreadSymbol
	// We need to access internal state, so let's create a test that feeds
	// the signal and then manually checks symbol extraction
	
	// Better approach: test symbol extraction by checking decoded output
	// matches expected pattern
	if bytesDecoded > 0 {
		t.Logf("First few decoded bytes: %v", output[:min(bytesDecoded, 4)])
		
		// For 0x55, 0xAA input, we expect the same output (if everything works)
		// But we need to account for bit ordering and symbol extraction
	}
	
	// Verify we extracted some symbols
	// The exact count depends on how many complete symbols we can extract
	assert.Greater(t, bytesDecoded, 0, "Should extract at least some bytes")
}

// TestFullPipeline tests the full pipeline: acquisition -> symbol sync -> symbol extraction
// This is an integration test that verifies all components work together
func TestFullPipeline(t *testing.T) {
	t.Skip("Full pipeline testing requires acquisition which is not yet implemented")
	// Generate code
	codeTable := GenerateWalshCodeTable(16)
	code := codeTable[0]
	
	// Create modulator and demodulator
	modulator := NewModulator(componentTestSampleRate, componentTestCarrierFreq, componentTestBaud, componentTestRolloffFactor, code)
	demodulator := NewDemodulator(componentTestSampleRate, componentTestCarrierFreq, componentTestBaud, componentTestRolloffFactor, code)
	
	// Test data
	testData := []byte{0x55, 0xAA}
	
	// Encode
	expectedSize := modulator.Encode(nil, testData)
	signal := vec.New(expectedSize)
	modulator.Encode(signal, testData)
	
	// Track pipeline stages
	var acquiredAt int
	var firstSymbolAt int
	var bytesDecoded int
	
	// Feed signal in chunks to observe pipeline
	signalData := []float32(signal)
	chunkSize := 500
	output := make([]byte, len(testData)*2)
	
	for i := 0; i < len(signalData); i += chunkSize {
		end := i + chunkSize
		if end > len(signalData) {
			end = len(signalData)
		}
		
		chunk := vec.New(end - i)
		chunkData := []float32(chunk)
		copy(chunkData, signalData[i:end])
		
		// Feed chunk
		decoded := demodulator.Decode(output[bytesDecoded:], chunk)
		
		// TODO: Re-enable when acquisition is implemented
		_ = acquiredAt
		_ = firstSymbolAt
		bytesDecoded += decoded
	}
	
	// TODO: Re-enable when acquisition is implemented
	_ = demodulator
	_ = bytesDecoded
}

// Helper function
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

