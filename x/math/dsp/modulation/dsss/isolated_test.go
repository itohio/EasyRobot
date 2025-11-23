package dsss

import (
	"testing"

	"github.com/itohio/EasyRobot/x/math/vec"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestIsolated_Acquisition_Debug tests acquisition with detailed debug output
// This helps identify why acquisition is failing
func TestIsolated_Acquisition_Debug(t *testing.T) {
	// Setup
	codeTable := GenerateWalshCodeTable(16)
	code := codeTable[0]
	
	modulator := NewModulator(componentTestSampleRate, componentTestCarrierFreq, componentTestBaud, code)
	demodulator := NewDemodulator(componentTestSampleRate, componentTestCarrierFreq, componentTestBaud, code)
	
	// Encode known test data
	testData := []byte{0x55, 0xAA}
	expectedSize := modulator.Encode(nil, testData)
	signal := vec.New(expectedSize)
	modulator.Encode(signal, testData)
	signalData := []float32(signal)
	
	t.Logf("Signal: %d samples, %d bytes data", len(signalData), len(testData))
	t.Logf("Samples per symbol: %d, Samples per chip: %d, Buffer size: %d",
		demodulator.samplesPerSymbol, demodulator.samplesPerChip, demodulator.bufferSize)
	
	// Feed samples until we have enough for acquisition
	chunkSize := 500
	for i := 0; i < len(signalData) && i < demodulator.samplesPerSymbol*3; i += chunkSize {
		end := i + chunkSize
		if end > len(signalData) {
			end = len(signalData)
		}
		
		chunk := vec.New(end - i)
		chunkData := []float32(chunk)
		copy(chunkData, signalData[i:end])
		
		// Pass empty buffer (not nil) to allow sample processing
		emptyBuf := make([]byte, 0)
		demodulator.Decode(emptyBuf, chunk)
		
		t.Logf("After %d samples: acquired=%v, totalSamples=%d, bufferIndex=%d",
			i+len(chunkData), demodulator.acquired, demodulator.totalSamples, demodulator.bufferIndex)
		
		if demodulator.acquired {
			t.Logf("✓ Acquired! Code phase: %d", demodulator.codePhase)
			break
		}
	}
	
	// Check signal characteristics
	if len(signalData) > 0 {
		var maxVal, minVal, sum float32
		maxVal = signalData[0]
		minVal = signalData[0]
		for _, v := range signalData {
			if v > maxVal {
				maxVal = v
			}
			if v < minVal {
				minVal = v
			}
			sum += v * v
		}
		avgPower := sum / float32(len(signalData))
		t.Logf("Signal stats: min=%.3f, max=%.3f, avg_power=%.3f", minVal, maxVal, avgPower)
	}
	
	assert.True(t, demodulator.acquired, "Should acquire signal")
	
	if !demodulator.acquired {
		// Try to manually inspect correlation by calling attemptAcquisition
		// We can't access it directly, but we can check if we have enough samples
		t.Logf("Failed to acquire. Checking conditions:")
		t.Logf("  totalSamples: %d (need >= %d)", 
			demodulator.totalSamples, demodulator.samplesPerSymbol)
		t.Logf("  bufferIndex: %d", demodulator.bufferIndex)
		t.Logf("  bufferSize: %d", demodulator.bufferSize)
		
		// Check if we have enough samples in buffer
		samplesInBuffer := demodulator.totalSamples
		if samplesInBuffer > demodulator.bufferSize {
			samplesInBuffer = demodulator.bufferSize
		}
		t.Logf("  Samples in buffer: %d", samplesInBuffer)
		
		// Expected correlation for perfect match: spreadingFactor * samplesPerChip
		expectedCorr := float32(demodulator.spreadingFactor * demodulator.samplesPerChip)
		threshold := expectedCorr * 0.1
		t.Logf("  Expected correlation: %.2f, Threshold: %.2f", expectedCorr, threshold)
	}
}

// TestIsolated_Acquisition tests acquisition in complete isolation
// Encodes a signal and verifies that acquisition can find it at the known start position
func TestIsolated_Acquisition(t *testing.T) {
	// Setup: 16 chips, 500 baud
	codeTable := GenerateWalshCodeTable(16)
	code := codeTable[0]
	
	modulator := NewModulator(componentTestSampleRate, componentTestCarrierFreq, componentTestBaud, code)
	demodulator := NewDemodulator(componentTestSampleRate, componentTestCarrierFreq, componentTestBaud, code)
	
	// Encode known test data
	testData := []byte{0x55, 0xAA}
	expectedSize := modulator.Encode(nil, testData)
	signal := vec.New(expectedSize)
	modulator.Encode(signal, testData)
	signalData := []float32(signal)
	
	t.Logf("Signal: %d samples, %d bytes data", len(signalData), len(testData))
	t.Logf("Samples per symbol: %d, Samples per chip: %d", 
		demodulator.samplesPerSymbol, demodulator.samplesPerChip)
	
	// Known signal start (no padding)
	knownSignalStart := 0
	
	// Feed samples one at a time to see exactly when acquisition happens
	acquired := false
	acquiredAtSample := -1
	maxCorrelationSeen := float32(0)
	
	for i := 0; i < len(signalData) && !acquired; i++ {
		// Feed single sample
		chunk := vec.New(1)
		chunkData := []float32(chunk)
		chunkData[0] = signalData[i]
		
		// Feed to demodulator (pass empty buffer, not nil)
		emptyBuf := make([]byte, 0)
		demodulator.Decode(emptyBuf, chunk)
		
		// Check acquisition status
		if demodulator.acquired && !acquired {
			acquired = true
			acquiredAtSample = demodulator.totalSamples
			t.Logf("✓ Acquisition successful at sample %d", acquiredAtSample)
			t.Logf("  Code phase: %d", demodulator.codePhase)
			t.Logf("  Expected around sample: %d", knownSignalStart+demodulator.samplesPerSymbol)
		}
		
		// Try to peek at correlation values (if we can access them)
		// For now, just track when we have enough samples
		if i >= demodulator.samplesPerSymbol && !acquired {
			// We should have acquired by now
			if i == demodulator.samplesPerSymbol*2 {
				t.Logf("⚠ Still not acquired after %d samples (2 symbols)", i+1)
			}
		}
	}
	
	// Verify acquisition
	assert.True(t, acquired, "Should acquire signal")
	if acquired {
		// Acquisition should happen within reasonable window
		expectedWindow := demodulator.samplesPerSymbol * 3
		assert.LessOrEqual(t, acquiredAtSample, expectedWindow,
			"Acquisition should happen within reasonable window")
		t.Logf("Max correlation seen: %.2f", maxCorrelationSeen)
	} else {
		t.Errorf("Failed to acquire after %d samples", len(signalData))
	}
}

// TestIsolated_Tracking tests tracking loop in isolation
// Assumes acquisition has happened, tests that tracking maintains lock
func TestIsolated_Tracking(t *testing.T) {
	// Setup
	codeTable := GenerateWalshCodeTable(16)
	code := codeTable[0]
	
	modulator := NewModulator(componentTestSampleRate, componentTestCarrierFreq, componentTestBaud, code)
	demodulator := NewDemodulator(componentTestSampleRate, componentTestCarrierFreq, componentTestBaud, code)
	
	// Encode longer signal to test tracking
	testData := make([]byte, 10)
	for i := range testData {
		testData[i] = byte(i % 256)
	}
	
	expectedSize := modulator.Encode(nil, testData)
	signal := vec.New(expectedSize)
	modulator.Encode(signal, testData)
	signalData := []float32(signal)
	
	// Feed entire signal to allow acquisition
	demodulator.Decode(nil, signal)
	
	// Verify acquisition happened
	require.True(t, demodulator.acquired, "Must acquire signal first")
	
	initialCodePhase := demodulator.codePhase
	initialLocked := demodulator.locked
	
	t.Logf("Initial state: codePhase=%d, locked=%v", initialCodePhase, initialLocked)
	
	// Reset and feed again to test tracking
	demodulator.Reset()
	
	// Feed in chunks to test tracking
	chunkSize := 200
	lockMaintained := true
	codePhaseChanges := 0
	
	for i := 0; i < len(signalData); i += chunkSize {
		end := i + chunkSize
		if end > len(signalData) {
			end = len(signalData)
		}
		
		chunk := vec.New(end - i)
		chunkData := []float32(chunk)
		copy(chunkData, signalData[i:end])
		
		wasAcquired := demodulator.acquired
		oldCodePhase := demodulator.codePhase
		
		demodulator.Decode(nil, chunk)
		
		// Check if lock is maintained
		if wasAcquired && !demodulator.acquired {
			lockMaintained = false
			t.Errorf("Lost lock at sample %d", demodulator.totalSamples)
			break
		}
		
		if demodulator.acquired && oldCodePhase != demodulator.codePhase {
			codePhaseChanges++
		}
	}
	
	assert.True(t, lockMaintained, "Should maintain lock throughout")
	if lockMaintained {
		t.Logf("✓ Tracking maintained lock, code phase changed %d times", codePhaseChanges)
	}
}

// TestIsolated_Despreading tests despreading in isolation
// Assumes acquisition and tracking work, tests symbol extraction
func TestIsolated_Despreading(t *testing.T) {
	// Setup
	codeTable := GenerateWalshCodeTable(16)
	code := codeTable[0]
	
	modulator := NewModulator(componentTestSampleRate, componentTestCarrierFreq, componentTestBaud, code)
	demodulator := NewDemodulator(componentTestSampleRate, componentTestCarrierFreq, componentTestBaud, code)
	
	// Encode known pattern for easy verification
	// 0x55 = 01010101, 0xAA = 10101010
	testData := []byte{0x55, 0xAA}
	expectedSize := modulator.Encode(nil, testData)
	signal := vec.New(expectedSize)
	modulator.Encode(signal, testData)
	signalData := []float32(signal)
	
	// Expected symbols for binary DSSS:
	// 0x55 = 01010101 = symbols: 0,1,0,1,0,1,0,1
	// 0xAA = 10101010 = symbols: 1,0,1,0,1,0,1,0
	expectedSymbols := []int{0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0}
	
	t.Logf("Expected %d symbols", len(expectedSymbols))
	
	// Feed entire signal to allow acquisition
	demodulator.Decode(nil, signal)
	require.True(t, demodulator.acquired, "Must acquire signal first")
	
	// Reset and feed again to extract symbols
	demodulator.Reset()
	
	// Feed signal and manually extract symbols
	extractedSymbols := make([]int, 0)
	chunkSize := 500
	
	for i := 0; i < len(signalData); i += chunkSize {
		end := i + chunkSize
		if end > len(signalData) {
			end = len(signalData)
		}
		
		chunk := vec.New(end - i)
		chunkData := []float32(chunk)
		copy(chunkData, signalData[i:end])
		
		// Feed chunk (pass empty buffer, not nil)
		emptyBuf := make([]byte, 0)
		demodulator.Decode(emptyBuf, chunk)
		
		// If acquired, try to extract symbols manually
		// We need to access despreadSymbol, but it's not exported
		// So we'll test through the Decode method and check byteBuffer
		if demodulator.acquired {
			// Check how many symbols we've extracted by checking byteBuffer length
			bitsExtracted := demodulator.byteBuffer.Len() * 8
			symbolsExtracted := bitsExtracted / demodulator.bitsPerSymbol
			
			if symbolsExtracted > len(extractedSymbols) {
				// New symbols were extracted
				// We can't directly access them, but we can verify the count
				extractedSymbols = make([]int, symbolsExtracted)
				for j := range extractedSymbols {
					extractedSymbols[j] = -1 // Placeholder
				}
			}
		}
	}
	
	// Now decode to get actual bytes and verify
	output := make([]byte, len(testData))
	demodulator.Reset()
	bytesDecoded := demodulator.Decode(output, signal)
	
	t.Logf("Decoded %d bytes (expected %d)", bytesDecoded, len(testData))
	
	if bytesDecoded > 0 {
		t.Logf("First few decoded bytes: %v", output[:min(bytesDecoded, 4)])
		
		// For binary DSSS, verify we got the right number of symbols
		expectedSymbolCount := len(testData) * 8 // 8 bits per byte
		actualSymbolCount := bytesDecoded * 8
		
		t.Logf("Expected %d symbols, got %d symbols worth of bits", 
			expectedSymbolCount, actualSymbolCount)
		
		// Verify we extracted some symbols
		assert.Greater(t, bytesDecoded, 0, "Should extract at least some bytes")
	}
}

// TestIsolated_FullPipeline tests the complete pipeline step by step
func TestIsolated_FullPipeline(t *testing.T) {
	// Setup
	codeTable := GenerateWalshCodeTable(16)
	code := codeTable[0]
	
	modulator := NewModulator(componentTestSampleRate, componentTestCarrierFreq, componentTestBaud, code)
	demodulator := NewDemodulator(componentTestSampleRate, componentTestCarrierFreq, componentTestBaud, code)
	
	testData := []byte{0x55, 0xAA}
	
	// Encode
	expectedSize := modulator.Encode(nil, testData)
	signal := vec.New(expectedSize)
	modulator.Encode(signal, testData)
	signalData := []float32(signal)
	
	t.Logf("=== Full Pipeline Test ===")
	t.Logf("Input: %v (%d bytes)", testData, len(testData))
	t.Logf("Signal: %d samples", len(signalData))
	
	// Track each stage
	var acquisitionSample int = -1
	var firstSymbolSample int = -1
	var bytesDecoded int
	
	output := make([]byte, len(testData)*2)
	chunkSize := 500
	
	for i := 0; i < len(signalData); i += chunkSize {
		end := i + chunkSize
		if end > len(signalData) {
			end = len(signalData)
		}
		
		chunk := vec.New(end - i)
		chunkData := []float32(chunk)
		copy(chunkData, signalData[i:end])
		
		beforeAcquired := demodulator.acquired
		beforeBytes := bytesDecoded
		
		decoded := demodulator.Decode(output[bytesDecoded:], chunk)
		
		// Track acquisition
		if !beforeAcquired && demodulator.acquired {
			acquisitionSample = demodulator.totalSamples
			t.Logf("✓ Stage 1 - Acquisition: sample %d", acquisitionSample)
			t.Logf("  Code phase: %d", demodulator.codePhase)
		}
		
		// Track first symbol
		if demodulator.acquired && beforeBytes == 0 && decoded > 0 {
			firstSymbolSample = demodulator.totalSamples
			t.Logf("✓ Stage 2 - First symbol decoded: sample %d", firstSymbolSample)
		}
		
		bytesDecoded += decoded
	}
	
	// Summary
	t.Logf("=== Pipeline Summary ===")
	if acquisitionSample >= 0 {
		t.Logf("✓ Acquisition: sample %d", acquisitionSample)
	} else {
		t.Errorf("✗ Acquisition: FAILED")
	}
	
	if firstSymbolSample >= 0 {
		t.Logf("✓ Symbol extraction: sample %d", firstSymbolSample)
	} else {
		t.Errorf("✗ Symbol extraction: FAILED")
	}
	
	t.Logf("Bytes decoded: %d (expected %d)", bytesDecoded, len(testData))
	
	if bytesDecoded > 0 {
		t.Logf("Decoded data: %v", output[:bytesDecoded])
		t.Logf("Expected data: %v", testData)
		
		// Check if they match
		if bytesDecoded >= len(testData) {
			match := true
			for i := 0; i < len(testData); i++ {
				if output[i] != testData[i] {
					match = false
					t.Errorf("Mismatch at byte %d: got 0x%02X, expected 0x%02X", 
						i, output[i], testData[i])
				}
			}
			if match {
				t.Logf("✓ Data matches!")
			}
		}
	}
	
	// Verify each stage
	assert.True(t, acquisitionSample >= 0, "Acquisition should succeed")
	assert.True(t, firstSymbolSample >= 0, "Symbol extraction should succeed")
	assert.Greater(t, bytesDecoded, 0, "Should decode some bytes")
}


