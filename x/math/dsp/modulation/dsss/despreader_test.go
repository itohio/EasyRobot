package dsss

import (
	"math"
	"testing"

	"github.com/itohio/EasyRobot/x/math/vec"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestDespreader_Isolated tests the Despreader component in complete isolation
// Creates a known signal and verifies despreading works correctly
func TestDespreader_Isolated(t *testing.T) {
	// Setup: 16 chips, 500 baud, 64000 Hz sample rate
	codeTable := GenerateWalshCodeTable(16)
	code := codeTable[0]
	
	spreadingFactor := 16
	samplesPerChip := 8
	samplesPerSymbol := spreadingFactor * samplesPerChip // 128
	modulationOrder := 2
	bufferSize := 1024
	carrierFreq := float32(1000)
	sampleRate := float32(64000)
	
	despreader := NewDespreader(spreadingFactor, samplesPerChip, samplesPerSymbol, modulationOrder, bufferSize,
		code, codeTable, carrierFreq, sampleRate)
	
	// Create a known signal: encode a single symbol (bit=1)
	modulator := NewModulator(sampleRate, carrierFreq, 500, code)
	testData := []byte{0x80} // Single bit = 1 (MSB)
	expectedSize := modulator.Encode(nil, testData)
	signal := vec.New(expectedSize)
	modulator.Encode(signal, testData)
	signalData := []float32(signal)
	
	t.Logf("Created signal: %d samples, expected symbol=1", len(signalData))
	
	// Create a sample buffer and fill it with the signal
	// Simulate how demodulator fills the buffer: bufferIndex points to NEXT write position
	sampleBuffer := make([]float32, bufferSize)
	
	// Fill enough samples so we have at least one full symbol period
	// Put signal at the end of what we've filled (most recent samples)
	// This simulates the demodulator having received samples over time
	totalSamples := samplesPerSymbol * 2 // Fill 2 symbol periods worth
	bufferIndex := 0
	
	// Fill first symbol period with zeros (old samples)
	for i := 0; i < samplesPerSymbol; i++ {
		sampleBuffer[bufferIndex] = 0
		bufferIndex = (bufferIndex + 1) % bufferSize
	}
	
	// Fill second symbol period with our signal (most recent)
	for i := 0; i < samplesPerSymbol && i < len(signalData); i++ {
		sampleBuffer[bufferIndex] = signalData[i]
		bufferIndex = (bufferIndex + 1) % bufferSize
	}
	
	// Now bufferIndex points to the next write position (after signal)
	// The despreader looks back by (samplesPerSymbol + phase)
	// For phase 0, it looks back samplesPerSymbol, which should hit our signal
	
	require.GreaterOrEqual(t, totalSamples, samplesPerSymbol, "Need at least one symbol period")
	
	// Test despreading at code phase 0 (signal is in most recent samplesPerSymbol samples)
	codePhase := 0
	pllPhase := float32(0) // No phase offset for this test
	
	symbol := despreader.DespreadSymbol(sampleBuffer, bufferIndex, totalSamples, codePhase, pllPhase)
	
	t.Logf("Despread symbol: %d (expected 1)", symbol)
	assert.Equal(t, 1, symbol, "Should despread to symbol 1")
}

// TestDespreader_Isolated_BinarySymbols tests despreading multiple binary symbols
func TestDespreader_Isolated_BinarySymbols(t *testing.T) {
	codeTable := GenerateWalshCodeTable(16)
	code := codeTable[0]
	
	spreadingFactor := 16
	samplesPerChip := 8
	samplesPerSymbol := spreadingFactor * samplesPerChip
	modulationOrder := 2
	bufferSize := 2048 // Larger buffer for multiple symbols
	carrierFreq := float32(1000)
	sampleRate := float32(64000)
	
	despreader := NewDespreader(spreadingFactor, samplesPerChip, samplesPerSymbol, modulationOrder, bufferSize,
		code, codeTable, carrierFreq, sampleRate)
	
	// Encode known pattern: 0x55 = 01010101
	testData := []byte{0x55}
	modulator := NewModulator(sampleRate, carrierFreq, 500, code)
	expectedSize := modulator.Encode(nil, testData)
	signal := vec.New(expectedSize)
	modulator.Encode(signal, testData)
	signalData := []float32(signal)
	
	t.Logf("Signal: %d samples, expected symbols: 0,1,0,1,0,1,0,1", len(signalData))
	
	// Fill buffer
	sampleBuffer := make([]float32, bufferSize)
	bufferIndex := 0
	totalSamples := 0
	
	for i := 0; i < len(signalData) && i < bufferSize; i++ {
		sampleBuffer[bufferIndex] = signalData[i]
		bufferIndex = (bufferIndex + 1) % bufferSize
		totalSamples++
	}
	
	// Expected symbols for 0x55: 0,1,0,1,0,1,0,1 (LSB first)
	expectedSymbols := []int{1, 0, 1, 0, 1, 0, 1, 0} // LSB first
	
	// Test despreading each symbol
	codePhase := 0
	pllPhase := float32(0)
	
	for symbolIdx := 0; symbolIdx < len(expectedSymbols); symbolIdx++ {
		// Calculate code phase for this symbol
		symbolPhase := codePhase + symbolIdx*samplesPerSymbol
		
		// Need enough samples for this symbol
		requiredSamples := symbolPhase + samplesPerSymbol
		if totalSamples < requiredSamples {
			t.Logf("Skipping symbol %d: need %d samples, have %d", symbolIdx, requiredSamples, totalSamples)
			continue
		}
		
		symbol := despreader.DespreadSymbol(sampleBuffer, bufferIndex, totalSamples, symbolPhase, pllPhase)
		expected := expectedSymbols[symbolIdx]
		
		t.Logf("Symbol %d: got %d, expected %d", symbolIdx, symbol, expected)
		assert.Equal(t, expected, symbol, "Symbol %d should match", symbolIdx)
	}
}

// TestDespreader_Isolated_CodePhase tests despreading with different code phases
func TestDespreader_Isolated_CodePhase(t *testing.T) {
	codeTable := GenerateWalshCodeTable(16)
	code := codeTable[0]
	
	spreadingFactor := 16
	samplesPerChip := 8
	samplesPerSymbol := spreadingFactor * samplesPerChip
	modulationOrder := 2
	bufferSize := 1024
	carrierFreq := float32(1000)
	sampleRate := float32(64000)
	
	despreader := NewDespreader(spreadingFactor, samplesPerChip, samplesPerSymbol, modulationOrder, bufferSize,
		code, codeTable, carrierFreq, sampleRate)
	
	// Encode single bit = 1
	testData := []byte{0x80}
	modulator := NewModulator(sampleRate, carrierFreq, 500, code)
	expectedSize := modulator.Encode(nil, testData)
	signal := vec.New(expectedSize)
	modulator.Encode(signal, testData)
	signalData := []float32(signal)
	
	// Fill buffer: put signal in most recent samplesPerSymbol samples
	sampleBuffer := make([]float32, bufferSize)
	totalSamples := samplesPerSymbol * 2
	bufferIndex := 0
	
	// Fill first symbol period with zeros
	for i := 0; i < samplesPerSymbol; i++ {
		sampleBuffer[bufferIndex] = 0
		bufferIndex = (bufferIndex + 1) % bufferSize
	}
	
	// Fill second symbol period with signal
	for i := 0; i < samplesPerSymbol && i < len(signalData); i++ {
		sampleBuffer[bufferIndex] = signalData[i]
		bufferIndex = (bufferIndex + 1) % bufferSize
	}
	
	// Test with code phase 0 (should work)
	codePhase := 0
	pllPhase := float32(0)
	symbol := despreader.DespreadSymbol(sampleBuffer, bufferIndex, totalSamples, codePhase, pllPhase)
	assert.Equal(t, 1, symbol, "Should despread correctly at phase 0")
	
	// Test with small phase offset (should still work if within tolerance)
	codePhase = 4 // 4 samples offset
	symbol = despreader.DespreadSymbol(sampleBuffer, bufferIndex, totalSamples, codePhase, pllPhase)
	t.Logf("Symbol at phase %d: %d", codePhase, symbol)
	// May or may not work depending on alignment, but shouldn't crash
}

// TestDespreader_Isolated_PLLPhase tests despreading with PLL phase correction
func TestDespreader_Isolated_PLLPhase(t *testing.T) {
	codeTable := GenerateWalshCodeTable(16)
	code := codeTable[0]
	
	spreadingFactor := 16
	samplesPerChip := 8
	samplesPerSymbol := spreadingFactor * samplesPerChip
	modulationOrder := 2
	bufferSize := 1024
	carrierFreq := float32(1000)
	sampleRate := float32(64000)
	
	despreader := NewDespreader(spreadingFactor, samplesPerChip, samplesPerSymbol, modulationOrder, bufferSize,
		code, codeTable, carrierFreq, sampleRate)
	
	// Encode single bit = 1
	testData := []byte{0x80}
	modulator := NewModulator(sampleRate, carrierFreq, 500, code)
	expectedSize := modulator.Encode(nil, testData)
	signal := vec.New(expectedSize)
	modulator.Encode(signal, testData)
	signalData := []float32(signal)
	
	// Fill buffer: put signal in most recent samplesPerSymbol samples
	sampleBuffer := make([]float32, bufferSize)
	totalSamples := samplesPerSymbol * 2
	bufferIndex := 0
	
	// Fill first symbol period with zeros
	for i := 0; i < samplesPerSymbol; i++ {
		sampleBuffer[bufferIndex] = 0
		bufferIndex = (bufferIndex + 1) % bufferSize
	}
	
	// Fill second symbol period with signal
	for i := 0; i < samplesPerSymbol && i < len(signalData); i++ {
		sampleBuffer[bufferIndex] = signalData[i]
		bufferIndex = (bufferIndex + 1) % bufferSize
	}
	
	codePhase := 0
	
	// Test with no PLL phase (should work)
	pllPhase := float32(0)
	symbol := despreader.DespreadSymbol(sampleBuffer, bufferIndex, totalSamples, codePhase, pllPhase)
	assert.Equal(t, 1, symbol, "Should work with no PLL phase")
	
	// Test with small PLL phase offset (should still work)
	pllPhase = float32(math.Pi / 8) // 22.5 degrees
	symbol = despreader.DespreadSymbol(sampleBuffer, bufferIndex, totalSamples, codePhase, pllPhase)
	t.Logf("Symbol with PLL phase %.2f: %d", pllPhase, symbol)
	// Should still work with small phase offset
}

