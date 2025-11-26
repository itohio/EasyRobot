package qam

import (
	"bytes"
	"testing"

	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/stretchr/testify/require"
)

// TestDebugPosition51 enables debug logging to investigate the quantization bug at position 51
func TestDebugPosition51(t *testing.T) {
	cfg := DefaultConfig()
	payload := makePayload(33) // Same as failing test
	symbols := bytesToSymbols(payload, cfg.BitsPerSymbol)
	
	t.Logf("Total symbols: %d", len(symbols))
	t.Logf("Symbol at position 51: %d (0x%x)", symbols[51], symbols[51])
	require.Equal(t, byte(10), symbols[51], "Position 51 should be symbol 10")
	
	mod, err := NewModulator(cfg)
	require.NoError(t, err)
	demod, err := NewDemodulator(cfg)
	require.NoError(t, err)
	
	// Enable debug logging
	demod.EnableDebugLogging(true)
	
	inputBits := buildBits(t, cfg.BitsPerSymbol, symbols)
	estimate := mod.Process(nil, inputBits)
	
	// Calculate total samples needed: signal + padding
	paddingSamples := int(0.5 * cfg.SampleRate) * 2
	signalSamples := estimate
	totalSamples := paddingSamples*2 + signalSamples
	
	sampleBuffer := make([]float32, totalSamples)
	signalStart := paddingSamples
	signalEnd := paddingSamples + signalSamples
	
	inputBits.Reset()
	signalBuf := buffer.New(sampleBuffer[signalStart:signalStart:signalEnd])
	written := mod.Process(signalBuf, inputBits)
	require.Equal(t, signalSamples, written)
	
	// Use infinite SNR (no noise)
	// addNoiseToBuffer(sampleBuffer, sampleBuffer[signalStart:signalEnd], math32.Inf(1))
	
	outputBits, err := buffer.NewBits(cfg.BitsPerSymbol)
	require.NoError(t, err)
	
	// Capture log output
	var logBuf bytes.Buffer
	// Note: This is a simple approach - in real code you might want to redirect log output properly
	_ = logBuf
	
	// Process in chunks like the real test
	chunks := splitIntoChunksFromBuffer(sampleBuffer, 3)
	totalDecoded := 0
	
	for chunkIdx, chunk := range chunks {
		reader := buffer.New(chunk)
		decoded := demod.Process(outputBits, reader)
		totalDecoded += decoded
		
		if chunkIdx == 1 {
			// After processing second chunk, check what we have so far
			outputBits.Reset()
			current := make([]byte, len(symbols))
			readBits, _ := outputBits.Read(current)
			decodedSymbols := current[:readBits/cfg.BitsPerSymbol]
			
			if len(decodedSymbols) > 51 {
				t.Logf("After chunk %d: decoded %d symbols", chunkIdx, len(decodedSymbols))
				t.Logf("Symbol at position 51: %d (0x%x)", decodedSymbols[51], decodedSymbols[51])
				
				// Check symbol at position 51 specifically
				if decodedSymbols[51] != symbols[51] {
					t.Errorf("Symbol mismatch at position 51: expected %d (0x%x), got %d (0x%x)",
						symbols[51], symbols[51], decodedSymbols[51], decodedSymbols[51])
				}
			}
		}
	}
	
	outputBits.Reset()
	decodedSymbols := make([]byte, len(symbols))
	readBits, err := outputBits.Read(decodedSymbols)
	require.NoError(t, err)
	require.Equal(t, len(symbols)*cfg.BitsPerSymbol, readBits)
	
	// Check all symbols
	if !equalSymbols(symbols, decodedSymbols) {
		t.Logf("Symbol mismatch: expected=%v actual=%v", symbols, decodedSymbols)
		for i := range symbols {
			if i >= len(decodedSymbols) || symbols[i] != decodedSymbols[i] {
				t.Logf("Position %d: expected %d (0x%x), got %d (0x%x)",
					i, symbols[i], symbols[i],
					func() byte {
						if i < len(decodedSymbols) {
							return decodedSymbols[i]
						}
						return 0xFF
					}(),
					func() byte {
						if i < len(decodedSymbols) {
							return decodedSymbols[i]
						}
						return 0xFF
					}())
				if i > 51 {
					break // Just show first few mismatches
				}
			}
		}
	}
	require.Equal(t, symbols, decodedSymbols, "All symbols should match")
}

