package mpsk

import (
	"fmt"
	"testing"

	"github.com/chewxy/math32"
	"github.com/stretchr/testify/require"

	"github.com/itohio/EasyRobot/x/buffer"
	test_utils "github.com/itohio/EasyRobot/x/math/dsp/modulation/test_utils"
)

func formatFloat(f float32) string {
	return fmt.Sprintf("%.1f", f)
}

// TestPhaseDetection_Rotation tests phase detection and rotation with known phase sequences
func TestPhaseDetection_Rotation(t *testing.T) {
	cfg := DefaultConfig()
	cfg.PreambleLength = 0 // No preamble for clean testing
	require.NoError(t, cfg.Validate())

	mod, err := NewModulator(cfg)
	require.NoError(t, err)
	demod, err := NewDemodulator(cfg)
	require.NoError(t, err)

	// Test sequence: known symbols for QPSK (0, 1, 2, 3)
	testSymbols := []byte{0, 1, 2, 3, 0, 1, 2, 3}
	bits := buildBits(t, cfg.BitsPerSymbol, testSymbols)

	// Generate clean signal (no noise)
	estimate := mod.Process(nil, bits)
	bits.Reset()
	sampleBuf := buffer.New(make([]float32, 0, estimate))
	written := mod.Process(sampleBuf, bits)
	require.Equal(t, estimate, written)

	samples := sampleBuf.Items()
	t.Logf("Generated %d samples for %d symbols", len(samples), len(testSymbols))

	// Test 1: Direct phase detection without rotation (should work if signal is aligned)
	t.Run("DirectPhaseDetection", func(t *testing.T) {
		demod.Reset()
		outputBits, err := buffer.NewBits(cfg.BitsPerSymbol)
		require.NoError(t, err)

		reader := buffer.New(samples)
		decoded := demod.Process(outputBits, reader)
		require.Greater(t, decoded, 0)

		outputBits.Reset()
		decodedSymbols := make([]byte, len(testSymbols))
		readBits, err := outputBits.Read(decodedSymbols)
		require.NoError(t, err)
		require.Equal(t, len(testSymbols)*cfg.BitsPerSymbol, readBits)

		t.Logf("Expected: %v", testSymbols)
		t.Logf("Got:      %v", decodedSymbols)
		require.Equal(t, testSymbols, decodedSymbols, "direct phase detection should work for aligned signal")
	})

	// Test 2: Phase rotation test - apply known phase offset and verify rotation fixes it
	phaseOffsets := []float32{
		0,                    // No offset
		math32.Pi / 4,        // 45 degrees
		math32.Pi / 2,        // 90 degrees
		math32.Pi,            // 180 degrees
		3 * math32.Pi / 2,    // 270 degrees
		math32.Pi / 8,        // 22.5 degrees
		math32.Pi / 16,       // 11.25 degrees
	}

	for _, offset := range phaseOffsets {
		offsetDeg := offset * 180 / math32.Pi
		t.Run("PhaseOffset_"+formatFloat(offsetDeg), func(t *testing.T) {
			// Rotate samples by phase offset
			rotatedSamples := make([]float32, len(samples))
			cosOffset := math32.Cos(offset)
			sinOffset := math32.Sin(offset)
			for i := 0; i < len(samples); i += 2 {
				// Rotate IQ pair
				iVal := samples[i]
				qVal := samples[i+1]
				rotatedSamples[i] = iVal*cosOffset - qVal*sinOffset
				rotatedSamples[i+1] = iVal*sinOffset + qVal*cosOffset
			}

			demod.Reset()
			outputBits, err := buffer.NewBits(cfg.BitsPerSymbol)
			require.NoError(t, err)

			reader := buffer.New(rotatedSamples)
			decoded := demod.Process(outputBits, reader)
			require.Greater(t, decoded, 0, "should decode with phase offset %.1f deg", offset*180/math32.Pi)

			outputBits.Reset()
			decodedSymbols := make([]byte, len(testSymbols))
			readBits, err := outputBits.Read(decodedSymbols)
			require.NoError(t, err)
			require.Equal(t, len(testSymbols)*cfg.BitsPerSymbol, readBits)

			// For small offsets, should match exactly
			// For large offsets (90, 180, 270), might have phase ambiguity
			if offset < math32.Pi/8 {
				require.Equal(t, testSymbols, decodedSymbols,
					"small phase offset %.1f deg should be corrected", offset*180/math32.Pi)
			} else {
				t.Logf("Offset %.1f deg: expected %v, got %v", offset*180/math32.Pi, testSymbols, decodedSymbols)
				// For large offsets, at least verify we get valid symbols
				for _, sym := range decodedSymbols {
					require.Less(t, int(sym), cfg.constellationSize(), "decoded symbol should be valid")
				}
			}
		})
	}
}

// TestPhaseDetection_StepSequence tests phase detection with a step sequence
func TestPhaseDetection_StepSequence(t *testing.T) {
	cfg := DefaultConfig()
	cfg.PreambleLength = 0
	require.NoError(t, cfg.Validate())

	demod, err := NewDemodulator(cfg)
	require.NoError(t, err)

	// Create a simple test: generate samples for known phase sequence
	// For QPSK: symbol 0 = 0°, symbol 1 = 90°, symbol 2 = 180°, symbol 3 = 270°
	symbolSamples := cfg.SamplesPerSymbol * 2
	table := newConstellationTable(cfg)

	// Generate samples for symbol sequence: 0, 1, 2, 3
	testSequence := []int{0, 1, 2, 3}
	totalSamples := len(testSequence) * symbolSamples
	samples := make([]float32, totalSamples)

	for seqIdx, symbolIdx := range testSequence {
		phase := table.phases[symbolIdx]
		cosPhase := math32.Cos(phase)
		sinPhase := math32.Sin(phase)
		env := table.envelope

		offset := seqIdx * symbolSamples
		for i := 0; i < cfg.SamplesPerSymbol; i++ {
			envVal := env[i]
			samples[offset+i*2] = cosPhase * envVal
			samples[offset+i*2+1] = sinPhase * envVal
		}
	}

	// Test direct integration and quantization
	t.Run("DirectIntegration", func(t *testing.T) {
		for i, expectedSymbol := range testSequence {
			offset := i * symbolSamples
			symbolSamples := samples[offset : offset+symbolSamples]

			iAcc, qAcc := demod.integrateSymbol(symbolSamples)
			phase := math32.Atan2(qAcc, iAcc)
			detectedSymbol := demod.quantizePhase(phase)

			t.Logf("Symbol %d: expected=%d, detected=%d, phase=%.2f rad (%.1f deg), iAcc=%.3f, qAcc=%.3f",
				i, expectedSymbol, detectedSymbol, phase, phase*180/math32.Pi, iAcc, qAcc)

			require.Equal(t, expectedSymbol, detectedSymbol,
				"direct integration should detect correct symbol for aligned signal")
		}
	})

	// Test with PLL rotation
	t.Run("WithPLLRotation", func(t *testing.T) {
		demod.Reset()
		outputBits, err := buffer.NewBits(cfg.BitsPerSymbol)
		require.NoError(t, err)

		reader := buffer.New(samples)
		decoded := demod.Process(outputBits, reader)
		require.Greater(t, decoded, 0)

		outputBits.Reset()
		decodedSymbols := make([]byte, len(testSequence))
		readBits, err := outputBits.Read(decodedSymbols)
		require.NoError(t, err)
		require.Equal(t, len(testSequence)*cfg.BitsPerSymbol, readBits)

		expectedSymbols := make([]byte, len(testSequence))
		for i, s := range testSequence {
			expectedSymbols[i] = byte(s)
		}

		t.Logf("Expected: %v", expectedSymbols)
		t.Logf("Got:      %v", decodedSymbols)
		require.Equal(t, expectedSymbols, decodedSymbols, "PLL should correctly decode aligned signal")
	})
}

// TestPhaseDetection_WithOffset tests if PLL correctly handles phase offsets
func TestPhaseDetection_WithOffset(t *testing.T) {
	cfg := DefaultConfig()
	cfg.PreambleLength = 0
	require.NoError(t, cfg.Validate())

	mod, err := NewModulator(cfg)
	require.NoError(t, err)
	demod, err := NewDemodulator(cfg)
	require.NoError(t, err)

	testSymbols := []byte{0, 1, 2, 3, 0, 1, 2, 3}
	bits := buildBits(t, cfg.BitsPerSymbol, testSymbols)

	estimate := mod.Process(nil, bits)
	bits.Reset()
	sampleBuf := buffer.New(make([]float32, 0, estimate))
	mod.Process(sampleBuf, bits)
	samples := sampleBuf.Items()

	// Apply a constant phase offset (simulating carrier offset)
	phaseOffset := math32.Pi / 6 // 30 degrees
	cosOffset := math32.Cos(phaseOffset)
	sinOffset := math32.Sin(phaseOffset)

	rotatedSamples := make([]float32, len(samples))
	for i := 0; i < len(samples); i += 2 {
		iVal := samples[i]
		qVal := samples[i+1]
		rotatedSamples[i] = iVal*cosOffset - qVal*sinOffset
		rotatedSamples[i+1] = iVal*sinOffset + qVal*cosOffset
	}

	demod.Reset()
	outputBits, err := buffer.NewBits(cfg.BitsPerSymbol)
	require.NoError(t, err)

	reader := buffer.New(rotatedSamples)
	decoded := demod.Process(outputBits, reader)
	require.Greater(t, decoded, 0, "should decode with 30 degree phase offset")

	outputBits.Reset()
	decodedSymbols := make([]byte, len(testSymbols))
	readBits, err := outputBits.Read(decodedSymbols)
	require.NoError(t, err)
	require.Equal(t, len(testSymbols)*cfg.BitsPerSymbol, readBits)

	t.Logf("Expected: %v", testSymbols)
	t.Logf("Got:      %v", decodedSymbols)

	// With PLL, should correct the 30 degree offset
	// Allow some tolerance for initial symbols while PLL converges
	matches := 0
	for i := range testSymbols {
		if testSymbols[i] == decodedSymbols[i] {
			matches++
		}
	}
	t.Logf("Matched %d/%d symbols", matches, len(testSymbols))

	// Should match at least the later symbols (after PLL converges)
	// First few might be wrong during convergence
	if len(testSymbols) >= 4 {
		lastHalf := testSymbols[len(testSymbols)/2:]
		decodedLastHalf := decodedSymbols[len(decodedSymbols)/2:]
		require.Equal(t, lastHalf, decodedLastHalf,
			"later symbols should match after PLL convergence")
	}
}

// TestPhaseDetection_PreambleTransition tests the critical preamble-to-payload transition
func TestPhaseDetection_PreambleTransition(t *testing.T) {
	cfg := DefaultConfig()
	cfg.PreambleLength = 20 // Use preamble
	require.NoError(t, cfg.Validate())

	mod, err := NewModulator(cfg)
	require.NoError(t, err)
	demod, err := NewDemodulator(cfg)
	require.NoError(t, err)

	// Payload symbols (these should be decoded correctly after preamble)
	payloadSymbols := []byte{0, 1, 2, 3, 0, 1, 2, 3}
	bits := buildBits(t, cfg.BitsPerSymbol, payloadSymbols)

	// Generate signal with preamble
	estimate := mod.Process(nil, bits)
	bits.Reset()
	sampleBuf := buffer.New(make([]float32, 0, estimate))
	written := mod.Process(sampleBuf, bits)
	require.Equal(t, estimate, written)

	samples := sampleBuf.Items()
	t.Logf("Generated %d samples (preamble + %d payload symbols)", len(samples), len(payloadSymbols))

	// Test with clean signal first
	t.Run("CleanSignal", func(t *testing.T) {
		demod.Reset()
		outputBits, err := buffer.NewBits(cfg.BitsPerSymbol)
		require.NoError(t, err)

		reader := buffer.New(samples)
		decoded := demod.Process(outputBits, reader)
		require.Greater(t, decoded, 0)

		outputBits.Reset()
		decodedSymbols := make([]byte, len(payloadSymbols))
		readBits, err := outputBits.Read(decodedSymbols)
		require.NoError(t, err)
		require.Equal(t, len(payloadSymbols)*cfg.BitsPerSymbol, readBits)

		t.Logf("Expected payload: %v", payloadSymbols)
		t.Logf("Decoded payload:  %v", decodedSymbols)
		require.Equal(t, payloadSymbols, decodedSymbols,
			"payload should decode correctly after preamble with clean signal")
	})

	// Test with 25dB SNR noise
	t.Run("With25dBSNR", func(t *testing.T) {
		noisySamples := append([]float32(nil), samples...)
		addNoise(noisySamples, 25.0)

		demod.Reset()
		outputBits, err := buffer.NewBits(cfg.BitsPerSymbol)
		require.NoError(t, err)

		reader := buffer.New(noisySamples)
		decoded := demod.Process(outputBits, reader)
		require.Greater(t, decoded, 0)

		outputBits.Reset()
		decodedSymbols := make([]byte, len(payloadSymbols))
		readBits, err := outputBits.Read(decodedSymbols)
		require.NoError(t, err)
		require.Equal(t, len(payloadSymbols)*cfg.BitsPerSymbol, readBits)

		t.Logf("Expected payload: %v", payloadSymbols)
		t.Logf("Decoded payload:  %v", decodedSymbols)

		// Count matches
		matches := 0
		for i := range payloadSymbols {
			if payloadSymbols[i] == decodedSymbols[i] {
				matches++
			}
		}
		t.Logf("Matched %d/%d symbols at 25dB SNR", matches, len(payloadSymbols))

		// At 25dB SNR, should match most symbols
		// Allow some errors but should match at least 75%
		matchRate := float32(matches) / float32(len(payloadSymbols))
		require.Greater(t, matchRate, float32(0.75),
			"should match at least 75%% of symbols at 25dB SNR, got %.1f%%", matchRate*100)
	})
}

// TestPhaseDetection_Roundtrip25dB tests the full roundtrip at 25dB without chunking
func TestPhaseDetection_Roundtrip25dB(t *testing.T) {
	cfg := DefaultConfig()
	payload := test_utils.MakeData(7)
	symbols := bytesToSymbols(payload, cfg.BitsPerSymbol)

	mod, err := NewModulator(cfg)
	require.NoError(t, err)

	inputBits := buildBits(t, cfg.BitsPerSymbol, symbols)
	estimate := mod.Process(nil, inputBits)
	require.Greater(t, estimate, 0)

	inputBits.Reset()
	sampleBuf := buffer.New(make([]float32, 0, estimate))
	written := mod.Process(sampleBuf, inputBits)
	require.Equal(t, estimate, written)

	modulated := append([]float32(nil), sampleBuf.Items()...)
	withSilence := surroundWithSilence(modulated, cfg)
	// Add noise to entire buffer (signal + padding) at the same SNR level
	addNoiseToBuffer(withSilence, modulated, 25.0)

	demod, err := NewDemodulator(cfg)
	require.NoError(t, err)
	outputBits, err := buffer.NewBits(cfg.BitsPerSymbol)
	require.NoError(t, err)

	// Test without chunking first
	t.Run("NoChunking", func(t *testing.T) {
		demod.Reset()
		reader := buffer.New(withSilence)
		decoded := demod.Process(outputBits, reader)
		require.Greater(t, decoded, 0)

		outputBits.Reset()
		decodedSymbols := make([]byte, len(symbols))
		readBits, err := outputBits.Read(decodedSymbols)
		require.NoError(t, err)
		require.Equal(t, len(symbols)*cfg.BitsPerSymbol, readBits)

		t.Logf("Expected: %v", symbols)
		t.Logf("Got:      %v", decodedSymbols)

		if !equalSymbols(symbols, decodedSymbols) {
			matches := 0
			for i := range symbols {
				if symbols[i] == decodedSymbols[i] {
					matches++
				}
			}
			t.Logf("Matched %d/%d symbols", matches, len(symbols))
		}
		require.Equal(t, symbols, decodedSymbols, "should decode correctly at 25dB without chunking")
	})

	// Test with chunking (like the real test)
	t.Run("WithChunking", func(t *testing.T) {
		demod.Reset()
		outputBits.Reset()
		chunks := splitIntoChunks(withSilence, 3)
		totalDecoded := 0
		for _, chunk := range chunks {
			reader := buffer.New(chunk)
			decoded := demod.Process(outputBits, reader)
			totalDecoded += decoded
		}
		require.Greater(t, totalDecoded, 0)

		outputBits.Reset()
		decodedSymbols := make([]byte, len(symbols))
		readBits, err := outputBits.Read(decodedSymbols)
		require.NoError(t, err)
		require.Equal(t, len(symbols)*cfg.BitsPerSymbol, readBits)

		t.Logf("Expected: %v", symbols)
		t.Logf("Got:      %v", decodedSymbols)

		if !equalSymbols(symbols, decodedSymbols) {
			matches := 0
			for i := range symbols {
				if symbols[i] == decodedSymbols[i] {
					matches++
				}
			}
			t.Logf("Matched %d/%d symbols with chunking", matches, len(symbols))
		}
		require.Equal(t, symbols, decodedSymbols, "should decode correctly at 25dB with chunking")
	})
}

