package mpsk

import (
	"testing"

	"github.com/chewxy/math32"
	"github.com/stretchr/testify/require"

	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/itohio/EasyRobot/x/math/dsp"
)

func TestRoundtrip_Basics(t *testing.T) {
	payloads := [][]byte{
		{0x5a},
		{0x10, 0x20, 0x30, 0x40},
		makePayload(33),
	}

	for _, payload := range payloads {
		cfg := DefaultConfig()
		runRoundtrip(t, cfg, payload, math32.Inf(1))
	}
}

func TestRoundtrip_NoiseScenarios(t *testing.T) {
	cfg := DefaultConfig()
	payload := makePayload(7)
	targetBER := float32(1e-5)
	marginDb := float32(2.5)

	// Test high SNR first, then sweep down to theoretical limit
	// Test 25dB first to verify tuned PLL parameters work
	for _, snr := range []float32{25, 30, 20, 15} {
		if !cfg.IsSNRSufficient(snr, targetBER, marginDb) {
			t.Logf("skipping snr=%.1fdB (minimum %.1fdB)", snr, cfg.MinimumSNR(targetBER, marginDb))
			continue
		}
		runRoundtrip(t, cfg, payload, snr)
	}

	// Test down to 10dB SNR (theoretical limit for QPSK)
	for _, snr := range []float32{10} {
		if !cfg.IsSNRSufficient(snr, targetBER, marginDb) {
			t.Logf("skipping snr=%.1fdB (minimum %.1fdB)", snr, cfg.MinimumSNR(targetBER, marginDb))
			continue
		}
		runRoundtrip(t, cfg, payload, snr)
	}

	for _, bits := range []int{1, 2, 3} {
		cfg := DefaultConfig()
		cfg.BitsPerSymbol = bits
		require.NoError(t, cfg.Validate())
		if !cfg.IsSNRSufficient(15, targetBER, marginDb) {
			t.Logf("skipping bits=%d at 15dB (minimum %.1fdB)", bits, cfg.MinimumSNR(targetBER, marginDb))
			continue
		}
		runRoundtrip(t, cfg, payload, 15)
	}
}

func TestRoundtrip_SingleSymbol_NoPreamble(t *testing.T) {
	cfg := DefaultConfig()
	cfg.PreambleLength = 0
	require.NoError(t, cfg.Validate())

	mod, err := NewModulator(cfg)
	require.NoError(t, err)
	demod, err := NewDemodulator(cfg)
	require.NoError(t, err)

	payload := []byte{0x2a}
	symbols := bytesToSymbols(payload, cfg.BitsPerSymbol)
	bits := buildBits(t, cfg.BitsPerSymbol, symbols)

	estimate := mod.Process(nil, bits)
	bits.Reset()
	sampleBuf := buffer.New(make([]float32, 0, estimate))
	written := mod.Process(sampleBuf, bits)
	require.Equal(t, estimate, written)

	outputBits, err := buffer.NewBits(cfg.BitsPerSymbol)
	require.NoError(t, err)
	reader := buffer.New(sampleBuf.Items())
	decoded := demod.Process(outputBits, reader)
	require.Greater(t, decoded, 0, "expected decoded bytes")
}

func runRoundtrip(t *testing.T, cfg Config, payload []byte, snrDb float32) {
	t.Helper()
	symbols := bytesToSymbols(payload, cfg.BitsPerSymbol)

	mod, err := NewModulator(cfg)
	require.NoError(t, err)

	inputBits := buildBits(t, cfg.BitsPerSymbol, symbols)

	estimate := mod.Process(nil, inputBits)
	require.Greaterf(t, estimate, 0, "bits=%d snr=%.1f", cfg.BitsPerSymbol, snrDb)

	// Calculate total samples needed: signal + padding (0.5s before and after)
	// estimateSamples already accounts for interleaved IQ (multiplies by 2)
	paddingSamples := int(0.5 * cfg.SampleRate) * 2 // IQ interleaved
	signalSamples := estimate
	totalSamples := paddingSamples*2 + signalSamples

	// Allocate single buffer for everything (no copying!)
	sampleBuffer := make([]float32, totalSamples)
	signalStart := paddingSamples
	signalEnd := paddingSamples + signalSamples

	inputBits.Reset()
	// Use slice of the buffer for modulation output (grow the slice as we write)
	// Use three-argument slice to set capacity limit
	signalBuf := buffer.New(sampleBuffer[signalStart:signalStart:signalEnd])
	written := mod.Process(signalBuf, inputBits)
	require.Equalf(t, signalSamples, written, "bits=%d snr=%.1f", cfg.BitsPerSymbol, snrDb)

	// Add noise to entire buffer (signal + padding) at the same SNR level
	// This is more realistic - padding should have noise too, not pure silence
	addNoiseToBuffer(sampleBuffer, sampleBuffer[signalStart:signalEnd], snrDb)

	demod, err := NewDemodulator(cfg)
	require.NoError(t, err)
	outputBits, err := buffer.NewBits(cfg.BitsPerSymbol)
	require.NoError(t, err)

	// Simulate streaming by splitting into chunks
	// Each chunk is a slice view of the same buffer (no copying!)
	chunks := splitIntoChunksFromBuffer(sampleBuffer, 3)
	totalDecoded := 0
	for _, chunk := range chunks {
		// chunk is already a slice view, use it directly
		reader := buffer.New(chunk)
		decoded := demod.Process(outputBits, reader)
		totalDecoded += decoded
	}
	require.Greaterf(t, totalDecoded, 0, "bits=%d snr=%.1f", cfg.BitsPerSymbol, snrDb)

	outputBits.Reset()
	decodedSymbols := make([]byte, len(symbols))
	readBits, err := outputBits.Read(decodedSymbols)
	require.NoError(t, err)
	require.Equalf(t, len(symbols)*cfg.BitsPerSymbol, readBits, "bits=%d snr=%.1f", cfg.BitsPerSymbol, snrDb)
	if !equalSymbols(symbols, decodedSymbols) {
		t.Logf("expected symbols=%v", symbols)
		t.Logf("actual symbols  =%v", decodedSymbols)
		// Show first 10 symbols for debugging
		maxShow := 10
		if len(symbols) < maxShow {
			maxShow = len(symbols)
		}
		t.Logf("first %d expected: %v", maxShow, symbols[:maxShow])
		t.Logf("first %d actual:   %v", maxShow, decodedSymbols[:maxShow])
	}
	require.Equalf(t, symbols, decodedSymbols, "symbol mismatch bits=%d snr=%.1f", cfg.BitsPerSymbol, snrDb)

	decodedBytes := symbolsToBytes(decodedSymbols, cfg.BitsPerSymbol, len(payload))
	require.Equalf(t, payload, decodedBytes, "bits=%d snr=%.1f", cfg.BitsPerSymbol, snrDb)
}

func buildBits(t *testing.T, bitsPerSymbol int, symbols []byte) *buffer.Bits {
	t.Helper()
	bits, err := buffer.NewBits(bitsPerSymbol)
	require.NoError(t, err)
	_, err = bits.Write(symbols)
	require.NoError(t, err)
	bits.Reset()
	return bits
}

func bytesToSymbols(data []byte, bitsPerSymbol int) []byte {
	if bitsPerSymbol <= 0 {
		return nil
	}
	mask := byte((1 << bitsPerSymbol) - 1)
	var symbols []byte
	acc := uint32(0)
	accBits := 0

	for _, b := range data {
		acc = (acc << 8) | uint32(b)
		accBits += 8
		for accBits >= bitsPerSymbol {
			shift := accBits - bitsPerSymbol
			symbol := byte((acc >> shift) & uint32(mask))
			symbols = append(symbols, symbol)
			accBits -= bitsPerSymbol
			acc &= (1 << shift) - 1
		}
	}

	if accBits > 0 {
		symbol := byte((acc << (bitsPerSymbol - accBits)) & uint32(mask))
		symbols = append(symbols, symbol)
	}
	return symbols
}

func symbolsToBytes(symbols []byte, bitsPerSymbol int, expectedBytes int) []byte {
	if bitsPerSymbol <= 0 {
		return nil
	}
	acc := uint32(0)
	accBits := 0
	result := make([]byte, 0, expectedBytes)

	for _, symbol := range symbols {
		acc = (acc << bitsPerSymbol) | uint32(symbol)
		accBits += bitsPerSymbol
		for accBits >= 8 {
			shift := accBits - 8
			result = append(result, byte((acc>>shift)&0xFF))
			accBits -= 8
			acc &= (1 << shift) - 1
			if len(result) == expectedBytes {
				return result
			}
		}
	}

	if accBits > 0 && len(result) < expectedBytes {
		result = append(result, byte(acc<<(8-accBits)))
	}
	if len(result) > expectedBytes {
		result = result[:expectedBytes]
	}
	return result
}

func makePayload(n int) []byte {
	payload := make([]byte, n)
	for i := range payload {
		payload[i] = byte(i*37 + 13)
	}
	return payload
}

func equalSymbols(a, b []byte) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func surroundWithSilence(samples []float32, cfg Config) []float32 {
	silence := int(0.5 * cfg.SampleRate)
	totalSilence := silence * 2
	total := make([]float32, totalSilence+len(samples)+totalSilence)
	copy(total[totalSilence:], samples)
	return total
}

// splitIntoChunksFromBuffer splits a buffer into chunk slices (views, no copying)
// All chunks are views into the same underlying buffer
func splitIntoChunksFromBuffer(samples []float32, parts int) [][]float32 {
	if parts <= 1 {
		return [][]float32{samples}
	}
	chunks := make([][]float32, 0, parts)
	chunkSize := len(samples) / parts
	for i := 0; i < parts; i++ {
		start := i * chunkSize
		end := start + chunkSize
		if i == parts-1 {
			end = len(samples)
		}
		// Use slice view (no copying) - all chunks point to same underlying buffer
		chunks = append(chunks, samples[start:end])
	}
	return chunks
}

// splitIntoChunks splits samples into chunks with copying (for backward compatibility)
func splitIntoChunks(samples []float32, parts int) [][]float32 {
	if parts <= 1 {
		return [][]float32{samples}
	}
	chunks := make([][]float32, 0, parts)
	chunkSize := len(samples) / parts
	for i := 0; i < parts; i++ {
		start := i * chunkSize
		end := start + chunkSize
		if i == parts-1 {
			end = len(samples)
		}
		chunk := make([]float32, end-start)
		copy(chunk, samples[start:end])
		chunks = append(chunks, chunk)
	}
	return chunks
}

// addNoise adds noise to samples, calculating SNR from the samples themselves.
func addNoise(samples []float32, snrDb float32) {
	if math32.IsInf(snrDb, 1) || snrDb <= 0 {
		return
	}
	signalPower := float32(0)
	for _, v := range samples {
		signalPower += v * v
	}
	if len(samples) == 0 {
		return
	}
	signalPower /= float32(len(samples))
	if signalPower == 0 {
		return
	}
	noisePower := signalPower / math32.Pow(10, snrDb/10)
	noiseAmp := math32.Sqrt(noisePower)
	gen := dsp.NewNoiseWhiteGenerator(noiseAmp)
	noise := make([]float32, len(samples))
	gen.Generate(noise)
	for i := range samples {
		samples[i] += noise[i]
	}
}

// addNoiseToBuffer adds noise to the entire buffer at the same SNR level as the signal region.
// This ensures padding has noise too, not pure silence, which is more realistic for noise tests.
func addNoiseToBuffer(buffer []float32, signalRegion []float32, snrDb float32) {
	if math32.IsInf(snrDb, 1) || snrDb <= 0 {
		return
	}
	if len(signalRegion) == 0 {
		// If no signal region, just add noise to buffer
		addNoise(buffer, snrDb)
		return
	}
	// Calculate signal power from the signal region only (non-zero samples)
	signalPower := float32(0)
	signalSamples := 0
	for _, v := range signalRegion {
		if v != 0 {
			signalPower += v * v
			signalSamples++
		}
	}
	if signalSamples == 0 {
		// No signal, just add noise
		addNoise(buffer, snrDb)
		return
	}
	signalPower /= float32(signalSamples)
	if signalPower == 0 {
		return
	}
	// Calculate noise power based on signal power and desired SNR
	noisePower := signalPower / math32.Pow(10, snrDb/10)
	noiseAmp := math32.Sqrt(noisePower)
	
	// Generate noise for entire buffer (signal + padding) at same level
	gen := dsp.NewNoiseWhiteGenerator(noiseAmp)
	noise := make([]float32, len(buffer))
	gen.Generate(noise)
	for i := range buffer {
		buffer[i] += noise[i]
	}
}
