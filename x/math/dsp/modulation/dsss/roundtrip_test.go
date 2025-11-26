package dsss

import (
	"fmt"
	"testing"

	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/itohio/EasyRobot/x/math/dsp"
	"github.com/stretchr/testify/require"
)

// Test helper: create standard test configuration
func createTestConfig(bitsPerSymbol int) Config {
	sampleRate := float32(96000)
	carrierFreq := float32(20000)
	symbolRate := float32(1000)
	spreadingFactor := 8

	codes := generateOrthogonalCodes(bitsPerSymbol, spreadingFactor)
	samplesPerChip := int(sampleRate / (symbolRate * float32(spreadingFactor)))

	return Config{
		BitsPerSymbol:       bitsPerSymbol,
		ModulationOrder:     1 << bitsPerSymbol,
		Codes:               codes,
		SpreadingFactor:     spreadingFactor,
		SampleRate:          sampleRate,
		SymbolRate:          symbolRate,
		CarrierFreq:         carrierFreq,
		SamplesPerChip:      samplesPerChip,
		PulseShape:          "raised_cosine",
		Rolloff:             0.35,
		PulseFilterSpan:     4,
		PreambleLength:      4,
		PreamblePattern:     []int{0, 1, 0, 1},
		EnergyThreshold:     0.001, // Will be overridden by SNR-based threshold
		DetectionWindow:     samplesPerChip * spreadingFactor,
		TransmissionTimeout: 3,
		Amplitude:           1.0,
	}
}

func generateOrthogonalCodes(numCodes, length int) [][]int8 {
	if numCodes <= 0 {
		numCodes = 1
	}
	codes := make([][]int8, numCodes)
	for i := range codes {
		codes[i] = make([]int8, length)
		for j := range codes[i] {
			if ((i + j) & 1) == 0 {
				codes[i][j] = 1
			} else {
				codes[i][j] = -1
			}
		}
	}
	return codes
}

func TestRoundtrip_Basics(t *testing.T) {
	payloads := [][]byte{
		{0x42},
		{0x01, 0x23, 0x45, 0x67},
		makePayload(33),
	}

	for _, payload := range payloads {
		config := createTestConfig(1)
		runRoundtrip(t, config, payload, math32.Inf(1))
	}
}

func TestRoundtrip_NoiseScenarios(t *testing.T) {
	config := createTestConfig(1)
	payload := makePayload(7)

	// Test high SNR first, then sweep down to theoretical limit
	// DSSS with spreading factor 8 has ~9dB processing gain
	// Theoretical limit for BPSK is ~10dB, so with processing gain we can go lower
	for _, snr := range []float32{30, 25, 20, 15, 12, 10, 8} {
		t.Run(fmt.Sprintf("SNR_%.0fdB", snr), func(t *testing.T) {
			runRoundtrip(t, config, payload, snr)
		})
	}
}

func TestRoundtrip_PacketSizes(t *testing.T) {
	config := createTestConfig(1)
	snrDb := float32(16.0) // Moderate SNR

	packetSizes := []int{1, 7, 33, 128, 512, 1024}
	for _, size := range packetSizes {
		t.Run(fmt.Sprintf("%d_bytes", size), func(t *testing.T) {
			payload := makePayload(size)
			runRoundtrip(t, config, payload, snrDb)
		})
	}
}

func TestRoundtrip_ModulationOrders(t *testing.T) {
	payload := makePayload(7)
	snrDb := float32(20.0) // Higher SNR for higher order modulations

	for _, bitsPerSymbol := range []int{1, 2, 3} {
		t.Run(fmt.Sprintf("%d_bits_per_symbol", bitsPerSymbol), func(t *testing.T) {
			config := createTestConfig(bitsPerSymbol)
			runRoundtrip(t, config, payload, snrDb)
		})
	}
}

func runRoundtrip(t *testing.T, cfg Config, payload []byte, snrDb float32) {
	t.Helper()

	mod, err := NewModulator(cfg)
	require.NoError(t, err)

	symbols := bytesToSymbols(payload, cfg.BitsPerSymbol)
	inputBits := buildBits(t, cfg.BitsPerSymbol, symbols)

	estimate := mod.Process(nil, inputBits)
	require.Greaterf(t, estimate, 0, "snr=%.1f", snrDb)

	// Calculate total samples needed: padding + signal + padding
	// DSSS uses real samples (not IQ interleaved)
	paddingSamples := int(0.1 * cfg.SampleRate) // 0.1s padding
	signalSamples := estimate
	totalSamples := paddingSamples*2 + signalSamples

	// Allocate single buffer for everything (zero-copy!)
	sampleBuffer := make([]float32, totalSamples)
	signalStart := paddingSamples
	signalEnd := paddingSamples + signalSamples

	inputBits.Reset()
	// Use slice view of the buffer for modulation output (zero-copy)
	signalBuf := buffer.New(sampleBuffer[signalStart:signalStart:signalEnd])
	written := mod.Process(signalBuf, inputBits)
	require.Equalf(t, signalSamples, written, "snr=%.1f", snrDb)

	// Add noise to entire buffer (signal + padding) at the same SNR level
	// This is more realistic - padding should have noise too, not pure silence
	addNoiseToBuffer(sampleBuffer, sampleBuffer[signalStart:signalEnd], snrDb)

	demod, err := NewDemodulator(cfg)
	require.NoError(t, err)
	// TODO: Set target SNR for SNR-based threshold calculation
	// demod.SetTargetSNR(snrDb)

	outputBits, err := buffer.NewBits(cfg.BitsPerSymbol)
	require.NoError(t, err)

	// Simulate streaming by splitting into chunks
	// Each chunk is a slice view of the same buffer (no copying!)
	chunks := splitIntoChunksFromBuffer(sampleBuffer, 10)
	totalDecoded := 0
	for _, chunk := range chunks {
		// chunk is already a slice view, use it directly
		reader := buffer.New(chunk)
		decoded := demod.Process(outputBits, reader)
		totalDecoded += decoded
	}
	require.Greaterf(t, totalDecoded, 0, "snr=%.1f", snrDb)

	outputBits.Reset()
	decodedSymbols := make([]byte, len(symbols))
	readBits, err := outputBits.Read(decodedSymbols)
	require.NoError(t, err)
	require.Equalf(t, len(symbols)*cfg.BitsPerSymbol, readBits, "snr=%.1f", snrDb)

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
	require.Equalf(t, symbols, decodedSymbols, "symbol mismatch snr=%.1f", snrDb)

	decodedBytes := symbolsToBytes(decodedSymbols, cfg.BitsPerSymbol, len(payload))
	require.Equalf(t, payload, decodedBytes, "snr=%.1f", snrDb)
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
	totalBits := len(data) * 8
	symbols := (totalBits + bitsPerSymbol - 1) / bitsPerSymbol

	out := make([]byte, symbols)
	bitIdx := 0
	for i := range symbols {
		var value byte
		for j := 0; j < bitsPerSymbol; j++ {
			value <<= 1
			if bitIdx < totalBits {
				byteIdx := bitIdx / 8
				bitPos := 7 - (bitIdx % 8)
				value |= (data[byteIdx] >> bitPos) & 0x1
			}
			bitIdx++
		}
		out[i] = value
	}
	return out
}

func symbolsToBytes(symbols []byte, bitsPerSymbol int, expectedBytes int) []byte {
	if bitsPerSymbol <= 0 || expectedBytes <= 0 {
		return nil
	}
	out := make([]byte, expectedBytes)
	totalBits := expectedBytes * 8
	bitIdx := 0

	for _, symbol := range symbols {
		for bit := bitsPerSymbol - 1; bit >= 0; bit-- {
			if bitIdx >= totalBits {
				return out
			}
			byteIdx := bitIdx / 8
			bitPos := 7 - (bitIdx % 8)
			bitVal := (symbol >> uint(bit)) & 0x1
			out[byteIdx] |= bitVal << uint(bitPos)
			bitIdx++
		}
	}
	return out
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

// splitIntoChunksFromBuffer splits a buffer into chunk slices (views, no copying)
// All chunks are views into the same underlying buffer
func splitIntoChunksFromBuffer(samples []float32, parts int) [][]float32 {
	if parts <= 1 {
		return [][]float32{samples}
	}
	chunks := make([][]float32, 0, parts)
	chunkSize := len(samples) / parts
	for i := range parts {
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
