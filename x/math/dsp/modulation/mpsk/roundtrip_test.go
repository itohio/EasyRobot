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

	for _, snr := range []float32{10, 5} {
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

	inputBits.Reset()
	sampleBuf := buffer.New(make([]float32, 0, estimate))
	written := mod.Process(sampleBuf, inputBits)
	require.Equalf(t, estimate, written, "bits=%d snr=%.1f", cfg.BitsPerSymbol, snrDb)

	modulated := append([]float32(nil), sampleBuf.Items()...)
	withSilence := surroundWithSilence(modulated, cfg)
	addNoise(withSilence, snrDb)

	demod, err := NewDemodulator(cfg)
	require.NoError(t, err)
	outputBits, err := buffer.NewBits(cfg.BitsPerSymbol)
	require.NoError(t, err)

	chunks := splitIntoChunks(withSilence, 3)
	totalDecoded := 0
	for _, chunk := range chunks {
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
