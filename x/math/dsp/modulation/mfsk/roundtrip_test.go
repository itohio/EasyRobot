package mfsk

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
	payload := makePayload(7)
	targetBER := float32(1e-5)
	marginDb := float32(2.5)

	// Test with SNR values above theoretical minimum with margin
	// Use SNR values that are at least 1 dB above theoretical minimum
	cfg := DefaultConfig()
	minSNR := cfg.MinimumSNR(targetBER, marginDb)
	testSNRs := []float32{minSNR + 5, minSNR + 2, minSNR + 0.5}
	for _, snr := range testSNRs {
		cfg := DefaultConfig()
		if snr < minSNR {
			t.Skipf("skipping SNR=%.1fdB test for bits=%d (below theoretical minimum %.1fdB)",
				snr, cfg.BitsPerSymbol, minSNR)
			continue
		}
		runRoundtrip(t, cfg, payload, snr)
	}

	// Test different M-FSK orders
	// M-FSK uses M frequencies: 2-FSK (1 bit), 4-FSK (2 bits), 8-FSK (3 bits), 16-FSK (4 bits)
	testConfigs := []struct {
		bitsPerSymbol int
		mFskOrder     int // M = number of frequencies
		description   string
	}{
		{1, 2, "2-FSK"},
		{2, 4, "4-FSK"},
		{3, 8, "8-FSK"},
		{4, 16, "16-FSK"},
	}

	for _, tc := range testConfigs {
		cfg := DefaultConfig()
		cfg.BitsPerSymbol = tc.bitsPerSymbol
		if err := cfg.Validate(); err != nil {
			t.Fatalf("invalid config for %s (bits=%d): %v", tc.description, tc.bitsPerSymbol, err)
		}
		// Verify tone count matches M-FSK order
		if cfg.toneCount() != tc.mFskOrder {
			t.Fatalf("tone count mismatch for %s: expected %d, got %d",
				tc.description, tc.mFskOrder, cfg.toneCount())
		}

		minSNR := cfg.MinimumSNR(targetBER, marginDb)
		// Use SNR that is well above theoretical minimum for reliable operation
		// Higher M-FSK needs more margin due to increased sensitivity to noise
		testSNR := minSNR + 2

		// Higher M-FSK (8-FSK, 16-FSK) requires more SNR due to frequency resolution limits
		// with non-coherent Goertzel detection. Use higher margins for reliable operation.
		if tc.mFskOrder >= 8 {
			// For 8-FSK and 16-FSK, use higher SNR margin
			testSNR = minSNR + 4
			if tc.mFskOrder >= 16 {
				// 16-FSK needs even more margin
				testSNR = minSNR + 5
			}
			// Ensure minimum SNR for high M-FSK
			if testSNR < 20 {
				testSNR = 20
			}
		}

		runRoundtrip(t, cfg, payload, testSNR)
	}
}

func runRoundtrip(t *testing.T, cfg Config, payload []byte, snrDb float32) {
	t.Helper()
	symbols := bytesToSymbols(payload, cfg.BitsPerSymbol)
	if testing.Verbose() {
		t.Logf("cfg bits=%d SPS=%d spacing=%.2fHz", cfg.BitsPerSymbol, cfg.SamplesPerSymbol, cfg.SymbolRate())
	}

	mod, err := NewModulator(cfg)
	require.NoError(t, err)
	if testing.Verbose() {
		t.Logf("validated bits=%d SPS=%d spacing=%.2fHz", mod.cfg.BitsPerSymbol, mod.cfg.SamplesPerSymbol, mod.cfg.SymbolRate())
	}

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

func surroundWithSilence(samples []float32, cfg Config) []float32 {
	silence := int(0.5 * cfg.SampleRate)
	total := make([]float32, silence+len(samples)+silence)
	copy(total[silence:], samples)
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
