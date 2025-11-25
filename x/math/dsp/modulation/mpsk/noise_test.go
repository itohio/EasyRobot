package mpsk

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/itohio/EasyRobot/x/buffer"
)

// TestRoundtrip_1Byte_Noise tests a single byte with various noise levels
// ensuring we stay within theoretical SNR limits
func TestRoundtrip_1Byte_Noise(t *testing.T) {
	cfg := DefaultConfig()
	require.NoError(t, cfg.Validate())

	payload := []byte{0x5a}
	targetBER := float32(1e-5)
	marginDb := float32(2.5)

	// Check theoretical limits
	minSNR := cfg.MinimumSNR(targetBER, marginDb)
	t.Logf("Config: bits=%d, symbolRate=%.1fHz, bandwidth=%.1fHz", 
		cfg.BitsPerSymbol, cfg.SymbolRate(), cfg.RequiredBandwidth())
	t.Logf("Theoretical minimum SNR: %.2f dB (BER=1e-5, margin=%.1fdB)", minSNR, marginDb)

	// Test at various SNR levels, starting high and working down
	testSNRs := []float32{30, 25, 20, 15, 12}
	for _, snr := range testSNRs {
		if snr < minSNR {
			t.Logf("Skipping SNR=%.1fdB (below theoretical minimum %.2fdB)", snr, minSNR)
			continue
		}
		t.Run(fmt.Sprintf("SNR_%.1fdB", snr), func(t *testing.T) {
			runRoundtrip1Byte(t, cfg, payload, snr)
		})
	}
}

func runRoundtrip1Byte(t *testing.T, cfg Config, payload []byte, snrDb float32) {
	t.Helper()
	symbols := bytesToSymbols(payload, cfg.BitsPerSymbol)

	mod, err := NewModulator(cfg)
	require.NoError(t, err)

	inputBits := buildBits(t, cfg.BitsPerSymbol, symbols)

	estimate := mod.Process(nil, inputBits)
	require.Greaterf(t, estimate, 0, "snr=%.1f", snrDb)

	inputBits.Reset()
	sampleBuf := buffer.New(make([]float32, 0, estimate))
	written := mod.Process(sampleBuf, inputBits)
	require.Equalf(t, estimate, written, "snr=%.1f", snrDb)

	modulated := append([]float32(nil), sampleBuf.Items()...)
	withSilence := surroundWithSilence(modulated, cfg)
	
	// Verify bandwidth before adding noise
	bw := cfg.RequiredBandwidth()
	nyquist := cfg.SampleRate / 2
	if bw > nyquist {
		t.Fatalf("Bandwidth %.1fHz exceeds Nyquist %.1fHz", bw, nyquist)
	}
	t.Logf("Bandwidth check: %.1fHz < %.1fHz (Nyquist) - OK", bw, nyquist)
	
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
	require.Greaterf(t, totalDecoded, 0, "snr=%.1f", snrDb)

	outputBits.Reset()
	decodedSymbols := make([]byte, len(symbols))
	readBits, err := outputBits.Read(decodedSymbols)
	require.NoError(t, err)
	require.Equalf(t, len(symbols)*cfg.BitsPerSymbol, readBits, "snr=%.1f", snrDb)
	require.Equalf(t, symbols, decodedSymbols, "symbol mismatch snr=%.1f", snrDb)

	decodedBytes := symbolsToBytes(decodedSymbols, cfg.BitsPerSymbol, len(payload))
	require.Equalf(t, payload, decodedBytes, "payload mismatch snr=%.1f", snrDb)
}

