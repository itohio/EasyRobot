package dsss

import (
	"fmt"
	"testing"

	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/x/buffer"
	test_utils "github.com/itohio/EasyRobot/x/math/dsp/modulation/test_utils"
	"github.com/stretchr/testify/require"
)

// Test helper: create standard test configuration
func createTestConfig(bitsPerSymbol int) Config {
	sampleRate := float32(96000)
	carrierFreq := float32(20000)
	symbolRate := float32(1000)
	spreadingFactor := 8

	modulationOrder := 1 << bitsPerSymbol
	codes := generateOrthogonalCodes(modulationOrder, spreadingFactor)
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

// generateOrthogonalCodes generates Walsh codes using Hadamard matrix construction.
// This ensures all codes are orthogonal to each other.
func generateOrthogonalCodes(numCodes, length int) [][]int8 {
	if numCodes <= 0 {
		numCodes = 1
	}
	// Ensure length is a power of 2 for Walsh codes
	actualLength := length
	if length <= 0 {
		actualLength = 1
	}
	// Find next power of 2 >= length
	nextPowerOf2 := 1
	for nextPowerOf2 < actualLength {
		nextPowerOf2 <<= 1
	}
	actualLength = nextPowerOf2

	// Generate Hadamard matrix (Walsh codes)
	hadamard := make([][]int8, actualLength)
	for i := range hadamard {
		hadamard[i] = make([]int8, actualLength)
	}

	// Initialize first element
	hadamard[0][0] = 1

	// Build Hadamard matrix recursively: H(2n) = [H(n) H(n); H(n) -H(n)]
	for n := 1; n < actualLength; n <<= 1 {
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				// Copy to top-right, top-left, bottom-left
				hadamard[i][j+n] = hadamard[i][j]
				hadamard[i+n][j] = hadamard[i][j]
				// Negate for bottom-right
				hadamard[i+n][j+n] = -hadamard[i][j]
			}
		}
	}

	// Extract requested number of codes, truncate to requested length
	codes := make([][]int8, numCodes)
	for i := 0; i < numCodes && i < actualLength; i++ {
		codes[i] = make([]int8, length)
		copy(codes[i], hadamard[i][:length])
	}

	// If we need more codes than the Hadamard matrix provides, repeat with different patterns
	// For now, we'll just use what we have (Walsh codes are limited to power-of-2 lengths)
	return codes
}

func TestRoundtrip_Basics(t *testing.T) {
	payloads := [][]byte{
		{0x42},
		{0x01, 0x23, 0x45, 0x67},
		test_utils.MakeData(33),
	}

	for _, payload := range payloads {
		config := createTestConfig(1)
		runRoundtrip(t, config, payload, math32.Inf(1))
	}
}

func TestRoundtrip_NoiseScenarios(t *testing.T) {
	config := createTestConfig(1)
	payload := test_utils.MakeData(7)

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
			payload := test_utils.MakeData(size)
			runRoundtrip(t, config, payload, snrDb)
		})
	}
}

func TestRoundtrip_ModulationOrders(t *testing.T) {
	payload := test_utils.MakeData(7)
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

	demod, err := NewDemodulator(cfg)
	require.NoError(t, err)
	// TODO: Set target SNR for SNR-based threshold calculation
	// demod.SetTargetSNR(snrDb)

	// Create bytesToBits converter for DSSS
	bytesToBits := func(data []byte) (buffer.Reader[byte], error) {
		symbols := bytesToSymbols(data, cfg.BitsPerSymbol)
		bits, err := buffer.NewBits(cfg.BitsPerSymbol)
		if err != nil {
			return nil, err
		}
		_, err = bits.Write(symbols)
		if err != nil {
			return nil, err
		}
		bits.Reset()
		return bits, nil
	}

	// Transmit: padding + payload + padding
	opts := test_utils.DefaultTransmitOptions()
	opts.IQInterleaved = false // DSSS uses real samples
	samples, _ := test_utils.Transmit(
		cfg.SampleRate,
		mod,
		[][]byte{payload},
		0.1, // 0.1s padding
		snrDb,
		bytesToBits,
		&opts,
	)
	require.NotNil(t, samples, "samples should not be nil")
	require.Greater(t, len(samples), 0, "samples should not be empty")

	// Receive: process samples through demodulator
	// Reset demodulator to ensure clean state
	demod.Reset()

	outputBits, err := buffer.NewBits(cfg.BitsPerSymbol)
	require.NoError(t, err)

	receiveOpts := test_utils.DefaultReceiveOptions()
	totalDecoded := test_utils.Receive(
		cfg.SampleRate,
		demod,
		outputBits,
		samples,
		&receiveOpts,
	)
	require.Greaterf(t, totalDecoded, 0, "snr=%.1f totalDecoded=%d", snrDb, totalDecoded)

	// Verify decoded data
	outputBits.Reset()
	symbols := bytesToSymbols(payload, cfg.BitsPerSymbol)

	// Read available symbols (may be less than expected if demodulator stopped early)
	availableSymbols := outputBits.Bits() / cfg.BitsPerSymbol
	if availableSymbols == 0 {
		t.Fatalf("no symbols decoded: Bits()=%d, BitsPerSymbol=%d, totalDecoded=%d", outputBits.Bits(), cfg.BitsPerSymbol, totalDecoded)
	}

	// Read what's available, but expect at least the payload symbols
	readSize := availableSymbols
	if readSize > len(symbols) {
		readSize = len(symbols)
	}
	decodedSymbols := make([]byte, readSize)
	readBits, err := outputBits.Read(decodedSymbols)
	if err != nil {
		t.Logf("outputBits state: Bits()=%d, Len()=%d, totalDecoded=%d, availableSymbols=%d", outputBits.Bits(), outputBits.Len(), totalDecoded, availableSymbols)
	}
	require.NoError(t, err)

	// We should have decoded at least the expected number of symbols
	require.GreaterOrEqualf(t, availableSymbols, len(symbols), "decoded %d symbols, expected at least %d (snr=%.1f)", availableSymbols, len(symbols), snrDb)
	require.Equalf(t, readSize*cfg.BitsPerSymbol, readBits, "snr=%.1f", snrDb)

	// Compare only the symbols we decoded
	expectedSlice := symbols[:readSize]
	if !equalSymbols(expectedSlice, decodedSymbols) {
		t.Logf("expected symbols=%v", expectedSlice)
		t.Logf("actual symbols  =%v", decodedSymbols)
		// Show first 10 symbols for debugging
		maxShow := 10
		if len(expectedSlice) < maxShow {
			maxShow = len(expectedSlice)
		}
		if maxShow > 0 {
			t.Logf("first %d expected: %v", maxShow, expectedSlice[:maxShow])
			t.Logf("first %d actual:   %v", maxShow, decodedSymbols[:maxShow])
		}
	}
	require.Equalf(t, expectedSlice, decodedSymbols, "symbol mismatch snr=%.1f", snrDb)

	// Convert back to bytes for final verification
	decodedBytes := symbolsToBytes(decodedSymbols, cfg.BitsPerSymbol, len(payload))
	expectedBytes := payload
	if len(decodedBytes) < len(expectedBytes) {
		expectedBytes = expectedBytes[:len(decodedBytes)]
	}
	require.Equalf(t, expectedBytes, decodedBytes, "byte mismatch snr=%.1f", snrDb)

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
