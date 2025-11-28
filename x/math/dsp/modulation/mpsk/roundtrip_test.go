package mpsk

import (
	"testing"

	"github.com/chewxy/math32"
	"github.com/stretchr/testify/require"

	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/itohio/EasyRobot/x/math/dsp"
	"github.com/itohio/EasyRobot/x/math/dsp/modulation"
	test_utils "github.com/itohio/EasyRobot/x/math/dsp/modulation/test_utils"
)

func TestRoundtrip_Basics(t *testing.T) {
	payloads := [][]byte{
		{0x5a},
		{0x10, 0x20, 0x30, 0x40},
		test_utils.MakeData(33),
	}

	for _, payload := range payloads {
		cfg := DefaultConfig()
		runRoundtrip(t, cfg, payload, math32.Inf(1))
	}
}

func TestRoundtrip_NoiseScenarios(t *testing.T) {
	cfg := DefaultConfig()
	payload := test_utils.MakeData(7)
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

	payload := []byte{0x2a}
	runRoundtrip(t, cfg, payload, math32.Inf(1))
}

func runRoundtrip(t *testing.T, cfg Config, payload []byte, snrDb float32) {
	t.Helper()

	mod, err := NewModulator(cfg)
	require.NoError(t, err)

	demod, err := NewDemodulator(cfg)
	require.NoError(t, err)

	// Create bytesToBits converter for MPSK
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

	// Calculate expected transmitter signal strength from config
	expectedTxPower := cfg.Amplitude * cfg.Amplitude
	expectedTxSignalStrength := float32(0)
	if expectedTxPower > 0 {
		expectedTxSignalStrength = 10.0 * math32.Log10(expectedTxPower)
	}

	// Track transmission statistics for inspection
	transmitSignalStart := -1
	transmitSignalEnd := -1

	// Transmit: padding + payload + padding
	opts := test_utils.DefaultTransmitOptions()
	opts.IQInterleaved = true // MPSK uses IQ interleaved samples

	// Inspect callback to verify transmitter stats
	opts.Inspect = func(startBufIdx, endBufIdx int, startSampleIdx, endSampleIdx int, tx modulation.Transmitter) {
		params := tx.Params()
		require.Equal(t, cfg.SymbolRate(), params.BaudRate, "BaudRate should match config")
		require.Equal(t, cfg.SampleRate, params.SampleRate, "SampleRate should match config")
		require.Equal(t, cfg.CarrierHz, params.CarrierFrequency, "CarrierFrequency should match config")
		require.InDeltaf(t, expectedTxSignalStrength, params.SignalStrength, 0.1,
			"SignalStrength should match expected value from config")

		transmitSignalStart = startSampleIdx
		transmitSignalEnd = endSampleIdx
	}

	samples, _ := test_utils.Transmit(
		cfg.SampleRate,
		mod,
		[][]byte{payload},
		0.5, // 0.5s padding (matching original test)
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
	receiveOpts.ChunkParts = 3 // Match original test chunk count

	// Inspect callback to verify receiver stats
	var lockedQuality modulation.QualityStats
	var finalTransmission modulation.TransmissionStats
	var wasLocked bool

	receiveOpts.Inspect = func(startSampleIdx, endSampleIdx int, startBufIdx, endBufIdx int, rx modulation.Receiver) {
		quality := rx.Quality()
		transmission := rx.Transmission()
		params := rx.Params()

		require.Equal(t, cfg.SymbolRate(), params.BaudRate, "BaudRate should match config")
		require.Equal(t, cfg.SampleRate, params.SampleRate, "SampleRate should match config")
		require.Equal(t, cfg.CarrierHz, params.CarrierFrequency, "CarrierFrequency should match config")

		if transmission.IsLocked {
			wasLocked = true
			lockedQuality = quality
			require.False(t, math32.IsNaN(quality.SNR), "SNR should not be NaN when locked")
			require.False(t, math32.IsInf(quality.SNR, 0), "SNR should not be Inf when locked")
			if snrDb < math32.Inf(1) {
				require.Greaterf(t, quality.SNR, float32(0),
					"SNR should be positive when locked (simulated SNR=%.2f dB)", snrDb)
			}
			require.False(t, math32.IsNaN(quality.SignalStrength), "SignalStrength should not be NaN when locked")
			require.False(t, math32.IsInf(quality.SignalStrength, 0), "SignalStrength should not be Inf when locked")
		}

		finalTransmission = transmission
	}

	totalDecoded := test_utils.Receive(
		cfg.SampleRate,
		demod,
		outputBits,
		samples,
		&receiveOpts,
	)
	require.Greaterf(t, totalDecoded, 0, "bits=%d snr=%.1f", cfg.BitsPerSymbol, snrDb)

	// Verify receiver stats
	require.Greater(t, finalTransmission.LockCount, 0, "Demodulator should have locked at least once during transmission")

	if wasLocked && snrDb < math32.Inf(1) {
		require.InDeltaf(t, snrDb, lockedQuality.SNR, 8.0,
			"Measured SNR should be close to simulated SNR (simulated=%.2f dB, measured=%.2f dB)",
			snrDb, lockedQuality.SNR)
	}

	if wasLocked {
		require.InDeltaf(t, expectedTxSignalStrength, lockedQuality.SignalStrength, 10.0,
			"Receiver SignalStrength should be in same ballpark as transmitter (tx=%.2f dB, rx=%.2f dB)",
			expectedTxSignalStrength, lockedQuality.SignalStrength)
	}

	// Verify transmission indices
	if transmitSignalStart >= 0 && transmitSignalEnd > transmitSignalStart {
		require.Greater(t, finalTransmission.TransmissionStart, 0, "TransmissionStart must be set")
		paddingSamples := int(0.5 * cfg.SampleRate * 2) // MPSK uses IQ interleaved, so double
		require.GreaterOrEqualf(t, finalTransmission.TransmissionStart, transmitSignalStart-paddingSamples,
			"TransmissionStart should be within reasonable range of actual signal start")
		require.LessOrEqualf(t, finalTransmission.TransmissionStart, transmitSignalEnd,
			"TransmissionStart should be before signal end")
	}

	// Verify decoded data
	outputBits.Reset()
	symbols := bytesToSymbols(payload, cfg.BitsPerSymbol)
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

// Internal helper functions for other test files in this package
func buildBits(t *testing.T, bitsPerSymbol int, symbols []byte) *buffer.Bits {
	t.Helper()
	bits, err := buffer.NewBits(bitsPerSymbol)
	require.NoError(t, err)
	_, err = bits.Write(symbols)
	require.NoError(t, err)
	bits.Reset()
	return bits
}

func surroundWithSilence(samples []float32, cfg Config) []float32 {
	silence := int(0.5*cfg.SampleRate) * 2 // IQ interleaved
	total := make([]float32, silence*2+len(samples))
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
	gen.Add(samples)
}

func addNoiseToBuffer(buffer []float32, signalRegion []float32, snrDb float32) {
	if math32.IsInf(snrDb, 1) || snrDb <= 0 {
		return
	}
	if len(signalRegion) == 0 {
		addNoise(buffer, snrDb)
		return
	}
	signalPower := float32(0)
	signalSamples := 0
	for _, v := range signalRegion {
		if v != 0 {
			signalPower += v * v
			signalSamples++
		}
	}
	if signalSamples == 0 {
		addNoise(buffer, snrDb)
		return
	}
	signalPower /= float32(signalSamples)
	if signalPower == 0 {
		return
	}
	noisePower := signalPower / math32.Pow(10, snrDb/10)
	noiseAmp := math32.Sqrt(noisePower)
	gen := dsp.NewNoiseWhiteGenerator(noiseAmp)
	gen.Add(buffer)
}

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
		chunks = append(chunks, samples[start:end])
	}
	return chunks
}
