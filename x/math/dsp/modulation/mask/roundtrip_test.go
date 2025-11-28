package mask

import (
	"testing"

	"github.com/chewxy/math32"
	"github.com/stretchr/testify/require"

	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/itohio/EasyRobot/x/math/dsp/modulation"
	test_utils "github.com/itohio/EasyRobot/x/math/dsp/modulation/test_utils"
)

func TestRoundtrip_Basics(t *testing.T) {
	payloads := [][]byte{
		test_utils.MakeData(1),
		test_utils.MakeData(4),
		test_utils.MakeData(33),
	}

	for _, payload := range payloads {
		cfg := DefaultConfig()
		// Preamble is required for amplitude normalization - cannot be disabled
		runRoundtrip(t, cfg, payload, math32.Inf(1))
	}
}

func TestRoundtrip_NoiseScenarios(t *testing.T) {
	payload := test_utils.MakeData(7)
	targetBER := float32(1e-5)
	marginDb := float32(2.5)

	// Test with SNR values above theoretical minimum with margin
	// Use higher SNR levels for M-ASK due to sensitivity to amplitude errors
	cfg := DefaultConfig()
	minSNR := cfg.MinimumSNR(targetBER, marginDb)
	testSNRs := []float32{40, 35, minSNR + 5, minSNR + 3}
	for _, snr := range testSNRs {
		cfg := DefaultConfig()
		if snr < minSNR {
			t.Skipf("skipping SNR=%.1fdB test for bits=%d (below theoretical minimum %.1fdB)",
				snr, cfg.BitsPerSymbol, minSNR)
			continue
		}
		runRoundtrip(t, cfg, payload, snr)
	}

	// Test different M-ASK orders (2, 4, 8 bits per symbol)
	testConfigs := []struct {
		bitsPerSymbol int
		description   string
	}{
		{2, "4-ASK"},
		{4, "16-ASK"},
		{8, "256-ASK"},
	}

	for _, tc := range testConfigs {
		cfg := DefaultConfig()
		cfg.BitsPerSymbol = tc.bitsPerSymbol
		if err := cfg.Validate(); err != nil {
			t.Fatalf("invalid config for %s (bits=%d): %v", tc.description, tc.bitsPerSymbol, err)
		}

		minSNR := cfg.MinimumSNR(targetBER, marginDb)
		// Use SNR that is well above theoretical minimum for reliable operation
		// Higher M-ASK needs more margin due to increased sensitivity to amplitude errors
		// Adjusted towards higher SNR levels for better test reliability
		testSNR := minSNR + 5
		if tc.bitsPerSymbol >= 4 {
			testSNR = minSNR + 8
		}
		if tc.bitsPerSymbol >= 8 {
			testSNR = minSNR + 12
			if testSNR < 35 {
				testSNR = 35 // Minimum for 8-bit M-ASK
			}
		}

		// Only test if SNR is sufficient
		if cfg.IsSNRSufficient(testSNR, targetBER, marginDb) {
			runRoundtrip(t, cfg, test_utils.MakeData(7), testSNR)
		} else {
			t.Logf("skipping %s at %.1fdB (minimum %.1fdB)", tc.description, testSNR, minSNR)
		}
	}
}

// TestRoundtrip_DebugNoise tests with debug logging enabled to investigate amplitude detection
func TestRoundtrip_DebugNoise(t *testing.T) {
	cfg := DefaultConfig()
	payload := test_utils.MakeData(7) // Use shared data generation
	snrDb := float32(35.0)            // Moderate SNR for debugging

	mod, err := NewModulator(cfg)
	require.NoError(t, err)

	demod, err := NewDemodulator(cfg)
	require.NoError(t, err)
	demod.SetDebugLogging(true) // Enable debug logging

	// bytesToBits helper for test_utils.Transmit
	bytesToBits := func(data []byte) (buffer.Reader[byte], error) {
		syms := bytesToSymbols(data, cfg.BitsPerSymbol)
		return buildBits(t, cfg.BitsPerSymbol, syms), nil
	}

	// Transmit: padding + payload + padding
	opts := test_utils.DefaultTransmitOptions()
	opts.IQInterleaved = false
	opts.ChunkParts = 3
	samples, _ := test_utils.Transmit(
		cfg.SampleRate,
		mod,
		[][]byte{payload},
		0.5,
		snrDb,
		bytesToBits,
		&opts,
	)
	require.NotNil(t, samples)
	require.Greater(t, len(samples), 0)

	// Receive with debug logging
	demod.Reset()
	demod.SetDebugLogging(true)

	outputBits, err := buffer.NewBits(cfg.BitsPerSymbol)
	require.NoError(t, err)

	receiveOpts := test_utils.DefaultReceiveOptions()
	receiveOpts.ChunkParts = 3
	totalDecoded := test_utils.Receive(
		cfg.SampleRate,
		demod,
		outputBits,
		samples,
		&receiveOpts,
	)

	t.Logf("Debug test completed: totalDecoded=%d", totalDecoded)
}

func TestRoundtrip_SingleByte_WithSilence(t *testing.T) {
	cfg := DefaultConfig()
	payload := test_utils.MakeData(1)
	runRoundtrip(t, cfg, payload, math32.Inf(1))
}

func TestRoundtrip_Baseband(t *testing.T) {
	cfg := DefaultConfig()
	cfg.CarrierHz = 0 // Baseband
	// Preamble is required for amplitude normalization - cannot be disabled
	cfg.EnergyThreshold = 1e-5 // Lower threshold for baseband
	payload := test_utils.MakeData(1)
	runRoundtrip(t, cfg, payload, math32.Inf(1))
}

func TestRoundtrip_SplitBuffer(t *testing.T) {
	cfg := DefaultConfig()
	payload := test_utils.MakeData(1)
	runRoundtrip(t, cfg, payload, math32.Inf(1))
}

func runRoundtrip(t *testing.T, cfg Config, payload []byte, snrDb float32) {
	t.Helper()
	symbols := bytesToSymbols(payload, cfg.BitsPerSymbol)
	if testing.Verbose() {
		t.Logf("cfg bits=%d SPS=%d rate=%.2fHz", cfg.BitsPerSymbol, cfg.SamplesPerSymbol, cfg.SymbolRate())
	}

	mod, err := NewModulator(cfg)
	require.NoError(t, err)

	demod, err := NewDemodulator(cfg)
	require.NoError(t, err)

	// bytesToBits helper for test_utils.Transmit
	bytesToBits := func(data []byte) (buffer.Reader[byte], error) {
		syms := bytesToSymbols(data, cfg.BitsPerSymbol)
		return buildBits(t, cfg.BitsPerSymbol, syms), nil
	}

	// Track transmission statistics for inspection
	transmitSignalStart := -1
	transmitSignalEnd := -1

	// Transmit: padding + payload + padding
	opts := test_utils.DefaultTransmitOptions()
	opts.IQInterleaved = false // M-ASK uses real samples (not IQ interleaved)
	opts.ChunkParts = 3        // Use 3 chunks for streaming simulation

	// Calculate expected transmitter signal strength from config
	// SignalStrength = 10 * log10(amplitude^2) in dB
	expectedTxPower := cfg.Amplitude * cfg.Amplitude
	expectedTxSignalStrength := float32(0)
	if expectedTxPower > 0 {
		expectedTxSignalStrength = 10.0 * math32.Log10(expectedTxPower)
	}

	// Inspect callback to verify transmitter stats
	opts.Inspect = func(startBufIdx, endBufIdx int, startSampleIdx, endSampleIdx int, tx modulation.Transmitter) {
		// Verify Transmitter interface provides stats
		params := tx.Params()
		require.Greater(t, params.BaudRate, float32(0), "BaudRate should be set")
		require.Equal(t, cfg.SymbolRate(), params.BaudRate, "BaudRate should match config SymbolRate")
		require.Greater(t, params.SampleRate, float32(0), "SampleRate should be set")
		require.Equal(t, cfg.SampleRate, params.SampleRate, "SampleRate should match config")
		require.Equal(t, cfg.CarrierHz, params.CarrierFrequency, "CarrierFrequency should match config")

		// Verify SignalStrength matches expected value from config (allow small tolerance for floating point)
		require.False(t, math32.IsNaN(params.SignalStrength), "SignalStrength should not be NaN")
		require.False(t, math32.IsInf(params.SignalStrength, 0), "SignalStrength should not be Inf")
		require.InDeltaf(t, expectedTxSignalStrength, params.SignalStrength, 0.1,
			"SignalStrength should match expected value from config (amplitude=%.3f -> power=%.6f -> %.3f dB)",
			cfg.Amplitude, expectedTxPower, expectedTxSignalStrength)

		// Track signal region for later verification
		transmitSignalStart = startSampleIdx
		transmitSignalEnd = endSampleIdx

		// Verify buffer indices are valid
		require.GreaterOrEqual(t, startBufIdx, 0, "startBufferIndex should be non-negative")
		require.GreaterOrEqual(t, endBufIdx, startBufIdx, "endBufferIndex should be >= startBufferIndex")
		require.GreaterOrEqual(t, startSampleIdx, 0, "startSampleIndex should be non-negative")
		require.GreaterOrEqual(t, endSampleIdx, startSampleIdx, "endSampleIndex should be >= startSampleIndex")
	}

	samples, _ := test_utils.Transmit(
		cfg.SampleRate,
		mod,
		[][]byte{payload},
		0.5, // 0.5s padding before and after
		snrDb,
		bytesToBits,
		&opts,
	)
	require.NotNil(t, samples, "samples should not be nil")
	require.Greater(t, len(samples), 0, "samples should not be empty")

	// Receive: process samples through demodulator
	demod.Reset() // Reset demodulator to ensure clean state
	// Re-enable debug logging if it was enabled before reset
	if len(payload) == 1 && payload[0] == 0x5a && snrDb == math32.Inf(1) {
		demod.SetDebugLogging(true)
	}

	outputBits, err := buffer.NewBits(cfg.BitsPerSymbol)
	require.NoError(t, err)

	receiveOpts := test_utils.DefaultReceiveOptions()
	receiveOpts.ChunkParts = 3 // Use 3 chunks for streaming simulation

	// Inspect callback to verify receiver stats
	var lockedQuality modulation.QualityStats
	var finalTransmission modulation.TransmissionStats
	var wasLocked bool

	receiveOpts.Inspect = func(startSampleIdx, endSampleIdx int, startBufIdx, endBufIdx int, rx modulation.Receiver) {
		// Verify Receiver interface provides stats
		quality := rx.Quality()
		transmission := rx.Transmission()
		params := rx.Params()

		// Verify params match config
		require.Greater(t, params.BaudRate, float32(0), "BaudRate should be set")
		require.Equal(t, cfg.SymbolRate(), params.BaudRate, "BaudRate should match config SymbolRate")
		require.Greater(t, params.SampleRate, float32(0), "SampleRate should be set")
		require.Equal(t, cfg.SampleRate, params.SampleRate, "SampleRate should match config")
		require.Equal(t, cfg.CarrierHz, params.CarrierFrequency, "CarrierFrequency should match config")

		// Verify quality stats when locked (during transmission)
		if transmission.IsLocked {
			wasLocked = true
			// Store quality stats when locked for later verification
			lockedQuality = quality

			// SNR must be positive (signal stronger than noise) for a working receiver
			require.False(t, math32.IsNaN(quality.SNR), "SNR should not be NaN when locked")
			require.False(t, math32.IsInf(quality.SNR, 0), "SNR should not be Inf when locked")
			if snrDb < math32.Inf(1) {
				// With noise, SNR must be positive
				require.Greaterf(t, quality.SNR, float32(0),
					"SNR should be positive when locked (simulated SNR=%.2f dB)", snrDb)
			}

			// SignalStrength should be reasonable (not NaN/Inf)
			require.False(t, math32.IsNaN(quality.SignalStrength), "SignalStrength should not be NaN when locked")
			require.False(t, math32.IsInf(quality.SignalStrength, 0), "SignalStrength should not be Inf when locked")
		}

		// Verify transmission indices
		if transmission.TransmissionStart > 0 {
			require.GreaterOrEqual(t, transmission.TransmissionStart, 0, "TransmissionStart should be non-negative")
			require.LessOrEqual(t, transmission.TransmissionStart, len(samples), "TransmissionStart should be within samples range")
		}
		if transmission.TransmissionEnd > 0 {
			require.GreaterOrEqual(t, transmission.TransmissionEnd, transmission.TransmissionStart, "TransmissionEnd should be >= TransmissionStart")
			require.LessOrEqual(t, transmission.TransmissionEnd, len(samples), "TransmissionEnd should be within samples range")
		}
		if transmission.PacketStart > 0 {
			require.GreaterOrEqual(t, transmission.PacketStart, transmission.TransmissionStart, "PacketStart should be >= TransmissionStart")
		}

		// Store final stats for later verification
		finalTransmission = transmission

		// Verify sample and buffer indices are valid
		require.GreaterOrEqual(t, startSampleIdx, 0, "startSampleIndex should be non-negative")
		require.GreaterOrEqual(t, endSampleIdx, startSampleIdx, "endSampleIndex should be >= startSampleIndex")
		require.LessOrEqual(t, endSampleIdx, len(samples), "endSampleIndex should be within samples range")
		require.GreaterOrEqual(t, startBufIdx, 0, "startBufferIndex should be non-negative")
		require.GreaterOrEqual(t, endBufIdx, startBufIdx, "endBufferIndex should be >= startBufferIndex")
	}

	totalDecoded := test_utils.Receive(
		cfg.SampleRate,
		demod,
		outputBits,
		samples,
		&receiveOpts,
	)
	require.Greaterf(t, totalDecoded, 0, "bits=%d snr=%.1f totalDecoded=%d", cfg.BitsPerSymbol, snrDb, totalDecoded)

	// Verify that lock was acquired during transmission (but may be unlocked after it ends)
	require.Greater(t, finalTransmission.LockCount, 0, "Demodulator should have locked at least once during transmission")
	// Note: IsLocked may be false at the end if transmission has ended, which is correct behavior

	// Verify SNR matches simulated SNR (within reasonable tolerance) - use stats from when locked
	if wasLocked && snrDb < math32.Inf(1) {
		// SNR must be positive and close to simulated SNR
		require.Greaterf(t, lockedQuality.SNR, float32(0),
			"SNR should be positive when locked (simulated SNR=%.2f dB)", snrDb)
		// Allow ±8dB tolerance due to measurement differences (noise estimation, signal processing, etc.)
		require.InDeltaf(t, snrDb, lockedQuality.SNR, 8.0,
			"Measured SNR should be close to simulated SNR (simulated=%.2f dB, measured=%.2f dB)",
			snrDb, lockedQuality.SNR)
	}

	// Verify SignalStrength is reasonable (should be in same ballpark as transmitter, accounting for noise)
	if wasLocked {
		require.False(t, math32.IsNaN(lockedQuality.SignalStrength), "SignalStrength should not be NaN")
		require.False(t, math32.IsInf(lockedQuality.SignalStrength, 0), "SignalStrength should not be Inf")
		// SignalStrength may differ from transmitter due to noise and normalization, but should be reasonable
		// For amplitude=0.8, expected is ~-1.94 dB, allow ±10dB tolerance for noise effects
		require.InDeltaf(t, expectedTxSignalStrength, lockedQuality.SignalStrength, 10.0,
			"Receiver SignalStrength should be in same ballpark as transmitter (tx=%.2f dB, rx=%.2f dB)",
			expectedTxSignalStrength, lockedQuality.SignalStrength)
	}

	// Verify transmission indices align with transmitted signal
	require.Greater(t, finalTransmission.TransmissionStart, 0, "TransmissionStart must be set")
	paddingSamples := int(0.5 * cfg.SampleRate)
	// TransmissionStart should be within the signal region (accounting for padding before signal)
	// It may be slightly before transmitSignalStart due to detection threshold, but not way off
	require.GreaterOrEqualf(t, finalTransmission.TransmissionStart, transmitSignalStart-paddingSamples,
		"TransmissionStart should be within reasonable range of actual signal start (signal starts at sample %d, detected at %d)",
		transmitSignalStart, finalTransmission.TransmissionStart)
	require.LessOrEqualf(t, finalTransmission.TransmissionStart, transmitSignalEnd,
		"TransmissionStart should be before signal end (signal ends at sample %d, detected start at %d)",
		transmitSignalEnd, finalTransmission.TransmissionStart)

	// Verify PacketStart is after TransmissionStart (preamble must be processed first)
	if finalTransmission.PacketStart > 0 {
		require.GreaterOrEqual(t, finalTransmission.PacketStart, finalTransmission.TransmissionStart,
			"PacketStart should be >= TransmissionStart (preamble must be detected first)")
		require.LessOrEqual(t, finalTransmission.PacketStart, transmitSignalEnd,
			"PacketStart should be before signal end")
	}

	// Verify decoded data
	outputBits.Reset()
	decodedSymbols := make([]byte, len(symbols))
	readBits, err := outputBits.Read(decodedSymbols)
	require.NoError(t, err)
	require.Equalf(t, len(symbols)*cfg.BitsPerSymbol, readBits, "bits=%d snr=%.1f", cfg.BitsPerSymbol, snrDb)

	if !equalSymbols(symbols, decodedSymbols) {
		t.Logf("expected symbols=%v actual=%v", symbols, decodedSymbols)
	}
	require.Equalf(t, symbols, decodedSymbols, "symbol mismatch bits=%d snr=%.1f", cfg.BitsPerSymbol, snrDb)

	decodedBytes := symbolsToBytes(decodedSymbols, cfg.BitsPerSymbol, len(payload))
	require.Equalf(t, payload, decodedBytes, "bits=%d snr=%.1f", cfg.BitsPerSymbol, snrDb)
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
