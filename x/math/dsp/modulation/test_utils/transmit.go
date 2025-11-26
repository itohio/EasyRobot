package test_utils

import (
	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/itohio/EasyRobot/x/chain"
	"github.com/itohio/EasyRobot/x/math/dsp"
)

// TransmitOptions configures the Transmit function behavior
type TransmitOptions struct {
	// IQInterleaved indicates if samples are IQ interleaved (true for MPSK/QAM, false for DSSS)
	IQInterleaved bool
	// ChunkParts is the number of chunks to split processing into for streaming simulation
	ChunkParts int
}

// DefaultTransmitOptions returns default options
func DefaultTransmitOptions() TransmitOptions {
	return TransmitOptions{
		IQInterleaved: false,
		ChunkParts:    10,
	}
}

// BytesToBits converts a byte payload into a Bits buffer.
// This function should be provided by the caller based on their modulation scheme's bitsPerSymbol.
type BytesToBits func([]byte) (buffer.Reader[byte], error)

// Transmit processes multiple data packets through a modulator with padding between them,
// adds noise, and returns the complete samples buffer.
//
// This function follows the zero-copy pattern from TUNING.md:
// - Allocates a single buffer for everything
// - Uses slice views (three-argument slices) for zero-copy operations
// - Buffer writes directly to the original array (no copying needed)
//
// Parameters:
//   - sampleRate: Sample rate in Hz
//   - modulator: Modulator processor (chain.Processor[float32, byte])
//   - data: List of data packets to transmit
//   - paddingDuration: Duration of padding (in seconds) before first packet, between packets, and after last packet
//   - snrDb: Signal-to-noise ratio in dB (use math32.Inf(1) for no noise)
//   - bytesToBits: Function to convert []byte to buffer.Reader[byte] (Bits) based on modulation scheme
//   - opts: Optional configuration (nil uses defaults)
//
// Returns:
//   - samples: Complete buffer with padding + data[0] + padding + data[1] + ... + padding
//   - signalRegions: Slice of slice views pointing to each data region in samples (for noise calculation)
func Transmit(
	sampleRate float32,
	modulator chain.Processor[float32, byte],
	data [][]byte,
	paddingDuration float32,
	snrDb float32,
	bytesToBits BytesToBits,
	opts *TransmitOptions,
) (samples []float32, signalRegions [][]float32) {
	if opts == nil {
		def := DefaultTransmitOptions()
		opts = &def
	}

	// Calculate padding samples
	paddingSamples := int(paddingDuration * sampleRate)
	if opts.IQInterleaved {
		paddingSamples *= 2
	}

	// Estimate total samples needed for all packets
	// Follow the pattern from working tests: estimate first, then allocate
	totalSamples := paddingSamples // Initial padding
	signalStarts := make([]int, len(data))
	signalEnds := make([]int, len(data))

	for i, payload := range data {
		// Build bits from payload for estimation
		bits, err := bytesToBits(payload)
		if err != nil || bits == nil {
			continue
		}

		// Estimate samples for this packet
		// Note: Don't reset modulator - keep state consistent between estimation and processing
		// This matches the working mfsk/mpsk test pattern
		estimate := modulator.Process(nil, bits)
		if estimate <= 0 {
			continue
		}

		signalStarts[i] = totalSamples
		totalSamples += estimate
		signalEnds[i] = totalSamples
		totalSamples += paddingSamples // Padding after each packet
	}

	if totalSamples == paddingSamples {
		// No data to transmit
		return nil, nil
	}

	// Allocate single buffer for everything (zero-copy!)
	// This matches the pattern: make([]float32, totalSamples)
	samples = make([]float32, totalSamples)
	signalRegions = make([][]float32, 0, len(data))

	// Process each data packet into its position
	// Follow the exact pattern from working tests
	for i, payload := range data {
		if signalStarts[i] >= signalEnds[i] {
			continue
		}

		// Build bits from payload for processing
		bits, err := bytesToBits(payload)
		if err != nil || bits == nil {
			continue
		}

		// Reset bits reader (matching working test pattern: inputBits.Reset())
		// Don't reset modulator - keep state consistent with estimation
		if bitsResetter, ok := bits.(interface{ Reset() }); ok {
			bitsResetter.Reset()
		}

		// Use three-argument slice view: samples[signalStart:signalStart:signalEnd]
		// This creates a slice view that shares the backing array with samples
		// The buffer will write directly into the original array - zero copy!
		// This is the exact pattern from mfsk/mpsk tests
		signalStart := signalStarts[i]
		signalEnd := signalEnds[i]
		// Create a slice view into the large samples buffer with the Go three-index slice syntax:
		// samples[low:high:max] creates a slice that starts at 'low', ends BEFORE 'high', and
		// has capacity up to 'max'. Here, signalStart is the index where this packet's signal should begin,
		// and signalEnd is where it ends in the overall samples buffer.
		// Using samples[signalStart:signalStart:signalEnd] makes a zero-length slice (for writing output),
		// whose underlying array can grow up to signalEnd, allowing the modulator to write directly into
		// the proper region of the buffer with no copying. This matches working test patterns (e.g. mfsk/mpsk).
		sliceView := samples[signalStart:signalStart:signalEnd]
		signalBuf := buffer.New(sliceView)
		written := modulator.Process(signalBuf, bits)

		if written > 0 {
			// The buffer wrote directly to samples[signalStart:signalStart+written]
			// because the slice view shares the backing array - zero copy!
			// Use the estimate (signalEnd - signalStart) for signal region, matching working tests
			estimatedSize := signalEnd - signalStart
			if signalStart+estimatedSize <= len(samples) {
				// Store signal region view for noise calculation (zero-copy slice view)
				signalRegions = append(signalRegions, samples[signalStart:signalStart+estimatedSize])
			}
		}

		// Reset modulator between packets (but not between estimation and processing of same packet)
		// This ensures each packet is processed independently
		if i < len(data)-1 {
			if resetter, ok := modulator.(interface{ Reset() }); ok {
				resetter.Reset()
			}
		}
	}

	// Add noise to entire buffer (signal + padding) at the same SNR level
	// This is more realistic - padding should have noise too, not pure silence
	if len(signalRegions) > 0 {
		// Combine all signal regions for noise calculation
		combinedSignal := make([]float32, 0)
		for _, region := range signalRegions {
			combinedSignal = append(combinedSignal, region...)
		}
		addNoiseToBuffer(samples, combinedSignal, snrDb)
	}

	return samples, signalRegions
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
	gen.Add(buffer)
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
	gen.Add(samples)
}
