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

	// Estimate total samples needed
	totalSamples := paddingSamples // Initial padding
	signalStarts := make([]int, len(data))
	signalEnds := make([]int, len(data))

	for i, payload := range data {
		// Build bits from payload using provided converter
		bits, err := bytesToBits(payload)
		if err != nil || bits == nil {
			continue
		}

		// Estimate samples for this packet
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
	samples = make([]float32, totalSamples)
	signalRegions = make([][]float32, 0, len(data))

	// Process each data packet into its position
	for i, payload := range data {
		if signalStarts[i] >= signalEnds[i] {
			continue
		}

		bits, err := bytesToBits(payload)
		if err != nil || bits == nil {
			continue
		}

		// Use slice view of the buffer for modulation output (zero-copy)
		signalStart := signalStarts[i]
		signalEnd := signalEnds[i]
		signalBuf := buffer.New(samples[signalStart:signalStart:signalEnd])
		written := modulator.Process(signalBuf, bits)

		if written > 0 && signalStart+written <= len(samples) {
			// Store signal region view for noise calculation
			signalRegions = append(signalRegions, samples[signalStart:signalStart+written])
		}

		// Reset modulator between packets
		if resetter, ok := modulator.(interface{ Reset() }); ok {
			resetter.Reset()
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

