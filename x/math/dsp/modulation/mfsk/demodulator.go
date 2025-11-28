package mfsk

import (
	"io"

	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/itohio/EasyRobot/x/chain"
	"github.com/itohio/EasyRobot/x/math/dsp"
	"github.com/itohio/EasyRobot/x/math/dsp/modulation"
	"github.com/itohio/EasyRobot/x/math/filter/pll"
	"github.com/itohio/EasyRobot/x/math/vec"
)

var (
	_ chain.Processor[byte, float32] = (*Demodulator)(nil)
	_ modulation.Receiver            = (*Demodulator)(nil)
)

type Demodulator struct {
	cfg   Config
	tones toneTable

	fifo     []float32
	buffered int
	readBuf  []float32

	active         bool
	preambleCount  int
	preambleErrors int
	silenceCount   int

	pllLoop   *pll.Loop
	targetSNR float32 // Target SNR for threshold calculation

	tmpSymbol [1]byte

	// Statistics tracking for Receiver interface
	totalSamplesProcessed int
	transmissionStartIdx  int
	transmissionEndIdx    int
	preambleStartIdx      int
	packetStartIdx        int
	lockAcquiredIdx       int
	lockLostIdx           int
	lockCount             int
	unlockCount           int
	rawSignalSum          float32
	rawSignalCount        int

	// SNR tracking
	signalPowerSum float32 // Sum of signal power (tone magnitudes)
	noisePowerSum  float32 // Sum of noise power (error between detected and ideal)
	snrSampleCount int     // Number of samples used for SNR calculation
}

func NewDemodulator(cfg Config) (*Demodulator, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	d := &Demodulator{
		cfg:       cfg,
		tones:     newToneTable(cfg),
		fifo:      make([]float32, cfg.SamplesPerSymbol*2),
		pllLoop:   pll.NewFromLoopBandwidth(cfg.LoopBandwidth, cfg.SymbolRate(), cfg.LoopDamping),
		targetSNR: 25.0, // Default target SNR for threshold calculation
	}
	return d, nil
}

func (d *Demodulator) Process(dst buffer.Writer[byte], src buffer.Reader[float32]) int {
	if dst == nil {
		return d.estimateBytes(src)
	}
	bitsWriter, ok := dst.(buffer.BitsWriter)
	if !ok || bitsWriter == nil {
		return 0
	}
	if bitsWriter.BitsLength() != d.cfg.BitsPerSymbol {
		return 0
	}

	totalBytes := 0
	sourceDone := false
	for {
		for d.buffered < d.cfg.SamplesPerSymbol && !sourceDone {
			needed := d.cfg.SamplesPerSymbol - d.buffered
			if len(d.readBuf) < needed {
				d.readBuf = make([]float32, needed)
			}
			n, err := src.Read(d.readBuf[:needed])
			if n > 0 {
				copy(d.fifo[d.buffered:], d.readBuf[:n])
				d.buffered += n
			}
			if err == io.EOF {
				sourceDone = true
			}
			if err != nil && err != io.EOF {
				sourceDone = true
			}
		}

		if d.buffered < d.cfg.SamplesPerSymbol {
			break
		}

		symbol := d.decodeSymbol(d.fifo[:d.cfg.SamplesPerSymbol])
		// Track total samples processed
		d.totalSamplesProcessed += d.cfg.SamplesPerSymbol
		if symbol >= 0 {
			d.tmpSymbol[0] = byte(symbol)
			writtenBits, _ := bitsWriter.Write(d.tmpSymbol[:])
			if writtenBits == 0 {
				break
			}
			totalBytes += writtenBits / d.cfg.BitsPerSymbol
		}

		// Advance buffer by one symbol period
		// Copy remaining samples (from SamplesPerSymbol to end) to beginning
		remaining := d.buffered - d.cfg.SamplesPerSymbol
		if remaining > 0 {
			copy(d.fifo, d.fifo[d.cfg.SamplesPerSymbol:d.buffered])
		}
		d.buffered = remaining
	}

	return totalBytes
}

func (d *Demodulator) decodeSymbol(samples []float32) int {
	if len(samples) == 0 {
		return -1
	}

	// Coarse detection: quick energy check to avoid wasting CPU on noise
	// When no lock on a signal - coarse detection avoids calculations
	// When coarse detection triggers and lock is achieved - disable coarse detection until lock is lost
	// Coarse detection should NOT be active during active lock
	energy := symbolEnergy(samples)
	coarseThreshold := d.cfg.EnergyThresholdForSNR(d.targetSNR) * 0.1 // Very lenient for coarse check

	// Coarse detection should NOT be active during active lock
	// Only use coarse detection when not active (no lock yet)
	useCoarseDetection := !d.active
	if useCoarseDetection && energy < coarseThreshold {
		// Not active and energy too low - likely noise, skip expensive processing
		return -1
	}

	// Full energy check with SNR-based adaptive threshold
	baseThreshold := d.cfg.EnergyThresholdForSNR(d.targetSNR)

	// Use adaptive threshold based on state
	var threshold float32
	if d.active {
		// During active transmission: be lenient (noise might reduce energy)
		threshold = baseThreshold * 0.3
	} else if d.preambleCount > 0 {
		// During preamble: moderate threshold
		threshold = baseThreshold * 0.5
	} else {
		// Initial detection: use full SNR-based threshold
		threshold = baseThreshold
		// Fallback to fixed threshold if SNR-based calculation fails
		if threshold <= 0 {
			threshold = d.cfg.EnergyThreshold
		}
	}

	if energy < threshold {
		// Only reset if we're NOT in active transmission
		// During active transmission, low energy might be due to noise
		if !d.active {
			d.silenceCount++
			if d.silenceCount >= d.cfg.TransmissionGap {
				// Enough silence detected - reset for new transmission
				// Track transmission end
				if d.active && d.transmissionEndIdx == 0 {
					d.transmissionEndIdx = d.totalSamplesProcessed
				}
				// Track lock loss
				if d.lockAcquiredIdx > 0 {
					d.lockLostIdx = d.totalSamplesProcessed
					d.unlockCount++
				}
				d.active = false
				d.preambleCount = 0
				d.preambleErrors = 0
				d.silenceCount = 0
				if d.pllLoop != nil {
					d.pllLoop.Reset()
				}
			}
			return -1
		}
		// During active transmission, low energy might be due to noise
		// Still try to decode - PLL might correct it
	}

	// Signal detected or continuing active transmission
	if !d.active {
		d.active = true
		// Track transmission start
		if d.transmissionStartIdx == 0 {
			d.transmissionStartIdx = d.totalSamplesProcessed
		}
	}
	d.silenceCount = 0

	// Track raw signal strength (before any processing)
	rawMag := math32.Sqrt(energy)
	d.rawSignalSum += rawMag
	d.rawSignalCount++

	toneIdx, phase, magnitude, secondBestMag := d.detectTone(samples)
	if toneIdx < 0 {
		// Invalid detection (noise or ambiguous)
		return -1
	}
	// Use energy threshold scaled by magnitude for better noise resilience
	// For high M-FSK, we need to be more lenient due to frequency resolution limits
	minMagnitudeFactor := float32(0.5)
	if d.cfg.BitsPerSymbol >= 8 {
		// For 8-FSK, be more lenient due to close tone spacing and noise sensitivity
		minMagnitudeFactor = 0.3
	} else if d.cfg.BitsPerSymbol > 4 {
		minMagnitudeFactor = 0.6
	}
	minMagnitude := d.cfg.EnergyThreshold * minMagnitudeFactor
	if magnitude < minMagnitude {
		return -1
	}

	// Use PLL to track carrier frequency offset during preamble
	// For M-FSK, the PLL tracks the phase error relative to expected preamble tones
	// This helps compensate for carrier frequency offset that affects all tones
	if d.cfg.PreambleLength > 0 && d.preambleCount < d.cfg.PreambleLength {
		// During preamble, use expected tone phase to compute error
		expected := 0
		if d.preambleCount&1 == 1 {
			expected = d.cfg.toneCount() - 1
		}
		if toneIdx == expected {
			// Use the phase error to update PLL (helps track carrier offset)
			// For M-FSK, phase error is less critical than for PSK, but still useful
			phaseError := wrapPi(phase)
			d.pllLoop.Update(phaseError)
		}
	} else {
		// During payload, use decision-directed phase error
		// The phase should be close to 0 for the detected tone (modulo 2π)
		phaseError := wrapPi(phase)
		d.pllLoop.Update(phaseError)
	}

	if d.cfg.PreambleLength > 0 && d.preambleCount < d.cfg.PreambleLength {
		expected := 0
		if d.preambleCount&1 == 1 {
			expected = d.cfg.toneCount() - 1
		}

		// For high M-FSK, allow detection within a small range of expected tone
		// due to frequency resolution limitations with noise
		toneMatch := (toneIdx == expected)
		if !toneMatch && d.cfg.BitsPerSymbol >= 8 {
			// Allow ±2 tones for 8-FSK due to noise and frequency resolution
			toneDiff := toneIdx - expected
			if toneDiff < 0 {
				toneDiff = -toneDiff
			}
			if toneDiff <= 2 || (expected == 0 && toneIdx < 3) ||
				(expected == d.cfg.toneCount()-1 && toneIdx >= d.cfg.toneCount()-3) {
				toneMatch = true
			}
		}

		if toneMatch {
			// Track preamble start on first preamble symbol
			if d.preambleCount == 0 && d.preambleStartIdx == 0 {
				d.preambleStartIdx = d.totalSamplesProcessed
			}
			d.preambleCount++
			d.preambleErrors = 0
			// Check if preamble is complete
			if d.preambleCount >= d.cfg.PreambleLength {
				// Track packet start (after preamble)
				if d.packetStartIdx == 0 {
					d.packetStartIdx = d.totalSamplesProcessed + d.cfg.SamplesPerSymbol
				}
				// Track lock acquisition
				if d.lockAcquiredIdx == 0 {
					d.lockAcquiredIdx = d.totalSamplesProcessed
					d.lockCount++
				}
			}
		} else {
			// Allow some preamble errors before resetting (more robust with noise)
			// For high M-FSK, allow more errors due to increased noise sensitivity
			maxErrors := 2
			if d.cfg.BitsPerSymbol >= 8 {
				maxErrors = 6 // More lenient for 8-FSK
			}
			d.preambleErrors++
			if d.preambleErrors > maxErrors {
				d.preambleCount = 0
				d.preambleErrors = 0
			}
		}
		return -1
	}

	// Track SNR for payload symbols (after preamble)
	// Signal power: detected tone magnitude squared
	signalPower := magnitude * magnitude
	d.signalPowerSum += signalPower

	// Noise power: second-best tone magnitude squared (interference/noise indicator)
	// For clean signals, second-best should be much smaller than best
	// For noisy signals, second-best will be closer to best
	noisePower := secondBestMag * secondBestMag
	if noisePower <= 0 {
		// Fallback: use a small fraction of signal power as baseline noise
		noisePower = signalPower * 0.0001
	}

	d.noisePowerSum += noisePower
	d.snrSampleCount++

	// Track transmission end: update whenever we successfully decode a payload symbol
	// This marks the last sample where transmission was successfully detected
	if d.active {
		d.transmissionEndIdx = d.totalSamplesProcessed
	}

	return toneIdx
}

func (d *Demodulator) detectTone(samples []float32) (idx int, phase float32, magnitude float32, secondBestMag float32) {
	count := d.cfg.toneCount()
	bestMag := float32(0)
	secondBestMagVal := float32(0)
	bestIdx := 0
	bestPhase := float32(0)
	window := vec.Vector(samples)

	for i := range count {
		measurement := dsp.Goertzel(window, d.cfg.ToneFrequency(i), d.cfg.SampleRate)
		if measurement.Amplitude > bestMag {
			secondBestMagVal = bestMag
			bestMag = measurement.Amplitude
			bestPhase = measurement.Phase
			bestIdx = i
		} else if measurement.Amplitude > secondBestMagVal {
			secondBestMagVal = measurement.Amplitude
		}
	}

	// Require best tone to be significantly stronger than second best (helps reject noise)
	// For high M-FSK, this check can be too strict due to frequency resolution limits
	// Only apply for moderate M-FSK (4-FSK, 8-FSK), skip for very high M-FSK
	if count > 4 && count <= 16 && bestMag > 0 && secondBestMagVal > 0 {
		ratio := bestMag / secondBestMagVal
		if ratio < 1.15 {
			// Too close, likely noise - return invalid
			return -1, 0, 0, 0
		}
	}
	// For 8-FSK and above, skip this check as frequency resolution limits make it unreliable

	return bestIdx, bestPhase, bestMag, secondBestMagVal
}

func (d *Demodulator) estimateBytes(src buffer.Reader[float32]) int {
	if src == nil {
		return 0
	}
	length := src.Len()
	if length <= 0 {
		return 0
	}
	symbols := length / d.cfg.SamplesPerSymbol
	return symbols * d.cfg.BitsPerSymbol / 8
}

func (d *Demodulator) Reset() {
	d.buffered = 0
	d.active = false
	d.preambleCount = 0
	d.preambleErrors = 0
	d.silenceCount = 0
	if d.pllLoop != nil {
		d.pllLoop.Reset()
	}

	// Reset statistics
	d.totalSamplesProcessed = 0
	d.transmissionStartIdx = 0
	d.transmissionEndIdx = 0
	d.preambleStartIdx = 0
	d.packetStartIdx = 0
	d.lockAcquiredIdx = 0
	d.lockLostIdx = 0
	d.lockCount = 0
	d.unlockCount = 0
}

// Quality implements modulation.Receiver.
func (d *Demodulator) Quality() modulation.QualityStats {
	stats := modulation.QualityStats{}

	// Calculate SNR
	if d.snrSampleCount > 0 && d.noisePowerSum > 0 {
		signalPower := d.signalPowerSum / float32(d.snrSampleCount)
		noisePower := d.noisePowerSum / float32(d.snrSampleCount)
		if noisePower > 0 {
			snrLinear := signalPower / noisePower
			if snrLinear > 0 {
				stats.SNR = 10.0 * math32.Log10(snrLinear)
			}
		}
	}

	// Calculate signal strength from tone magnitudes (more accurate for MFSK)
	// Use average signal power from SNR tracking
	if d.snrSampleCount > 0 {
		avgSignalPower := d.signalPowerSum / float32(d.snrSampleCount)
		if avgSignalPower > 0 {
			// Signal strength in dB: 10*log10(power) for power signals
			// For amplitude signals, use 20*log10(amplitude)
			// Since we have power, use 10*log10
			stats.SignalStrength = 10.0 * math32.Log10(avgSignalPower)
		}
	} else if d.rawSignalCount > 0 {
		// Fallback to raw signal magnitude
		avgMagnitude := d.rawSignalSum / float32(d.rawSignalCount)
		power := avgMagnitude * avgMagnitude
		if power > 0 {
			stats.SignalStrength = 10.0 * math32.Log10(power)
		}
	}

	return stats
}

// Transmission implements modulation.Receiver.
func (d *Demodulator) Transmission() modulation.TransmissionStats {
	isLocked := d.active && d.preambleCount >= d.cfg.PreambleLength
	return modulation.TransmissionStats{
		TransmissionStart: d.transmissionStartIdx,
		TransmissionEnd:   d.transmissionEndIdx,
		PacketStart:       d.packetStartIdx,
		PreambleStart:     d.preambleStartIdx,
		IsLocked:          isLocked,
		LockStart:         d.lockAcquiredIdx,
		LockEnd:           d.lockLostIdx,
		LockCount:         d.lockCount,
		UnlockCount:       d.unlockCount,
	}
}

// Params implements modulation.Receiver.
func (d *Demodulator) Params() modulation.TransmitterParams {
	power := d.cfg.Amplitude * d.cfg.Amplitude
	signalStrength := float32(0)
	if power > 0 {
		signalStrength = 10.0 * math32.Log10(power)
	}
	return modulation.TransmitterParams{
		BaudRate:         d.cfg.SymbolRate(),
		SampleRate:       d.cfg.SampleRate,
		SignalStrength:   signalStrength,
		Bandwidth:        d.cfg.RequiredBandwidth(),
		CarrierFrequency: d.cfg.CarrierHz,
	}
}

func symbolEnergy(samples []float32) float32 {
	var sum float32
	for _, v := range samples {
		sum += v * v
	}
	if len(samples) == 0 {
		return 0
	}
	return sum / float32(len(samples))
}
