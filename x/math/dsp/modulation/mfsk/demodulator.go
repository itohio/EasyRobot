package mfsk

import (
	"io"

	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/itohio/EasyRobot/x/chain"
	"github.com/itohio/EasyRobot/x/math/dsp"
	"github.com/itohio/EasyRobot/x/math/filter/pll"
	"github.com/itohio/EasyRobot/x/math/vec"
)

var _ chain.Processor[byte, float32] = (*Demodulator)(nil)

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
	}
	d.silenceCount = 0

	toneIdx, phase, magnitude := d.detectTone(samples)
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
			d.preambleCount++
			d.preambleErrors = 0
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

	return toneIdx
}

func (d *Demodulator) detectTone(samples []float32) (idx int, phase float32, magnitude float32) {
	count := d.cfg.toneCount()
	bestMag := float32(0)
	secondBestMag := float32(0)
	bestIdx := 0
	bestPhase := float32(0)
	window := vec.Vector(samples)

	for i := range count {
		measurement := dsp.Goertzel(window, d.cfg.ToneFrequency(i), d.cfg.SampleRate)
		if measurement.Amplitude > bestMag {
			secondBestMag = bestMag
			bestMag = measurement.Amplitude
			bestPhase = measurement.Phase
			bestIdx = i
		} else if measurement.Amplitude > secondBestMag {
			secondBestMag = measurement.Amplitude
		}
	}

	// Require best tone to be significantly stronger than second best (helps reject noise)
	// For high M-FSK, this check can be too strict due to frequency resolution limits
	// Only apply for moderate M-FSK (4-FSK, 8-FSK), skip for very high M-FSK
	if count > 4 && count <= 16 && bestMag > 0 && secondBestMag > 0 {
		ratio := bestMag / secondBestMag
		if ratio < 1.15 {
			// Too close, likely noise - return invalid
			return -1, 0, 0
		}
	}
	// For 8-FSK and above, skip this check as frequency resolution limits make it unreliable

	return bestIdx, bestPhase, bestMag
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
