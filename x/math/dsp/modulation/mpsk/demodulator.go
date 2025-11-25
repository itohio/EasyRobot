package mpsk

import (
	"io"

	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/itohio/EasyRobot/x/chain"
	"github.com/itohio/EasyRobot/x/math/filter/agc"
	"github.com/itohio/EasyRobot/x/math/filter/pll"
)

var _ chain.Processor[byte, float32] = (*Demodulator)(nil)

// Demodulator converts complex baseband samples back to M-PSK symbols.
type Demodulator struct {
	cfg   Config
	table constellationTable

	fifo     []float32
	buffered int
	readBuf  []float32

	active          bool
	preambleCount   int
	preambleErrors  int
	silenceCount    int
	pllLocked       bool    // Track if PLL has converged
	phaseErrorSum   float32 // Sum of recent phase errors for lock validation
	phaseErrorCount int     // Count of phase errors for averaging

	pllLoop                *pll.Loop
	agc                    *agc.Filter
	loopBandwidthAcq       float32
	loopBandwidthTrack     float32
	loopDamping            float32
	usingTrackingBandwidth bool

	tmpSymbol [1]byte
}

// NewDemodulator builds a streaming demodulator instance.
func NewDemodulator(cfg Config) (*Demodulator, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	symbolSamples := cfg.SamplesPerSymbol * 2
	// Use optimal PLL parameters based on target SNR (25dB) and preamble length
	// Acquisition bandwidth is wider for faster convergence during preamble
	// Tracking bandwidth is narrower for better noise filtering during payload
	targetSNR := float32(25.0) // Target 25dB SNR operation
	acqBw := cfg.OptimalAcquisitionBandwidth(targetSNR)
	trackBw := cfg.OptimalTrackingBandwidth(targetSNR)
	loopBw := acqBw // Start with acquisition bandwidth
	damping := cfg.OptimalLoopDamping(targetSNR)
	agcFilter := agc.New(1.0, 0.001)
	agcFilter.SetSampleRate(cfg.SymbolRate())
	d := &Demodulator{
		cfg:                    cfg,
		table:                  newConstellationTable(cfg),
		fifo:                   make([]float32, symbolSamples*2),
		pllLoop:                pll.NewFromLoopBandwidth(loopBw, cfg.SymbolRate(), damping),
		agc:                    agcFilter,
		loopBandwidthAcq:       acqBw,
		loopBandwidthTrack:     trackBw,
		loopDamping:            damping,
		usingTrackingBandwidth: false,
		pllLocked:              cfg.PreambleLength == 0, // No preamble means start locked
	}
	return d, nil
}

// Process implements chain.Processor.
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
	symbolSamples := d.cfg.SamplesPerSymbol * 2

	for {
		for d.buffered < symbolSamples && !sourceDone {
			needed := symbolSamples - d.buffered
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

		if d.buffered < symbolSamples {
			break
		}

		symbol := d.decodeSymbol(d.fifo[:symbolSamples])
		if symbol >= 0 {
			d.tmpSymbol[0] = byte(symbol)
			writtenBits, _ := bitsWriter.Write(d.tmpSymbol[:])
			if writtenBits == 0 {
				break
			}
			totalBytes += writtenBits / d.cfg.BitsPerSymbol
		}

		copy(d.fifo, d.fifo[symbolSamples:symbolSamples+d.buffered])
		d.buffered -= symbolSamples
	}

	return totalBytes
}

func (d *Demodulator) decodeSymbol(samples []float32) int {
	if len(samples) == 0 {
		return -1
	}
	energy := symbolEnergy(samples)

	// Use adaptive threshold: if we're in an active transmission, be more lenient
	threshold := d.cfg.EnergyThreshold
	if d.active {
		// During active transmission, allow lower energy (noise might reduce it)
		threshold = d.cfg.EnergyThreshold * 0.3
	} else if d.preambleCount > 0 {
		// During preamble acquisition, be lenient to catch weak signals
		threshold = d.cfg.EnergyThreshold * 0.4
	}

	if energy < threshold {
		// Only reset if we've seen enough silence
		d.silenceCount++
		if d.silenceCount >= d.cfg.TransmissionGap {
			d.active = false
			d.preambleCount = 0
			d.preambleErrors = 0
			d.silenceCount = 0
			d.unlockPLL()
			d.pllLoop.Reset()
		}
		return -1
	}

	d.active = true
	d.silenceCount = 0

	// Integrate IQ samples over symbol period (baseband, no carrier mixing needed)
	iAccum, qAccum := d.integrateSymbol(samples)

	// During preamble, use expected phase to compute error and update PLL
	if d.cfg.PreambleLength > 0 && d.preambleCount < d.cfg.PreambleLength {
		expected := 0
		if d.preambleCount&1 == 1 {
			expected = d.cfg.constellationSize() / 2
		}
		expectedPhase := d.table.phases[expected]

		// Use raw phase error for first few symbols to help PLL converge quickly
		// Then switch to PLL-corrected phase error for better accuracy
		var phase float32
		var phaseError float32

		// Strategy: Use raw phase for initial convergence, then switch to corrected phase
		// This balances fast convergence with noise rejection
		// Strategy: Use raw phase for initial convergence (first 16 symbols),
		// then switch to PLL-corrected phase for refinement
		// More symbols with raw phase helps ensure PLL converges to correct phase
		if d.preambleCount < 8 {
			// First 8 symbols: use raw phase to initialize PLL quickly
			// This helps the PLL converge even in the presence of noise
			rawPhase := math32.Atan2(qAccum, iAccum)
			phaseError = wrapPi(rawPhase - expectedPhase)
			// Update PLL - always update during initial convergence
			// Use full update to get good initial lock
			d.pllLoop.Update(phaseError)
			// Apply correction only (no AGC during very early acquisition)
			iRot, qRot := d.pllLoop.Rotate(iAccum, qAccum)
			phase = math32.Atan2(qRot, iRot)
			// Update AGC with magnitude for convergence, but don't apply yet
			mag := math32.Hypot(iRot, qRot)
			if mag > 0 {
				d.agc.Process(mag)
			}
		} else {
			// Later symbols: use PLL-corrected phase for better accuracy
			// This refines the PLL estimate as it converges
			iRot, qRot := d.pllLoop.Rotate(iAccum, qAccum)
			// Don't apply AGC during preamble - just update it for convergence
			phase = math32.Atan2(qRot, iRot)
			// Update AGC with magnitude for convergence, but don't apply yet
			mag := math32.Hypot(iRot, qRot)
			if mag > 0 {
				d.agc.Process(mag)
			}
			phaseError = wrapPi(phase - expectedPhase)
			// Always update PLL during preamble refinement
			// Use full update for reasonable errors, reduced for large errors
			maxPhaseError := twoPi / float32(d.cfg.constellationSize()) * 0.6
			if math32.Abs(phaseError) < maxPhaseError {
				d.pllLoop.Update(phaseError)
			} else {
				// Large error - likely noise, use reduced update to prevent pulling
				d.pllLoop.Update(phaseError * 0.3)
			}
			// Re-apply with updated PLL for symbol detection (no AGC during preamble)
			iRot, qRot = d.pllLoop.Rotate(iAccum, qAccum)
			phase = math32.Atan2(qRot, iRot)
			// Update AGC with magnitude for convergence
			mag = math32.Hypot(iRot, qRot)
			if mag > 0 {
				d.agc.Process(mag)
			}
		}

		symbol := d.quantizePhase(phase)

		// Recompute phase error for matching check
		phaseError = wrapPi(phase - expectedPhase)

		// Check if symbol matches expected (with tolerance for noise)
		symbolMatch := (symbol == expected)
		if !symbolMatch {
			// Allow small phase errors (more lenient for noise)
			phaseDiff := math32.Abs(phaseError)
			step := twoPi / float32(d.cfg.constellationSize())
			// Allow up to a full symbol step in phase error (very lenient for noise)
			if phaseDiff < step*1.0 {
				symbolMatch = true
			}
			// Also check if we're close to the expected symbol (adjacent symbols)
			size := d.cfg.constellationSize()
			expectedNext := (expected + 1) % size
			expectedPrev := (expected - 1 + size) % size
			if symbol == expectedNext || symbol == expectedPrev {
				// If we're on an adjacent symbol and phase error is reasonable, accept it
				if phaseDiff < step*1.2 {
					symbolMatch = true
				}
			}
		}

		// Track phase errors for lock validation (only recent errors)
		// Reset tracking periodically to focus on recent convergence
		if d.preambleCount > 0 && d.preambleCount%8 == 0 {
			// Reset phase error tracking every 8 symbols to focus on recent errors
			d.phaseErrorSum = 0
			d.phaseErrorCount = 0
		}
		d.phaseErrorSum += math32.Abs(phaseError)
		d.phaseErrorCount++

		if symbolMatch {
			d.preambleCount++
			d.preambleErrors = 0
			// Consider PLL locked after we've seen enough correct preamble symbols
			// Use recent phase error average to validate convergence
			if d.preambleCount >= 16 {
				// Check average phase error over recent symbols
				// Require more symbols and stricter error bounds for better lock
				if d.phaseErrorCount >= 12 {
					avgPhaseError := d.phaseErrorSum / float32(d.phaseErrorCount)
					maxAllowedError := twoPi / float32(d.cfg.constellationSize()) * 0.3
					// Require both average and current error to be very small
					// This ensures PLL has actually converged, not just oscillating
					if avgPhaseError < maxAllowedError && math32.Abs(phaseError) < maxAllowedError*1.5 {
						if !d.pllLocked {
							d.pllLocked = true
							d.setTrackingBandwidth()
						}
					}
				} else if d.preambleCount >= d.cfg.PreambleLength-2 {
					// Near end of preamble: be more lenient but still require good lock
					maxAllowedError := twoPi / float32(d.cfg.constellationSize()) * 0.5
					if math32.Abs(phaseError) < maxAllowedError {
						if !d.pllLocked {
							d.pllLocked = true
							d.setTrackingBandwidth()
						}
					}
				}
			}
		} else {
			d.preambleErrors++
			// Allow more errors before resetting (more robust with noise)
			// Increase tolerance as we get more preamble symbols (PLL is locking)
			maxErrors := 4
			if d.preambleCount > 4 {
				maxErrors = 6 // More lenient once we've seen some correct symbols
			}
			if d.preambleCount > d.cfg.PreambleLength/2 {
				maxErrors = 8 // Even more lenient once PLL is partially locked
			}
			if d.preambleErrors > maxErrors {
				d.preambleCount = 0
				d.preambleErrors = 0
				d.unlockPLL()
			}
		}
		return -1
	}

	// Only decode payload if PLL is locked AND preamble is complete
	if !d.pllLocked {
		return -1
	}
	// Ensure preamble is fully processed before decoding payload
	if d.cfg.PreambleLength > 0 && d.preambleCount < d.cfg.PreambleLength {
		return -1
	}

	// For payload symbols, apply PLL correction then AGC normalization
	iRot, qRot := d.pllLoop.Rotate(iAccum, qAccum)
	iNorm, qNorm := d.applyAGC(iRot, qRot)
	phase := math32.Atan2(qNorm, iNorm)

	// Quantize to nearest constellation point
	symbol := d.quantizePhase(phase)

	// Decision-directed PLL update for payload symbols
	// Use the corrected phase to compute error
	idealPhase := d.table.phases[symbol]
	phaseError := wrapPi(phase - idealPhase)

	// Always update PLL - the loop bandwidth naturally filters noise
	// Use adaptive update: full for small errors, reduced for large errors
	// This allows PLL to track phase drift while rejecting noise spikes
	maxPhaseError := twoPi / float32(d.cfg.constellationSize()) * 0.5
	if math32.Abs(phaseError) < maxPhaseError {
		// Normal update - PLL should track small phase errors
		d.pllLoop.Update(phaseError)
	} else {
		// Large error - likely noise spike, use very reduced update
		// This prevents PLL from being pulled too far by noise
		// But still allows slow tracking of actual phase drift
		d.pllLoop.Update(phaseError * 0.1)
	}

	return symbol

}

func (d *Demodulator) integrateSymbol(samples []float32) (float32, float32) {
	env := d.table.envelope
	var iAcc, qAcc float32

	// Integrate interleaved IQ samples with matched filter (envelope)
	for i := range d.cfg.SamplesPerSymbol {
		base := i * 2
		envVal := env[i]
		iAcc += samples[base] * envVal
		qAcc += samples[base+1] * envVal
	}

	// Normalize by envelope energy to match QAM pattern
	if d.table.envelopeEnergy > 0 {
		inv := 1 / d.table.envelopeEnergy
		iAcc *= inv
		qAcc *= inv
	}

	return iAcc, qAcc
}

func (d *Demodulator) quantizePhase(phi float32) int {
	size := d.cfg.constellationSize()
	step := twoPi / float32(size)

	// Normalize phase to [0, 2π)
	phi = wrapTwoPi(phi)

	// Find nearest constellation point by computing distance to all points
	// This is more robust than simple rounding, especially near boundaries
	bestIdx := 0
	bestDist := float32(1e10)

	for i := range size {
		idealPhase := d.table.phases[i]
		// Compute angular distance (accounting for wrap-around)
		diff := wrapPi(phi - idealPhase)
		dist := math32.Abs(diff)
		if dist < bestDist {
			bestDist = dist
			bestIdx = i
		}
	}

	// Verify the best match is reasonable (within half a symbol step)
	// If not, there might be a phase ambiguity issue - check adjacent symbols
	idealPhase := d.table.phases[bestIdx]
	phaseDiff := wrapPi(phi - idealPhase)
	if math32.Abs(phaseDiff) > step*0.6 {
		// Phase is far from ideal - might be noise or phase ambiguity
		// Try adjacent symbols to see if they're closer
		nextIdx := (bestIdx + 1) % size
		prevIdx := (bestIdx - 1 + size) % size
		nextPhase := d.table.phases[nextIdx]
		prevPhase := d.table.phases[prevIdx]
		nextDiff := math32.Abs(wrapPi(phi - nextPhase))
		prevDiff := math32.Abs(wrapPi(phi - prevPhase))
		if nextDiff < math32.Abs(phaseDiff) {
			bestIdx = nextIdx
		} else if prevDiff < math32.Abs(phaseDiff) {
			bestIdx = prevIdx
		}
	}

	return bestIdx
}

func (d *Demodulator) applyAGC(i, q float32) (float32, float32) {
	mag := math32.Hypot(i, q)
	if mag <= 0 {
		return i, q
	}
	scaled := d.agc.Process(mag)
	gain := scaled / (mag + 1e-6)
	return i * gain, q * gain
}

func (d *Demodulator) estimateBytes(src buffer.Reader[float32]) int {
	if src == nil {
		return 0
	}
	length := src.Len()
	if length <= 0 {
		return 0
	}
	symbols := length / (d.cfg.SamplesPerSymbol * 2)
	return symbols * d.cfg.BitsPerSymbol / 8
}

// Reset clears internal state.
func (d *Demodulator) Reset() {
	d.buffered = 0
	d.active = false
	d.preambleCount = 0
	d.preambleErrors = 0
	d.silenceCount = 0
	d.unlockPLL()
	if d.pllLoop != nil {
		d.pllLoop.SetLoopBandwidth(d.loopBandwidthAcq, d.loopDamping)
		d.pllLoop.Reset()
	}
	if d.agc != nil {
		d.agc.Reset()
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

func (d *Demodulator) setTrackingBandwidth() {
	if d.pllLoop == nil || d.usingTrackingBandwidth {
		return
	}
	d.pllLoop.SetLoopBandwidth(d.loopBandwidthTrack, d.loopDamping)
	d.usingTrackingBandwidth = true
}

func (d *Demodulator) setAcquisitionBandwidth() {
	if d.pllLoop == nil || !d.usingTrackingBandwidth {
		return
	}
	d.pllLoop.SetLoopBandwidth(d.loopBandwidthAcq, d.cfg.LoopDamping)
	d.usingTrackingBandwidth = false
}

func (d *Demodulator) unlockPLL() {
	d.setAcquisitionBandwidth()
	d.pllLocked = false
	d.phaseErrorSum = 0
	d.phaseErrorCount = 0
}
