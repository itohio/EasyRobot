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
	targetSNR              float32 // Target SNR for threshold calculation

	// SNR tracking
	signalPowerSum float32 // Sum of signal power (ideal symbol magnitudes)
	noisePowerSum  float32 // Sum of noise power (error between received and ideal)
	snrSampleCount int     // Number of samples used for SNR calculation

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
		targetSNR:              targetSNR,
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

		// Advance buffer by one symbol period
		// Copy remaining samples (from symbolSamples to end) to beginning
		remaining := d.buffered - symbolSamples
		if remaining > 0 {
			copy(d.fifo, d.fifo[symbolSamples:d.buffered])
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
	// This is a fast check before any expensive operations
	energy := symbolEnergy(samples)
	coarseThreshold := d.cfg.EnergyThresholdForSNR(d.targetSNR) * 0.1 // Very lenient for coarse check
	if !d.active && energy < coarseThreshold {
		// Not active and energy too low - likely noise, skip expensive processing
		return -1
	}

	// Full energy check with adaptive threshold
	energy = symbolEnergy(samples)

	// Calculate threshold based on expected SNR
	// This distinguishes signal+noise from noise-only samples
	baseThreshold := d.cfg.EnergyThresholdForSNR(d.targetSNR)

	// Use adaptive threshold: if we're in an active transmission, be more lenient
	var threshold float32
	if d.active {
		// During active transmission, allow lower energy (noise might reduce it)
		// Also, during active transmission, don't reset on silence - only reset between transmissions
		threshold = baseThreshold * 0.3
	} else if d.preambleCount > 0 {
		// During preamble acquisition, be lenient to catch weak signals
		// But still use SNR-based threshold to avoid false detections
		threshold = baseThreshold * 0.5
	} else {
		// Not yet active and no preamble started - use SNR-based threshold
		// This ensures we reject noise-only padding but detect signal+noise
		threshold = baseThreshold
	}

	if energy < threshold {
		// Only reset if we're NOT in active transmission
		// During active transmission, low energy symbols might be due to noise/phase errors,
		// not actual silence - don't reset during transmission
		if !d.active {
			// Not in active transmission - this might be silence between transmissions
			d.silenceCount++
			if d.silenceCount >= d.cfg.TransmissionGap {
				// Enough silence detected - reset for new transmission
				// This prepares the demodulator to re-acquire the next transmission
				d.active = false
				d.preambleCount = 0
				d.preambleErrors = 0
				d.silenceCount = 0
				d.unlockPLL()
				if d.pllLoop != nil {
					// Reset PLL to acquisition bandwidth for fast re-acquisition
					d.pllLoop.SetLoopBandwidth(d.loopBandwidthAcq, d.loopDamping)
					d.pllLoop.Reset()
					d.usingTrackingBandwidth = false
				}
				if d.agc != nil {
					d.agc.Reset()
				}
			}
			// Not in active transmission and low energy - likely silence, skip
			return -1
		}
		// During active transmission, low energy might be due to noise
		// Still try to decode - PLL/AGC might correct it
		// Don't skip symbols during active transmission
	}

	// Signal detected or continuing active transmission
	if !d.active {
		// Just detected signal - mark as active
		d.active = true
	}
	d.silenceCount = 0

	// Integrate IQ samples over symbol period (baseband, no carrier mixing needed)
	iAccum, qAccum := d.integrateSymbol(samples)

	// Apply AGC BEFORE PLL lock - normalize signal amplitude so PLL can lock faster
	// AGC trains on raw integrated samples, then normalizes before PLL processing
	mag := math32.Hypot(iAccum, qAccum)
	if mag > 0 {
		// Update AGC with magnitude for convergence
		d.agc.Process(mag)
		// Apply AGC normalization BEFORE PLL - this helps PLL lock on normalized signal
		iNorm, qNorm := d.applyAGC(iAccum, qAccum)
		iAccum, qAccum = iNorm, qNorm
	}

	// During preamble, use expected phase to compute error and update PLL
	if d.cfg.PreambleLength > 0 && d.preambleCount < d.cfg.PreambleLength {
		// Use optimized preamble pattern for fast PLL lock (same as modulator)
		expected := d.preamblePattern(d.preambleCount)
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
			// Apply correction
			iRot, qRot := d.pllLoop.Rotate(iAccum, qAccum)
			phase = math32.Atan2(qRot, iRot)
			// Recompute phase error after rotation for tracking
			phaseError = wrapPi(phase - expectedPhase)
			// Track phase errors for lock validation
			d.phaseErrorSum += math32.Abs(phaseError)
			d.phaseErrorCount++
		} else {
			// Later symbols: use PLL-corrected phase for better accuracy
			// This refines the PLL estimate as it converges
			iRot, qRot := d.pllLoop.Rotate(iAccum, qAccum)
			// Apply AGC during later preamble to avoid sudden gain changes at payload transition
			// Start applying AGC earlier (at 50% of preamble) to ensure convergence
			if d.pllLocked || d.preambleCount >= d.cfg.PreambleLength/2 {
				// Apply AGC if locked or past halfway point of preamble
				iNorm, qNorm := d.applyAGC(iRot, qRot)
				phase = math32.Atan2(qNorm, iNorm)
			} else {
				// Update AGC but don't apply yet during early preamble
				phase = math32.Atan2(qRot, iRot)
				mag := math32.Hypot(iRot, qRot)
				if mag > 0 {
					d.agc.Process(mag)
				}
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
			// Re-apply with updated PLL for symbol detection
			iRot, qRot = d.pllLoop.Rotate(iAccum, qAccum)
			// Only apply AGC when PLL is locked (consistent with earlier logic)
			if d.pllLocked {
				iNorm, qNorm := d.applyAGC(iRot, qRot)
				phase = math32.Atan2(qNorm, iNorm)
			} else {
				phase = math32.Atan2(qRot, iRot)
				// Update AGC with magnitude for convergence, but don't apply
				mag := math32.Hypot(iRot, qRot)
				if mag > 0 {
					d.agc.Process(mag)
				}
			}
		}

		symbol := d.quantizePhase(phase)

		// Recompute phase error for matching check
		phaseError = wrapPi(phase - expectedPhase)

		// Check if symbol matches expected (with tolerance for noise)
		// At 25dB SNR, be more lenient about matching, especially near end of preamble
		symbolMatch := (symbol == expected)
		if !symbolMatch {
			// Allow small phase errors (more lenient for noise)
			phaseDiff := math32.Abs(phaseError)
			step := twoPi / float32(d.cfg.constellationSize())

			// More lenient matching near end of preamble (last 4 symbols)
			// This helps ensure we complete preamble even with some symbol errors
			leniencyFactor := float32(1.0)
			if d.cfg.PreambleLength > 0 && d.preambleCount >= d.cfg.PreambleLength-4 {
				leniencyFactor = 1.5 // More lenient near end
			}

			// Allow up to a full symbol step in phase error (very lenient for noise)
			if phaseDiff < step*leniencyFactor {
				symbolMatch = true
			}
			// Also check if we're close to the expected symbol (adjacent symbols)
			size := d.cfg.constellationSize()
			expectedNext := (expected + 1) % size
			expectedPrev := (expected - 1 + size) % size
			if symbol == expectedNext || symbol == expectedPrev {
				// If we're on an adjacent symbol and phase error is reasonable, accept it
				if phaseDiff < step*leniencyFactor*1.2 {
					symbolMatch = true
				}
			}
		}

		// Track phase errors for lock validation (only recent errors)
		// Don't reset too frequently - we need enough samples for lock detection
		// Only reset if we have too many samples (to keep window manageable)
		if d.phaseErrorCount >= 20 {
			// Reset if we have too many samples, but keep recent ones
			// Keep last 12 samples worth of data
			keepCount := 12
			if d.phaseErrorCount > keepCount {
				// Approximate: keep average of recent samples
				avg := d.phaseErrorSum / float32(d.phaseErrorCount)
				d.phaseErrorSum = avg * float32(keepCount)
				d.phaseErrorCount = keepCount
			}
		}
		d.phaseErrorSum += math32.Abs(phaseError)
		d.phaseErrorCount++

		if symbolMatch {
			d.preambleCount++
			d.preambleErrors = 0
			// Consider PLL locked after we've seen enough correct preamble symbols
			// Use recent phase error average to validate convergence
			// More lenient thresholds for better performance at 25dB SNR
			if d.preambleCount >= 12 {
				// Check average phase error over recent symbols
				// Reduced requirements for faster lock at moderate SNR
				if d.phaseErrorCount >= 8 {
					avgPhaseError := d.phaseErrorSum / float32(d.phaseErrorCount)
					// More lenient error bounds for 25dB SNR operation
					maxAllowedError := twoPi / float32(d.cfg.constellationSize()) * 0.5
					// Require both average and current error to be reasonable
					// This ensures PLL has converged without being too strict
					if avgPhaseError < maxAllowedError && math32.Abs(phaseError) < maxAllowedError*2.0 {
						if !d.pllLocked {
							d.pllLocked = true
							d.setTrackingBandwidth()
						}
					}
				} else if d.preambleCount >= d.cfg.PreambleLength-2 {
					// Near end of preamble: be more lenient but still require reasonable lock
					maxAllowedError := twoPi / float32(d.cfg.constellationSize()) * 0.7
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

	// Only decode payload if preamble is complete
	// Allow decoding even if lock detection hasn't triggered, as long as we've processed enough preamble
	if d.cfg.PreambleLength > 0 && d.preambleCount < d.cfg.PreambleLength {
		return -1
	}
	// If we have a preamble but PLL isn't locked yet, check if we can safely start decoding
	// Don't force lock if PLL hasn't converged - this causes phase errors
	if d.cfg.PreambleLength > 0 && !d.pllLocked && d.preambleCount >= d.cfg.PreambleLength {
		// Check if PLL has reasonable phase error before forcing lock
		// Use the last integrated symbol to check phase alignment
		// If phase error is too large, don't force lock - PLL needs more time
		if d.phaseErrorCount > 0 {
			avgPhaseError := d.phaseErrorSum / float32(d.phaseErrorCount)
			maxAllowedError := twoPi / float32(d.cfg.constellationSize()) * 0.8
			// Only force lock if average phase error is reasonable
			// Be more strict - don't force lock if error is too large
			// This prevents decoding with wrong phase
			if avgPhaseError < maxAllowedError {
				d.pllLocked = true
				d.setTrackingBandwidth()
			} else {
				// PLL not converged yet - don't decode payload
				// At 25dB SNR, PLL should converge, but if it doesn't, don't force it
				return -1
			}
		} else {
			// No phase error data - don't force lock
			return -1
		}
	}
	// If no preamble, allow decoding immediately (like QAM)
	// PLL will track during payload decoding

	// For payload symbols, AGC has already been applied to iAccum, qAccum
	// Now apply PLL correction to AGC-normalized samples
	iRot, qRot := d.pllLoop.Rotate(iAccum, qAccum)
	phase := math32.Atan2(qRot, iRot)

	// Quantize to nearest constellation point
	symbol := d.quantizePhase(phase)

	// Track SNR: signal power (ideal) and noise power (error)
	idealPhase := d.table.phases[symbol]
	idealI := math32.Cos(idealPhase)
	idealQ := math32.Sin(idealPhase)
	idealMag := math32.Hypot(idealI, idealQ)

	// Signal power: magnitude of ideal symbol
	d.signalPowerSum += idealMag * idealMag

	// Noise power: error between received and ideal
	errorI := iRot - idealI
	errorQ := qRot - idealQ
	noiseMag := math32.Hypot(errorI, errorQ)
	d.noisePowerSum += noiseMag * noiseMag
	d.snrSampleCount++

	// Decision-directed PLL update for payload symbols
	// Use the corrected phase to compute error
	phaseError := wrapPi(phase - idealPhase)

	// Always update PLL - the loop bandwidth naturally filters noise
	// Use adaptive update: full for small errors, reduced for large errors
	// This allows PLL to track phase drift while rejecting noise spikes
	// At 25dB SNR, be more lenient with error thresholds
	maxPhaseError := twoPi / float32(d.cfg.constellationSize()) * 0.7
	if math32.Abs(phaseError) < maxPhaseError {
		// Normal update - PLL should track small phase errors
		d.pllLoop.Update(phaseError)
	} else {
		// Large error - likely noise spike, use reduced update
		// At 25dB SNR, be less aggressive about rejecting large errors
		// as they might be legitimate phase drift
		d.pllLoop.Update(phaseError * 0.3)
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

// Reset clears internal state and prepares for new transmission.
func (d *Demodulator) Reset() {
	d.buffered = 0
	d.active = false
	d.preambleCount = 0
	d.preambleErrors = 0
	d.silenceCount = 0
	d.unlockPLL()
	if d.pllLoop != nil {
		// Ensure PLL is set to acquisition bandwidth for fast re-acquisition
		d.pllLoop.SetLoopBandwidth(d.loopBandwidthAcq, d.loopDamping)
		d.pllLoop.Reset()
		d.usingTrackingBandwidth = false
	}
	if d.agc != nil {
		d.agc.Reset()
	}
	// Reset SNR tracking
	d.signalPowerSum = 0
	d.noisePowerSum = 0
	d.snrSampleCount = 0
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
	// Reset SNR tracking when unlocking
	d.signalPowerSum = 0
	d.noisePowerSum = 0
	d.snrSampleCount = 0
}

// SNR returns the estimated SNR (in dB) of the last decoded symbols.
// Returns 0 if insufficient samples have been processed.
func (d *Demodulator) SNR() float32 {
	if d.snrSampleCount == 0 || d.noisePowerSum <= 0 {
		return 0
	}
	signalPower := d.signalPowerSum / float32(d.snrSampleCount)
	noisePower := d.noisePowerSum / float32(d.snrSampleCount)
	if noisePower <= 0 {
		return 0
	}
	snrLinear := signalPower / noisePower
	if snrLinear <= 0 {
		return 0
	}
	return 10 * math32.Log10(snrLinear)
}

// preamblePattern returns the expected preamble symbol index for fast PLL lock.
// Must match the pattern used in modulator.go.
// Pattern design:
// - First M symbols: cover all M phases (0, 1, 2, 3 for QPSK) - resolves phase ambiguity
// - Next symbols: maximize phase transitions to help PLL track quickly
// - For QPSK: 0, 1, 2, 3, 0, 2, 1, 3, 0, 1, 2, 3, ...
func (d *Demodulator) preamblePattern(index int) int {
	M := d.cfg.constellationSize()

	// First M symbols: cover all phases (0, 1, 2, 3 for QPSK)
	if index < M {
		return index
	}

	// After covering all phases, use optimized pattern for fast PLL tracking
	// Pattern repeats every 2*M symbols: [0, 1, 2, 3, 0, 2, 1, 3] for QPSK
	patternIndex := (index - M) % (2 * M)
	if patternIndex < M {
		// First half: sequential 0, 1, 2, 3
		return patternIndex
	} else {
		// Second half: optimized transitions 0, 2, 1, 3
		offset := patternIndex - M
		switch offset {
		case 0:
			return 0
		case 1:
			return (M / 2) % M // 2 for QPSK
		case 2:
			return 1
		case 3:
			return (M/2 + 1) % M // 3 for QPSK
		default:
			// For M > 4, use a pattern that maximizes phase differences
			if offset%2 == 0 {
				return offset / 2
			} else {
				return (offset/2 + M/2) % M
			}
		}
	}
}
