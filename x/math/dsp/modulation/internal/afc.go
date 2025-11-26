package internal

import (
	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/x/buffer"
)

var _ Processor = (*AFC)(nil)

// AFC implements Automatic Frequency Control for tracking slow frequency drift.
// It estimates frequency offset from phase derivatives and applies correction.
type AFC struct {
	symbolRate float32 // Symbol rate in Hz
	sampleRate float32 // Sample rate in Hz

	// Loop parameters
	kp           float32 // Proportional gain
	ki           float32 // Integral gain
	dampingRatio float32

	// Frequency tracking state
	estimatedOffset float32 // Estimated frequency offset in Hz
	phaseHistory    [3]float32 // Phase history for derivative estimation
	phaseHistoryIdx int

	// Phase accumulator for correction
	phaseAccumulator float32
	phaseIncrement   float32
	sin              float32
	cos              float32

	batchSize int // Batch size for processing
}

// NewAFC creates a new Automatic Frequency Control processor.
// symbolRate: symbol rate in Hz
// sampleRate: sample rate in Hz
// loopBandwidth: loop bandwidth in Hz (typical: 0.0001 to 0.001 * symbolRate)
// dampingRatio: damping ratio (typical: 0.707)
func NewAFC(symbolRate, sampleRate, loopBandwidth, dampingRatio float32) AFC {
	return NewAFCWithBatchSize(symbolRate, sampleRate, loopBandwidth, dampingRatio, 16)
}

// NewAFCWithBatchSize creates a new AFC processor with custom batch size.
func NewAFCWithBatchSize(symbolRate, sampleRate, loopBandwidth, dampingRatio float32, batchSize int) AFC {
	if dampingRatio <= 0 {
		dampingRatio = 0.707
	}
	if loopBandwidth <= 0 {
		loopBandwidth = symbolRate * 0.0001
	}
	if batchSize <= 0 {
		batchSize = 16
	}

	// Calculate gains for slow tracking
	omegaN := 2 * math32.Pi * loopBandwidth / symbolRate
	kp := 2 * dampingRatio * omegaN
	ki := omegaN * omegaN

	phaseIncrement := 2 * math32.Pi * 0 / sampleRate // Start at 0 offset

	afc := AFC{
		symbolRate:     symbolRate,
		sampleRate:     sampleRate,
		kp:             kp,
		ki:             ki,
		dampingRatio:   dampingRatio,
		phaseIncrement: phaseIncrement,
		cos:            1.0,
		batchSize:      batchSize,
	}
	afc.updateTrig()
	return afc
}

// updateTrig updates cached sin/cos values.
func (a *AFC) updateTrig() {
	a.sin, a.cos = math32.Sincos(a.phaseAccumulator)
}

// Process implements Processor.
// Tracks frequency drift and applies correction.
func (a *AFC) Process(dst buffer.Writer[complex64], src buffer.Reader[complex64]) int {
	return processInPlace(dst, src, a.batchSize, func(items []complex64) {
		for idx := range items {
			sample := items[idx]
			iVal := float32(real(sample))
			qVal := float32(imag(sample))

			// Estimate phase from sample
			phase := math32.Atan2(qVal, iVal)

			// Update phase history for derivative estimation
			a.phaseHistory[a.phaseHistoryIdx] = phase
			a.phaseHistoryIdx = (a.phaseHistoryIdx + 1) % len(a.phaseHistory)

			// Estimate frequency offset from phase derivative
			// Use unwrapped phase differences
			if idx > 0 || a.phaseHistoryIdx > 1 {
				// Simple derivative: phase difference
				prevIdx := (a.phaseHistoryIdx - 2 + len(a.phaseHistory)) % len(a.phaseHistory)
				phaseDiff := phase - a.phaseHistory[prevIdx]
				phaseDiff = wrapPi(phaseDiff)

				// Update frequency estimate using PI controller
				freqError := phaseDiff * a.symbolRate / (2 * math32.Pi)
				a.estimatedOffset += a.ki * phaseDiff * a.symbolRate / (2 * math32.Pi)
				a.estimatedOffset += a.kp * freqError

				// Update phase increment for correction
				a.phaseIncrement = 2 * math32.Pi * a.estimatedOffset / a.sampleRate
			}

			// Apply frequency correction
			iCorrected := iVal*a.cos + qVal*a.sin
			qCorrected := -iVal*a.sin + qVal*a.cos

			items[idx] = complex64(complex(iCorrected, qCorrected))

			// Update phase accumulator
			a.phaseAccumulator += a.phaseIncrement
			for a.phaseAccumulator > math32.Pi {
				a.phaseAccumulator -= 2 * math32.Pi
				a.updateTrig()
			}
			for a.phaseAccumulator < -math32.Pi {
				a.phaseAccumulator += 2 * math32.Pi
				a.updateTrig()
			}
		}
	})
}

// Reset clears the internal state.
func (a *AFC) Reset() {
	a.phaseAccumulator = 0
	a.estimatedOffset = 0
	a.phaseHistoryIdx = 0
	for i := range a.phaseHistory {
		a.phaseHistory[i] = 0
	}
	a.updateTrig()
}

// EstimatedOffset returns the estimated frequency offset in Hz.
func (a *AFC) EstimatedOffset() float32 {
	return a.estimatedOffset
}

// SetLoopBandwidth updates the loop bandwidth and recalculates gains.
func (a *AFC) SetLoopBandwidth(loopBandwidth, dampingRatio float32) {
	if dampingRatio <= 0 {
		dampingRatio = 0.707
	}
	if loopBandwidth <= 0 {
		loopBandwidth = a.symbolRate * 0.0001
	}

	omegaN := 2 * math32.Pi * loopBandwidth / a.symbolRate
	a.kp = 2 * dampingRatio * omegaN
	a.ki = omegaN * omegaN
	a.dampingRatio = dampingRatio
}

// SetEstimatedOffset sets the initial frequency offset estimate.
func (a *AFC) SetEstimatedOffset(offsetHz float32) {
	a.estimatedOffset = offsetHz
	a.phaseIncrement = 2 * math32.Pi * offsetHz / a.sampleRate
}


