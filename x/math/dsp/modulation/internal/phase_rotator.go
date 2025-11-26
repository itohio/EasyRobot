package internal

import (
	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/x/buffer"
)

var _ Processor = (*PhaseRotator)(nil)

// PhaseRotator rotates IQ samples by a phase angle.
// Can be used for phase correction, constellation rotation, or fixed phase shifts.
type PhaseRotator struct {
	phase     float32 // Rotation phase in radians
	sin       float32 // Cached sin(phase)
	cos       float32 // Cached cos(phase)
	batchSize int     // Batch size for processing
}

// NewPhaseRotator creates a new phase rotator with initial phase.
func NewPhaseRotator(phase float32) PhaseRotator {
	return NewPhaseRotatorWithBatchSize(phase, 16)
}

// NewPhaseRotatorWithBatchSize creates a new phase rotator with custom batch size.
func NewPhaseRotatorWithBatchSize(phase float32, batchSize int) PhaseRotator {
	if batchSize <= 0 {
		batchSize = 16
	}
	pr := PhaseRotator{
		phase:     phase,
		batchSize: batchSize,
	}
	pr.updateTrig()
	return pr
}

// updateTrig updates cached sin/cos values.
func (p *PhaseRotator) updateTrig() {
	p.sin, p.cos = math32.Sincos(p.phase)
}

// Process implements Processor.
// Rotates samples by the current phase.
func (p *PhaseRotator) Process(dst buffer.Writer[complex64], src buffer.Reader[complex64]) int {
	return processInPlace(dst, src, p.batchSize, func(items []complex64) {
		for i := range items {
			sample := items[i]
			iVal := float32(real(sample))
			qVal := float32(imag(sample))

			// Rotation: I' = I*cos(θ) - Q*sin(θ), Q' = I*sin(θ) + Q*cos(θ)
			iRot := iVal*p.cos - qVal*p.sin
			qRot := iVal*p.sin + qVal*p.cos

			items[i] = complex64(complex(iRot, qRot))
		}
	})
}

// Reset clears the internal state (phase is preserved).
func (p *PhaseRotator) Reset() {
	// Nothing to reset - phase is preserved
}

// SetPhase sets the rotation phase in radians.
func (p *PhaseRotator) SetPhase(phase float32) {
	p.phase = phase
	p.updateTrig()
}

// Phase returns the current rotation phase in radians.
func (p *PhaseRotator) Phase() float32 {
	return p.phase
}

// RotatePhase adds a phase increment to the current phase.
func (p *PhaseRotator) RotatePhase(deltaPhase float32) {
	p.phase += deltaPhase
	// Wrap to [-π, π)
	for p.phase > math32.Pi {
		p.phase -= 2 * math32.Pi
	}
	for p.phase < -math32.Pi {
		p.phase += 2 * math32.Pi
	}
	p.updateTrig()
}

