package internal

import (
	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/x/buffer"
)

var _ Processor = (*CostasLoop)(nil)

// CostasLoop implements a Costas loop for carrier recovery in QAM/PSK demodulation.
// It's a type of PLL that extracts carrier phase and frequency from suppressed-carrier signals.
// The loop uses a decision-directed phase detector for QAM/PSK constellations.
type CostasLoop struct {
	// Loop parameters
	kp           float32 // Proportional gain
	ki           float32 // Integral gain
	symbolRate   float32
	modulationOrder int // Modulation order (2 for QPSK, 4 for 16-QAM, etc.)

	// Internal state
	phase     float32 // Current phase estimate
	frequency float32 // Frequency offset estimate
	sin       float32 // Cached sin(phase)
	cos       float32 // Cached cos(phase)

	batchSize int // Batch size for processing
}

// NewCostasLoop creates a new Costas loop processor.
// symbolRate: symbol rate in Hz
// loopBandwidth: loop bandwidth in Hz (typical: 0.01 to 0.1 * symbolRate)
// dampingRatio: damping ratio (typical: 0.707 for critical damping)
// modulationOrder: modulation order for phase detector (2 for QPSK, 4 for 16-QAM)
func NewCostasLoop(symbolRate, loopBandwidth, dampingRatio float32, modulationOrder int) CostasLoop {
	return NewCostasLoopWithBatchSize(symbolRate, loopBandwidth, dampingRatio, modulationOrder, 16)
}

// NewCostasLoopWithBatchSize creates a new Costas loop processor with custom batch size.
func NewCostasLoopWithBatchSize(symbolRate, loopBandwidth, dampingRatio float32, modulationOrder int, batchSize int) CostasLoop {
	if dampingRatio <= 0 {
		dampingRatio = 0.707
	}
	if loopBandwidth <= 0 {
		loopBandwidth = symbolRate * 0.05
	}
	if batchSize <= 0 {
		batchSize = 16
	}

	// Calculate gains from loop bandwidth and damping
	omegaN := 2 * math32.Pi * loopBandwidth / symbolRate
	kp := 2 * dampingRatio * omegaN
	ki := omegaN * omegaN

	return CostasLoop{
		kp:              kp,
		ki:              ki,
		symbolRate:      symbolRate,
		modulationOrder: modulationOrder,
		cos:             1.0,
		batchSize:       batchSize,
	}
}

// Process implements Processor.
// Performs carrier recovery using Costas loop.
func (c *CostasLoop) Process(dst buffer.Writer[complex64], src buffer.Reader[complex64]) int {
	return processInPlace(dst, src, c.batchSize, func(items []complex64) {
		for i := range items {
			sample := items[i]
			iVal := float32(real(sample))
			qVal := float32(imag(sample))

			// Rotate by negative phase to correct carrier offset
			// This derotates the signal to baseband
			iRot := iVal*c.cos + qVal*c.sin
			qRot := -iVal*c.sin + qVal*c.cos

			// Costas phase detector for QAM/PSK
			// For M-PSK/M-QAM, the phase error is: phase_error = (1/M) * atan2(Q, I) for small errors
			// For decision-directed: use hard decision symbol phase
			phaseError := c.phaseDetector(iRot, qRot)

			// Update loop: integrate frequency, update phase
			c.frequency += c.ki * phaseError
			c.phase = wrapPi(c.phase + c.kp*phaseError + c.frequency)

			// Update cached sin/cos
			c.sin, c.cos = math32.Sincos(c.phase)

			// Output derotated sample
			items[i] = complex64(complex(iRot, qRot))
		}
	})
}

// phaseDetector computes the phase error using a decision-directed approach.
// For M-PSK/M-QAM, this uses the modulation order to compute the error.
func (c *CostasLoop) phaseDetector(i, q float32) float32 {
	// For M-PSK: phase error = (1/M) * angle(sample^M)
	// For M-QAM: use decision-directed approach
	phase := math32.Atan2(q, i)

	if c.modulationOrder > 0 {
		// For M-PSK: multiply phase by M and unwrap
		phase *= float32(c.modulationOrder)
		phase = wrapPi(phase)
		phase /= float32(c.modulationOrder)
	}

	return phase
}

// Reset clears the internal state.
func (c *CostasLoop) Reset() {
	c.phase = 0
	c.frequency = 0
	c.sin = 0
	c.cos = 1.0
}

// Phase returns the current phase estimate in radians.
func (c *CostasLoop) Phase() float32 {
	return c.phase
}

// FrequencyOffset returns the estimated carrier frequency offset in Hz.
func (c *CostasLoop) FrequencyOffset() float32 {
	return c.frequency * c.symbolRate / (2 * math32.Pi)
}
