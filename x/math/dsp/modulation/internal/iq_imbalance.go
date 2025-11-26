package internal

import (
	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/x/buffer"
)

var _ Processor = (*IQImbalance)(nil)

// IQImbalance corrects gain and phase imbalance between I and Q channels.
// This compensates for hardware imperfections in mixers/ADCs.
type IQImbalance struct {
	// Correction matrix: [I' Q'] = [a b; c d] * [I Q]
	// For balanced channels: a=1, b=0, c=0, d=1
	a float32 // I channel gain
	b float32 // I→Q coupling
	c float32 // Q→I coupling
	d float32 // Q channel gain

	// Auto-adaptation parameters
	adaptive       bool
	alpha          float32 // Adaptation rate
	estimatedPhase float32 // Estimated phase imbalance
	estimatedGain  float32 // Estimated gain imbalance
	batchSize      int     // Batch size for processing
}

// NewIQImbalance creates a new I/Q imbalance corrector.
// gainImbalance: gain difference (dB) between I and Q (0 = balanced)
// phaseImbalance: phase difference (radians) between I and Q (0 = balanced)
func NewIQImbalance(gainImbalanceDb, phaseImbalanceRad float32) IQImbalance {
	return NewIQImbalanceWithBatchSize(gainImbalanceDb, phaseImbalanceRad, 16)
}

// NewIQImbalanceWithBatchSize creates a new I/Q imbalance corrector with custom batch size.
func NewIQImbalanceWithBatchSize(gainImbalanceDb, phaseImbalanceRad float32, batchSize int) IQImbalance {
	if batchSize <= 0 {
		batchSize = 16
	}
	// Convert gain imbalance from dB to linear
	gainRatio := math32.Pow(10, gainImbalanceDb/20)

	// Compute correction matrix for gain and phase imbalance
	// For phase imbalance: sin(θ) and cos(θ) terms
	cosPhi := math32.Cos(phaseImbalanceRad)
	sinPhi := math32.Sin(phaseImbalanceRad)

	// Correction matrix (inverse of imbalance)
	// Balanced case: gainRatio=1, phaseImbalance=0 → a=1, b=0, c=0, d=1
	// General case accounts for both gain and phase errors
	det := gainRatio * cosPhi
	if math32.Abs(det) < 1e-6 {
		det = 1.0
	}

	// Inverse matrix for correction
	a := cosPhi / det
	b := -sinPhi / det
	c := sinPhi * gainRatio / det
	d := cosPhi * gainRatio / det

	return IQImbalance{
		a:              a,
		b:              b,
		c:              c,
		d:              d,
		estimatedPhase: phaseImbalanceRad,
		estimatedGain:  gainRatio,
		batchSize:      batchSize,
	}
}

// NewAdaptiveIQImbalance creates an adaptive I/Q imbalance corrector.
// alpha: adaptation rate (typical: 0.001 to 0.01)
func NewAdaptiveIQImbalance(alpha float32) IQImbalance {
	return NewAdaptiveIQImbalanceWithBatchSize(alpha, 16)
}

// NewAdaptiveIQImbalanceWithBatchSize creates an adaptive I/Q imbalance corrector with custom batch size.
func NewAdaptiveIQImbalanceWithBatchSize(alpha float32, batchSize int) IQImbalance {
	if batchSize <= 0 {
		batchSize = 16
	}
	return IQImbalance{
		a:        1.0,
		b:        0.0,
		c:        0.0,
		d:        1.0,
		adaptive: true,
		alpha:    alpha,
		batchSize: batchSize,
	}
}

// Process implements Processor.
// Corrects I/Q imbalance.
func (i *IQImbalance) Process(dst buffer.Writer[complex64], src buffer.Reader[complex64]) int {
	return processInPlace(dst, src, i.batchSize, func(items []complex64) {
		for idx := range items {
			sample := items[idx]
			iVal := float32(real(sample))
			qVal := float32(imag(sample))

			// Apply correction matrix: [I' Q'] = [a b; c d] * [I Q]
			iCorrected := i.a*iVal + i.b*qVal
			qCorrected := i.c*iVal + i.d*qVal

			if i.adaptive {
				// Update estimates using gradient descent
				// Simple adaptation: minimize I*Q cross-correlation
				error := iCorrected * qCorrected // Should be zero for balanced channels
				i.b += i.alpha * error * iVal
				i.c += i.alpha * error * qVal
			}

			items[idx] = complex64(complex(iCorrected, qCorrected))
		}
	})
}

// Reset clears the internal state.
func (i *IQImbalance) Reset() {
	if i.adaptive {
		i.a = 1.0
		i.b = 0.0
		i.c = 0.0
		i.d = 1.0
	}
}

// SetCorrectionMatrix sets the correction matrix directly.
func (i *IQImbalance) SetCorrectionMatrix(a, b, c, d float32) {
	i.a = a
	i.b = b
	i.c = c
	i.d = d
	i.adaptive = false
}

// CorrectionMatrix returns the current correction matrix.
func (i *IQImbalance) CorrectionMatrix() (a, b, c, d float32) {
	return i.a, i.b, i.c, i.d
}

// EstimatedImbalance returns the estimated gain and phase imbalance.
func (i *IQImbalance) EstimatedImbalance() (gainDb float32, phaseRad float32) {
	gainDb = 20 * math32.Log10(i.estimatedGain)
	phaseRad = i.estimatedPhase
	return
}

