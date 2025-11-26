package dsp

import (
	"math/rand"

	"github.com/chewxy/math32"
)

// GeneratorAssignOp is the operation function type for GenerateWithOp
// It allows direct modification of destination elements via pointer
type GeneratorAssignOp func(d *float32, value float32)

// Generator is the interface for all signal generators
type Generator interface {
	// Overwrites dst with generated samples
	Generate(dst []float32) int
	// Adds generated samples to dst samples
	Add(dst []float32) int
	// Applies assignment operation to dst samples
	// reverse: if true, generates samples in reverse order
	// op: the assignment operation to apply to each destination element
	GenerateWithOp(dst []float32, reverse bool, op GeneratorAssignOp) int
}

// Correlator is the interface for generators that support correlation
type Correlator interface {
	Correlate(dst, samples []float32) int
}

// SquareGenerator generates a square wave signal
type SquareGenerator struct {
	sampleRate float32
	frequency  float32
	amplitude  float32
	period     float32
	time       float32
}

// NewSquareGenerator creates a new square wave generator
func NewSquareGenerator(sampleRate, frequency, amplitude float32) SquareGenerator {
	period := sampleRate / frequency
	return SquareGenerator{
		sampleRate: sampleRate,
		frequency:  frequency,
		amplitude:  amplitude,
		period:     period,
		time:       0,
	}
}

// GenerateWithOp is the common generation function that takes a binary operation closure and reverse flag
func (g *SquareGenerator) GenerateWithOp(dst []float32, reverse bool, op GeneratorAssignOp) int {
	n := len(dst)
	if n == 0 {
		return 0
	}

	// Bounds check elimination: prove to compiler that dst has at least n elements
	_ = dst[n-1]
	dst = dst[:n]

	dt := 1.0 / g.sampleRate
	period := 1.0 / g.frequency
	halfPeriod := period * 0.5

	if reverse {
		// Reverse path: generate samples in reverse time order
		startTime := g.time + float32(n-1)*dt
		for i := range n {
			t := startTime - float32(i)*dt
			// Wrap time to [0, period)
			for t >= period {
				t -= period
			}
			for t < 0 {
				t += period
			}
			var value float32
			if t < halfPeriod {
				value = g.amplitude
			} else {
				value = -g.amplitude
			}
			op(&dst[n-1-i], value)
		}
		g.time += float32(n) * dt
	} else {
		// Forward path: generate samples in normal time order
		for i := range n {
			// Avoid Mod by using direct time wrapping (faster than Mod)
			t := g.time
			if t >= period {
				// Fast path: subtract period until in range
				t -= period * math32.Floor(t/period)
			} else if t < 0 {
				// Handle negative (shouldn't happen normally)
				t += period * math32.Ceil(-t/period)
			}

			var value float32
			if t < halfPeriod {
				value = g.amplitude
			} else {
				value = -g.amplitude
			}
			op(&dst[i], value)
			g.time += dt
		}
	}

	return n
}

// Generate fills dst with square wave samples
func (g *SquareGenerator) Generate(dst []float32) int {
	return g.GenerateWithOp(dst, false, func(d *float32, value float32) {
		*d = value
	})
}

// Add adds square wave samples to dst (additive generation)
func (g *SquareGenerator) Add(dst []float32) int {
	return g.GenerateWithOp(dst, false, func(d *float32, value float32) {
		*d += value
	})
}

// TriangleGenerator generates a triangle wave signal
type TriangleGenerator struct {
	sampleRate float32
	frequency  float32
	amplitude  float32
	period     float32
	time       float32
}

// NewTriangleGenerator creates a new triangle wave generator
func NewTriangleGenerator(sampleRate, frequency, amplitude float32) TriangleGenerator {
	period := sampleRate / frequency
	return TriangleGenerator{
		sampleRate: sampleRate,
		frequency:  frequency,
		amplitude:  amplitude,
		period:     period,
		time:       0,
	}
}

// GenerateWithOp is the common generation function that takes a binary operation closure and reverse flag
func (g *TriangleGenerator) GenerateWithOp(dst []float32, reverse bool, op GeneratorAssignOp) int {
	n := len(dst)
	if n == 0 {
		return 0
	}

	// Bounds check elimination: prove to compiler that dst has at least n elements
	_ = dst[n-1]
	dst = dst[:n]

	dt := 1.0 / g.sampleRate

	if reverse {
		// Reverse path: generate samples in reverse time order
		startTime := g.time + float32(n-1)*dt
		for i := range n {
			t := (startTime - float32(i)*dt) * g.frequency
			// Wrap to [0, 1)
			for t >= 1.0 {
				t -= 1.0
			}
			for t < 0 {
				t += 1.0
			}
			value := triangleValue(g.amplitude, t)
			op(&dst[n-1-i], value)
		}
		g.time += float32(n) * dt
	} else {
		// Forward path: generate samples in normal time order
		for i := range n {
			// Avoid Mod by using direct time wrapping
			t := g.time * g.frequency
			if t >= 1.0 {
				// Fast path: subtract period until in range
				t -= math32.Floor(t)
			} else if t < 0 {
				// Handle negative (shouldn't happen normally)
				t += math32.Ceil(-t)
			}
			value := triangleValue(g.amplitude, t)
			op(&dst[i], value)
			g.time += dt
		}
	}

	return n
}

// Generate fills dst with triangle wave samples
func (g *TriangleGenerator) Generate(dst []float32) int {
	return g.GenerateWithOp(dst, false, func(d *float32, value float32) {
		*d = value
	})
}

// Add adds triangle wave samples to dst (additive generation)
func (g *TriangleGenerator) Add(dst []float32) int {
	return g.GenerateWithOp(dst, false, func(d *float32, value float32) {
		*d += value
	})
}

func triangleValue(amplitude, phase float32) float32 {
	if phase < 0.5 {
		return -amplitude + 4*amplitude*phase
	}
	return 3*amplitude - 4*amplitude*phase
}

// NoiseWhiteGenerator generates white noise
type NoiseWhiteGenerator struct {
	amplitude float32
}

// NewNoiseWhiteGenerator creates a new white noise generator
func NewNoiseWhiteGenerator(amplitude float32) NoiseWhiteGenerator {
	return NoiseWhiteGenerator{
		amplitude: amplitude,
	}
}

// GenerateWithOp is the common generation function that takes a binary operation closure
func (g *NoiseWhiteGenerator) GenerateWithOp(dst []float32, reverse bool, op GeneratorAssignOp) int {
	n := len(dst)
	if n == 0 {
		return 0
	}

	// Bounds check elimination: prove to compiler that dst has at least n elements
	_ = dst[n-1]
	dst = dst[:n]

	// Reverse doesn't make sense for noise, but we support it for interface consistency
	if reverse {
		for i := range n {
			idx := n - 1 - i
			value := g.amplitude * (rand.Float32()*2 - 1)
			op(&dst[idx], value)
		}
	} else {
		for i := range n {
			value := g.amplitude * (rand.Float32()*2 - 1)
			op(&dst[i], value)
		}
	}
	return n
}

// Generate fills dst with white noise samples (overwrite)
func (g *NoiseWhiteGenerator) Generate(dst []float32) int {
	return g.GenerateWithOp(dst, false, func(d *float32, value float32) {
		*d = value
	})
}

// Add adds white noise samples to the dst slice (in-place addition)
func (g *NoiseWhiteGenerator) Add(dst []float32) int {
	return g.GenerateWithOp(dst, false, func(d *float32, value float32) {
		*d += value
	})
}

// NoisePinkGenerator generates pink noise (1/f noise)
type NoisePinkGenerator struct {
	amplitude float32
	// Filter state
	y1, y2, y3, y4, y5 float32
	// Filter coefficients
	b0, b1, b2, b3, b4, b5 float32
}

// NewNoisePinkGenerator creates a new pink noise generator
func NewNoisePinkGenerator(amplitude float32) NoisePinkGenerator {
	return NoisePinkGenerator{
		amplitude: amplitude,
		b0:        0.99886,
		b1:        -2.49482,
		b2:        1.44331,
		b3:        -0.44419,
		b4:        0.05081,
		b5:        -0.00226,
	}
}

// GenerateWithOp is the common generation function that takes a binary operation closure
func (g *NoisePinkGenerator) GenerateWithOp(dst []float32, reverse bool, op GeneratorAssignOp) int {
	n := len(dst)
	if n == 0 {
		return 0
	}

	// Bounds check elimination: prove to compiler that dst has at least n elements
	_ = dst[n-1]
	dst = dst[:n]

	// Reverse doesn't make sense for noise, but we support it for interface consistency
	if reverse {
		for i := range n {
			idx := n - 1 - i
			white := g.amplitude * (rand.Float32()*2 - 1)
			y0 := g.b0*white + g.b1*g.y1 + g.b2*g.y2 + g.b3*g.y3 + g.b4*g.y4 + g.b5*g.y5
			op(&dst[idx], y0)

			// Update filter state
			g.y5 = g.y4
			g.y4 = g.y3
			g.y3 = g.y2
			g.y2 = g.y1
			g.y1 = y0
		}
	} else {
		for i := range n {
			white := g.amplitude * (rand.Float32()*2 - 1)
			y0 := g.b0*white + g.b1*g.y1 + g.b2*g.y2 + g.b3*g.y3 + g.b4*g.y4 + g.b5*g.y5
			op(&dst[i], y0)

			// Update filter state
			g.y5 = g.y4
			g.y4 = g.y3
			g.y3 = g.y2
			g.y2 = g.y1
			g.y1 = y0
		}
	}
	return n
}

// Generate fills dst with pink noise samples (overwrite contents)
func (g *NoisePinkGenerator) Generate(dst []float32) int {
	return g.GenerateWithOp(dst, false, func(d *float32, value float32) {
		*d = value
	})
}

// Add fills dst with pink noise samples (sums into existing buffer)
func (g *NoisePinkGenerator) Add(dst []float32) int {
	return g.GenerateWithOp(dst, false, func(d *float32, value float32) {
		*d += value
	})
}

// NoiseBrownGenerator generates brown noise (1/f^2 noise)
type NoiseBrownGenerator struct {
	amplitude   float32
	accumulator float32
	prev        float32
}

// NewNoiseBrownGenerator creates a new brown noise generator
func NewNoiseBrownGenerator(amplitude float32) NoiseBrownGenerator {
	return NoiseBrownGenerator{
		amplitude:   amplitude,
		accumulator: 0,
		prev:        0,
	}
}

// GenerateWithOp is the common generation function that takes a binary operation closure
func (g *NoiseBrownGenerator) GenerateWithOp(dst []float32, reverse bool, op GeneratorAssignOp) int {
	n := len(dst)
	if n == 0 {
		return 0
	}

	// Bounds check elimination: prove to compiler that dst has at least n elements
	_ = dst[n-1]
	dst = dst[:n]

	// Reverse doesn't make sense for noise, but we support it for interface consistency
	if reverse {
		for i := range n {
			idx := n - 1 - i
			white := g.amplitude * (rand.Float32()*2 - 1)
			g.accumulator += white * 0.02

			// High-pass filter to remove DC component
			var value float32
			if idx < n-1 {
				value = g.accumulator - g.prev*0.99
			} else {
				value = g.accumulator
			}
			op(&dst[idx], value)
			g.prev = value

			// Normalize to prevent drift
			if g.accumulator > g.amplitude {
				g.accumulator = g.amplitude
			} else if g.accumulator < -g.amplitude {
				g.accumulator = -g.amplitude
			}
		}
	} else {
		for i := range n {
			white := g.amplitude * (rand.Float32()*2 - 1)
			g.accumulator += white * 0.02

			// High-pass filter to remove DC component
			var value float32
			if i > 0 {
				value = g.accumulator - g.prev*0.99
			} else {
				value = g.accumulator
			}
			op(&dst[i], value)
			g.prev = value

			// Normalize to prevent drift
			if g.accumulator > g.amplitude {
				g.accumulator = g.amplitude
			} else if g.accumulator < -g.amplitude {
				g.accumulator = -g.amplitude
			}
		}
	}
	return n
}

// Generate fills dst with brown noise samples
func (g *NoiseBrownGenerator) Generate(dst []float32) int {
	return g.GenerateWithOp(dst, false, func(d *float32, value float32) {
		*d = value
	})
}

// Add adds brown noise samples to dst (additive generation)
func (g *NoiseBrownGenerator) Add(dst []float32) int {
	return g.GenerateWithOp(dst, false, func(d *float32, value float32) {
		*d += value
	})
}

// ConstantGenerator generates a constant value
type ConstantGenerator struct {
	value float32
}

// NewConstantGenerator creates a new constant generator
func NewConstantGenerator(value float32) ConstantGenerator {
	return ConstantGenerator{
		value: value,
	}
}

// GenerateWithOp is the common generation function that takes a binary operation closure
func (g *ConstantGenerator) GenerateWithOp(dst []float32, reverse bool, op GeneratorAssignOp) int {
	n := len(dst)
	if n == 0 {
		return 0
	}

	// Bounds check elimination: prove to compiler that dst has at least n elements
	_ = dst[n-1]
	dst = dst[:n]

	// Reverse doesn't make sense for constant, but we support it for interface consistency
	if reverse {
		for i := range n {
			idx := n - 1 - i
			op(&dst[idx], g.value)
		}
	} else {
		for i := range n {
			op(&dst[i], g.value)
		}
	}
	return n
}

// Generate fills dst with constant value
func (g *ConstantGenerator) Generate(dst []float32) int {
	return g.GenerateWithOp(dst, false, func(d *float32, value float32) {
		*d = value
	})
}

// Add adds constant value to dst (additive generation)
func (g *ConstantGenerator) Add(dst []float32) int {
	return g.GenerateWithOp(dst, false, func(d *float32, value float32) {
		*d += value
	})
}

// RampGenerator generates a linear ramp signal
type RampGenerator struct {
	start     float32
	increment float32
	current   float32
}

// NewRampGenerator creates a new ramp generator
func NewRampGenerator(start, increment float32) RampGenerator {
	return RampGenerator{
		start:     start,
		increment: increment,
		current:   start,
	}
}

// GenerateWithOp is the common generation function that takes a binary operation closure and reverse flag
func (g *RampGenerator) GenerateWithOp(dst []float32, reverse bool, op GeneratorAssignOp) int {
	n := len(dst)
	if n == 0 {
		return 0
	}

	// Bounds check elimination: prove to compiler that dst has at least n elements
	_ = dst[n-1]
	dst = dst[:n]

	if reverse {
		// Reverse path: generate samples in reverse order
		// Start from end value and work backwards
		endValue := g.current + float32(n-1)*g.increment
		for i := range n {
			idx := n - 1 - i
			value := endValue - float32(i)*g.increment
			op(&dst[idx], value)
		}
		g.current += float32(n) * g.increment
	} else {
		// Forward path: generate samples in normal order
		for i := range n {
			value := g.current
			op(&dst[i], value)
			g.current += g.increment
		}
	}
	return n
}

// Generate fills dst with ramp samples
func (g *RampGenerator) Generate(dst []float32) int {
	return g.GenerateWithOp(dst, false, func(d *float32, value float32) {
		*d = value
	})
}

// Add adds ramp samples to dst (additive generation)
func (g *RampGenerator) Add(dst []float32) int {
	return g.GenerateWithOp(dst, false, func(d *float32, value float32) {
		*d += value
	})
}

// Reset resets the ramp to its start value
func (g *RampGenerator) Reset() {
	g.current = g.start
}

// GaussianGenerator generates a Gaussian (normal) distribution function
// Used for pulse shaping and symbol envelope shaping in digital communications
// Gaussian(x) = amplitude × exp(-(x - center)² / (2 × sigma²))
type GaussianGenerator struct {
	center    float32 // Center position (mean)
	sigma     float32 // Standard deviation (controls width)
	amplitude float32 // Amplitude scaling
}

// NewGaussianGenerator creates a new Gaussian generator
// center: center position of the Gaussian (mean)
// sigma: standard deviation (controls the width, larger = wider)
// amplitude: amplitude scaling factor
func NewGaussianGenerator(center, sigma, amplitude float32) GaussianGenerator {
	return GaussianGenerator{
		center:    center,
		sigma:     sigma,
		amplitude: amplitude,
	}
}

// GenerateWithOp is the common generation function that takes a binary operation closure and reverse flag
func (g *GaussianGenerator) GenerateWithOp(dst []float32, reverse bool, op GeneratorAssignOp) int {
	n := len(dst)
	if n == 0 {
		return 0
	}

	// Bounds check elimination: prove to compiler that dst has at least n elements
	_ = dst[n-1]
	dst = dst[:n]

	if g.sigma <= 0 {
		// Invalid sigma, return zeros
		if reverse {
			for i := range n {
				idx := n - 1 - i
				op(&dst[idx], 0)
			}
		} else {
			for i := range n {
				op(&dst[i], 0)
			}
		}
		return n
	}

	// Pre-compute 1/(2*sigma²) for efficiency
	invTwoSigmaSq := 1.0 / (2.0 * g.sigma * g.sigma)

	if reverse {
		// Reverse path: generate samples in reverse order (from end to start)
		// Start at i=n-1: x = (n-1) - center
		x := float32(n-1) - g.center
		xSq := x * x
		twoX := 2.0 * x

		for i := range n {
			idx := n - 1 - i
			// Gaussian: exp(-x²/(2σ²))
			exponent := -xSq * invTwoSigmaSq
			value := g.amplitude * math32.Exp(exponent)
			op(&dst[idx], value)
			// Decrement for next iteration: x²_new = x²_old - 2*x_old + 1 (going backwards)
			// (x-1)² = x² - 2x + 1, so xSq_new = xSq_old - 2*x_old + 1
			xSq -= twoX - 1.0
			x -= 1.0
			twoX -= 2.0
		}
	} else {
		// Forward path: generate samples in normal order
		// Incremental computation: avoid float32(i) and x*x multiplication in hot path
		// Start at i=0: x = 0 - center
		x := -g.center
		xSq := x * x // Initial x² = center²
		// When x increments by 1: x_new = x_old + 1
		// x²_new = (x_old + 1)² = x_old² + 2*x_old + 1
		// So: xSq_new = xSq_old + 2*x_old + 1
		// We compute this BEFORE incrementing x, so we use x_old
		twoX := 2.0 * x // Pre-compute 2*x for the increment formula

		for i := range n {
			// Gaussian: exp(-x²/(2σ²))
			exponent := -xSq * invTwoSigmaSq
			value := g.amplitude * math32.Exp(exponent)
			op(&dst[i], value)
			// Increment for next iteration: x²_new = x²_old + 2*x_old + 1
			// Compute using OLD x value before incrementing
			xSq += twoX + 1.0
			x += 1.0
			twoX += 2.0 // twoX_new = 2*(x_old + 1) = 2*x_old + 2 = twoX_old + 2
		}
	}
	return n
}

// Generate fills dst with Gaussian function samples
func (g *GaussianGenerator) Generate(dst []float32) int {
	return g.GenerateWithOp(dst, false, func(d *float32, value float32) {
		*d = value
	})
}

// Add adds Gaussian function samples to dst (additive generation)
func (g *GaussianGenerator) Add(dst []float32) int {
	return g.GenerateWithOp(dst, false, func(d *float32, value float32) {
		*d += value
	})
}
