package dsp

import (
	"github.com/chewxy/math32"
)

// SineGenerator generates a sinusoidal signal with state tracking
type SineGenerator struct {
	sampleRate float32
	frequency  float32
	amplitude  float32
	phase      float32
	omega      float32
	time       float32
}

// NewSineGenerator creates a new sinusoid generator
func NewSineGenerator(sampleRate, frequency, amplitude, phase float32) SineGenerator {
	omega := 2 * math32.Pi * frequency
	return SineGenerator{
		sampleRate: sampleRate,
		frequency:  frequency,
		amplitude:  amplitude,
		phase:      phase,
		omega:      omega,
		time:       0,
	}
}

// GenerateWithOp is the common generation function that takes a binary operation closure and reverse flag
func (g *SineGenerator) GenerateWithOp(dst []float32, reverse bool, op GeneratorAssignOp) int {
	n := len(dst)
	if n == 0 {
		return 0
	}

	// Bounds check elimination: prove to compiler that dst has at least n elements
	_ = dst[n-1]
	dst = dst[:n]

	dt := 1.0 / g.sampleRate
	twoPi := 2 * math32.Pi
	phaseIncrement := g.omega * dt // Pre-compute phase increment

	if reverse {
		// Reverse path: generate samples in reverse time order
		startTime := g.time + float32(n-1)*dt
		// Initialize phase for first sample (last in output)
		phase := g.omega*startTime + g.phase
		// Wrap phase to [0, 2π) - use faster method than Mod
		// Only wrap if phase is significantly outside range to avoid expensive loops
		if phase >= twoPi || phase < 0 {
			phase = math32.Mod(phase, twoPi)
			if phase < 0 {
				phase += twoPi
			}
		}
		for i := range n {
			idx := n - 1 - i
			value := g.amplitude * math32.Sin(phase)
			op(&dst[idx], value)
			// Decrement phase for next iteration (going backwards in time)
			phase -= phaseIncrement
			// Wrap phase only when necessary (not every iteration)
			if phase < 0 {
				phase += twoPi
			}
		}
		g.time += float32(n) * dt
	} else {
		// Forward path: generate samples in normal time order
		// Initialize phase for first sample
		phase := g.omega*g.time + g.phase
		// Wrap phase to [0, 2π) - use Mod only if needed
		// Only wrap if phase is significantly outside range to avoid expensive operations
		if phase >= twoPi || phase < 0 {
			phase = math32.Mod(phase, twoPi)
			if phase < 0 {
				phase += twoPi
			}
		}

		for i := range n {
			value := g.amplitude * math32.Sin(phase)
			op(&dst[i], value)
			// Increment phase for next iteration
			phase += phaseIncrement
			// Wrap phase only when necessary (not every iteration)
			// Since phaseIncrement is small, we only need to wrap occasionally
			if phase >= twoPi {
				phase -= twoPi
			}
		}
		g.time += float32(n) * dt
	}

	return n
}

// Generate fills dst with sinusoidal samples and returns the number of samples filled
func (g *SineGenerator) Generate(dst []float32) int {
	return g.GenerateWithOp(dst, false, func(d *float32, value float32) {
		*d = value
	})
}

// Add adds sinusoidal samples to dst (additive generation)
func (g *SineGenerator) Add(dst []float32) int {
	return g.GenerateWithOp(dst, false, func(d *float32, value float32) {
		*d += value
	})
}

// GenerateReverse fills dst with sinusoidal samples in reverse time order
func (g *SineGenerator) GenerateReverse(dst []float32) int {
	return g.GenerateWithOp(dst, true, func(d *float32, value float32) {
		*d = value
	})
}

// AddReverse adds sinusoidal samples in reverse time order to dst
func (g *SineGenerator) AddReverse(dst []float32) int {
	return g.GenerateWithOp(dst, true, func(d *float32, value float32) {
		*d += value
	})
}

// Correlate computes correlation with the provided samples using reversed signal
func (g *SineGenerator) Correlate(dst, samples []float32) int {
	refLen := len(samples)
	resultLen := len(samples) + refLen - 1
	if len(dst) < resultLen {
		resultLen = len(dst)
		if resultLen < refLen {
			refLen = resultLen
		}
	}

	// Use dst as temporary buffer for reversed signal if possible
	var reversed []float32
	needsAlloc := len(dst) < resultLen+refLen
	if !needsAlloc {
		reversed = dst[resultLen : resultLen+refLen]
	} else {
		// Stack array for small buffers
		if refLen <= 64 {
			var buf [64]float32
			reversed = buf[:refLen]
		} else {
			reversed = make([]float32, refLen)
		}
	}

	// Generate reversed signal directly
	revGen := *g
	revGen.GenerateReverse(reversed)

	// Compute correlation
	nSamples := len(samples)
	nRef := len(reversed)
	for i := range resultLen {
		sum := float32(0)
		for j := range nSamples {
			k := i - j
			if k >= 0 && k < nRef {
				sum += samples[j] * reversed[k]
			}
		}
		dst[i] = sum
	}

	return resultLen
}

// CosineGenerator generates a cosine signal with state tracking
type CosineGenerator struct {
	sampleRate float32
	frequency  float32
	amplitude  float32
	phase      float32
	omega      float32
	time       float32
}

// NewCosineGenerator creates a new cosine generator
func NewCosineGenerator(sampleRate, frequency, amplitude, phase float32) CosineGenerator {
	omega := 2 * math32.Pi * frequency
	return CosineGenerator{
		sampleRate: sampleRate,
		frequency:  frequency,
		amplitude:  amplitude,
		phase:      phase,
		omega:      omega,
		time:       0,
	}
}

// GenerateWithOp is the common generation function that takes a binary operation closure and reverse flag
func (g *CosineGenerator) GenerateWithOp(dst []float32, reverse bool, op GeneratorAssignOp) int {
	n := len(dst)
	if n == 0 {
		return 0
	}

	// Bounds check elimination: prove to compiler that dst has at least n elements
	_ = dst[n-1]
	dst = dst[:n]

	dt := 1.0 / g.sampleRate
	twoPi := 2 * math32.Pi
	phaseIncrement := g.omega * dt // Pre-compute phase increment

	if reverse {
		// Reverse path: generate samples in reverse time order
		startTime := g.time + float32(n-1)*dt
		// Initialize phase for first sample (last in output)
		phase := g.omega*startTime + g.phase
		// Wrap phase to [0, 2π) - use Mod only if needed
		if phase >= twoPi || phase < 0 {
			phase = math32.Mod(phase, twoPi)
			if phase < 0 {
				phase += twoPi
			}
		}
		for i := range n {
			idx := n - 1 - i
			value := g.amplitude * math32.Cos(phase)
			op(&dst[idx], value)
			// Decrement phase for next iteration (going backwards in time)
			phase -= phaseIncrement
			// Wrap phase only when necessary (not every iteration)
			if phase < 0 {
				phase += twoPi
			}
		}
		g.time += float32(n) * dt
	} else {
		// Forward path: generate samples in normal time order
		// Initialize phase for first sample
		phase := g.omega*g.time + g.phase
		// Wrap phase to [0, 2π) - use Mod only if needed
		if phase >= twoPi || phase < 0 {
			phase = math32.Mod(phase, twoPi)
			if phase < 0 {
				phase += twoPi
			}
		}
		for i := range n {
			value := g.amplitude * math32.Cos(phase)
			op(&dst[i], value)
			// Increment phase for next iteration
			phase += phaseIncrement
			// Wrap phase only when necessary (not every iteration)
			if phase >= twoPi {
				phase -= twoPi
			}
		}
		g.time += float32(n) * dt
	}

	return n
}

// Generate fills dst with cosine samples and returns the number of samples filled
func (g *CosineGenerator) Generate(dst []float32) int {
	return g.GenerateWithOp(dst, false, func(d *float32, value float32) {
		*d = value
	})
}

// Add adds cosine samples to dst (additive generation)
func (g *CosineGenerator) Add(dst []float32) int {
	return g.GenerateWithOp(dst, false, func(d *float32, value float32) {
		*d += value
	})
}

// GenerateReverse fills dst with cosine samples in reverse time order
func (g *CosineGenerator) GenerateReverse(dst []float32) int {
	return g.GenerateWithOp(dst, true, func(d *float32, value float32) {
		*d = value
	})
}

// AddReverse adds cosine samples in reverse time order to dst
func (g *CosineGenerator) AddReverse(dst []float32) int {
	return g.GenerateWithOp(dst, true, func(d *float32, value float32) {
		*d += value
	})
}

// Correlate computes correlation with the provided samples using reversed signal
func (g *CosineGenerator) Correlate(dst, samples []float32) int {
	refLen := len(samples)
	resultLen := len(samples) + refLen - 1
	if len(dst) < resultLen {
		resultLen = len(dst)
		if resultLen < refLen {
			refLen = resultLen
		}
	}

	// Use dst as temporary buffer for reversed signal if possible
	var reversed []float32
	needsAlloc := len(dst) < resultLen+refLen
	if !needsAlloc {
		reversed = dst[resultLen : resultLen+refLen]
	} else {
		if refLen <= 64 {
			var buf [64]float32
			reversed = buf[:refLen]
		} else {
			reversed = make([]float32, refLen)
		}
	}

	// Generate reversed signal directly
	revGen := *g
	revGen.GenerateReverse(reversed)

	// Compute correlation
	nSamples := len(samples)
	nRef := len(reversed)
	for i := range resultLen {
		sum := float32(0)
		for j := range nSamples {
			k := i - j
			if k >= 0 && k < nRef {
				sum += samples[j] * reversed[k]
			}
		}
		dst[i] = sum
	}

	return resultLen
}

// SincGenerator generates a sinc function (sin(πx)/(πx))
// Used for pulse shaping in digital communications
type SincGenerator struct {
	center    float32 // Center position (typically 0)
	scale     float32 // Scale factor (typically 1.0)
	amplitude float32 // Amplitude scaling
}

// NewSincGenerator creates a new sinc generator
// center: center position of the sinc function (typically 0)
// scale: scale factor for the argument (typically 1.0)
// amplitude: amplitude scaling factor
func NewSincGenerator(center, scale, amplitude float32) SincGenerator {
	return SincGenerator{
		center:    center,
		scale:     scale,
		amplitude: amplitude,
	}
}

// GenerateWithOp is the common generation function that takes a binary operation closure and reverse flag
func (g *SincGenerator) GenerateWithOp(dst []float32, reverse bool, op GeneratorAssignOp) int {
	n := len(dst)
	if n == 0 {
		return 0
	}

	// Bounds check elimination: prove to compiler that dst has at least n elements
	_ = dst[n-1]
	dst = dst[:n]

	pi := math32.Pi

	if reverse {
		// Reverse path: generate samples in reverse order (from end to start)
		// Start at i=n-1: x = (n-1) - center
		x := float32(n-1) - g.center
		xScaled := x * g.scale
		xScaledDecrement := g.scale // When x decrements by 1, xScaled decrements by scale

		for i := range n {
			idx := n - 1 - i
			var value float32
			if xScaled == 0 {
				value = g.amplitude
			} else {
				value = g.amplitude * math32.Sin(pi*xScaled) / (pi * xScaled)
			}
			op(&dst[idx], value)
			// Decrement for next iteration (going backwards)
			x -= 1.0
			xScaled -= xScaledDecrement
		}
	} else {
		// Forward path: generate samples in normal order
		// Incremental computation: avoid float32(i) and multiplication in hot path
		// Start at i=0: x = 0 - center
		x := -g.center
		xScaled := x * g.scale
		xScaledIncrement := g.scale // When x increments by 1, xScaled increments by scale

		for i := range n {
			var value float32
			if xScaled == 0 {
				value = g.amplitude
			} else {
				value = g.amplitude * math32.Sin(pi*xScaled) / (pi * xScaled)
			}
			op(&dst[i], value)
			// Increment for next iteration
			x += 1.0
			xScaled += xScaledIncrement
		}
	}
	return n
}

// Generate fills dst with sinc function samples
func (g *SincGenerator) Generate(dst []float32) int {
	return g.GenerateWithOp(dst, false, func(d *float32, value float32) {
		*d = value
	})
}

// Add adds sinc function samples to dst (additive generation)
func (g *SincGenerator) Add(dst []float32) int {
	return g.GenerateWithOp(dst, false, func(d *float32, value float32) {
		*d += value
	})
}

// RaisedCosineGenerator generates a raised cosine pulse shape
// Commonly used for pulse shaping in digital communications
type RaisedCosineGenerator struct {
	center    float32
	scale     float32
	rolloff   float32
	amplitude float32
}

// NewRaisedCosineGenerator creates a new raised cosine generator
// center: center sample position
// scale: scaling factor for normalized time (typically 1/Tc)
// rolloff: excess bandwidth factor (0 <= rolloff <= 1)
// amplitude: amplitude scaling factor
func NewRaisedCosineGenerator(center, scale, rolloff, amplitude float32) RaisedCosineGenerator {
	if scale == 0 {
		scale = 1
	}
	if rolloff < 0 {
		rolloff = -rolloff
	}
	if rolloff > 1 {
		rolloff = 1
	}
	return RaisedCosineGenerator{
		center:    center,
		scale:     scale,
		rolloff:   rolloff,
		amplitude: amplitude,
	}
}

// GenerateWithOp generates raised cosine samples with optional reverse traversal
func (g *RaisedCosineGenerator) GenerateWithOp(dst []float32, reverse bool, op GeneratorAssignOp) int {
	n := len(dst)
	if n == 0 {
		return 0
	}

	_ = dst[n-1]
	dst = dst[:n]

	scale := g.scale
	if scale == 0 {
		scale = 1
	}
	alpha := g.rolloff
	if alpha < 0 {
		alpha = -alpha
	}

	start := -g.center
	step := scale
	if reverse {
		start = float32(n-1) - g.center
		step = -scale
	}

	xScaled := start * scale
	twoAlpha := 2 * alpha
	xAlpha := twoAlpha * xScaled

	theta := math32.Pi * xScaled
	deltaTheta := math32.Pi * step

	phi := math32.Pi * alpha * xScaled
	deltaPhi := math32.Pi * alpha * step

	sinTheta := math32.Sin(theta)
	cosTheta := math32.Cos(theta)
	sinDeltaTheta := math32.Sin(deltaTheta)
	cosDeltaTheta := math32.Cos(deltaTheta)

	useAlpha := alpha >= 1e-6
	sinPhi := float32(0)
	cosPhi := float32(1)
	sinDeltaPhi := float32(0)
	cosDeltaPhi := float32(1)
	if useAlpha {
		sinPhi = math32.Sin(phi)
		cosPhi = math32.Cos(phi)
		sinDeltaPhi = math32.Sin(deltaPhi)
		cosDeltaPhi = math32.Cos(deltaPhi)
	}
	deltaAlpha := twoAlpha * step

	idx := 0
	incrementIndex := 1
	if reverse {
		idx = n - 1
		incrementIndex = -1
	}

	const thetaEps = 1e-6
	const denomEps = 1e-5
	amp := g.amplitude
	limitValue := float32(0)
	if useAlpha && alpha > denomEps {
		limitValue = amp * (alpha / 2) * math32.Sin(math32.Pi/(2*alpha))
	}

	for i := 0; i < n; i++ {
		value := amp
		if useAlpha {
			denom := 1 - xAlpha*xAlpha
			if math32.Abs(denom) < denomEps {
				value = limitValue
			} else {
				sinc := float32(1)
				if math32.Abs(theta) >= thetaEps {
					sinc = sinTheta / theta
				}
				value = amp * (sinc * cosPhi / denom)
			}
		} else if math32.Abs(theta) >= thetaEps {
			value = amp * sinTheta / theta
		}

		op(&dst[idx], value)

		idx += incrementIndex
		theta += deltaTheta
		xScaled += step
		xAlpha += deltaAlpha

		nextSinTheta := sinTheta*cosDeltaTheta + cosTheta*sinDeltaTheta
		nextCosTheta := cosTheta*cosDeltaTheta - sinTheta*sinDeltaTheta
		sinTheta = nextSinTheta
		cosTheta = nextCosTheta

		if useAlpha {
			nextSinPhi := sinPhi*cosDeltaPhi + cosPhi*sinDeltaPhi
			nextCosPhi := cosPhi*cosDeltaPhi - sinPhi*sinDeltaPhi
			sinPhi = nextSinPhi
			cosPhi = nextCosPhi
		}
	}

	return n
}

// Generate fills dst with raised cosine samples
func (g *RaisedCosineGenerator) Generate(dst []float32) int {
	return g.GenerateWithOp(dst, false, func(d *float32, value float32) {
		*d = value
	})
}

// Add adds raised cosine samples to dst
func (g *RaisedCosineGenerator) Add(dst []float32) int {
	return g.GenerateWithOp(dst, false, func(d *float32, value float32) {
		*d += value
	})
}

// AddReverse adds raised cosine samples in reverse order to dst
func (g *RaisedCosineGenerator) AddReverse(dst []float32) int {
	return g.GenerateWithOp(dst, true, func(d *float32, value float32) {
		*d += value
	})
}

// Correlate computes correlation with the provided samples using reversed signal
func (g *RaisedCosineGenerator) Correlate(dst, samples []float32) int {
	refLen := len(samples)
	resultLen := len(samples) + refLen - 1
	if len(dst) < resultLen {
		resultLen = len(dst)
		if resultLen < refLen {
			refLen = resultLen
		}
	}

	var reversed []float32
	needsAlloc := len(dst) < resultLen+refLen
	if !needsAlloc {
		reversed = dst[resultLen : resultLen+refLen]
	} else {
		if refLen <= 64 {
			var buf [64]float32
			reversed = buf[:refLen]
		} else {
			reversed = make([]float32, refLen)
		}
	}

	revGen := *g
	revGen.GenerateWithOp(reversed, true, func(d *float32, value float32) {
		*d = value
	})

	nSamples := len(samples)
	nRef := len(reversed)
	for i := range resultLen {
		sum := float32(0)
		for j := range nSamples {
			k := i - j
			if k >= 0 && k < nRef {
				sum += samples[j] * reversed[k]
			}
		}
		dst[i] = sum
	}

	return resultLen
}
