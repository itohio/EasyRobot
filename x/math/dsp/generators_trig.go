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
func (g *SineGenerator) GenerateWithOp(dst []float32, reverse bool, op func(i int, value float32)) int {
	n := len(dst)
	if n == 0 {
		return 0
	}

	dt := 1.0 / g.sampleRate
	var startTime float32
	if reverse {
		startTime = g.time + float32(n-1)*dt
	}

	for i := range n {
		var t float32
		var idx int
		if reverse {
			t = startTime - float32(i)*dt
			idx = n - 1 - i
		} else {
			t = g.time
			idx = i
			g.time += dt
		}
		// Wrap phase to [0, 2π) to avoid calculation errors
		phase := g.omega*t + g.phase
		phase = math32.Mod(phase, 2*math32.Pi)
		if phase < 0 {
			phase += 2 * math32.Pi
		}
		value := g.amplitude * math32.Sin(phase)
		op(idx, value)
	}

	if reverse {
		g.time += float32(n) * dt
	}

	return n
}

// Generate fills dst with sinusoidal samples and returns the number of samples filled
func (g *SineGenerator) Generate(dst []float32) int {
	return g.GenerateWithOp(dst, false, func(i int, value float32) {
		dst[i] = value
	})
}

// Add adds sinusoidal samples to dst (additive generation)
func (g *SineGenerator) Add(dst []float32) int {
	return g.GenerateWithOp(dst, false, func(i int, value float32) {
		dst[i] += value
	})
}

// GenerateReverse fills dst with sinusoidal samples in reverse time order
func (g *SineGenerator) GenerateReverse(dst []float32) int {
	return g.GenerateWithOp(dst, true, func(i int, value float32) {
		dst[i] = value
	})
}

// AddReverse adds sinusoidal samples in reverse time order to dst
func (g *SineGenerator) AddReverse(dst []float32) int {
	return g.GenerateWithOp(dst, true, func(i int, value float32) {
		dst[i] += value
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
func (g *CosineGenerator) GenerateWithOp(dst []float32, reverse bool, op func(i int, value float32)) int {
	n := len(dst)
	if n == 0 {
		return 0
	}

	dt := 1.0 / g.sampleRate
	var startTime float32
	if reverse {
		startTime = g.time + float32(n-1)*dt
	}

	for i := range n {
		var t float32
		var idx int
		if reverse {
			t = startTime - float32(i)*dt
			idx = n - 1 - i
		} else {
			t = g.time
			idx = i
			g.time += dt
		}
		// Wrap phase to [0, 2π) to avoid calculation errors
		phase := g.omega*t + g.phase
		phase = math32.Mod(phase, 2*math32.Pi)
		if phase < 0 {
			phase += 2 * math32.Pi
		}
		value := g.amplitude * math32.Cos(phase)
		op(idx, value)
	}

	if reverse {
		g.time += float32(n) * dt
	}

	return n
}

// Generate fills dst with cosine samples and returns the number of samples filled
func (g *CosineGenerator) Generate(dst []float32) int {
	return g.GenerateWithOp(dst, false, func(i int, value float32) {
		dst[i] = value
	})
}

// Add adds cosine samples to dst (additive generation)
func (g *CosineGenerator) Add(dst []float32) int {
	return g.GenerateWithOp(dst, false, func(i int, value float32) {
		dst[i] += value
	})
}

// GenerateReverse fills dst with cosine samples in reverse time order
func (g *CosineGenerator) GenerateReverse(dst []float32) int {
	return g.GenerateWithOp(dst, true, func(i int, value float32) {
		dst[i] = value
	})
}

// AddReverse adds cosine samples in reverse time order to dst
func (g *CosineGenerator) AddReverse(dst []float32) int {
	return g.GenerateWithOp(dst, true, func(i int, value float32) {
		dst[i] += value
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
