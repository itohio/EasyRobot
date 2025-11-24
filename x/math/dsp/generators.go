package dsp

import (
	"math/rand"

	"github.com/chewxy/math32"
)

// Generator is the interface for all signal generators
type Generator interface {
	Generate(dst []float32) int
}

// Correlator is the interface for generators that support correlation
type Correlator interface {
	Correlate(dst, samples []float32) int
}

// ChirpGenerator generates a linear chirp signal (frequency sweep)
type ChirpGenerator struct {
	sampleRate float32
	f0         float32 // start frequency
	f1         float32 // end frequency
	amplitude  float32
	time       float32
	dt         float32
	k          float32 // frequency sweep rate
}

// NewChirpGenerator creates a new linear chirp generator
func NewChirpGenerator(sampleRate, f0, f1, amplitude float32) ChirpGenerator {
	dt := 1.0 / sampleRate
	// For linear chirp: f(t) = f0 + k*t, where k = (f1-f0)/T
	// We'll use time as the sweep parameter
	return ChirpGenerator{
		sampleRate: sampleRate,
		f0:         f0,
		f1:         f1,
		amplitude:  amplitude,
		time:       0,
		dt:         dt,
		k:          0, // Will be set when duration is known or computed on-the-fly
	}
}

// generate is the common generation function that takes a binary operation closure and reverse flag
// For reverse: generates f1->f0 chirp (time-reversed version of f0->f1)
func (g *ChirpGenerator) generate(dst []float32, reverse bool, op func(i int, value float32)) int {
	n := len(dst)
	if n == 0 {
		return 0
	}

	T := float32(n) * g.dt
	var fStart float32
	var k float32
	var time float32
	if reverse {
		// For correlation, we need the time-reversed chirp
		// Time-reversed f0->f1 is equivalent to forward f1->f0
		fStart = g.f1
		if T > 0 {
			k = (g.f0 - g.f1) / T // Reverse sweep rate
		}
		time = 0
	} else {
		fStart = g.f0
		if T > 0 {
			g.k = (g.f1 - g.f0) / T
		}
		k = g.k
		time = g.time
	}

	for i := range n {
		// For linear chirp, phase = 2*pi*(fStart*t + 0.5*k*t^2)
		phase := 2 * math32.Pi * (fStart*time + 0.5*k*time*time)
		value := g.amplitude * math32.Sin(phase)
		op(i, value)
		time += g.dt
		if !reverse {
			g.time = time
		}
	}

	if reverse {
		g.time += float32(n) * g.dt
	}

	return n
}

// Generate fills dst with chirp samples and returns the number of samples filled
func (g *ChirpGenerator) Generate(dst []float32) int {
	return g.generate(dst, false, func(i int, value float32) {
		dst[i] = value
	})
}

// Add adds chirp samples to dst (additive generation)
func (g *ChirpGenerator) Add(dst []float32) int {
	return g.generate(dst, false, func(i int, value float32) {
		dst[i] += value
	})
}

// GenerateReverse fills dst with chirp samples in reverse time order (for correlation)
// This generates the time-reversed chirp directly: generates f1->f0 which is equivalent to
// time-reversed f0->f1, avoiding the need to reverse the array
func (g *ChirpGenerator) GenerateReverse(dst []float32) int {
	return g.generate(dst, true, func(i int, value float32) {
		dst[i] = value
	})
}

// AddReverse adds chirp samples in reverse time order to dst (additive generation)
func (g *ChirpGenerator) AddReverse(dst []float32) int {
	return g.generate(dst, true, func(i int, value float32) {
		dst[i] += value
	})
}

// Correlate computes correlation with the provided samples using reversed chirp
// Generates reversed chirp directly without extra allocations
func (g *ChirpGenerator) Correlate(dst, samples []float32) int {
	refLen := len(samples)
	resultLen := len(samples) + refLen - 1
	if len(dst) < resultLen {
		resultLen = len(dst)
		// Adjust refLen if dst is too small
		if resultLen < refLen {
			refLen = resultLen
		}
	}

	// Try to use dst as temporary buffer for reversed chirp
	var reversed []float32
	needsAlloc := len(dst) < resultLen+refLen
	if !needsAlloc {
		reversed = dst[resultLen : resultLen+refLen]
	} else {
		// Stack array for small buffers, heap for large
		if refLen <= 64 {
			var buf [64]float32
			reversed = buf[:refLen]
		} else {
			reversed = make([]float32, refLen)
		}
	}

	// Generate reversed chirp directly (time-reversed f0->f1 = f1->f0 forward)
	revGen := NewChirpGenerator(g.sampleRate, g.f1, g.f0, g.amplitude)
	revGen.GenerateReverse(reversed)

	// Compute correlation manually: corr[i] = sum(samples[j] * reversed[i-j])
	// Convolve1D doesn't work for correlation (output size is N+M-1, not N-M+1)
	nSamples := len(samples)
	nReversed := len(reversed)

	// Initialize result
	for i := range resultLen {
		dst[i] = 0
	}

	// Compute cross-correlation
	for i := range resultLen {
		sum := float32(0)
		for j := range nSamples {
			k := i - j
			if k >= 0 && k < nReversed {
				sum += samples[j] * reversed[k]
			}
		}
		dst[i] = sum
	}

	return resultLen
}

// LogarithmicChirpGenerator generates a logarithmic chirp signal
type LogarithmicChirpGenerator struct {
	sampleRate float32
	f0         float32 // start frequency
	f1         float32 // end frequency
	amplitude  float32
	time       float32
	dt         float32
	phase      float32
}

// NewLogarithmicChirpGenerator creates a new logarithmic chirp generator
func NewLogarithmicChirpGenerator(sampleRate, f0, f1, amplitude float32) LogarithmicChirpGenerator {
	dt := 1.0 / sampleRate
	return LogarithmicChirpGenerator{
		sampleRate: sampleRate,
		f0:         f0,
		f1:         f1,
		amplitude:  amplitude,
		time:       0,
		dt:         dt,
		phase:      0,
	}
}

// generate is the common generation function that takes a binary operation closure and reverse flag
// For reverse: generates f1->f0 logarithmic chirp (time-reversed version of f0->f1)
func (g *LogarithmicChirpGenerator) generate(dst []float32, reverse bool, op func(i int, value float32)) int {
	n := len(dst)
	if n == 0 {
		return 0
	}

	T := float32(n) * g.dt
	if T <= 0 || g.f0 <= 0 || g.f1 <= 0 {
		return n
	}

	var fStart, fBase float32
	var ratio, lnRatio float32
	var phase float32
	var time float32
	useConstantFreq := false

	if reverse {
		// For correlation, we need the time-reversed logarithmic chirp
		// Time-reversed f0->f1 is equivalent to forward f1->f0
		fStart = g.f1
		fBase = g.f1
		ratio = g.f0 / g.f1
		lnRatio = math32.Log(ratio)
		phase = 0
		time = 0
		useConstantFreq = lnRatio == 0
	} else {
		fStart = g.f0
		fBase = g.f0
		ratio = g.f1 / g.f0
		lnRatio = math32.Log(ratio)
		phase = g.phase
		time = g.time
		useConstantFreq = lnRatio == 0
	}

	var omega float32
	if useConstantFreq {
		omega = 2 * math32.Pi * fStart / g.sampleRate
	}

	for i := range n {
		var value float32
		if useConstantFreq {
			value = g.amplitude * math32.Sin(phase)
			phase += omega * g.dt
		} else {
			freq := fBase * math32.Pow(ratio, time/T)
			phaseIncrement := 2 * math32.Pi * freq * g.dt
			phase += phaseIncrement
			value = g.amplitude * math32.Sin(phase)
		}
		op(i, value)
		time += g.dt
		if !reverse {
			g.time = time
			g.phase = phase
		}
	}

	if reverse {
		g.time += float32(n) * g.dt
	}

	return n
}

// Generate fills dst with logarithmic chirp samples
func (g *LogarithmicChirpGenerator) Generate(dst []float32) int {
	return g.generate(dst, false, func(i int, value float32) {
		dst[i] = value
	})
}

// Add adds logarithmic chirp samples to dst (additive generation)
func (g *LogarithmicChirpGenerator) Add(dst []float32) int {
	return g.generate(dst, false, func(i int, value float32) {
		dst[i] += value
	})
}

// GenerateReverse fills dst with logarithmic chirp samples in reverse time order (for correlation)
// This generates the time-reversed chirp directly: generates f1->f0 which is equivalent to
// time-reversed f0->f1, avoiding the need to reverse the array
func (g *LogarithmicChirpGenerator) GenerateReverse(dst []float32) int {
	return g.generate(dst, true, func(i int, value float32) {
		dst[i] = value
	})
}

// AddReverse adds logarithmic chirp samples in reverse time order to dst (additive generation)
func (g *LogarithmicChirpGenerator) AddReverse(dst []float32) int {
	return g.generate(dst, true, func(i int, value float32) {
		dst[i] += value
	})
}

// Correlate computes correlation with reversed logarithmic chirp
// Generates reversed chirp directly without extra allocations
func (g *LogarithmicChirpGenerator) Correlate(dst, samples []float32) int {
	refLen := len(samples)
	resultLen := len(samples) + refLen - 1
	if len(dst) < resultLen {
		resultLen = len(dst)
		// Adjust refLen if dst is too small
		if resultLen < refLen {
			refLen = resultLen
		}
	}

	// Try to use dst as temporary buffer for reversed chirp
	var reversed []float32
	needsAlloc := len(dst) < resultLen+refLen
	if !needsAlloc {
		// Use end of dst as temporary buffer for reversed chirp
		reversed = dst[resultLen : resultLen+refLen]
	} else {
		// Minimum allocation needed - can't avoid this for the kernel
		reversed = make([]float32, refLen)
	}

	// Generate reversed logarithmic chirp directly (time-reversed f0->f1 = f1->f0 forward)
	revGen := NewLogarithmicChirpGenerator(g.sampleRate, g.f1, g.f0, g.amplitude)
	revGen.GenerateReverse(reversed)

	// Compute correlation manually: corr[i] = sum(samples[j] * reversed[i-j])
	nSamples := len(samples)
	nReversed := len(reversed)

	// Initialize result
	for i := range resultLen {
		dst[i] = 0
	}

	// Compute cross-correlation
	for i := range resultLen {
		sum := float32(0)
		for j := range nSamples {
			k := i - j
			if k >= 0 && k < nReversed {
				sum += samples[j] * reversed[k]
			}
		}
		dst[i] = sum
	}

	return resultLen
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

// Generate fills dst with square wave samples
func (g *SquareGenerator) Generate(dst []float32) int {
	n := len(dst)
	if n == 0 {
		return 0
	}

	dt := 1.0 / g.sampleRate
	for i := range n {
		// Use period to determine square wave value
		periodPos := math32.Mod(g.time*g.frequency, 1.0)
		if periodPos < 0.5 {
			dst[i] = g.amplitude
		} else {
			dst[i] = -g.amplitude
		}
		g.time += dt
	}

	return n
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

// Generate fills dst with triangle wave samples
func (g *TriangleGenerator) Generate(dst []float32) int {
	n := len(dst)
	if n == 0 {
		return 0
	}

	dt := 1.0 / g.sampleRate
	for i := range n {
		phase := math32.Mod(g.time*g.frequency, 1.0)
		if phase < 0 {
			phase += 1
		}
		dst[i] = triangleValue(g.amplitude, phase)
		g.time += dt
	}
	if n > 0 {
		cycles := g.time * g.frequency
		frac := cycles - math32.Floor(cycles)
		if frac < 1e-6 || 1-frac < 1e-6 {
			dst[n-1] = triangleValue(g.amplitude, 0)
		}
	}

	return n
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

// Generate fills dst with white noise samples
func (g *NoiseWhiteGenerator) Generate(dst []float32) int {
	n := len(dst)
	for i := range n {
		dst[i] = g.amplitude * (rand.Float32()*2 - 1)
	}
	return n
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

// Generate fills dst with pink noise samples
func (g *NoisePinkGenerator) Generate(dst []float32) int {
	n := len(dst)
	for i := range n {
		white := g.amplitude * (rand.Float32()*2 - 1)
		y0 := g.b0*white + g.b1*g.y1 + g.b2*g.y2 + g.b3*g.y3 + g.b4*g.y4 + g.b5*g.y5
		dst[i] = y0

		// Update filter state
		g.y5 = g.y4
		g.y4 = g.y3
		g.y3 = g.y2
		g.y2 = g.y1
		g.y1 = y0
	}
	return n
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

// Generate fills dst with brown noise samples
func (g *NoiseBrownGenerator) Generate(dst []float32) int {
	n := len(dst)
	for i := range n {
		white := g.amplitude * (rand.Float32()*2 - 1)
		g.accumulator += white * 0.02

		// High-pass filter to remove DC component
		if i > 0 {
			dst[i] = g.accumulator - g.prev*0.99
		} else {
			dst[i] = g.accumulator
		}
		g.prev = dst[i]

		// Normalize to prevent drift
		if g.accumulator > g.amplitude {
			g.accumulator = g.amplitude
		} else if g.accumulator < -g.amplitude {
			g.accumulator = -g.amplitude
		}
	}
	return n
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

// Generate fills dst with constant value
func (g *ConstantGenerator) Generate(dst []float32) int {
	n := len(dst)
	for i := range n {
		dst[i] = g.value
	}
	return n
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

// Generate fills dst with ramp samples
func (g *RampGenerator) Generate(dst []float32) int {
	n := len(dst)
	for i := range n {
		dst[i] = g.current
		g.current += g.increment
	}
	return n
}

// Reset resets the ramp to its start value
func (g *RampGenerator) Reset() {
	g.current = g.start
}
