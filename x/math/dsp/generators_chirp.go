package dsp

import (
	"github.com/chewxy/math32"
)

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

// GenerateWithOp is the common generation function that takes a binary operation closure and reverse flag
// For reverse: generates f1->f0 chirp (time-reversed version of f0->f1)
func (g *ChirpGenerator) GenerateWithOp(dst []float32, reverse bool, op GeneratorAssignOp) int {
	n := len(dst)
	if n == 0 {
		return 0
	}

	// Bounds check elimination: prove to compiler that dst has at least n elements
	_ = dst[n-1]
	dst = dst[:n]

	T := float32(n) * g.dt
	twoPi := 2 * math32.Pi

	if reverse {
		// Reverse path: generate samples in reverse time order
		// For correlation, we need the time-reversed chirp
		// Time-reversed f0->f1 is equivalent to forward f1->f0
		fStart := g.f1
		var k float32
		if T > 0 {
			k = (g.f0 - g.f1) / T // Reverse sweep rate
		}
		time := float32(0)
		// Pre-compute phase increment components
		phaseRate := twoPi * fStart
		phaseAccel := twoPi * 0.5 * k
		phase := phaseRate*time + phaseAccel*time*time
		// Pre-compute incremental phase update components
		// phase(t+dt) = phase(t) + 2π*dt*(fStart + k*t) + 2π*0.5*k*dt²
		// = phase(t) + phaseRate*dt + 2π*k*t*dt + phaseAccel*dt²
		// But we can simplify: phase += phaseRate*dt + 2π*k*t*dt + phaseAccel*dt²
		// Since k*t changes, we track it incrementally: k*t += k*dt
		dt := g.dt
		phaseIncrementBase := phaseRate * dt
		phaseIncrementAccel := twoPi * k * dt * dt // k*dt² term
		kt := float32(0)                           // k*t, incremented by k*dt each iteration

		for i := range n {
			value := g.amplitude * math32.Sin(phase)
			op(&dst[i], value)
			// Incremental phase update: phase += 2π*dt*(fStart + k*t) + 2π*0.5*k*dt²
			// = phaseRate*dt + 2π*k*t*dt + phaseAccel*dt²
			phase += phaseIncrementBase + twoPi*kt*dt + phaseIncrementAccel
			// Wrap phase to [0, 2π) to maintain precision
			if phase >= twoPi {
				phase -= twoPi
			} else if phase < 0 {
				phase += twoPi
			}
			kt += k * dt
			time += dt
		}
		g.time += float32(n) * dt
	} else {
		// Forward path: generate samples in normal time order
		fStart := g.f0
		if T > 0 {
			g.k = (g.f1 - g.f0) / T
		}
		k := g.k
		time := g.time
		// Pre-compute phase increment components
		phaseRate := twoPi * fStart
		phaseAccel := twoPi * 0.5 * k
		phase := phaseRate*time + phaseAccel*time*time
		// Pre-compute incremental phase update components
		dt := g.dt
		phaseIncrementBase := phaseRate * dt
		phaseIncrementAccel := twoPi * k * dt * dt // k*dt² term
		kt := k * time                             // k*t, incremented by k*dt each iteration

		for i := range n {
			value := g.amplitude * math32.Sin(phase)
			op(&dst[i], value)
			// Incremental phase update: phase += 2π*dt*(fStart + k*t) + 2π*0.5*k*dt²
			// = phaseRate*dt + 2π*k*t*dt + phaseAccel*dt²
			phase += phaseIncrementBase + twoPi*kt*dt + phaseIncrementAccel
			// Wrap phase to [0, 2π) to maintain precision
			if phase >= twoPi {
				phase -= twoPi
			} else if phase < 0 {
				phase += twoPi
			}
			kt += k * dt
			time += dt
		}
		g.time = time
	}

	return n
}

// Generate fills dst with chirp samples and returns the number of samples filled
func (g *ChirpGenerator) Generate(dst []float32) int {
	return g.GenerateWithOp(dst, false, func(d *float32, value float32) {
		*d = value
	})
}

// Add adds chirp samples to dst (additive generation)
func (g *ChirpGenerator) Add(dst []float32) int {
	return g.GenerateWithOp(dst, false, func(d *float32, value float32) {
		*d += value
	})
}

// GenerateReverse fills dst with chirp samples in reverse time order (for correlation)
// This generates the time-reversed chirp directly: generates f1->f0 which is equivalent to
// time-reversed f0->f1, avoiding the need to reverse the array
func (g *ChirpGenerator) GenerateReverse(dst []float32) int {
	return g.GenerateWithOp(dst, true, func(d *float32, value float32) {
		*d = value
	})
}

// AddReverse adds chirp samples in reverse time order to dst (additive generation)
func (g *ChirpGenerator) AddReverse(dst []float32) int {
	return g.GenerateWithOp(dst, true, func(d *float32, value float32) {
		*d += value
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

// GenerateWithOp is the common generation function that takes a binary operation closure and reverse flag
// For reverse: generates f1->f0 logarithmic chirp (time-reversed version of f0->f1)
func (g *LogarithmicChirpGenerator) GenerateWithOp(dst []float32, reverse bool, op GeneratorAssignOp) int {
	n := len(dst)
	if n == 0 {
		return 0
	}

	// Bounds check elimination: prove to compiler that dst has at least n elements
	_ = dst[n-1]
	dst = dst[:n]

	T := float32(n) * g.dt
	twoPi := 2 * math32.Pi
	if T <= 0 || g.f0 <= 0 || g.f1 <= 0 {
		return n
	}

	if reverse {
		// Reverse path: generate samples in reverse time order
		// For correlation, we need the time-reversed logarithmic chirp
		// Time-reversed f0->f1 is equivalent to forward f1->f0
		fStart := g.f1
		fBase := g.f1
		ratio := g.f0 / g.f1
		lnRatio := math32.Log(ratio)
		useConstantFreq := lnRatio == 0

		var omega float32
		if useConstantFreq {
			omega = 2 * math32.Pi * fStart / g.sampleRate
		}

		phase := float32(0)
		time := float32(0)

		// Pre-compute incremental frequency update for logarithmic chirp
		// freq(t+dt) = fBase * ratio^((t+dt)/T) = freq(t) * ratio^(dt/T)
		var freq float32
		var ratioPowerIncrement float32
		var phaseIncrementBase float32
		if !useConstantFreq {
			ratioPowerIncrement = math32.Pow(ratio, g.dt/T) // Pre-compute once
			freq = fBase * math32.Pow(ratio, time/T)        // Initial frequency
			phaseIncrementBase = 2 * math32.Pi * g.dt       // Pre-compute 2π*dt
		}

		for i := range n {
			var value float32
			if useConstantFreq {
				value = g.amplitude * math32.Sin(phase)
				phase += omega * g.dt
			} else {
				// Incremental frequency update: freq *= ratio^(dt/T)
				phaseIncrement := phaseIncrementBase * freq
				phase += phaseIncrement
				value = g.amplitude * math32.Sin(phase)
				// Update frequency incrementally (avoid Pow in hot path)
				freq *= ratioPowerIncrement
			}
			// Wrap phase to [0, 2π) to maintain precision
			if phase >= twoPi {
				phase -= twoPi
			} else if phase < 0 {
				phase += twoPi
			}
			op(&dst[i], value)
			time += g.dt
		}
		g.time += float32(n) * g.dt
	} else {
		// Forward path: generate samples in normal time order
		fStart := g.f0
		fBase := g.f0
		ratio := g.f1 / g.f0
		lnRatio := math32.Log(ratio)
		useConstantFreq := lnRatio == 0

		var omega float32
		if useConstantFreq {
			omega = 2 * math32.Pi * fStart / g.sampleRate
		}

		phase := g.phase
		time := g.time

		// Pre-compute incremental frequency update for logarithmic chirp
		var freq float32
		var ratioPowerIncrement float32
		var phaseIncrementBase float32
		if !useConstantFreq {
			ratioPowerIncrement = math32.Pow(ratio, g.dt/T) // Pre-compute once
			freq = fBase * math32.Pow(ratio, time/T)        // Initial frequency
			phaseIncrementBase = 2 * math32.Pi * g.dt       // Pre-compute 2π*dt
		}

		for i := range n {
			var value float32
			if useConstantFreq {
				value = g.amplitude * math32.Sin(phase)
				phase += omega * g.dt
			} else {
				// Incremental frequency update: freq *= ratio^(dt/T)
				phaseIncrement := phaseIncrementBase * freq
				phase += phaseIncrement
				value = g.amplitude * math32.Sin(phase)
				// Update frequency incrementally (avoid Pow in hot path)
				freq *= ratioPowerIncrement
			}
			// Wrap phase to [0, 2π) to maintain precision
			if phase >= twoPi {
				phase -= twoPi
			} else if phase < 0 {
				phase += twoPi
			}
			op(&dst[i], value)
			time += g.dt
		}
		g.time = time
		g.phase = phase
	}

	return n
}

// Generate fills dst with logarithmic chirp samples
func (g *LogarithmicChirpGenerator) Generate(dst []float32) int {
	return g.GenerateWithOp(dst, false, func(d *float32, value float32) {
		*d = value
	})
}

// Add adds logarithmic chirp samples to dst (additive generation)
func (g *LogarithmicChirpGenerator) Add(dst []float32) int {
	return g.GenerateWithOp(dst, false, func(d *float32, value float32) {
		*d += value
	})
}

// GenerateReverse fills dst with logarithmic chirp samples in reverse time order (for correlation)
// This generates the time-reversed chirp directly: generates f1->f0 which is equivalent to
// time-reversed f0->f1, avoiding the need to reverse the array
func (g *LogarithmicChirpGenerator) GenerateReverse(dst []float32) int {
	return g.GenerateWithOp(dst, true, func(d *float32, value float32) {
		*d = value
	})
}

// AddReverse adds logarithmic chirp samples in reverse time order to dst (additive generation)
func (g *LogarithmicChirpGenerator) AddReverse(dst []float32) int {
	return g.GenerateWithOp(dst, true, func(d *float32, value float32) {
		*d += value
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

// PinkChirpGenerator generates a linear chirp signal with constant energy per frequency
// (pink noise spectrum: -6dB per octave). The amplitude envelope decreases as frequency increases
// to maintain constant energy per frequency bin.
type PinkChirpGenerator struct {
	sampleRate float32
	f0         float32 // start frequency
	f1         float32 // end frequency
	amplitude  float32
	time       float32
	dt         float32
	k          float32 // frequency sweep rate
}

// NewPinkChirpGenerator creates a new pink linear chirp generator
// The generated chirp has constant energy per frequency (pink noise spectrum)
func NewPinkChirpGenerator(sampleRate, f0, f1, amplitude float32) PinkChirpGenerator {
	dt := 1.0 / sampleRate
	return PinkChirpGenerator{
		sampleRate: sampleRate,
		f0:         f0,
		f1:         f1,
		amplitude:  amplitude,
		time:       0,
		dt:         dt,
		k:          0, // Will be set when duration is known
	}
}

// GenerateWithOp generates a pink chirp with constant energy per frequency
func (g *PinkChirpGenerator) GenerateWithOp(dst []float32, reverse bool, op GeneratorAssignOp) int {
	n := len(dst)
	if n == 0 {
		return 0
	}

	// Bounds check elimination
	_ = dst[n-1]
	dst = dst[:n]

	T := float32(n) * g.dt
	twoPi := 2 * math32.Pi

	// Compute pink noise envelope parameters
	// Kend = 10^((-6 * log2(f_max/f_min)) / 20.0)
	// K = log(Kend) / T
	fMin := g.f0
	fMax := g.f1
	if fMin > fMax {
		fMin, fMax = fMax, fMin
	}
	if fMin <= 0 || fMax <= 0 {
		return n
	}

	// Compute Kend: 10^((-6 * log2(fMax/fMin)) / 20.0)
	// = 10^(-0.3 * log2(fMax/fMin))
	// = 10^(-0.3 * log(fMax/fMin) / log(2))
	// = exp(-0.3 * log(fMax/fMin) / log(2) * ln(10))
	// = exp(-0.3 * ln(10) / ln(2) * log(fMax/fMin))
	// = exp(-0.3 * 2.302585 / 0.693147 * log(fMax/fMin))
	// = exp(-0.996578 * log(fMax/fMin))
	// = (fMax/fMin)^(-0.996578)
	// For float32, we use math32.Log and math32.Exp
	logRatio := math32.Log(fMax / fMin)
	Kend := math32.Exp(-0.3 * math32.Log(10) / math32.Log(2) * logRatio)

	var K float32
	if T > 0 {
		K = math32.Log(Kend) / T
	}

	if reverse {
		// Reverse path: generate samples in reverse time order
		fStart := g.f1
		var k float32
		if T > 0 {
			k = (g.f0 - g.f1) / T
		}
		time := float32(0)
		phaseRate := twoPi * fStart
		phaseAccel := twoPi * 0.5 * k
		phase := phaseRate*time + phaseAccel*time*time
		dt := g.dt
		phaseIncrementBase := phaseRate * dt
		phaseIncrementAccel := twoPi * k * dt * dt
		kt := float32(0)
		// For reverse, envelope starts at Kend and decreases
		// envelope(t) = exp((T - t) * K)
		// envelope(t+dt) = exp((T - t - dt) * K) = envelope(t) * exp(-dt * K)
		envelope := Kend
		envelopeDecrement := math32.Exp(-dt * K) // Pre-compute decrement factor

		for i := range n {
			value := g.amplitude * math32.Sin(phase) * envelope
			op(&dst[i], value)
			phase += phaseIncrementBase + twoPi*kt*dt + phaseIncrementAccel
			// Wrap phase to [0, 2π) to maintain precision
			if phase >= twoPi {
				phase -= twoPi
			} else if phase < 0 {
				phase += twoPi
			}
			kt += k * dt
			time += dt
			// Incremental envelope update
			envelope *= envelopeDecrement
		}
		g.time += float32(n) * dt
	} else {
		// Forward path: generate samples in normal time order
		fStart := g.f0
		if T > 0 {
			g.k = (g.f1 - g.f0) / T
		}
		k := g.k
		time := g.time
		phaseRate := twoPi * fStart
		phaseAccel := twoPi * 0.5 * k
		phase := phaseRate*time + phaseAccel*time*time
		dt := g.dt
		phaseIncrementBase := phaseRate * dt
		phaseIncrementAccel := twoPi * k * dt * dt
		kt := k * time
		// Envelope starts at exp(time * K) and increases incrementally
		// envelope(t+dt) = exp((t+dt) * K) = exp(t*K) * exp(dt*K) = envelope(t) * exp(dt*K)
		envelope := math32.Exp(time * K)
		envelopeIncrement := math32.Exp(dt * K) // Pre-compute increment factor

		for i := range n {
			value := g.amplitude * math32.Sin(phase) * envelope
			op(&dst[i], value)
			phase += phaseIncrementBase + twoPi*kt*dt + phaseIncrementAccel
			// Wrap phase to [0, 2π) to maintain precision
			if phase >= twoPi {
				phase -= twoPi
			} else if phase < 0 {
				phase += twoPi
			}
			kt += k * dt
			time += dt
			// Incremental envelope update
			envelope *= envelopeIncrement
		}
		g.time = time
	}

	return n
}

// Generate fills dst with pink chirp samples
func (g *PinkChirpGenerator) Generate(dst []float32) int {
	return g.GenerateWithOp(dst, false, func(d *float32, value float32) {
		*d = value
	})
}

// Add adds pink chirp samples to dst
func (g *PinkChirpGenerator) Add(dst []float32) int {
	return g.GenerateWithOp(dst, false, func(d *float32, value float32) {
		*d += value
	})
}

// PinkLogarithmicChirpGenerator generates a logarithmic chirp signal with constant energy per frequency
// (pink noise spectrum: -6dB per octave). The amplitude envelope decreases as frequency increases
// to maintain constant energy per frequency bin.
type PinkLogarithmicChirpGenerator struct {
	sampleRate float32
	f0         float32 // start frequency
	f1         float32 // end frequency
	amplitude  float32
	time       float32
	dt         float32
	phase      float32
}

// NewPinkLogarithmicChirpGenerator creates a new pink logarithmic chirp generator
// The generated chirp has constant energy per frequency (pink noise spectrum)
func NewPinkLogarithmicChirpGenerator(sampleRate, f0, f1, amplitude float32) PinkLogarithmicChirpGenerator {
	dt := 1.0 / sampleRate
	return PinkLogarithmicChirpGenerator{
		sampleRate: sampleRate,
		f0:         f0,
		f1:         f1,
		amplitude:  amplitude,
		time:       0,
		dt:         dt,
		phase:      0,
	}
}

// GenerateWithOp generates a pink logarithmic chirp with constant energy per frequency
func (g *PinkLogarithmicChirpGenerator) GenerateWithOp(dst []float32, reverse bool, op GeneratorAssignOp) int {
	n := len(dst)
	if n == 0 {
		return 0
	}

	// Bounds check elimination
	_ = dst[n-1]
	dst = dst[:n]

	T := float32(n) * g.dt
	twoPi := 2 * math32.Pi
	if T <= 0 || g.f0 <= 0 || g.f1 <= 0 {
		return n
	}

	// Compute pink noise envelope parameters
	fMin := g.f0
	fMax := g.f1
	if fMin > fMax {
		fMin, fMax = fMax, fMin
	}

	// Kend = 10^((-6 * log2(fMax/fMin)) / 20.0)
	logRatio := math32.Log(fMax / fMin)
	Kend := math32.Exp(-0.3 * math32.Log(10) / math32.Log(2) * logRatio)

	var K float32
	if T > 0 {
		K = math32.Log(Kend) / T
	}

	if reverse {
		// Reverse path: generate samples in reverse time order
		fStart := g.f1
		fBase := g.f1
		ratio := g.f0 / g.f1
		lnRatio := math32.Log(ratio)
		useConstantFreq := lnRatio == 0

		var omega float32
		if useConstantFreq {
			omega = 2 * math32.Pi * fStart / g.sampleRate
		}

		phase := float32(0)
		time := float32(0)
		// For reverse, envelope starts at Kend and decreases
		// envelope(t) = exp((T - t) * K)
		// envelope(t+dt) = exp((T - t - dt) * K) = envelope(t) * exp(-dt * K)
		envelope := Kend
		envelopeDecrement := math32.Exp(-g.dt * K) // Pre-compute decrement factor

		// Pre-compute incremental frequency update for logarithmic chirp
		var freq float32
		var ratioPowerIncrement float32
		var phaseIncrementBase float32
		if !useConstantFreq {
			ratioPowerIncrement = math32.Pow(ratio, g.dt/T) // Pre-compute once
			freq = fBase * math32.Pow(ratio, time/T)        // Initial frequency
			phaseIncrementBase = 2 * math32.Pi * g.dt       // Pre-compute 2π*dt
		}

		for i := range n {
			var value float32
			if useConstantFreq {
				value = g.amplitude * math32.Sin(phase) * envelope
				phase += omega * g.dt
			} else {
				// Incremental frequency update: freq *= ratio^(dt/T)
				phaseIncrement := phaseIncrementBase * freq
				phase += phaseIncrement
				value = g.amplitude * math32.Sin(phase) * envelope
				// Update frequency incrementally (avoid Pow in hot path)
				freq *= ratioPowerIncrement
			}
			// Wrap phase to [0, 2π) to maintain precision
			if phase >= twoPi {
				phase -= twoPi
			} else if phase < 0 {
				phase += twoPi
			}
			op(&dst[i], value)
			time += g.dt
			// Incremental envelope update
			envelope *= envelopeDecrement
		}
		g.time += float32(n) * g.dt
	} else {
		// Forward path: generate samples in normal time order
		fStart := g.f0
		fBase := g.f0
		ratio := g.f1 / g.f0
		lnRatio := math32.Log(ratio)
		useConstantFreq := lnRatio == 0

		var omega float32
		if useConstantFreq {
			omega = 2 * math32.Pi * fStart / g.sampleRate
		}

		phase := g.phase
		time := g.time
		// Envelope starts at exp(time * K) and increases incrementally
		// envelope(t+dt) = exp((t+dt) * K) = exp(t*K) * exp(dt*K) = envelope(t) * exp(dt*K)
		envelope := math32.Exp(time * K)
		envelopeIncrement := math32.Exp(g.dt * K) // Pre-compute increment factor

		// Pre-compute incremental frequency update for logarithmic chirp
		var freq float32
		var ratioPowerIncrement float32
		var phaseIncrementBase float32
		if !useConstantFreq {
			ratioPowerIncrement = math32.Pow(ratio, g.dt/T) // Pre-compute once
			freq = fBase * math32.Pow(ratio, time/T)        // Initial frequency
			phaseIncrementBase = 2 * math32.Pi * g.dt       // Pre-compute 2π*dt
		}

		for i := range n {
			var value float32
			if useConstantFreq {
				value = g.amplitude * math32.Sin(phase) * envelope
				phase += omega * g.dt
			} else {
				// Incremental frequency update: freq *= ratio^(dt/T)
				phaseIncrement := phaseIncrementBase * freq
				phase += phaseIncrement
				value = g.amplitude * math32.Sin(phase) * envelope
				// Update frequency incrementally (avoid Pow in hot path)
				freq *= ratioPowerIncrement
			}
			// Wrap phase to [0, 2π) to maintain precision
			if phase >= twoPi {
				phase -= twoPi
			} else if phase < 0 {
				phase += twoPi
			}
			op(&dst[i], value)
			time += g.dt
			// Incremental envelope update
			envelope *= envelopeIncrement
		}
		g.time = time
		g.phase = phase
	}

	return n
}

// Generate fills dst with pink logarithmic chirp samples
func (g *PinkLogarithmicChirpGenerator) Generate(dst []float32) int {
	return g.GenerateWithOp(dst, false, func(d *float32, value float32) {
		*d = value
	})
}

// Add adds pink logarithmic chirp samples to dst
func (g *PinkLogarithmicChirpGenerator) Add(dst []float32) int {
	return g.GenerateWithOp(dst, false, func(d *float32, value float32) {
		*d += value
	})
}
