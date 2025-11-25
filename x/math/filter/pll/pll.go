package pll

import "github.com/chewxy/math32"

// Loop implements a float32 PLL (phase + frequency tracking) suitable for coherent demodulators.
// It exposes helpers to rotate complex samples and to estimate residual CFO.
type Loop struct {
	kp float32
	ki float32

	symbolRate float32

	phase float32
	freq  float32

	sin float32
	cos float32
}

// New returns a loop configured with explicit proportional / integral gains.
func New(kp, ki, symbolRate float32) *Loop {
	return &Loop{
		kp:         kp,
		ki:         ki,
		symbolRate: symbolRate,
		cos:        1,
	}
}

// NewFromLoopBandwidth derives gains from loop bandwidth (Hz) and damping ratio (default 0.707).
func NewFromLoopBandwidth(loopBandwidth, symbolRate, dampingRatio float32) *Loop {
	if dampingRatio <= 0 {
		dampingRatio = 0.707
	}
	if loopBandwidth <= 0 {
		loopBandwidth = symbolRate * 0.05
	}

	omegaN := 2 * math32.Pi * loopBandwidth / symbolRate
	kp := 2 * dampingRatio * omegaN
	ki := omegaN * omegaN
	return New(kp, ki, symbolRate)
}

// SetLoopBandwidth updates the loop gains based on a desired bandwidth and damping ratio.
func (l *Loop) SetLoopBandwidth(loopBandwidth, dampingRatio float32) {
	if l == nil {
		return
	}
	if dampingRatio <= 0 {
		dampingRatio = 0.707
	}
	if loopBandwidth <= 0 {
		loopBandwidth = l.symbolRate * 0.05
	}
	omegaN := 2 * math32.Pi * loopBandwidth / l.symbolRate
	l.kp = 2 * dampingRatio * omegaN
	l.ki = omegaN * omegaN
}

// Update ingests a phase error (radians) and advances the loop. Returns the new phase estimate.
func (l *Loop) Update(phaseError float32) float32 {
	phaseError = wrapPi(phaseError)
	l.freq += l.ki * phaseError
	l.phase = wrapPi(l.phase + l.kp*phaseError + l.freq)
	l.sin, l.cos = math32.Sincos(l.phase)
	return l.phase
}

// Phase returns the current phase estimate.
func (l *Loop) Phase() float32 {
	return l.phase
}

// FrequencyOffset returns the estimated carrier offset in Hz.
func (l *Loop) FrequencyOffset() float32 {
	return l.freq * l.symbolRate / (2 * math32.Pi)
}

// Rotate rotates a complex sample (I,Q) by the negative loop phase.
func (l *Loop) Rotate(i, q float32) (float32, float32) {
	return i*l.cos + q*l.sin, -i*l.sin + q*l.cos
}

// RotateInterleaved rotates interleaved IQ samples in-place.
func (l *Loop) RotateInterleaved(samples []float32) {
	for i := 0; i+1 < len(samples); i += 2 {
		samples[i], samples[i+1] = l.Rotate(samples[i], samples[i+1])
	}
}

// Reset clears the loop state.
func (l *Loop) Reset() {
	l.phase = 0
	l.freq = 0
	l.sin = 0
	l.cos = 1
}

func wrapPi(x float32) float32 {
	switch {
	case x > math32.Pi:
		return x - 2*math32.Pi
	case x < -math32.Pi:
		return x + 2*math32.Pi
	default:
		return x
	}
}
