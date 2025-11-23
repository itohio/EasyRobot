package pll

import (
	"math"
)

// Filter implements a Phase-Locked Loop for carrier phase and frequency tracking
// Used by M-FSK, M-PSK, QAM, and M-ASK demodulators
type Filter struct {
	// PI controller gains
	kp float32 // Proportional gain
	ki float32 // Integral gain

	// State (no allocations - all pre-allocated)
	phase      float32 // Current phase estimate (radians)
	freqOffset float32 // Frequency offset accumulator (radians/symbol)

	// Symbol rate for frequency conversion (Hz)
	symbolRate float32
}

// New creates a new PLL filter with specified gains
// kp: proportional gain (typically 0.05-0.2)
// ki: integral gain (typically 0.005-0.02)
// symbolRate: symbol rate in Hz (for frequency offset conversion)
func New(kp, ki, symbolRate float32) *Filter {
	return &Filter{
		kp:         kp,
		ki:         ki,
		symbolRate: symbolRate,
	}
}

// NewFromLoopBandwidth creates a PLL filter with gains computed from loop bandwidth
// loopBandwidth: desired loop bandwidth in Hz (typically 0.01-0.1 * symbol_rate)
// symbolRate: symbol rate in Hz
// dampingRatio: damping ratio (typically 0.707 for critical damping)
func NewFromLoopBandwidth(loopBandwidth, symbolRate float32, dampingRatio float32) *Filter {
	// Natural frequency: ωn = 2π * B_loop / R_sym
	omegaN := 2 * math.Pi * loopBandwidth / symbolRate

	// PI controller gains from control theory
	kp := 2 * dampingRatio * omegaN
	ki := omegaN * omegaN

	return &Filter{
		kp:         float32(kp),
		ki:         float32(ki),
		symbolRate: symbolRate,
	}
}

// Update updates the PLL with a phase error (in radians)
// phaseError: phase error from correlator output, typically arg(Z) where Z is complex correlator output
// The phase error is the difference between the actual phase and the estimated phase
func (p *Filter) Update(phaseError float32) {
	// Normalize phase error to [-π, π] range
	// This prevents large phase jumps from causing instability
	if phaseError > math.Pi {
		phaseError -= 2 * math.Pi
	} else if phaseError < -math.Pi {
		phaseError += 2 * math.Pi
	}

	// Update frequency offset (integral term)
	// This accumulates the phase error to track frequency offset
	p.freqOffset += p.ki * phaseError

	// Update phase estimate (proportional + integral)
	// Proportional term: immediate correction
	// Integral term: accumulated frequency offset
	p.phase += p.kp*phaseError + p.freqOffset

	// Normalize phase to [-π, π] range
	// This prevents phase wrapping issues
	if p.phase > math.Pi {
		p.phase -= 2 * math.Pi
	} else if p.phase < -math.Pi {
		p.phase += 2 * math.Pi
	}
}

// Phase returns the current phase estimate (in radians)
// Use this to correct samples: corrected = sample * e^(-j*phase)
func (p *Filter) Phase() float32 {
	return p.phase
}

// FrequencyOffset returns the current frequency offset estimate (in Hz)
// This is the estimated CFO (Carrier Frequency Offset)
func (p *Filter) FrequencyOffset() float32 {
	// Convert from radians/symbol to Hz
	// freqOffset is in radians/symbol, so Hz = (radians/symbol) * (symbols/sec) / (2π)
	return p.freqOffset * p.symbolRate / (2 * math.Pi)
}

// SetGains sets the proportional and integral gains
func (p *Filter) SetGains(kp, ki float32) {
	p.kp = kp
	p.ki = ki
}

// Reset resets the PLL state
func (p *Filter) Reset() {
	p.phase = 0
	p.freqOffset = 0
}

// RotatePhase rotates a complex sample by the negative of the current phase estimate
// This corrects for phase offset: corrected = sample * e^(-j*phase)
// Returns I and Q components
func (p *Filter) RotatePhase(i, q float32) (correctedI, correctedQ float32) {
	phase := -p.phase
	cosPhase := float32(math.Cos(float64(phase)))
	sinPhase := float32(math.Sin(float64(phase)))

	// Rotation: [I'] = [cos -sin] [I]
	//           [Q']   [sin  cos] [Q]
	correctedI = i*cosPhase - q*sinPhase
	correctedQ = i*sinPhase + q*cosPhase

	return correctedI, correctedQ
}

// RotatePhaseBatch rotates a batch of complex samples
// This is more efficient than calling RotatePhase for each sample
// samples: interleaved I/Q samples [I0, Q0, I1, Q1, ...]
func (p *Filter) RotatePhaseBatch(samples []float32) {
	phase := -p.phase
	cosPhase := float32(math.Cos(float64(phase)))
	sinPhase := float32(math.Sin(float64(phase)))

	// Rotate each I/Q pair
	for i := 0; i < len(samples)-1; i += 2 {
		iVal := samples[i]
		qVal := samples[i+1]

		samples[i] = iVal*cosPhase - qVal*sinPhase
		samples[i+1] = iVal*sinPhase + qVal*cosPhase
	}
}

