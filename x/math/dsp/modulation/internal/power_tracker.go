package internal

import (
	"github.com/itohio/EasyRobot/x/types"
)

// PowerTracker tracks running signal power from samples with exponential decay.
// It tracks minimum, maximum, exponential moving average (EMA), and peak power of power.
// Peak power decays exponentially but immediately updates when new power exceeds it.
// Supports both real (NumberReal) and complex (NumberComplex) sample types.
type PowerTracker[T types.NumberReal | types.NumberComplex] struct {
	ema   float32 // Exponential moving average
	peak  float32 // Peak power with exponential decay
	alpha float32 // Decay factor (0-1), higher = faster decay
	min   float32
	max   float32
	count int
}

// NewPowerTracker creates a new power tracker with the specified decay factor.
// alpha: decay factor (0-1), typical values: 0.01-0.1
//   - 0.01 = slow decay (long memory)
//   - 0.1 = fast decay (short memory)
func NewPowerTracker[T types.NumberReal | types.NumberComplex](alpha float32) PowerTracker[T] {
	if alpha < 0 {
		alpha = 0
	}
	if alpha > 1 {
		alpha = 1
	}
	return PowerTracker[T]{
		alpha: alpha,
	}
}

// Update calculates power from samples and updates the tracker with exponential decay.
func (pt *PowerTracker[T]) Update(samples []T) {
	power := Power(samples)
	pt.update(power)
}

// UpdatePower updates the tracker with a power value directly.
// Useful when power is already calculated from a single sample.
func (pt *PowerTracker[T]) UpdatePower(power float32) {
	pt.update(power)
}

// update updates internal state with a power value.
func (pt *PowerTracker[T]) update(power float32) {
	if power <= 0 {
		return
	}
	pt.count++
	if pt.count == 1 {
		// First sample: initialize
		pt.ema = power
		pt.peak = power
		pt.min = power
		pt.max = power
	} else {
		// Update EMA with decay
		pt.ema = pt.ema*(1.0-pt.alpha) + power*pt.alpha
		// Update peak power: if new power > peak, update immediately; otherwise decay
		if power > pt.peak {
			pt.peak = power
		} else {
			pt.peak = pt.peak*(1.0-pt.alpha) + power*pt.alpha
		}
		// Update min/max
		if power < pt.min {
			pt.min = power
		}
		if power > pt.max {
			pt.max = power
		}
	}
}

// Power calculates the average power from a slice of samples.
// For real samples, power = sum(sample^2) / length
// For complex samples, power = sum(|sample|^2) / length
func Power[T types.NumberReal | types.NumberComplex](samples []T) float32 {
	if len(samples) == 0 {
		return 0
	}
	var sum float32
	for _, s := range samples {
		sum += powerValue(s)
	}
	return sum / float32(len(samples))
}

// powerValue calculates the power contribution of a single sample.
func powerValue[T types.NumberReal | types.NumberComplex](s T) float32 {
	// Check if it's a complex type
	if c, ok := any(s).(complex64); ok {
		real := float32(real(c))
		imag := float32(imag(c))
		return real*real + imag*imag
	}

	// Handle real types
	switch v := any(s).(type) {
	case float32:
		return v * v
	case float64:
		val := float32(v)
		return val * val
	case int:
		val := float32(v)
		return val * val
	case int8:
		val := float32(v)
		return val * val
	case int16:
		val := float32(v)
		return val * val
	case int32:
		val := float32(v)
		return val * val
	case int64:
		val := float32(v)
		return val * val
	}
	// Should not reach here if T is properly constrained
	return 0
}

// Power returns the accumulated power (EMA).
func (pt *PowerTracker[T]) Power() float32 {
	return pt.ema
}

// EMA returns the exponential moving average of power.
func (pt *PowerTracker[T]) EMA() float32 {
	return pt.ema
}

// Min returns the minimum power observed.
func (pt *PowerTracker[T]) Min() float32 {
	return pt.min
}

// Max returns the maximum power observed.
func (pt *PowerTracker[T]) Max() float32 {
	return pt.max
}

// Peak returns the peak power with exponential decay.
// Peak power immediately updates when new power exceeds it, otherwise decays exponentially.
func (pt *PowerTracker[T]) Peak() float32 {
	return pt.peak
}

// Count returns the number of samples processed.
func (pt *PowerTracker[T]) Count() int {
	return pt.count
}

// SetAlpha sets the decay factor for exponential smoothing.
func (pt *PowerTracker[T]) SetAlpha(alpha float32) {
	if alpha < 0 {
		alpha = 0
	}
	if alpha > 1 {
		alpha = 1
	}
	pt.alpha = alpha
}

// Reset clears all accumulated power measurements.
func (pt *PowerTracker[T]) Reset() {
	pt.ema = 0
	pt.peak = 0
	pt.min = 0
	pt.max = 0
	pt.count = 0
}
