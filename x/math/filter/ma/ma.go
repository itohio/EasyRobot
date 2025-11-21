package ma

// Moving Average and Exponential Moving Average filters

import (
	"github.com/chewxy/math32"
)

// Filter implements a simple moving average filter.
type Filter struct {
	buffer []float32
	index  int
	sum    float32
	size   int
	count  int // number of samples added so far
}

// New creates a new moving average filter with the specified window size.
func New(size int) *Filter {
	if size <= 0 {
		panic("moving average window size must be > 0")
	}

	return &Filter{
		buffer: make([]float32, size),
		size:   size,
	}
}

// Reset resets the moving average filter.
func (ma *Filter) Reset() {
	ma.index = 0
	ma.sum = 0
	ma.count = 0
	for i := range ma.buffer {
		ma.buffer[i] = 0
	}
}

// Process adds a new sample and returns the current moving average.
func (ma *Filter) Process(sample float32) float32 {
	// Remove the oldest sample from sum
	ma.sum -= ma.buffer[ma.index]

	// Add new sample to buffer and sum
	ma.buffer[ma.index] = sample
	ma.sum += sample

	// Update index
	ma.index = (ma.index + 1) % ma.size

	// Update count (for partial windows)
	if ma.count < ma.size {
		ma.count++
	}

	// Return average
	return ma.sum / float32(ma.count)
}

// Exponential implements an exponential moving average filter.
type Exponential struct {
	alpha       float32 // smoothing factor (0 < alpha <= 1)
	value       float32 // current EMA value
	initialized bool    // whether we have processed at least one sample
}

// NewEMA creates a new EMA filter with the specified smoothing factor.
// Alpha should be between 0 and 1, where smaller values give more smoothing.
func NewEMA(alpha float32) *Exponential {
	if alpha <= 0 || alpha > 1 {
		panic("EMA alpha must be in range (0, 1]")
	}

	return &Exponential{
		alpha: alpha,
	}
}

// Reset resets the EMA filter.
func (ema *Exponential) Reset() {
	ema.value = 0
	ema.initialized = false
}

// Process processes a new sample and returns the updated EMA.
func (ema *Exponential) Process(sample float32) float32 {
	if !ema.initialized {
		// First sample
		ema.value = sample
		ema.initialized = true
	} else {
		// EMA formula: value = alpha * sample + (1 - alpha) * previous_value
		ema.value = ema.alpha*sample + (1-ema.alpha)*ema.value
	}

	return ema.value
}

// SetAlpha changes the smoothing factor.
func (ema *Exponential) SetAlpha(alpha float32) {
	if alpha <= 0 || alpha > 1 {
		panic("EMA alpha must be in range (0, 1]")
	}
	ema.alpha = alpha
}

// GetAlpha returns the current smoothing factor.
func (ema *Exponential) GetAlpha() float32 {
	return ema.alpha
}

// AlphaFromHalfLife calculates alpha from a desired half-life.
// Half-life is the number of samples for the influence to decay by half.
func AlphaFromHalfLife(halfLife float32) float32 {
	if halfLife <= 0 {
		panic("half-life must be > 0")
	}
	return 1.0 - math32.Pow(0.5, 1.0/halfLife)
}

// AlphaFromTimeConstant calculates alpha from a time constant.
// Time constant is the number of samples to reach ~63% of the final value.
func AlphaFromTimeConstant(timeConstant float32) float32 {
	if timeConstant <= 0 {
		panic("time constant must be > 0")
	}
	return 1.0 - math32.Exp(-1.0/timeConstant)
}
