package internal

import "github.com/chewxy/math32"

// wrapPi wraps phase to [-π, π) range.
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

// wrapTwoPi wraps phase to [0, 2π) range.
func wrapTwoPi(x float32) float32 {
	switch {
	case x >= 2*math32.Pi:
		return x - 2*math32.Pi
	case x < 0:
		return x + 2*math32.Pi
	default:
		return x
	}
}

