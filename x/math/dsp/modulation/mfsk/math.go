package mfsk

import "github.com/chewxy/math32"

const twoPi = 2 * math32.Pi

func mathSin(x float32) float32 {
	return math32.Sin(x)
}

func wrapPi(x float32) float32 {
	switch {
	case x > math32.Pi:
		return x - twoPi
	case x < -math32.Pi:
		return x + twoPi
	default:
		return x
	}
}
