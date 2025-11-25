package mpsk

import "github.com/chewxy/math32"

const twoPi = 2 * math32.Pi

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

func wrapTwoPi(x float32) float32 {
	switch {
	case x >= twoPi:
		return x - twoPi
	case x < 0:
		return x + twoPi
	default:
		return x
	}
}
