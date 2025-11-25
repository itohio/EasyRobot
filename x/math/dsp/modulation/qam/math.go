package qam

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

func binaryToGray(v int) int {
	return v ^ (v >> 1)
}

func grayToBinary(v, bits int) int {
	result := v
	for shift := 1; shift < bits; shift <<= 1 {
		result ^= result >> shift
	}
	return result
}
