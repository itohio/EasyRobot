package interpolation

import "math"

func Cosine1D(a, b, t float32) float32 {
	mu2 := (1 - float32(math.Cos(float64(t)*math.Pi))) / 2
	return (a*(1-mu2) + b*mu2)
}
