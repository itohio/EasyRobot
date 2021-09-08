package interpolation

// Linear interpolation between a and b. t = 0..1
func Lerp(a, b, t float32) float32 {
	return a + (b-a)*t
}

// Linear interpolation from a in interval d. t = 0..1
func LerpD(a, d, t float32) float32 {
	return a + d*t
}
