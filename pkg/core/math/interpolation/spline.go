// The below code was borrowed and ported from Paul Bourke blog:
// http://paulbourke.net/miscellaneous/interpolation/

package interpolation

func CubicSpline1D(p1, p2, p3, p4, t float32) float32 {
	mu2 := t * t

	a0 := p4 - p3 - p1 + p2
	a1 := p1 - p2 - a0
	a2 := p3 - p1
	a3 := p2

	return (a0*t+a1)*mu2 + a2*t + a3
}

func CubicCatmulRomSpline1D(p1, p2, p3, p4 float32, t float32) float32 {
	mu2 := t * t

	a0 := -0.5*p1 + 1.5*p2 - 1.5*p3 + 0.5*p4
	a1 := p1 - 2.5*p2 + 2*p3 - 0.5*p4
	a2 := -0.5*p1 + 0.5*p3
	a3 := p2

	return (a0*t+a1)*mu2 + a2*t + a3
}

// Tension: 0 is high, 0.5 normal, 1 is low
// Bias: 0 is even,
// 		positive is towards first segment,
// 		negative towards the other
func CubicHermiteSpline1D(
	p1, p2,
	p3, p4,
	t,
	tension,
	bias float32) float32 {

	tb := (1 + bias) * tension
	tb1 := (1 - bias) * tension
	mu2 := t * t
	mu3 := mu2 * t
	m0 := (p2 - p1) * tb
	m0 += (p3 - p2) * tb1
	m1 := (p3 - p2) * tb
	m1 += (p4 - p3) * tb1
	a0 := 2*mu3 - 3*mu2 + 1
	a2 := mu3 - mu2
	a1 := a2 + t - mu2 //mu3 - 2*mu2 + t
	a3 := 1 - a1       //-2*mu3 + 3*mu2

	return a0*p2 + a1*m0 + a2*m1 + a3*p3
}
