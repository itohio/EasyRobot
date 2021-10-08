package math

import "github.com/chewxy/math32"

//go:generate go run ../../../cmd/codegen -i gen/vec.tpl -c gen/vec.json
//go:generate go run ../../../cmd/codegen -i gen/mat.tpl -c gen/mat.json

const magic32 = 0x5F375A86

func SQR(a float32) float32 {
	return a * a
}

func Clamp(a, min, max float32) float32 {
	switch {
	case a > max:
		return max
	case a < min:
		return min
	default:
		return a
	}
}

// (a^2+b^2)^(1/2) without Owerflow
func Pytag(a, b float32) float32 {
	absa := math32.Abs(a)
	absb := math32.Abs(b)
	if absa > absb {
		return absa * math32.Sqrt(1.0+SQR(absb/absa))
	} else {
		if absb > 0 {
			return absb * math32.Sqrt(1.0+SQR(absa/absb))
		}
		return 0
	}
}

// Quadratic equation solver
func Quad(a, b, c, eps float32) (float32, float32) {
	if a == 0 {
		if c == 0 {
			return 0, 0
		}
		return b / c, b / c
	}

	if b == 0 {
		t := -c / a
		if t <= 0 {
			return 0, 0
		}
		t = math32.Sqrt(t)
		return t, t
	}

	r := -b
	z := b*b - 4*a*c
	if z < eps {
		z = 0
	} else if z < 0 {
		return 0, 0
	}
	z = math32.Sqrt(z)
	return (r + z) / (2 * a), (r - z) / (2 * a)
}

// https://medium.com/@adrien.za/fast-inverse-square-root-in-go-and-javascript-for-fun-6b891e74e5a8
func FastISqrt(x float32) float32 {
	// If n is negative return NaN
	// if x < 0 {
	// 	return float32(math.NaN())
	// }

	// n2 and th are for one iteration of Newton's method later
	n2, th := x*0.5, float32(1.5)
	// Use math.Float32bits to represent the float32, n, as
	// an uint32 without modification.
	b := math32.Float32bits(x)
	// Use the new uint32 view of the float32 to shift the bits
	// of the float32 1 to the right, chopping off 1 bit from
	// the fraction part of the float32.
	b = magic32 - (b >> 1)
	// Use math.Float32frombits to convert the uint32 bits back
	// into their float32 representation, again no actual change
	// in the bits, just a change in how we treat them in memory.
	// f is now our answer of 1 / sqrt(n)
	f := math32.Float32frombits(b)
	// Perform one iteration of Newton's method on f to improve
	// accuracy
	f *= th - (n2 * f * f)

	// And return our fast inverse square root result
	return f
}
