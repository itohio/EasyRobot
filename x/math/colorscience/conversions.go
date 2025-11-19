package colorscience

import (
	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/x/math"
)

// XYZToLAB converts CIE XYZ to CIE LAB color space.
func XYZToLAB(X, Y, Z float32, illuminant WhitePoint) (float32, float32, float32) {
	// Reference white point
	Xn, Yn, Zn := illuminant.XYZ()

	// Normalize by white point
	x := X / Xn
	y := Y / Yn
	z := Z / Zn

	// Apply f function
	f := func(t float32) float32 {
		const delta float32 = 6.0 / 29.0
		const delta3 float32 = delta * delta * delta
		const threeDelta2 float32 = 3.0 * delta * delta

		if t > delta3 {
			return math32.Pow(t, 1.0/3.0)
		}
		return t/(threeDelta2) + 4.0/29.0
	}

	fx := f(x)
	fy := f(y)
	fz := f(z)

	L := 116.0*fy - 16.0
	a := 500.0 * (fx - fy)
	b := 200.0 * (fy - fz)

	return L, a, b
}

// LABToXYZ converts CIE LAB to CIE XYZ color space.
func LABToXYZ(L, a, b float32, illuminant WhitePoint) (float32, float32, float32) {
	// Reference white point
	Xn, Yn, Zn := illuminant.XYZ()

	// Calculate fy
	fy := (L + 16.0) / 116.0
	fx := a/500.0 + fy
	fz := fy - b/200.0

	// Inverse f function
	finv := func(t float32) float32 {
		const delta float32 = 6.0 / 29.0
		const threeDelta2 float32 = 3.0 * delta * delta

		if t > delta {
			return t * t * t
		}
		return threeDelta2 * (t - 4.0/29.0)
	}

	x := finv(fx) * Xn
	y := finv(fy) * Yn
	z := finv(fz) * Zn

	return x, y, z
}

// XYZToRGB converts CIE XYZ (D65) to sRGB.
// out255: if true, return values 0-255; if false, return 0-1.
func XYZToRGB(X, Y, Z float32, out255 bool) (float32, float32, float32) {
	// sRGB transformation matrix (D65 white point)
	// Matrix from XYZ to linear RGB
	r := 3.2406*X + -1.5372*Y + -0.4986*Z
	g := -0.9689*X + 1.8758*Y + 0.0415*Z
	b := 0.0557*X + -0.2040*Y + 1.0570*Z

	// Gamma correction
	gamma := func(c float32) float32 {
		if c <= 0.0031308 {
			return 12.92 * c
		}
		return 1.055*math32.Pow(c, 1.0/2.4) - 0.055
	}

	r = gamma(r)
	g = gamma(g)
	b = gamma(b)

	// Clip to valid range
	r = math.Clamp(r, 0.0, 1.0)
	g = math.Clamp(g, 0.0, 1.0)
	b = math.Clamp(b, 0.0, 1.0)

	if out255 {
		r *= 255.0
		g *= 255.0
		b *= 255.0
	}

	return r, g, b
}

// RGBToXYZ converts sRGB (0-255 or 0-1) to CIE XYZ (D65).
func RGBToXYZ(r, g, b float32) (float32, float32, float32) {
	// Normalize to 0-1 if needed
	if r > 1.0 || g > 1.0 || b > 1.0 {
		r /= 255.0
		g /= 255.0
		b /= 255.0
	}

	// Clip to valid range
	r = math.Clamp(r, 0.0, 1.0)
	g = math.Clamp(g, 0.0, 1.0)
	b = math.Clamp(b, 0.0, 1.0)

	// Inverse gamma correction
	invGamma := func(c float32) float32 {
		if c <= 0.04045 {
			return c / 12.92
		}
		return math32.Pow((c+0.055)/1.055, 2.4)
	}

	r = invGamma(r)
	g = invGamma(g)
	b = invGamma(b)

	// Transformation matrix from linear RGB to XYZ (D65 white point)
	X := 0.4124*r + 0.3576*g + 0.1805*b
	Y := 0.2126*r + 0.7152*g + 0.0722*b
	Z := 0.0193*r + 0.1192*g + 0.9505*b

	return X, Y, Z
}

// RGBToLAB converts sRGB to CIE LAB by chaining RGBToXYZ → XYZToLAB.
func RGBToLAB(r, g, b float32, illuminant WhitePoint) (float32, float32, float32) {
	X, Y, Z := RGBToXYZ(r, g, b)
	return XYZToLAB(X, Y, Z, illuminant)
}

// LABToRGB converts CIE LAB to sRGB by chaining LABToXYZ → XYZToRGB.
func LABToRGB(L, a, b float32, illuminant WhitePoint, out255 bool) (float32, float32, float32) {
	X, Y, Z := LABToXYZ(L, a, b, illuminant)
	return XYZToRGB(X, Y, Z, out255)
}

