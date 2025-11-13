package interpolation

import (
	"github.com/itohio/EasyRobot/pkg/core/math/vec"
)

// LinearUpsample upsamples a vector from size N to size M using linear interpolation.
// dst must be pre-allocated with size M. Returns dst on success, nil on invalid input.
func LinearUpsample(src vec.Vector, dst vec.Vector) vec.Vector {
	if len(src) == 0 || len(dst) == 0 {
		return nil
	}

	srcSize := len(src)
	dstSize := len(dst)

	// Handle edge case: single input value
	if srcSize == 1 {
		for i := range dst {
			dst[i] = src[0]
		}
		return dst
	}

	// Handle edge case: same size
	if srcSize == dstSize {
		copy(dst, src)
		return dst
	}

	// Linear upsampling with scale factor
	scale := float32(srcSize-1) / float32(dstSize-1)

	for i := range dst {
		pos := float32(i) * scale
		idx := int(pos)
		frac := pos - float32(idx)

		if idx >= srcSize-1 {
			dst[i] = src[srcSize-1]
		} else {
			dst[i] = Lerp(src[idx], src[idx+1], frac)
		}
	}

	return dst
}

// CubicUpsample upsamples a vector from size N to size M using cubic spline interpolation.
// dst must be pre-allocated with size M. Returns dst on success, nil on invalid input.
func CubicUpsample(src vec.Vector, dst vec.Vector) vec.Vector {
	if len(src) == 0 || len(dst) == 0 {
		return nil
	}

	srcSize := len(src)
	dstSize := len(dst)

	// Handle edge case: single input value
	if srcSize == 1 {
		for i := range dst {
			dst[i] = src[0]
		}
		return dst
	}

	// Handle edge case: same size
	if srcSize == dstSize {
		copy(dst, src)
		return dst
	}

	// For cubic interpolation, we need at least 2 points
	if srcSize == 2 {
		return LinearUpsample(src, dst)
	}

	// Compute natural cubic spline coefficients
	coeffs := computeCubicSpline(src)
	if coeffs == nil {
		return nil
	}

	// Perform cubic upsampling
	scale := float32(srcSize-1) / float32(dstSize-1)

	for i := range dst {
		pos := float32(i) * scale
		idx := int(pos)
		frac := pos - float32(idx)

		if idx >= srcSize-1 {
			dst[i] = src[srcSize-1]
		} else {
			dst[i] = evaluateCubicSpline(src[idx], coeffs[idx*3:], frac)
		}
	}

	return dst
}

// computeCubicSpline computes natural cubic spline coefficients for a given vector.
// Returns a slice of coefficients [a, b, c] for each segment [i, i+1].
// Each segment has 3 coefficients: a*t^3 + b*t^2 + c*t + y0
func computeCubicSpline(y vec.Vector) []float32 {
	n := len(y)
	if n < 2 {
		return nil
	}

	// For 2 points, just linear interpolation
	if n == 2 {
		coeffs := make([]float32, 3)
		coeffs[0] = 0           // a
		coeffs[1] = 0           // b
		coeffs[2] = y[1] - y[0] // c
		return coeffs
	}

	// Use Catmull-Rom spline for simplicity and good results
	// This is a cubic interpolation that passes through all points
	coeffs := make([]float32, 3*(n-1))

	for i := 0; i < n-1; i++ {
		// Get control points with boundary handling
		var p0, p1, p2, p3 float32
		if i == 0 {
			p0 = y[0]
			p1 = y[0]
			p2 = y[i+1]
			if i+2 < n {
				p3 = y[i+2]
			} else {
				p3 = y[i+1]
			}
		} else if i == n-2 {
			p0 = y[i-1]
			p1 = y[i]
			p2 = y[i+1]
			p3 = y[i+1]
		} else {
			p0 = y[i-1]
			p1 = y[i]
			p2 = y[i+1]
			p3 = y[i+2]
		}

		// Catmull-Rom coefficients
		// S(t) = (-0.5*p0 + 1.5*p1 - 1.5*p2 + 0.5*p3)*t^3 +
		//        (p0 - 2.5*p1 + 2*p2 - 0.5*p3)*t^2 +
		//        (-0.5*p0 + 0.5*p2)*t + p1
		a := -0.5*p0 + 1.5*p1 - 1.5*p2 + 0.5*p3
		b := p0 - 2.5*p1 + 2*p2 - 0.5*p3
		c := -0.5*p0 + 0.5*p2
		// d = p1, stored as y0 in evaluateCubicSpline

		coeffs[i*3+0] = a
		coeffs[i*3+1] = b
		coeffs[i*3+2] = c
	}

	return coeffs
}

// evaluateCubicSpline evaluates a cubic spline segment at fractional position t [0,1].
func evaluateCubicSpline(y0 float32, coeffs []float32, t float32) float32 {
	if len(coeffs) < 3 {
		return y0
	}

	a := coeffs[0]
	b := coeffs[1]
	c := coeffs[2]

	// S(t) = a*t^3 + b*t^2 + c*t + y0
	t2 := t * t
	t3 := t2 * t

	return a*t3 + b*t2 + c*t + y0
}
