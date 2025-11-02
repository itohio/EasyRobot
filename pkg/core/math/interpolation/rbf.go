package interpolation

import (
	"fmt"

	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/pkg/core/math/mat"
	"github.com/itohio/EasyRobot/pkg/core/math/vec"
)

// RBFKernel defines a radial basis function kernel.
type RBFKernel func(r float32) float32

// RBF implements radial basis function interpolation for 2D scattered data.
type RBF struct {
	samples      []Sample
	kernel       RBFKernel
	epsilon      float32
	coefficients vec.Vector
	fitted       bool
}

// NewRBF creates a new RBF interpolator with the given kernel and shape parameter.
func NewRBF(kernel RBFKernel, epsilon float32) *RBF {
	return &RBF{
		samples: make([]Sample, 0),
		kernel:  kernel,
		epsilon: epsilon,
		fitted:  false,
	}
}

// AddSample adds a known data point at location (x, y) with value v.
func (r *RBF) AddSample(x, y, v float32) {
	r.samples = append(r.samples, Sample{X: x, Y: y, V: v})
	r.fitted = false // Invalidate fitted coefficients
}

// Samples returns all registered samples.
func (r *RBF) Samples() []Sample {
	return r.samples
}

// Fit prepares the RBF system for interpolation by computing coefficients.
func (r *RBF) Fit() error {
	n := len(r.samples)
	if n == 0 {
		return fmt.Errorf("no samples provided")
	}

	// Single sample: trivial case
	if n == 1 {
		r.coefficients = vec.New(1)
		r.coefficients[0] = r.samples[0].V
		r.fitted = true
		return nil
	}

	// Build interpolation matrix A where A[i][j] = kernel(distance(i, j))
	A := mat.New(n, n)
	if A == nil {
		return fmt.Errorf("failed to allocate matrix")
	}

	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			dist := distance(r.samples[i], r.samples[j])
			scaledDist := r.epsilon * dist
			A[i][j] = r.kernel(scaledDist)
		}
	}

	// Right-hand side is the sample values
	b := vec.New(n)
	for i := 0; i < n; i++ {
		b[i] = r.samples[i].V
	}

	// Solve for coefficients: A * coefficients = b
	r.coefficients = vec.New(n)
	if err := solveLinear(A, b, r.coefficients); err != nil {
		return fmt.Errorf("failed to solve RBF system: %w", err)
	}

	r.fitted = true
	return nil
}

// Interpolate computes interpolated value at location (x, y).
func (r *RBF) Interpolate(x, y float32) (float32, error) {
	n := len(r.samples)
	if n == 0 {
		return 0, fmt.Errorf("no samples provided")
	}

	// Fit if not already fitted
	if !r.fitted {
		if err := r.Fit(); err != nil {
			return 0, err
		}
	}

	// Single sample: return its value
	if n == 1 {
		return r.samples[0].V, nil
	}

	// Evaluate RBF: sum of coefficients * kernel(distance)
	var value float32
	for i := 0; i < n; i++ {
		dist := distanceToPoint(r.samples[i].X, r.samples[i].Y, x, y)
		scaledDist := r.epsilon * dist
		value += r.coefficients[i] * r.kernel(scaledDist)
	}

	return value, nil
}

// Common RBF kernels

// GaussianKernel creates a Gaussian (exponential) RBF kernel.
func GaussianKernel() RBFKernel {
	return func(r float32) float32 {
		return exp(-r * r)
	}
}

// MultiquadricKernel creates a multiquadric RBF kernel.
func MultiquadricKernel() RBFKernel {
	return func(r float32) float32 {
		return sqrt(1 + r*r)
	}
}

// InverseMultiquadricKernel creates an inverse multiquadric RBF kernel.
func InverseMultiquadricKernel() RBFKernel {
	return func(r float32) float32 {
		return 1.0 / sqrt(1+r*r)
	}
}

// ThinPlateSplineKernel creates a thin plate spline RBF kernel.
func ThinPlateSplineKernel() RBFKernel {
	return func(r float32) float32 {
		if r < 1e-6 {
			return 0
		}
		return r * r * log(r)
	}
}

// BiharmonicKernel creates a biharmonic RBF kernel.
func BiharmonicKernel() RBFKernel {
	return func(r float32) float32 {
		if r < 1e-6 {
			return 0
		}
		return r
	}
}

// TriharmonicKernel creates a triharmonic RBF kernel.
func TriharmonicKernel() RBFKernel {
	return func(r float32) float32 {
		if r < 1e-6 {
			return 0
		}
		return r * r * r
	}
}

// Mathematical helper functions

func log(x float32) float32 {
	if x <= 0 {
		return -1e10
	}
	return math32.Log(x)
}
