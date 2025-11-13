package interpolation

import (
	"fmt"

	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/x/math/mat"
	"github.com/itohio/EasyRobot/x/math/vec"
)

// Sample represents a known data point with location and value.
type Sample struct {
	X, Y, V float32
}

// VariogramModel defines a function that computes the covariance/semivariance.
type VariogramModel func(distance float32) float32

// ExponentialVariogram creates an exponential variogram model.
// Range is the distance at which correlation drops to ~5%.
func ExponentialVariogram(rangeVal float32) VariogramModel {
	return func(d float32) float32 {
		if d == 0 {
			return 0
		}
		return 1 - exp(-3*d/rangeVal)
	}
}

// GaussianVariogram creates a Gaussian variogram model.
func GaussianVariogram(rangeVal float32) VariogramModel {
	return func(d float32) float32 {
		if d == 0 {
			return 0
		}
		return 1 - exp(-(3*d*d)/(rangeVal*rangeVal))
	}
}

// SphericalVariogram creates a spherical variogram model.
func SphericalVariogram(rangeVal float32) VariogramModel {
	return func(d float32) float32 {
		if d == 0 {
			return 0
		}
		if d >= rangeVal {
			return 1
		}
		return 1 - (1.5*d/rangeVal - 0.5*(d*d*d)/(rangeVal*rangeVal*rangeVal))
	}
}

// Kriging implements ordinary Kriging interpolation for 2D scattered data.
type Kriging struct {
	samples   []Sample
	variogram VariogramModel
	solution  []float32
	matrix    mat.Matrix
}

// NewKriging creates a new Kriging interpolator with the given variogram model.
func NewKriging(variogram VariogramModel) *Kriging {
	return &Kriging{
		samples:   make([]Sample, 0),
		variogram: variogram,
	}
}

// AddSample adds a known data point at location (x, y) with value v.
func (k *Kriging) AddSample(x, y, v float32) {
	k.samples = append(k.samples, Sample{X: x, Y: y, V: v})
	k.solution = nil // Invalidate cached solution
	k.matrix = nil
}

// Samples returns all registered samples.
func (k *Kriging) Samples() []Sample {
	return k.samples
}

// Fit prepares the Kriging system for interpolation.
// This computes weights that will be reused for all queries.
func (k *Kriging) Fit() error {
	n := len(k.samples)
	if n == 0 {
		return fmt.Errorf("no samples provided")
	}

	// Build covariance matrix: (n+1) x (n+1)
	// Last row and column are for Lagrange multiplier (sum of weights = 1)
	k.matrix = mat.New(n+1, n+1)

	// Build covariance between sample points
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			dist := distance(k.samples[i], k.samples[j])
			k.matrix[i][j] = k.variogram(dist)
		}
		// Lagrange constraint: sum of weights = 1
		k.matrix[i][n] = 1
		k.matrix[n][i] = 1
	}
	k.matrix[n][n] = 0

	// Solve for weights (will be computed during first interpolation)

	return nil
}

// Interpolate computes interpolated value at location (x, y).
func (k *Kriging) Interpolate(x, y float32) (float32, error) {
	n := len(k.samples)
	if n == 0 {
		return 0, fmt.Errorf("no samples provided")
	}

	// Single sample: return its value
	if n == 1 {
		return k.samples[0].V, nil
	}

	// Build right-hand side for this query point
	rhs := vec.New(n + 1)
	for i := 0; i < n; i++ {
		dist := distanceToPoint(k.samples[i].X, k.samples[i].Y, x, y)
		rhs[i] = k.variogram(dist)
	}
	rhs[n] = 1 // Lagrange constraint

	// Build or reuse covariance matrix
	if k.matrix == nil {
		if err := k.Fit(); err != nil {
			return 0, err
		}
	}

	// Solve linear system: matrix * weights = rhs
	weights := vec.New(n + 1)
	if err := solveLinear(k.matrix, rhs, weights); err != nil {
		return 0, err
	}

	// Compute interpolated value
	var value float32
	for i := 0; i < n; i++ {
		value += weights[i] * k.samples[i].V
	}

	return value, nil
}

// Distance computation helpers

func distance(s1, s2 Sample) float32 {
	dx := s2.X - s1.X
	dy := s2.Y - s1.Y
	return sqrt(dx*dx + dy*dy)
}

func distanceToPoint(x1, y1, x2, y2 float32) float32 {
	dx := x2 - x1
	dy := y2 - y1
	return sqrt(dx*dx + dy*dy)
}

// Mathematical helper functions

func exp(x float32) float32 {
	// For very large negative values, return 0
	if x < -20 {
		return 0
	}
	// Use math32.Exp for proper implementation
	return math32.Exp(x)
}

func sqrt(x float32) float32 {
	if x < 0 {
		return 0
	}
	return math32.Sqrt(x)
}

// solveLinear solves a system of linear equations using Gaussian elimination.
func solveLinear(A mat.Matrix, b, x vec.Vector) error {
	n := len(b)
	if len(A) != n || len(A[0]) != n {
		return fmt.Errorf("matrix size mismatch")
	}

	// Create augmented matrix
	augmented := make([]vec.Vector, n)
	for i := 0; i < n; i++ {
		augmented[i] = vec.New(n + 1)
		copy(augmented[i][:n], A[i])
		augmented[i][n] = b[i]
	}

	// Gaussian elimination with partial pivoting
	for i := 0; i < n; i++ {
		// Find pivot
		maxRow := i
		maxVal := abs(augmented[i][i])
		for k := i + 1; k < n; k++ {
			if abs(augmented[k][i]) > maxVal {
				maxRow = k
				maxVal = abs(augmented[k][i])
			}
		}

		if maxVal < 1e-6 {
			return fmt.Errorf("singular or ill-conditioned matrix")
		}

		// Swap rows
		if maxRow != i {
			augmented[i], augmented[maxRow] = augmented[maxRow], augmented[i]
		}

		// Eliminate
		pivot := augmented[i][i]
		for k := i + 1; k < n; k++ {
			factor := augmented[k][i] / pivot
			for j := i; j <= n; j++ {
				augmented[k][j] -= factor * augmented[i][j]
			}
		}
	}

	// Back substitution
	for i := n - 1; i >= 0; i-- {
		x[i] = augmented[i][n]
		for j := i + 1; j < n; j++ {
			x[i] -= augmented[i][j] * x[j]
		}
		x[i] /= augmented[i][i]
	}

	return nil
}

func abs(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}
