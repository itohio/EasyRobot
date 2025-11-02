package interpolation

import (
	"testing"
)

func TestRBF_SingleSample(t *testing.T) {
	r := NewRBF(GaussianKernel(), 1.0)

	r.AddSample(0, 0, 42)

	value, err := r.Interpolate(0, 0)
	if err != nil {
		t.Fatalf("Interpolate failed: %v", err)
	}

	if abs(value-42) > 0.01 {
		t.Errorf("Expected 42, got %v", value)
	}
}

func TestRBF_ExactReproduction(t *testing.T) {
	r := NewRBF(GaussianKernel(), 1.0)

	samples := []Sample{
		{X: 0, Y: 0, V: 1},
		{X: 1, Y: 0, V: 2},
		{X: 0, Y: 1, V: 3},
		{X: 1, Y: 1, V: 4},
		{X: 0.5, Y: 0.5, V: 5},
	}

	for _, s := range samples {
		r.AddSample(s.X, s.Y, s.V)
	}

	// Fit explicitly
	err := r.Fit()
	if err != nil {
		t.Fatalf("Fit failed: %v", err)
	}

	// Interpolate at each sample point - should reproduce exactly
	for _, s := range samples {
		value, err := r.Interpolate(s.X, s.Y)
		if err != nil {
			t.Fatalf("Interpolate failed at (%v, %v): %v", s.X, s.Y, err)
		}

		tolerance := float32(0.001)
		if abs(value-s.V) > tolerance {
			t.Errorf("At (%v, %v): expected %v, got %v", s.X, s.Y, s.V, value)
		}
	}
}

func TestRBF_Interpolation(t *testing.T) {
	r := NewRBF(InverseMultiquadricKernel(), 2.0)

	// Create known function z = x^2 + y^2
	r.AddSample(0, 0, 0)
	r.AddSample(1, 0, 1)
	r.AddSample(0, 1, 1)
	r.AddSample(1, 1, 2)
	r.AddSample(2, 0, 4)

	// Test at (1.5, 0.5) - should be close to 1.5^2 + 0.5^2 = 2.5
	value, err := r.Interpolate(1.5, 0.5)
	if err != nil {
		t.Fatalf("Interpolate failed: %v", err)
	}

	expected := float32(2.5)
	tolerance := float32(0.2)
	if abs(value-expected) > tolerance {
		t.Errorf("Expected ~%v, got %v", expected, value)
	}
}

func TestRBFKernels(t *testing.T) {
	tests := []struct {
		name   string
		kernel RBFKernel
		radius float32
	}{
		{"Gaussian", GaussianKernel(), 1.0},
		{"Multiquadric", MultiquadricKernel(), 1.0},
		{"InverseMultiquadric", InverseMultiquadricKernel(), 1.0},
		{"ThinPlateSpline", ThinPlateSplineKernel(), 1.0},
		{"Biharmonic", BiharmonicKernel(), 1.0},
		{"Triharmonic", TriharmonicKernel(), 1.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Check kernel at center (radius 0)
			v := tt.kernel(0)
			// Most kernels should return specific values at 0
			if v < -1e10 {
				t.Errorf("Kernel at 0: got invalid value %v", v)
			}

			// Check that kernel is defined for reasonable radii
			v1 := tt.kernel(0.5)
			v2 := tt.kernel(1.0)
			v3 := tt.kernel(2.0)

			// Values should be finite
			if v1 != v1 || v2 != v2 || v3 != v3 {
				t.Errorf("Kernel returned NaN at some radius")
			}
		})
	}
}

func TestRBF_NoSamples(t *testing.T) {
	r := NewRBF(GaussianKernel(), 1.0)

	_, err := r.Interpolate(0, 0)
	if err == nil {
		t.Error("Expected error for no samples")
	}
}

func TestRBF_ImplicitFit(t *testing.T) {
	r := NewRBF(GaussianKernel(), 1.0)

	r.AddSample(0, 0, 1)
	r.AddSample(1, 0, 2)

	// First interpolate should trigger automatic fit
	_, err := r.Interpolate(0.5, 0)
	if err != nil {
		t.Fatalf("Automatic fit failed: %v", err)
	}

	// Should now be fitted
	if !r.fitted {
		t.Error("Expected implicit fit to set fitted flag")
	}
}
