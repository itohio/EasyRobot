package interpolation

import (
	"testing"
)

func TestKriging_SingleSample(t *testing.T) {
	k := NewKriging(ExponentialVariogram(1.0))

	k.AddSample(0, 0, 42)

	value, err := k.Interpolate(0, 0)
	if err != nil {
		t.Fatalf("Interpolate failed: %v", err)
	}

	if value != 42 {
		t.Errorf("Expected 42, got %v", value)
	}
}

func TestKriging_LinearData(t *testing.T) {
	k := NewKriging(GaussianVariogram(10.0))

	// Create a linear function z = x + y
	k.AddSample(0, 0, 0)
	k.AddSample(1, 0, 1)
	k.AddSample(0, 1, 1)
	k.AddSample(2, 0, 2)
	k.AddSample(0, 2, 2)

	// Test interpolation at (1.5, 0.5) - should be close to 2
	value, err := k.Interpolate(1.5, 0.5)
	if err != nil {
		t.Fatalf("Interpolate failed: %v", err)
	}

	expected := float32(2.0)
	tolerance := float32(0.3)
	if abs(value-expected) > tolerance {
		t.Errorf("Expected ~%v, got %v (diff: %v)", expected, value, abs(value-expected))
	}
}

func TestKriging_ExactReproduction(t *testing.T) {
	k := NewKriging(SphericalVariogram(5.0))

	samples := []Sample{
		{X: 0, Y: 0, V: 1},
		{X: 1, Y: 0, V: 2},
		{X: 0, Y: 1, V: 3},
		{X: 1, Y: 1, V: 4},
	}

	for _, s := range samples {
		k.AddSample(s.X, s.Y, s.V)
	}

	// Interpolate at each sample point - should reproduce exactly
	for _, s := range samples {
		value, err := k.Interpolate(s.X, s.Y)
		if err != nil {
			t.Fatalf("Interpolate failed at (%v, %v): %v", s.X, s.Y, err)
		}

		tolerance := float32(0.01)
		if abs(value-s.V) > tolerance {
			t.Errorf("At (%v, %v): expected %v, got %v", s.X, s.Y, s.V, value)
		}
	}
}

func TestKriging_NoSamples(t *testing.T) {
	k := NewKriging(ExponentialVariogram(1.0))

	_, err := k.Interpolate(0, 0)
	if err == nil {
		t.Error("Expected error for no samples")
	}
}

func TestVariogramModels(t *testing.T) {
	tests := []struct {
		name  string
		model VariogramModel
	}{
		{"Exponential", ExponentialVariogram(2.0)},
		{"Gaussian", GaussianVariogram(2.0)},
		{"Spherical", SphericalVariogram(2.0)},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Check that model returns 0 at distance 0
			v := tt.model(0)
			if v != 0 {
				t.Errorf("At distance 0: got %v, want 0", v)
			}

			// Check that model increases with distance
			v1 := tt.model(1.0)
			v2 := tt.model(2.0)
			if v2 <= v1 {
				t.Errorf("Model should increase with distance: %v -> %v", v1, v2)
			}
		})
	}
}
