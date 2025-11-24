package dsp

import (
	"math"
	"testing"
)

func assertFloatSliceClose(t *testing.T, got, want []float32, tol float64) {
	t.Helper()
	if len(got) < len(want) {
		t.Fatalf("slice too short: got %d want %d", len(got), len(want))
	}
	for i := range want {
		if diff := math.Abs(float64(got[i] - want[i])); diff > tol {
			t.Fatalf("value mismatch at %d: got %f want %f (diff %f)", i, got[i], want[i], diff)
		}
	}
}

func assertComplexSliceClose(t *testing.T, got, want []complex64, tol float64) {
	t.Helper()
	if len(got) < len(want) {
		t.Fatalf("slice too short: got %d want %d", len(got), len(want))
	}
	for i := range want {
		diff := cmplxAbsComplex(got[i] - want[i])
		if diff > tol {
			t.Fatalf("value mismatch at %d: got %v want %v (diff %f)", i, got[i], want[i], diff)
		}
	}
}

func cmplxAbsComplex(v complex64) float64 {
	return math.Hypot(float64(real(v)), float64(imag(v)))
}
