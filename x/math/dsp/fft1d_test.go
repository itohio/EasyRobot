package dsp

import (
	"fmt"
	"math"
	"testing"

	"github.com/itohio/EasyRobot/x/math/vec"
)

func TestFFT1D_ForwardImpulse(t *testing.T) {
	fft := NewFFT1D(4)
	impulse := vec.NewFrom(1, 0, 0, 0)
	spectrum := vec.New(8)

	fft.Forward(spectrum, impulse)

	if abs := math.Abs(float64(spectrum[0] - 1.0)); abs > 1e-6 {
		t.Fatalf("expected DC component 1, got %f", spectrum[0])
	}
	if abs := math.Abs(float64(spectrum[4])); abs > 1e-6 {
		t.Fatalf("expected imaginary part 0, got %f", spectrum[4])
	}
}

func TestFFT1D_ForwardBackward(t *testing.T) {
	cases := []vec.Vector{
		vec.NewFrom(1, 2, 3, 4),
		vec.NewFrom(1, 2, 3, 4, 5, 6),
	}

	for _, data := range cases {
		t.Run(
			fmtShape(len(data)),
			func(t *testing.T) {
				fft := NewFFT1D(len(data))
				spectrum := vec.New(4 * len(data))
				reconstructed := vec.New(len(data))

				fft.Forward(spectrum, data)
				fft.Backward(reconstructed, spectrum)

				assertCloseVector(t, reconstructed, []float32(data), 1e-4)
			})
	}
}

func TestFFT1D_ConvolveMatchesNaive(t *testing.T) {
	signal := vec.NewFrom(1, 2, 3, 4, 5)
	kernel := vec.NewFrom(0.25, -0.5, 1)
	resultLen := len(signal) + len(kernel) - 1
	result := vec.New(resultLen)

	fft := NewFFT1D(resultLen)
	fft.Convolve(result, signal, kernel)

	expected := naiveConvolve(signal, kernel)
	assertCloseVector(t, result, expected, 1e-3)
}

func TestFFT1D_Convolve_LengthMismatch(t *testing.T) {
	fft := NewFFT1D(4)
	signal := vec.NewFrom(1, 2, 3, 4)
	kernel := vec.NewFrom(1, 1, 1, 1)
	result := vec.New(len(signal) + len(kernel) - 1)

	defer func() {
		if recover() == nil {
			t.Fatal("expected panic for FFT length mismatch")
		}
	}()
	fft.Convolve(result, signal, kernel)
}

func TestFFT1D_CorrelateMatchesNaive(t *testing.T) {
	signal := vec.NewFrom(1, 3, -2, 4)
	reference := vec.NewFrom(2, -1, 0)
	resultLen := len(signal) + len(reference) - 1
	result := vec.New(resultLen)

	fft := NewFFT1D(resultLen)
	fft.Correlate(result, signal, reference)

	expected := naiveCorrelate(signal, reference)
	assertCloseVector(t, result, expected, 1e-3)
}

func TestFFT1D_Correlate_LengthMismatch(t *testing.T) {
	fft := NewFFT1D(4)
	signal := vec.NewFrom(1, 2, 3, 4)
	reference := vec.NewFrom(1, 2, 3, 4)
	result := vec.New(len(signal) + len(reference) - 1)

	defer func() {
		if recover() == nil {
			t.Fatal("expected panic for FFT length mismatch")
		}
	}()
	fft.Correlate(result, signal, reference)
}

func TestFFT1D_EdgeCases(t *testing.T) {
	fft := NewFFT1D(1)
	signal := vec.NewFrom(5.0)
	spectrum := vec.New(2)
	fft.Forward(spectrum, signal)
	if math.Abs(float64(spectrum[0]-5.0)) > 1e-6 {
		t.Fatalf("expected FFT of single element to be 5, got %f", spectrum[0])
	}

	fft = NewFFT1D(3)
	signal = vec.NewFrom(1, 2, 3)
	spectrum = vec.New(12)
	fft.Forward(spectrum, signal)
	hasNonZero := false
	for _, v := range spectrum {
		if math.Abs(float64(v)) > 1e-6 {
			hasNonZero = true
			break
		}
	}
	if !hasNonZero {
		t.Fatal("expected FFT output to contain non-zero values")
	}
}

func fmtShape(n int) string {
	return fmt.Sprintf("len_%d", n)
}

func naiveConvolve(a, b vec.Vector) []float32 {
	result := make([]float32, len(a)+len(b)-1)
	for i := range a {
		for j := range b {
			result[i+j] += a[i] * b[j]
		}
	}
	return result
}

func naiveCorrelate(signal, reference vec.Vector) []float32 {
	resultLen := len(signal) + len(reference) - 1
	result := make([]float32, resultLen)
	for lag := 0; lag < resultLen; lag++ {
		sum := float32(0)
		for n := 0; n < len(reference); n++ {
			m := n + lag
			if m >= 0 && m < len(signal) {
				sum += signal[m] * reference[n]
			}
		}
		result[lag] = sum
	}
	return result
}

func assertCloseVector(t *testing.T, got vec.Vector, want []float32, tol float64) {
	t.Helper()
	if len(got) < len(want) {
		t.Fatalf("destination shorter than expected: got %d want %d", len(got), len(want))
	}
	for i := range want {
		if diff := math.Abs(float64(got[i] - want[i])); diff > tol {
			t.Fatalf("value mismatch at %d: got %f want %f (diff %f)", i, got[i], want[i], diff)
		}
	}
}
