package dsp

import (
	"math"
	"testing"

	"github.com/itohio/EasyRobot/x/math/mat"
	"github.com/itohio/EasyRobot/x/math/vec"
)

func TestMeasurements_Measure1D(t *testing.T) {
	tests := []struct {
		name     string
		signal   []float32
		expected MeasurementResult
		tol      float64
	}{
		{
			name:   "empty signal",
			signal: []float32{},
			expected: MeasurementResult{},
			tol:    1e-6,
		},
		{
			name:   "constant signal",
			signal: []float32{2.0, 2.0, 2.0, 2.0},
			expected: MeasurementResult{
				RMS:  2.0,
				Peak: 2.0,
				Mean: 2.0,
			},
			tol: 1e-6,
		},
		{
			name:   "alternating signal",
			signal: []float32{1.0, -1.0, 1.0, -1.0},
			expected: MeasurementResult{
				RMS:  1.0,
				Peak: 1.0,
				Mean: 0.0,
			},
			tol: 1e-6,
		},
		{
			name:   "ramp signal",
			signal: []float32{0.0, 1.0, 2.0, 3.0},
			expected: MeasurementResult{
				Mean: 1.5,
				Peak: 3.0,
			},
			tol: 1e-5,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			signal := vec.NewFrom(tt.signal...)
			result := Measure1D(signal)

			if len(tt.signal) == 0 {
				if result.RMS != 0 || result.Peak != 0 {
					t.Errorf("Expected zero result for empty signal")
				}
				return
			}

			if math.Abs(float64(result.RMS-tt.expected.RMS)) > tt.tol {
				t.Errorf("RMS: expected %f, got %f", tt.expected.RMS, result.RMS)
			}
			if math.Abs(float64(result.Peak-tt.expected.Peak)) > tt.tol {
				t.Errorf("Peak: expected %f, got %f", tt.expected.Peak, result.Peak)
			}
			if math.Abs(float64(result.Mean-tt.expected.Mean)) > tt.tol {
				t.Errorf("Mean: expected %f, got %f", tt.expected.Mean, result.Mean)
			}
		})
	}
}

func TestMeasurements_Measure2D(t *testing.T) {
	matrix := mat.New(2, 2)
	matrix[0][0] = 1.0
	matrix[0][1] = 2.0
	matrix[1][0] = 3.0
	matrix[1][1] = 4.0

	result := Measure2D(matrix)

	expectedMean := float32(2.5)
	expectedPeak := float32(4.0)

	if math.Abs(float64(result.Mean-expectedMean)) > 1e-5 {
		t.Errorf("Mean: expected %f, got %f", expectedMean, result.Mean)
	}
	if math.Abs(float64(result.Peak-expectedPeak)) > 1e-5 {
		t.Errorf("Peak: expected %f, got %f", expectedPeak, result.Peak)
	}
}

func TestMeasurements_Goertzel(t *testing.T) {
	// Generate a 10Hz sine wave at 1000Hz sample rate
	gen := NewSinusoidGenerator(1000.0, 10.0, 1.0, 0.0)
	data := make([]float32, 1000)
	gen.Generate(data)
	signal := vec.NewFrom(data...)

	result := Goertzel(signal, 10.0, 1000.0)

	// Should detect 10Hz with high amplitude
	if result.Amplitude < 0.9 {
		t.Errorf("Expected amplitude > 0.9, got %f", result.Amplitude)
	}
	if math.Abs(float64(result.Frequency-10.0)) > 0.1 {
		t.Errorf("Expected frequency ~10.0, got %f", result.Frequency)
	}
}


func TestCalculateRMSStatistics(t *testing.T) {
	tests := []struct {
		name     string
		signal   []float32
		checkMin bool
		checkMax bool
		checkAvg bool
	}{
		{
			name:     "empty signal",
			signal:   []float32{},
			checkMin: false,
			checkMax: false,
			checkAvg: false,
		},
		{
			name:     "constant signal",
			signal:   []float32{1.0, 1.0, 1.0, 1.0},
			checkMin: true,
			checkMax: true,
			checkAvg: true,
		},
		{
			name:     "varying signal",
			signal:   []float32{0.1, 0.5, 1.0, 0.5, 0.1},
			checkMin: true,
			checkMax: true,
			checkAvg: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			signal := vec.NewFrom(tt.signal...)
			stats := CalculateRMSStatistics(signal)

			if len(tt.signal) == 0 {
				if !math.IsInf(stats.MinRMSDb, -1) {
					t.Errorf("Expected -Inf for empty signal MinRMSDb")
				}
				return
			}

			if tt.checkMin && math.IsInf(stats.MinRMSDb, -1) {
				t.Errorf("MinRMSDb should not be -Inf for non-empty signal")
			}
			if tt.checkMax && math.IsInf(stats.MaxRMSDb, -1) {
				t.Errorf("MaxRMSDb should not be -Inf for non-empty signal")
			}
			if tt.checkAvg && math.IsInf(stats.AvgRMSDb, -1) {
				t.Errorf("AvgRMSDb should not be -Inf for non-empty signal")
			}

			// Max should be >= Min
			if stats.MaxRMSDb < stats.MinRMSDb {
				t.Errorf("MaxRMSDb (%f) should be >= MinRMSDb (%f)", stats.MaxRMSDb, stats.MinRMSDb)
			}
		})
	}
}

