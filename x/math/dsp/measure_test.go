package dsp

import (
	"math"
	"testing"

	"github.com/itohio/EasyRobot/x/math/mat"
	"github.com/itohio/EasyRobot/x/math/shape"
	"github.com/itohio/EasyRobot/x/math/tensor"
	"github.com/itohio/EasyRobot/x/math/tensor/types"
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
				RMS:  float32(math.Sqrt(3.5)),
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

func TestMeasurements_Measure1DC(t *testing.T) {
	tests := []struct {
		name     string
		signal   []complex64
		expected MeasurementResult
		tol      float64
	}{
		{
			name:   "empty signal",
			signal: []complex64{},
			expected: MeasurementResult{},
			tol:    1e-6,
		},
		{
			name:   "constant signal",
			signal: []complex64{2.0 + 0i, 2.0 + 0i, 2.0 + 0i, 2.0 + 0i},
			expected: MeasurementResult{
				RMS:  2.0,
				Peak: 2.0,
				Mean: 2.0,
			},
			tol: 1e-6,
		},
		{
			name:   "complex values",
			signal: []complex64{1.0 + 1i, -1.0 + 1i, 1.0 - 1i, -1.0 - 1i},
			expected: MeasurementResult{
				RMS:  float32(math.Sqrt(2.0)), // magnitude of 1+1i is sqrt(2)
				Peak: float32(math.Sqrt(2.0)),
				Mean: float32(math.Sqrt(2.0)),
			},
			tol: 1e-5,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			signal := tensor.New(types.C64, shape.NewShape(len(tt.signal)))
			data := types.GetTensorData[[]complex64](signal)
			copy(data, tt.signal)
			result := Measure1DC(signal)

			if len(tt.signal) == 0 {
				if result.RMS != 0 || result.Peak != 0 {
					t.Errorf("Expected zero result for empty signal")
				}
				return
			}

			if math.Abs(float64(result.RMS-float32(tt.expected.RMS))) > tt.tol {
				t.Errorf("RMS: expected %f, got %f", tt.expected.RMS, result.RMS)
			}
			if math.Abs(float64(result.Peak-float32(tt.expected.Peak))) > tt.tol {
				t.Errorf("Peak: expected %f, got %f", tt.expected.Peak, result.Peak)
			}
			if math.Abs(float64(result.Mean-float32(tt.expected.Mean))) > tt.tol {
				t.Errorf("Mean: expected %f, got %f", tt.expected.Mean, result.Mean)
			}
		})
	}
}

func TestMeasurements_Measure2DC(t *testing.T) {
	signal := tensor.New(types.C64, shape.NewShape(2, 2))
	data := types.GetTensorData[[]complex64](signal)
	data[0] = 1.0 + 0i
	data[1] = 2.0 + 0i
	data[2] = 3.0 + 0i
	data[3] = 4.0 + 0i

	result := Measure2DC(signal)

	expectedMean := float32(2.5)
	expectedPeak := float32(4.0)

	if math.Abs(float64(result.Mean-expectedMean)) > 1e-5 {
		t.Errorf("Mean: expected %f, got %f", expectedMean, result.Mean)
	}
	if math.Abs(float64(result.Peak-expectedPeak)) > 1e-5 {
		t.Errorf("Peak: expected %f, got %f", expectedPeak, result.Peak)
	}
}

func TestMeasurements_GoertzelC(t *testing.T) {
	// Generate a 10Hz complex sine wave at 1000Hz sample rate
	n := 1000
	signal := tensor.New(types.C64, shape.NewShape(n))
	data := types.GetTensorData[[]complex64](signal)
	sampleRate := float32(1000.0)
	freq := float32(10.0)
	amplitude := float32(1.0)
	for i := range n {
		t := float32(i) / sampleRate
		phase := 2 * math.Pi * freq * t
		data[i] = complex64(complex(amplitude*float32(math.Cos(float64(phase))), amplitude*float32(math.Sin(float64(phase)))))
	}

	result := GoertzelC(signal, 10.0, 1000.0)

	// Should detect 10Hz with high amplitude
	if result.Amplitude < 0.9 {
		t.Errorf("Expected amplitude > 0.9, got %f", result.Amplitude)
	}
	if math.Abs(float64(result.Frequency-10.0)) > 0.1 {
		t.Errorf("Expected frequency ~10.0, got %f", result.Frequency)
	}
}

func TestCalculateRMSStatisticsC(t *testing.T) {
	tests := []struct {
		name     string
		signal   []complex64
		checkMin bool
		checkMax bool
		checkAvg bool
	}{
		{
			name:     "empty signal",
			signal:   []complex64{},
			checkMin: false,
			checkMax: false,
			checkAvg: false,
		},
		{
			name:     "constant signal",
			signal:   []complex64{1.0 + 0i, 1.0 + 0i, 1.0 + 0i, 1.0 + 0i},
			checkMin: true,
			checkMax: true,
			checkAvg: true,
		},
		{
			name:     "varying signal",
			signal:   []complex64{0.1 + 0i, 0.5 + 0i, 1.0 + 0i, 0.5 + 0i, 0.1 + 0i},
			checkMin: true,
			checkMax: true,
			checkAvg: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			signal := tensor.New(types.C64, shape.NewShape(len(tt.signal)))
			data := types.GetTensorData[[]complex64](signal)
			copy(data, tt.signal)
			stats := CalculateRMSStatisticsC(signal)

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

