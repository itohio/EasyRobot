package dsp

import (
	"math"
	"testing"

	"github.com/itohio/EasyRobot/x/math/mat"
	"github.com/itohio/EasyRobot/x/math/vec"
)

func TestCorrelation_CrossCorrelate1D(t *testing.T) {

	tests := []struct {
		name     string
		signal1  []float32
		signal2  []float32
		expected int // expected result length
	}{
		{
			name:     "basic correlation",
			signal1:  []float32{1, 2, 3},
			signal2:  []float32{1, 0, 0},
			expected: 5, // n1 + n2 - 1
		},
		{
			name:     "identical signals",
			signal1:  []float32{1, 2, 3},
			signal2:  []float32{1, 2, 3},
			expected: 5,
		},
		{
			name:     "empty signal1",
			signal1:  []float32{},
			signal2:  []float32{1, 2},
			expected: 1, // 0 + 2 - 1 = 1
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s1 := vec.NewFrom(tt.signal1...)
			s2 := vec.NewFrom(tt.signal2...)
			dst := vec.New(tt.expected)
			n := CrossCorrelate1D(dst, s1, s2)

			if n != tt.expected {
				t.Errorf("Expected correlation length %d, got %d", tt.expected, n)
			}
		})
	}
}

func TestCorrelation_CrossCorrelate1D_ByConvolution(t *testing.T) {

	signal1 := vec.NewFrom(1.0, 2.0, 3.0, 4.0)
	signal2 := vec.NewFrom(1.0, 0.5)

	// Standard correlation
	resultLen := 5 // len(signal1) + len(signal2) - 1
	corr1 := vec.New(resultLen)
	CrossCorrelate1D(corr1, signal1, signal2)

	// Correlation by convolution
	corr2 := vec.New(resultLen)
	CrossCorrelate1DByConvolution(corr2, signal1, signal2)

	corr1Data := corr1.View().(vec.Vector)
	corr2Data := corr2.View().(vec.Vector)

	if len(corr1Data) != len(corr2Data) {
		t.Fatalf("Length mismatch: %d vs %d", len(corr1Data), len(corr2Data))
	}

	// Results should be approximately equal
	for i := range corr1Data {
		diff := math.Abs(float64(corr1Data[i] - corr2Data[i]))
		if diff > 1e-5 {
			t.Errorf("Mismatch at index %d: %f vs %f (diff: %f)", i, corr1Data[i], corr2Data[i], diff)
		}
	}
}

func TestCorrelation_CrossCorrelate2D(t *testing.T) {

	matrix1 := mat.New(3, 3)
	matrix2 := mat.New(2, 2)

	// Fill matrices
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			matrix1[i][j] = float32(i*3 + j + 1)
		}
	}
	matrix2[0][0] = 1.0
	matrix2[0][1] = 0.0
	matrix2[1][0] = 0.0
	matrix2[1][1] = 1.0

	expectedRows := 4 // 3 + 2 - 1
	expectedCols := 4 // 3 + 2 - 1
	result := mat.New(expectedRows, expectedCols)
	n := CrossCorrelate2D(result, matrix1, matrix2)

	if n != expectedRows {
		t.Errorf("Expected %d rows, got %d", expectedRows, n)
	}
	if result.Rows() != expectedRows {
		t.Errorf("Expected %d rows, got %d", expectedRows, result.Rows())
	}
	if result.Cols() != expectedCols {
		t.Errorf("Expected %d cols, got %d", expectedCols, result.Cols())
	}
}

func TestCorrelation_FindPeakCorrelation(t *testing.T) {

	tests := []struct {
		name     string
		corrData []float32
		expected CorrelationResult
	}{
		{
			name:     "peak at center",
			corrData: []float32{0.1, 0.5, 1.0, 0.5, 0.1},
			expected: CorrelationResult{
				Coefficient: 1.0,
				Lag:         0, // (5-1)/2 = 2, peak at 2, lag = 2-2 = 0
			},
		},
		{
			name:     "peak at start",
			corrData: []float32{1.0, 0.5, 0.1},
			expected: CorrelationResult{
				Coefficient: 1.0,
				Lag:         -1, // (3-1)/2 = 1, peak at 0, lag = 0-1 = -1
			},
		},
		{
			name:     "empty correlation",
			corrData: []float32{},
			expected: CorrelationResult{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			corrVec := vec.NewFrom(tt.corrData...)
			result := FindPeakCorrelation(corrVec)

			if len(tt.corrData) == 0 {
				if result.Coefficient != 0 {
					t.Errorf("Expected zero coefficient for empty correlation")
				}
				return
			}

			if math.Abs(float64(result.Coefficient-tt.expected.Coefficient)) > 1e-6 {
				t.Errorf("Coefficient: expected %f, got %f", tt.expected.Coefficient, result.Coefficient)
			}
			if result.Lag != tt.expected.Lag {
				t.Errorf("Lag: expected %d, got %d", tt.expected.Lag, result.Lag)
			}
		})
	}
}

func TestCorrelation_FindPeakCorrelation2D(t *testing.T) {

	matrix := mat.New(5, 5)
	// Set peak at center
	matrix[2][2] = 1.0

	maxCorr, dy, dx := FindPeakCorrelation2D(matrix)

	if math.Abs(float64(maxCorr-1.0)) > 1e-6 {
		t.Errorf("Expected max correlation 1.0, got %f", maxCorr)
	}
	if dy != 0 || dx != 0 {
		t.Errorf("Expected shift (0,0), got (%d,%d)", dy, dx)
	}
}

func TestCorrelation_PhaseCorrelation2D(t *testing.T) {

	// Create two identical matrices
	matrix1 := mat.New(8, 8)
	matrix2 := mat.New(8, 8)

	for i := 0; i < 8; i++ {
		for j := 0; j < 8; j++ {
			val := float32(i*8 + j)
			matrix1[i][j] = val
			matrix2[i][j] = val
		}
	}

	result := mat.New(8, 8)
	PhaseCorrelation2D(result, matrix1, matrix2)

	// Phase correlation of identical signals should have peak at center
	_, dy, dx := FindPeakCorrelation2D(result)

	if dy != 0 || dx != 0 {
		t.Errorf("Expected zero shift for identical signals, got (%d,%d)", dy, dx)
	}
}

func TestCorrelation_EstimateShift2D(t *testing.T) {

	// Create shifted version of a matrix
	matrix1 := mat.New(8, 8)
	matrix2 := mat.New(8, 8)

	// Fill matrix1
	for i := 0; i < 8; i++ {
		for j := 0; j < 8; j++ {
			matrix1[i][j] = float32(i*8 + j)
		}
	}

	// Create matrix2 as shifted version (shift by 2,2)
	for i := 0; i < 6; i++ {
		for j := 0; j < 6; j++ {
			matrix2[i+2][j+2] = matrix1[i][j]
		}
	}

	dy, dx := EstimateShift2D(matrix1, matrix2)

	// Should detect shift of approximately (2, 2)
	if math.Abs(float64(dy-2)) > 1 || math.Abs(float64(dx-2)) > 1 {
		t.Errorf("Expected shift approximately (2,2), got (%d,%d)", dy, dx)
	}
}
