package fp32

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestCrossCorrelate1D(t *testing.T) {
	tests := []struct {
		name     string
		signal1  []float32
		signal2  []float32
		expected []float32
	}{
		{
			name:     "impulse_with_impulse",
			signal1:  []float32{1, 0, 0, 0},
			signal2:  []float32{1, 0, 0, 0},
			expected: []float32{0, 0, 0, 1, 0, 0, 0},
		},
		{
			name:     "simple",
			signal1:  []float32{1, 2, 3},
			signal2:  []float32{1, 2},
			expected: []float32{2, 5, 8, 6, 0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, len(tt.signal1)+len(tt.signal2)-1)
			n := CrossCorrelate1D(dst, tt.signal1, tt.signal2)

			assert.Equal(t, len(tt.expected), n)
			for i, exp := range tt.expected {
				if i < len(dst) {
					assert.InDelta(t, exp, dst[i], 1e-6, "Correlation at index %d", i)
				}
			}
		})
	}
}

func TestCrossCorrelate1DByConvolution(t *testing.T) {
	signal1 := []float32{1, 2, 3, 4}
	signal2 := []float32{1, 2}

	// Both methods should give same result
	corr1 := make([]float32, len(signal1)+len(signal2)-1)
	corr2 := make([]float32, len(signal1)+len(signal2)-1)

	n1 := CrossCorrelate1D(corr1, signal1, signal2)
	n2 := CrossCorrelate1DByConvolution(corr2, signal1, signal2)

	assert.Equal(t, n1, n2)
	for i := range n1 {
		assert.InDelta(t, corr1[i], corr2[i], 1e-6, "Methods should match at index %d", i)
	}
}

func TestCrossCorrelate2D(t *testing.T) {
	signal1 := []float32{1, 2, 3, 4} // 2x2
	rows1, cols1 := 2, 2
	signal2 := []float32{1, 2} // 1x2
	rows2, cols2 := 1, 2

	dstRows := rows1 + rows2 - 1
	dstCols := cols1 + cols2 - 1
	dst := make([]float32, dstRows*dstCols)

	n := CrossCorrelate2D(dst, dstRows, dstCols, signal1, rows1, cols1, signal2, rows2, cols2)

	assert.Equal(t, dstRows, n)
	// Basic sanity check - result should be non-zero
	hasNonZero := false
	for _, v := range dst {
		if v != 0 {
			hasNonZero = true
			break
		}
	}
	assert.True(t, hasNonZero, "2D correlation should produce non-zero result")
}

func TestFindPeakCorrelation(t *testing.T) {
	tests := []struct {
		name     string
		corr     []float32
		expected float32
		expectedIdx int
	}{
		{
			name:        "simple",
			corr:        []float32{1, 2, 5, 3, 1},
			expected:    5.0,
			expectedIdx: 2,
		},
		{
			name:        "all_same",
			corr:        []float32{1, 1, 1, 1},
			expected:    1.0,
			expectedIdx: 0,
		},
		{
			name:        "negative",
			corr:        []float32{-5, -2, -1},
			expected:    -1.0,
			expectedIdx: 2,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			maxCorr, maxIdx := FindPeakCorrelation(tt.corr)
			assert.InDelta(t, tt.expected, maxCorr, 1e-6, "Peak correlation value")
			assert.Equal(t, tt.expectedIdx, maxIdx, "Peak correlation index")
		})
	}
}

func TestFindPeakCorrelation2D(t *testing.T) {
	rows, cols := 3, 3
	corr := []float32{
		1, 2, 3,
		4, 9, 5, // peak at (1,1)
		6, 7, 8,
	}

	maxCorr, maxRow, maxCol := FindPeakCorrelation2D(corr, rows, cols)

	assert.InDelta(t, 9.0, maxCorr, 1e-6, "Peak correlation value")
	assert.Equal(t, 1, maxRow, "Peak row")
	assert.Equal(t, 1, maxCol, "Peak col")
}

func TestCrossCorrelate1D_Empty(t *testing.T) {
	dst := make([]float32, 10)
	n := CrossCorrelate1D(dst, []float32{}, []float32{1, 2})
	assert.Equal(t, 0, n)
}

func TestCrossCorrelate1D_SmallDst(t *testing.T) {
	signal1 := []float32{1, 2, 3, 4, 5}
	signal2 := []float32{1, 2, 3}
	dst := make([]float32, 3) // Smaller than needed (7)

	n := CrossCorrelate1D(dst, signal1, signal2)
	assert.Equal(t, 3, n) // Should fill what it can
}

