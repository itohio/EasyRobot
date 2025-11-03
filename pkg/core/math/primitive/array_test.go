package primitive

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestSumArr(t *testing.T) {
	a := []float32{1, 2, 3, 4}
	b := []float32{5, 6, 7, 8}
	dst := make([]float32, 4)

	SumArr(dst, a, b, 4, 1, 1)
	assert.Equal(t, []float32{6, 8, 10, 12}, dst)
}

func TestDiffArr(t *testing.T) {
	a := []float32{5, 6, 7, 8}
	b := []float32{1, 2, 3, 4}
	dst := make([]float32, 4)

	DiffArr(dst, a, b, 4, 1, 1)
	assert.Equal(t, []float32{4, 4, 4, 4}, dst)
}

func TestMulArr(t *testing.T) {
	a := []float32{1, 2, 3, 4}
	b := []float32{2, 2, 2, 2}
	dst := make([]float32, 4)

	MulArr(dst, a, b, 4, 1, 1)
	assert.Equal(t, []float32{2, 4, 6, 8}, dst)
}

func TestDivArr(t *testing.T) {
	a := []float32{4, 6, 8, 10}
	b := []float32{2, 2, 2, 2}
	dst := make([]float32, 4)

	DivArr(dst, a, b, 4, 1, 1)
	assert.Equal(t, []float32{2, 3, 4, 5}, dst)
}

func TestSum(t *testing.T) {
	a := []float32{1, 2, 3, 4}
	result := Sum(a, 4, 1)
	assert.Equal(t, float32(10), result)
}

func TestSqrSum(t *testing.T) {
	a := []float32{1, 2, 3}
	result := SqrSum(a, 3, 1)
	assert.Equal(t, float32(14), result) // 1*1 + 2*2 + 3*3 = 1 + 4 + 9 = 14
}

func TestStatsArr(t *testing.T) {
	tests := []struct {
		name            string
		a               []float32
		num             int
		stride          int
		wantMin, wantMax float32
		wantMean, wantStddev float32
	}{
		{
			name: "simple array",
			a:    []float32{1, 2, 3, 4, 5},
			num:  5,
			stride: 1,
			wantMin: 1,
			wantMax: 5,
			wantMean: 3,
			wantStddev: 1.4142135623730951, // approximate
		},
		{
			name: "single element",
			a:    []float32{42},
			num:  1,
			stride: 1,
			wantMin: 42,
			wantMax: 42,
			wantMean: 42,
			wantStddev: 0,
		},
		{
			name: "all same values",
			a:    []float32{5, 5, 5, 5},
			num:  4,
			stride: 1,
			wantMin: 5,
			wantMax: 5,
			wantMean: 5,
			wantStddev: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var min, max, mean, stddev float32
			StatsArr(&min, &max, &mean, &stddev, tt.a, tt.num, tt.stride)
			assert.InDelta(t, tt.wantMin, min, 1e-5)
			assert.InDelta(t, tt.wantMax, max, 1e-5)
			assert.InDelta(t, tt.wantMean, mean, 1e-5)
			assert.InDelta(t, tt.wantStddev, stddev, 1e-4)
		})
	}
}

func TestPercentileArr(t *testing.T) {
	tests := []struct {
		name           string
		a              []float32
		num            int
		stride         int
		p              float32
		wantPercentile float32
		wantSumAboveP  float32
	}{
		{
			name: "median (p50)",
			a:    []float32{1, 2, 3, 4, 5},
			num:  5,
			stride: 1,
			p:     0.5,
			wantPercentile: 3,
			wantSumAboveP: 9, // 4 + 5
		},
		{
			name: "p25",
			a:    []float32{1, 2, 3, 4, 5},
			num:  5,
			stride: 1,
			p:     0.25,
			wantPercentile: 2,
			wantSumAboveP: 12, // 3 + 4 + 5
		},
		{
			name: "p75",
			a:    []float32{1, 2, 3, 4, 5},
			num:  5,
			stride: 1,
			p:     0.75,
			wantPercentile: 4,
			wantSumAboveP: 5, // 5
		},
		{
			name: "single element",
			a:    []float32{42},
			num:  1,
			stride: 1,
			p:     0.5,
			wantPercentile: 42,
			wantSumAboveP: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var sumAboveP float32
			percentile := PercentileArr(tt.p, &sumAboveP, tt.a, tt.num, tt.stride)
			assert.InDelta(t, tt.wantPercentile, percentile, 1e-5)
			assert.InDelta(t, tt.wantSumAboveP, sumAboveP, 1e-5)
		})
	}
}

func TestSumArrInPlace(t *testing.T) {
	dst := []float32{1, 2, 3, 4}
	SumArrInPlace(dst, 10, 4)
	assert.Equal(t, []float32{11, 12, 13, 14}, dst)
}

func TestMulArrInPlace(t *testing.T) {
	dst := []float32{1, 2, 3, 4}
	MulArrInPlace(dst, 2.0, 4)
	assert.Equal(t, []float32{2, 4, 6, 8}, dst)
}
