package fp32

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

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
		name                 string
		a                    []float32
		num                  int
		stride               int
		wantMin, wantMax     float32
		wantMean, wantStddev float32
	}{
		{
			name:       "simple array",
			a:          []float32{1, 2, 3, 4, 5},
			num:        5,
			stride:     1,
			wantMin:    1,
			wantMax:    5,
			wantMean:   3,
			wantStddev: 1.4142135623730951, // approximate
		},
		{
			name:       "single element",
			a:          []float32{42},
			num:        1,
			stride:     1,
			wantMin:    42,
			wantMax:    42,
			wantMean:   42,
			wantStddev: 0,
		},
		{
			name:       "all same values",
			a:          []float32{5, 5, 5, 5},
			num:        4,
			stride:     1,
			wantMin:    5,
			wantMax:    5,
			wantMean:   5,
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
			name:           "median (p50)",
			a:              []float32{1, 2, 3, 4, 5},
			num:            5,
			stride:         1,
			p:              0.5,
			wantPercentile: 3,
			wantSumAboveP:  9, // 4 + 5
		},
		{
			name:           "p25",
			a:              []float32{1, 2, 3, 4, 5},
			num:            5,
			stride:         1,
			p:              0.25,
			wantPercentile: 2,
			wantSumAboveP:  12, // 3 + 4 + 5
		},
		{
			name:           "p75",
			a:              []float32{1, 2, 3, 4, 5},
			num:            5,
			stride:         1,
			p:              0.75,
			wantPercentile: 4,
			wantSumAboveP:  5, // 5
		},
		{
			name:           "single element",
			a:              []float32{42},
			num:            1,
			stride:         1,
			p:              0.5,
			wantPercentile: 42,
			wantSumAboveP:  0,
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

func TestSumArrScalar(t *testing.T) {
	tests := []struct {
		name     string
		src      []float32
		c        float32
		num      int
		strideDst int
		strideSrc int
		wantDst  []float32
	}{
		{
			name:      "simple addition",
			src:       []float32{1, 2, 3, 4},
			c:         10,
			num:       4,
			strideDst: 1,
			strideSrc: 1,
			wantDst:   []float32{11, 12, 13, 14},
		},
		{
			name:      "with stride",
			src:       []float32{1, 0, 2, 0, 3, 0, 4},
			c:         5,
			num:       4,
			strideDst: 1,
			strideSrc: 2,
			wantDst:   []float32{6, 7, 8, 9},
		},
		{
			name:      "zero scalar",
			src:       []float32{1, 2, 3, 4},
			c:         0,
			num:       4,
			strideDst: 1,
			strideSrc: 1,
			wantDst:   []float32{1, 2, 3, 4},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, len(tt.wantDst))
			SumArrScalar(dst, tt.src, tt.c, tt.num, tt.strideDst, tt.strideSrc)
			assert.InDeltaSlice(t, tt.wantDst, dst, 1e-5)
		})
	}
}

func TestDiffArrScalar(t *testing.T) {
	tests := []struct {
		name      string
		src       []float32
		c         float32
		num       int
		strideDst int
		strideSrc int
		wantDst   []float32
	}{
		{
			name:      "simple subtraction",
			src:       []float32{11, 12, 13, 14},
			c:         10,
			num:       4,
			strideDst: 1,
			strideSrc: 1,
			wantDst:   []float32{1, 2, 3, 4},
		},
		{
			name:      "with stride",
			src:       []float32{6, 0, 7, 0, 8, 0, 9},
			c:         5,
			num:       4,
			strideDst: 1,
			strideSrc: 2,
			wantDst:   []float32{1, 2, 3, 4},
		},
		{
			name:      "zero scalar",
			src:       []float32{1, 2, 3, 4},
			c:         0,
			num:       4,
			strideDst: 1,
			strideSrc: 1,
			wantDst:   []float32{1, 2, 3, 4},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, len(tt.wantDst))
			DiffArrScalar(dst, tt.src, tt.c, tt.num, tt.strideDst, tt.strideSrc)
			assert.InDeltaSlice(t, tt.wantDst, dst, 1e-5)
		})
	}
}
