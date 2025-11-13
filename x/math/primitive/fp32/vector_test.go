package fp32

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestHadamardProduct(t *testing.T) {
	tests := []struct {
		name      string
		a         []float32
		b         []float32
		num       int
		strideDst int
		strideA   int
		strideB   int
		wantDst   []float32
	}{
		{
			name:      "simple product",
			a:         []float32{1, 2, 3, 4},
			b:         []float32{2, 3, 4, 5},
			num:       4,
			strideDst: 1,
			strideA:   1,
			strideB:   1,
			wantDst:   []float32{2, 6, 12, 20},
		},
		{
			name:      "with stride",
			a:         []float32{1, 0, 2, 0, 3, 0, 4},
			b:         []float32{2, 0, 3, 0, 4, 0, 5},
			num:       4,
			strideDst: 1,
			strideA:   2,
			strideB:   2,
			wantDst:   []float32{2, 6, 12, 20},
		},
		{
			name:      "single element",
			a:         []float32{5},
			b:         []float32{3},
			num:       1,
			strideDst: 1,
			strideA:   1,
			strideB:   1,
			wantDst:   []float32{15},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, len(tt.wantDst))
			HadamardProduct(dst, tt.a, tt.b, tt.num, tt.strideDst, tt.strideA, tt.strideB)
			assert.InDeltaSlice(t, tt.wantDst, dst, 1e-5)
		})
	}
}

func TestHadamardProductVsAdd(t *testing.T) {
	// Test that HadamardProduct produces same result as HadamardProductAdd with zero-initialized dst
	a := []float32{1, 2, 3, 4}
	b := []float32{2, 3, 4, 5}
	num := 4

	dst1 := make([]float32, 4)
	dst2 := make([]float32, 4)

	HadamardProduct(dst1, a, b, num, 1, 1, 1)
	HadamardProductAdd(dst2, a, b, num, 1, 1)

	assert.InDeltaSlice(t, dst1, dst2, 1e-5, "HadamardProduct should match HadamardProductAdd with zero dst")
}

func TestNormalizeVec(t *testing.T) {
	tests := []struct {
		name      string
		src       []float32
		num       int
		strideDst int
		strideSrc int
		wantDst   []float32
	}{
		{
			name:      "simple normalization",
			src:       []float32{3, 4},
			num:       2,
			strideDst: 1,
			strideSrc: 1,
			wantDst:   []float32{0.6, 0.8}, // [3, 4] / 5 = [0.6, 0.8]
		},
		{
			name:      "unit vector",
			src:       []float32{1, 0, 0},
			num:       3,
			strideDst: 1,
			strideSrc: 1,
			wantDst:   []float32{1, 0, 0},
		},
		{
			name:      "zero vector",
			src:       []float32{0, 0, 0},
			num:       3,
			strideDst: 1,
			strideSrc: 1,
			wantDst:   []float32{0, 0, 0}, // Should copy src when norm is zero
		},
		{
			name:      "with stride",
			src:       []float32{3, 0, 4, 0},
			num:       2,
			strideDst: 1,
			strideSrc: 2,
			wantDst:   []float32{0.6, 0.8},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, len(tt.wantDst))
			NormalizeVec(dst, tt.src, tt.num, tt.strideDst, tt.strideSrc)
			assert.InDeltaSlice(t, tt.wantDst, dst, 1e-5)
		})
	}
}

func TestNormalizeVecVsInPlace(t *testing.T) {
	// Test that NormalizeVec produces same result as NormalizeVecInPlace when src == dst
	src := []float32{3, 4}
	dst1 := make([]float32, 2)
	dst2 := make([]float32, 2)
	copy(dst1, src)
	copy(dst2, src)

	NormalizeVec(dst1, src, 2, 1, 1)
	NormalizeVecInPlace(dst2, 2, 1)

	assert.InDeltaSlice(t, dst1, dst2, 1e-5, "NormalizeVec should match NormalizeVecInPlace when src == dst")
}

