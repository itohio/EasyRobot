package fp32

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestConvolve1DAdd(t *testing.T) {
	tests := []struct {
		name       string
		dst        []float32
		vec        []float32
		kernel     []float32
		N, M       int
		stride     int
		transposed bool
		wantDst    []float32
	}{
		{
			name:       "forward convolution simple",
			dst:        make([]float32, 3),
			vec:        []float32{1, 2, 3, 4, 5},
			kernel:     []float32{1, 1},
			N:          5,
			M:          2,
			stride:     1,
			transposed: false,
			wantDst:    []float32{3, 5, 7}, // [1+2, 2+3, 3+4]
		},
		{
			name:       "forward convolution with stride 2",
			dst:        make([]float32, 2),
			vec:        []float32{1, 2, 3, 4, 5},
			kernel:     []float32{1, 1},
			N:          5,
			M:          2,
			stride:     2,
			transposed: false,
			wantDst:    []float32{3, 7}, // [1+2, 3+4]
		},
		{
			name:       "forward convolution with identity kernel",
			dst:        make([]float32, 4),
			vec:        []float32{1, 2, 3, 4, 5},
			kernel:     []float32{1},
			N:          5,
			M:          1,
			stride:     1,
			transposed: false,
			wantDst:    []float32{1, 2, 3, 4}, // [1, 2, 3, 4]
		},
		{
			name:       "transposed convolution simple",
			dst:        make([]float32, 6),
			vec:        []float32{1, 2},
			kernel:     []float32{1, 1},
			N:          2,
			M:          2,
			stride:     2,
			transposed: true,
			wantDst:    []float32{1, 1, 2, 2, 0, 0}, // [1*1, 1*1, 2*1, 2*1, ...]
		},
		{
			name:       "forward convolution empty vectors",
			dst:        make([]float32, 1),
			vec:        []float32{1, 2, 3},
			kernel:     []float32{},
			N:          3,
			M:          0,
			stride:     1,
			transposed: false,
			wantDst:    []float32{0}, // No operation when kernel is empty
		},
		{
			name:       "transposed convolution with stride 1",
			dst:        make([]float32, 5),
			vec:        []float32{1, 2},
			kernel:     []float32{1, 1},
			N:          2,
			M:          2,
			stride:     1,
			transposed: true,
			wantDst:    []float32{1, 3, 2, 0, 0}, // vec[0]*kernel[0]=1->dst[0], vec[0]*kernel[1]=1->dst[1], vec[1]*kernel[0]=2->dst[1], vec[1]*kernel[1]=2->dst[2]
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, len(tt.dst))
			copy(dst, tt.dst)
			Convolve1DAdd(dst, tt.vec, tt.kernel, tt.N, tt.M, tt.stride, tt.transposed)
			assert.Equal(t, tt.wantDst, dst, "Convolve1DAdd result mismatch")
		})
	}
}

func TestConvolve1DAddAccumulation(t *testing.T) {
	// Test that Convolve1DAdd accumulates (adds) to existing dst values
	dst := []float32{10, 20, 30}
	vec := []float32{1, 2, 3, 4, 5}
	kernel := []float32{1, 1}
	Convolve1DAdd(dst, vec, kernel, 5, 2, 1, false)
	// dst should be [10+3, 20+5, 30+7] = [13, 25, 37]
	expected := []float32{13, 25, 37}
	assert.Equal(t, expected, dst, "Convolve1DAdd should accumulate")
}

func TestConvolve1DAddEdgeCases(t *testing.T) {
	tests := []struct {
		name   string
		dst    []float32
		vec    []float32
		kernel []float32
		N, M   int
		stride int
	}{
		{
			name:   "kernel larger than vec",
			dst:    make([]float32, 1),
			vec:    []float32{1, 2},
			kernel: []float32{1, 1, 1},
			N:      2,
			M:      3,
			stride: 1,
		},
		{
			name:   "stride 1 minimum",
			dst:    make([]float32, 3), // dstSize = (3-1)/1 = 2, but need space
			vec:    []float32{1, 2, 3},
			kernel: []float32{1},
			N:      3,
			M:      1,
			stride: 1,
		},
		{
			name:   "large stride",
			dst:    make([]float32, 1),
			vec:    []float32{1, 2, 3, 4, 5},
			kernel: []float32{1, 1},
			N:      5,
			M:      2,
			stride: 10,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Just verify it doesn't panic
			dst := make([]float32, len(tt.dst))
			Convolve1DAdd(dst, tt.vec, tt.kernel, tt.N, tt.M, tt.stride, false)
			// No assertion - just checking no panic
		})
	}
}

