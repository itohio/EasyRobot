package fp32

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestAxpy(t *testing.T) {
	tests := []struct {
		name                string
		y, x                []float32
		alpha               float32
		strideY, strideX, n int
		wantY               []float32
	}{
		{
			name:    "simple",
			y:       []float32{1, 2, 3},
			x:       []float32{4, 5, 6},
			alpha:   2.0,
			strideY: 1,
			strideX: 1,
			n:       3,
			wantY:   []float32{9, 12, 15}, // [1+2*4, 2+2*5, 3+2*6]
		},
		{
			name:    "with stride",
			y:       make([]float32, 6),
			x:       []float32{1, 2, 3},
			alpha:   2.0,
			strideY: 2,
			strideX: 1,
			n:       3,
			wantY:   []float32{2, 0, 4, 0, 6, 0}, // y[0]=2, y[2]=4, y[4]=6
		},
		{
			name:    "zero alpha",
			y:       []float32{1, 2, 3},
			x:       []float32{4, 5, 6},
			alpha:   0.0,
			strideY: 1,
			strideX: 1,
			n:       3,
			wantY:   []float32{1, 2, 3}, // y unchanged
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			Axpy(tt.y, tt.x, tt.strideY, tt.strideX, tt.n, tt.alpha)
			assert.Equal(t, tt.wantY, tt.y)
		})
	}
}

func TestDot(t *testing.T) {
	tests := []struct {
		name                string
		x, y                []float32
		strideX, strideY, n int
		want                float32
	}{
		{
			name:    "simple",
			x:       []float32{1, 2, 3},
			y:       []float32{4, 5, 6},
			strideX: 1,
			strideY: 1,
			n:       3,
			want:    32, // 1*4 + 2*5 + 3*6 = 4 + 10 + 18
		},
		{
			name:    "with stride",
			x:       []float32{1, 0, 2, 0, 3},
			y:       []float32{4, 5, 6},
			strideX: 2,
			strideY: 1,
			n:       3,
			want:    32,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Dot(tt.x, tt.y, tt.strideX, tt.strideY, tt.n)
			assert.InDelta(t, float64(tt.want), float64(got), 1e-5)
		})
	}
}

func TestNrm2(t *testing.T) {
	tests := []struct {
		name      string
		x         []float32
		stride, n int
		want      float32
	}{
		{
			name:   "simple",
			x:      []float32{3, 4},
			stride: 1,
			n:      2,
			want:   5, // sqrt(3^2 + 4^2) = 5
		},
		{
			name:   "zero vector",
			x:      []float32{0, 0, 0},
			stride: 1,
			n:      3,
			want:   0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Nrm2(tt.x, tt.stride, tt.n)
			assert.InDelta(t, float64(tt.want), float64(got), 1e-5)
		})
	}
}

func TestAsum(t *testing.T) {
	tests := []struct {
		name      string
		x         []float32
		stride, n int
		want      float32
	}{
		{
			name:   "all positive",
			x:      []float32{1, 2, 3},
			stride: 1,
			n:      3,
			want:   6,
		},
		{
			name:   "mixed signs",
			x:      []float32{-1, 2, -3, 4},
			stride: 1,
			n:      4,
			want:   10, // 1 + 2 + 3 + 4
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Asum(tt.x, tt.stride, tt.n)
			assert.InDelta(t, float64(tt.want), float64(got), 1e-5)
		})
	}
}

func TestScal(t *testing.T) {
	tests := []struct {
		name      string
		x         []float32
		stride, n int
		alpha     float32
		want      []float32
	}{
		{
			name:   "simple",
			x:      []float32{1, 2, 3},
			stride: 1,
			n:      3,
			alpha:  2.0,
			want:   []float32{2, 4, 6},
		},
		{
			name:   "negative alpha",
			x:      []float32{1, 2, 3},
			stride: 1,
			n:      3,
			alpha:  -1.0,
			want:   []float32{-1, -2, -3},
		},
		{
			name:   "zero alpha",
			x:      []float32{1, 2, 3},
			stride: 1,
			n:      3,
			alpha:  0.0,
			want:   []float32{0, 0, 0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			Scal(tt.x, tt.stride, tt.n, tt.alpha)
			assert.Equal(t, tt.want, tt.x)
		})
	}
}

func TestCopy(t *testing.T) {
	tests := []struct {
		name                string
		y, x                []float32
		strideY, strideX, n int
		want                []float32
	}{
		{
			name:    "simple",
			y:       make([]float32, 3),
			x:       []float32{1, 2, 3},
			strideY: 1,
			strideX: 1,
			n:       3,
			want:    []float32{1, 2, 3},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			Copy(tt.y, tt.x, tt.strideY, tt.strideX, tt.n)
			assert.Equal(t, tt.want, tt.y)
		})
	}
}

func TestSwap(t *testing.T) {
	tests := []struct {
		name                string
		x, y                []float32
		strideX, strideY, n int
		wantX, wantY        []float32
	}{
		{
			name:    "simple",
			x:       []float32{1, 2, 3},
			y:       []float32{4, 5, 6},
			strideX: 1,
			strideY: 1,
			n:       3,
			wantX:   []float32{4, 5, 6},
			wantY:   []float32{1, 2, 3},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			Swap(tt.x, tt.y, tt.strideX, tt.strideY, tt.n)
			assert.Equal(t, tt.wantX, tt.x)
			assert.Equal(t, tt.wantY, tt.y)
		})
	}
}

func TestIamax(t *testing.T) {
	tests := []struct {
		name      string
		x         []float32
		stride, n int
		want      int
	}{
		{
			name:   "simple",
			x:      []float32{1, 5, 2},
			stride: 1,
			n:      3,
			want:   1, // index of 5 (max abs value)
		},
		{
			name:   "negative max",
			x:      []float32{1, -5, 2},
			stride: 1,
			n:      3,
			want:   1, // index of -5 (max abs value)
		},
		{
			name:   "first element",
			x:      []float32{10, 5, 2},
			stride: 1,
			n:      3,
			want:   0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Iamax(tt.x, tt.stride, tt.n)
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestEmptyVectors(t *testing.T) {
	x := []float32{1, 2, 3}
	y := []float32{4, 5, 6}

	require.NotPanics(t, func() {
		Axpy(y, x, 1, 1, 0, 1.0)
		_ = Dot(x, y, 1, 1, 0)
		_ = Nrm2(x, 1, 0)
		_ = Asum(x, 1, 0)
		Scal(x, 1, 0, 2.0)
		Copy(y, x, 1, 1, 0)
		Swap(x, y, 1, 1, 0)
		_ = Iamax(x, 1, 0)
	})
}
