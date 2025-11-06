package generics

import (
	"testing"
)

func TestCopy(t *testing.T) {
	tests := []struct {
		name string
		x    []float32
		n    int
		want []float32
	}{
		{
			name: "contiguous",
			x:    []float32{1, 2, 3, 4},
			n:    4,
			want: []float32{1, 2, 3, 4},
		},
		{
			name: "partial",
			x:    []float32{1, 2, 3, 4, 5},
			n:    3,
			want: []float32{1, 2, 3},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			y := make([]float32, len(tt.want))
			Copy(y, tt.x, tt.n)
			for i := 0; i < tt.n; i++ {
				if y[i] != tt.want[i] {
					t.Errorf("Copy() y[%d] = %v, want %v", i, y[i], tt.want[i])
				}
			}
		})
	}
}

func TestCopyStrided(t *testing.T) {
	tests := []struct {
		name    string
		x       []float32
		strideY int
		strideX int
		n       int
		want    []float32
	}{
		{
			name:    "contiguous",
			x:       []float32{1, 2, 3, 4},
			strideY: 1,
			strideX: 1,
			n:       4,
			want:    []float32{1, 2, 3, 4},
		},
		{
			name:    "strided",
			x:       []float32{1, 0, 2, 0, 3, 0, 4},
			strideY: 1,
			strideX: 2,
			n:       4,
			want:    []float32{1, 2, 3, 4},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			y := make([]float32, len(tt.want))
			CopyStrided(y, tt.x, tt.strideY, tt.strideX, tt.n)
			for i := 0; i < tt.n; i++ {
				if y[i] != tt.want[i] {
					t.Errorf("CopyStrided() y[%d] = %v, want %v", i, y[i], tt.want[i])
				}
			}
		})
	}
}

func TestSwap(t *testing.T) {
	tests := []struct {
		name string
		x    []float32
		y    []float32
		n    int
		wantX []float32
		wantY []float32
	}{
		{
			name:  "contiguous",
			x:     []float32{1, 2, 3, 4},
			y:     []float32{5, 6, 7, 8},
			n:     4,
			wantX: []float32{5, 6, 7, 8},
			wantY: []float32{1, 2, 3, 4},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			x := make([]float32, len(tt.x))
			y := make([]float32, len(tt.y))
			copy(x, tt.x)
			copy(y, tt.y)
			Swap(x, y, tt.n)
			for i := 0; i < tt.n; i++ {
				if x[i] != tt.wantX[i] {
					t.Errorf("Swap() x[%d] = %v, want %v", i, x[i], tt.wantX[i])
				}
				if y[i] != tt.wantY[i] {
					t.Errorf("Swap() y[%d] = %v, want %v", i, y[i], tt.wantY[i])
				}
			}
		})
	}
}

func TestSwapStrided(t *testing.T) {
	tests := []struct {
		name    string
		x       []float32
		y       []float32
		strideX int
		strideY int
		n       int
		wantX   []float32
		wantY   []float32
	}{
		{
			name:    "contiguous",
			x:       []float32{1, 2, 3, 4},
			y:       []float32{5, 6, 7, 8},
			strideX: 1,
			strideY: 1,
			n:       4,
			wantX:   []float32{5, 6, 7, 8},
			wantY:   []float32{1, 2, 3, 4},
		},
		{
			name:    "strided",
			x:       []float32{1, 0, 2, 0, 3},
			y:       []float32{5, 0, 6, 0, 7},
			strideX: 2,
			strideY: 2,
			n:       3,
			wantX:   []float32{5, 0, 6, 0, 7},
			wantY:   []float32{1, 0, 2, 0, 3},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			x := make([]float32, len(tt.x))
			y := make([]float32, len(tt.y))
			copy(x, tt.x)
			copy(y, tt.y)
			SwapStrided(x, y, tt.strideX, tt.strideY, tt.n)
			for i := 0; i < len(tt.wantX); i++ {
				if x[i] != tt.wantX[i] {
					t.Errorf("SwapStrided() x[%d] = %v, want %v", i, x[i], tt.wantX[i])
				}
			}
			for i := 0; i < len(tt.wantY); i++ {
				if y[i] != tt.wantY[i] {
					t.Errorf("SwapStrided() y[%d] = %v, want %v", i, y[i], tt.wantY[i])
				}
			}
		})
	}
}

