package generics

import (
	"testing"
)

func TestElemCopy(t *testing.T) {
	tests := []struct {
		name     string
		src      []float32
		shape    []int
		stridesD []int
		stridesS []int
		want     []float32
	}{
		{
			name:     "1D contiguous",
			src:      []float32{1, 2, 3, 4, 5},
			shape:    []int{5},
			stridesD: nil,
			stridesS: nil,
			want:     []float32{1, 2, 3, 4, 5},
		},
		{
			name:     "2D contiguous",
			src:      []float32{1, 2, 3, 4, 5, 6},
			shape:    []int{2, 3},
			stridesD: nil,
			stridesS: nil,
			want:     []float32{1, 2, 3, 4, 5, 6},
		},
		{
			name:     "2D strided",
			src:      []float32{1, 2, 3, 4, 5, 6, 7, 8, 9},
			shape:    []int{2, 2},
			stridesD: []int{3, 1}, // stride 3 for rows, writes to positions 0,1,3,4
			stridesS: []int{3, 1},
			want:     []float32{1, 2, 0, 4, 5, 0, 0, 0, 0}, // positions 0,1,3,4 are set
		},
		{
			name:     "empty shape",
			src:      []float32{1, 2, 3},
			shape:    []int{},
			stridesD: nil,
			stridesS: nil,
			want:     []float32{0, 0, 0},
		},
		{
			name:     "zero size",
			src:      []float32{1, 2, 3},
			shape:    []int{0},
			stridesD: nil,
			stridesS: nil,
			want:     []float32{0, 0, 0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, len(tt.want))
			if tt.stridesD == nil && tt.stridesS == nil && IsContiguous(nil, tt.shape) {
				// Use simple ElemCopy for contiguous
				size := SizeFromShape(tt.shape)
				ElemCopy(dst, tt.src, size)
			} else {
				// Use ElemCopyStrided for strided
				ElemCopyStrided(dst, tt.src, tt.shape, tt.stridesD, tt.stridesS)
			}

			// Check all elements in the destination array
			for i := 0; i < len(tt.want); i++ {
				if dst[i] != tt.want[i] {
					t.Errorf("ElemCopy() dst[%d] = %v, want %v", i, dst[i], tt.want[i])
				}
			}
		})
	}
}

func TestElemCopyMultipleTypes(t *testing.T) {
	// Test with different numeric types
	testCopyType[int32](t, []int32{1, 2, 3, 4}, []int{4})
	testCopyType[int64](t, []int64{10, 20, 30}, []int{3})
	testCopyType[float64](t, []float64{1.5, 2.5, 3.5}, []int{3})
}

func testCopyType[T Numeric](t *testing.T, src []T, shape []int) {
	dst := make([]T, len(src))
	size := SizeFromShape(shape)
	ElemCopy(dst, src, size)

	for i := range src {
		if dst[i] != src[i] {
			t.Errorf("ElemCopy() for type %T: dst[%d] = %v, want %v", *new(T), i, dst[i], src[i])
		}
	}
}

func TestElemSwap(t *testing.T) {
	tests := []struct {
		name    string
		dst     []float32
		src     []float32
		n       int
		wantDst []float32
		wantSrc []float32
	}{
		{
			name:    "1D contiguous",
			dst:     []float32{1, 2, 3},
			src:     []float32{4, 5, 6},
			n:       3,
			wantDst: []float32{4, 5, 6},
			wantSrc: []float32{1, 2, 3},
		},
		{
			name:    "partial swap",
			dst:     []float32{1, 2, 3, 4, 5},
			src:     []float32{10, 20, 30, 40, 50},
			n:       3,
			wantDst: []float32{10, 20, 30, 4, 5},
			wantSrc: []float32{1, 2, 3, 40, 50},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, len(tt.dst))
			src := make([]float32, len(tt.src))
			copy(dst, tt.dst)
			copy(src, tt.src)

			ElemSwap(dst, src, tt.n)

			for i := 0; i < tt.n; i++ {
				if dst[i] != tt.wantDst[i] {
					t.Errorf("ElemSwap() dst[%d] = %v, want %v", i, dst[i], tt.wantDst[i])
				}
				if src[i] != tt.wantSrc[i] {
					t.Errorf("ElemSwap() src[%d] = %v, want %v", i, src[i], tt.wantSrc[i])
				}
			}
		})
	}
}

func TestElemSwapStrided(t *testing.T) {
	tests := []struct {
		name     string
		dst      []float32
		src      []float32
		shape    []int
		stridesD []int
		stridesS []int
		wantDst  []float32
		wantSrc  []float32
	}{
		{
			name:     "2D contiguous",
			dst:      []float32{1, 2, 3, 4},
			src:      []float32{5, 6, 7, 8},
			shape:    []int{2, 2},
			stridesD: nil,
			stridesS: nil,
			wantDst:  []float32{5, 6, 7, 8},
			wantSrc:  []float32{1, 2, 3, 4},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, len(tt.dst))
			src := make([]float32, len(tt.src))
			copy(dst, tt.dst)
			copy(src, tt.src)

			ElemSwapStrided(dst, src, tt.shape, tt.stridesD, tt.stridesS)

			size := SizeFromShape(tt.shape)
			for i := 0; i < size; i++ {
				if dst[i] != tt.wantDst[i] {
					t.Errorf("ElemSwapStrided() dst[%d] = %v, want %v", i, dst[i], tt.wantDst[i])
				}
				if src[i] != tt.wantSrc[i] {
					t.Errorf("ElemSwapStrided() src[%d] = %v, want %v", i, src[i], tt.wantSrc[i])
				}
			}
		})
	}
}
