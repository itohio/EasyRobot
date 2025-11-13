package generics

import (
	"testing"
)

func TestElemWhere(t *testing.T) {
	tests := []struct {
		name      string
		condition []float32
		a         []float32
		b         []float32
		shape     []int
		stridesD  []int
		stridesC  []int
		stridesA  []int
		stridesB  []int
		want      []float32
	}{
		{
			name:      "contiguous",
			condition: []float32{1, 0, 1, 0},
			a:         []float32{10, 20, 30, 40},
			b:         []float32{100, 200, 300, 400},
			shape:     []int{4},
			stridesD:  nil,
			stridesC:  nil,
			stridesA:  nil,
			stridesB:  nil,
			want:      []float32{10, 200, 30, 400},
		},
		{
			name:      "2D contiguous",
			condition: []float32{1, 0, 0, 1},
			a:         []float32{10, 20, 30, 40},
			b:         []float32{100, 200, 300, 400},
			shape:     []int{2, 2},
			stridesD:  nil,
			stridesC:  nil,
			stridesA:  nil,
			stridesB:  nil,
			want:      []float32{10, 200, 300, 40},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, len(tt.want))
			ElemWhere(dst, tt.condition, tt.a, tt.b, tt.shape, tt.stridesD, tt.stridesC, tt.stridesA, tt.stridesB)
			size := SizeFromShape(tt.shape)
			for i := 0; i < size; i++ {
				if dst[i] != tt.want[i] {
					t.Errorf("ElemWhere() dst[%d] = %v, want %v", i, dst[i], tt.want[i])
				}
			}
		})
	}
}
