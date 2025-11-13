package generics

import (
	"testing"
)

func TestElemVecCopyStrided(t *testing.T) {
	tests := []struct {
		name      string
		src       []float32
		n         int
		strideDst int
		strideSrc int
		want      []float32
	}{
		{
			name:      "contiguous",
			src:       []float32{1, 2, 3, 4},
			n:         4,
			strideDst: 1,
			strideSrc: 1,
			want:      []float32{1, 2, 3, 4},
		},
		{
			name:      "strided",
			src:       []float32{1, 0, 2, 0, 3, 0, 4},
			n:         4,
			strideDst: 1,
			strideSrc: 2,
			want:      []float32{1, 2, 3, 4},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, len(tt.want))
			ElemVecCopyStrided(dst, tt.src, tt.n, tt.strideDst, tt.strideSrc)
			for i := 0; i < tt.n; i++ {
				if dst[i] != tt.want[i] {
					t.Errorf("ElemVecCopyStrided() dst[%d] = %v, want %v", i, dst[i], tt.want[i])
				}
			}
		})
	}
}

func TestElemMatCopyStrided(t *testing.T) {
	tests := []struct {
		name  string
		src   []float32
		rows  int
		cols  int
		ldDst int
		ldSrc int
		want  []float32
	}{
		{
			name:  "contiguous",
			src:   []float32{1, 2, 3, 4, 5, 6},
			rows:  2,
			cols:  3,
			ldDst: 3,
			ldSrc: 3,
			want:  []float32{1, 2, 3, 4, 5, 6},
		},
		{
			name:  "strided",
			src:   []float32{1, 2, 0, 3, 4, 0, 5, 6},
			rows:  2,
			cols:  2,
			ldDst: 2,
			ldSrc: 3,
			want:  []float32{1, 2, 3, 4},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, len(tt.want))
			ElemMatCopyStrided(dst, tt.src, tt.rows, tt.cols, tt.ldDst, tt.ldSrc)
			for i := 0; i < len(tt.want); i++ {
				if dst[i] != tt.want[i] {
					t.Errorf("ElemMatCopyStrided() dst[%d] = %v, want %v", i, dst[i], tt.want[i])
				}
			}
		})
	}
}
