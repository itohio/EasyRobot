package primitive

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestGemv_N(t *testing.T) {
	tests := []struct {
		name        string
		y, a, x     []float32
		ldA, M, N   int
		alpha, beta float32
		wantY       []float32
	}{
		{
			name: "simple 2x3 matrix",
			// A = [1 2 3]
			//     [4 5 6]
			// x = [1, 2, 3]^T
			// y = [0, 0]^T
			// Result: y = A*x = [1*1+2*2+3*3, 4*1+5*2+6*3] = [14, 32]
			a:     []float32{1, 2, 3, 4, 5, 6}, // row-major: row0, row1
			x:     []float32{1, 2, 3},
			y:     []float32{0, 0},
			ldA:   3,
			M:     2,
			N:     3,
			alpha: 1.0,
			beta:  0.0,
			wantY: []float32{14, 32}, // [1*1+2*2+3*3, 4*1+5*2+6*3]
		},
		{
			name: "with alpha and beta",
			// y = alpha*A*x + beta*y
			// alpha=2, beta=3, y=[1,1]^T initially
			a:     []float32{1, 2, 3, 4, 5, 6},
			x:     []float32{1, 1, 1},
			y:     []float32{1, 1},
			ldA:   3,
			M:     2,
			N:     3,
			alpha: 2.0,
			beta:  3.0,
			// y[0] = 3*1 + 2*(1*1+2*1+3*1) = 3 + 2*6 = 15
			// y[1] = 3*1 + 2*(4*1+5*1+6*1) = 3 + 2*15 = 33
			wantY: []float32{15, 33},
		},
		{
			name:  "alpha=0",
			a:     []float32{1, 2, 3, 4, 5, 6},
			x:     []float32{1, 1, 1},
			y:     []float32{1, 2},
			ldA:   3,
			M:     2,
			N:     3,
			alpha: 0.0,
			beta:  2.0,
			// y = 2*[1,2] = [2, 4]
			wantY: []float32{2, 4},
		},
		{
			name:  "beta=1 (no scaling)",
			a:     []float32{1, 2, 3, 4, 5, 6},
			x:     []float32{1, 0, 0},
			y:     []float32{10, 20},
			ldA:   3,
			M:     2,
			N:     3,
			alpha: 1.0,
			beta:  1.0,
			// y[0] = 10 + (1*1+2*0+3*0) = 11
			// y[1] = 20 + (4*1+5*0+6*0) = 24
			wantY: []float32{11, 24},
		},
		{
			name: "with leading dimension padding",
			// A is 2x2 but stored with ldA=4
			// A = [1 2]
			//     [3 4]
			// stored as: [1, 2, _, _, 3, 4, _, _]
			a:     []float32{1, 2, 0, 0, 3, 4, 0, 0},
			x:     []float32{1, 1},
			y:     []float32{0, 0},
			ldA:   4,
			M:     2,
			N:     2,
			alpha: 1.0,
			beta:  0.0,
			// y[0] = 1*1 + 2*1 = 3
			// y[1] = 3*1 + 4*1 = 7
			wantY: []float32{3, 7},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Make a copy of y for testing
			yCopy := make([]float32, len(tt.y))
			copy(yCopy, tt.y)

			Gemv_N(yCopy, tt.a, tt.x, tt.ldA, tt.M, tt.N, tt.alpha, tt.beta)
			assert.InDeltaSlice(t, tt.wantY, yCopy, 1e-5)
		})
	}
}

func TestGemv_T(t *testing.T) {
	tests := []struct {
		name        string
		y, a, x     []float32
		ldA, M, N   int
		alpha, beta float32
		wantY       []float32
	}{
		{
			name: "simple 2x3 matrix transpose",
			// A = [1 2 3]
			//     [4 5 6]
			// A^T = [1 4]
			//       [2 5]
			//       [3 6]
			// x = [1, 2]^T
			// y = [0, 0, 0]^T
			// Result: y = A^T*x = [1*1+4*2, 2*1+5*2, 3*1+6*2] = [9, 12, 15]
			a:     []float32{1, 2, 3, 4, 5, 6}, // row-major storage of A
			x:     []float32{1, 2},
			y:     []float32{0, 0, 0},
			ldA:   3,
			M:     2,
			N:     3,
			alpha: 1.0,
			beta:  0.0,
			wantY: []float32{9, 12, 15}, // [1*1+4*2, 2*1+5*2, 3*1+6*2]
		},
		{
			name:  "with alpha and beta",
			a:     []float32{1, 2, 3, 4, 5, 6},
			x:     []float32{1, 1},
			y:     []float32{1, 1, 1},
			ldA:   3,
			M:     2,
			N:     3,
			alpha: 2.0,
			beta:  3.0,
			// Column 0 of A: [1, 4]^T, column 1: [2, 5]^T, column 2: [3, 6]^T
			// y[0] = 3*1 + 2*(1*1+4*1) = 3 + 10 = 13
			// y[1] = 3*1 + 2*(2*1+5*1) = 3 + 14 = 17
			// y[2] = 3*1 + 2*(3*1+6*1) = 3 + 18 = 21
			wantY: []float32{13, 17, 21},
		},
		{
			name:  "alpha=0",
			a:     []float32{1, 2, 3, 4, 5, 6},
			x:     []float32{1, 1},
			y:     []float32{1, 2, 3},
			ldA:   3,
			M:     2,
			N:     3,
			alpha: 0.0,
			beta:  2.0,
			// y = 2*[1,2,3] = [2, 4, 6]
			wantY: []float32{2, 4, 6},
		},
		{
			name:  "beta=1 (no scaling)",
			a:     []float32{1, 2, 3, 4, 5, 6},
			x:     []float32{1, 0},
			y:     []float32{10, 20, 30},
			ldA:   3,
			M:     2,
			N:     3,
			alpha: 1.0,
			beta:  1.0,
			// Column 0 of A: [1, 4]^T
			// y[0] = 10 + (1*1+4*0) = 11
			// y[1] = 20 + (2*1+5*0) = 22
			// y[2] = 30 + (3*1+6*0) = 33
			wantY: []float32{11, 22, 33},
		},
		{
			name: "with leading dimension padding",
			// A is 2x2 but stored with ldA=4
			// A = [1 2]
			//     [3 4]
			// A^T = [1 3]
			//       [2 4]
			// x = [1, 1]^T
			// y = [0, 0]^T
			a:     []float32{1, 2, 0, 0, 3, 4, 0, 0},
			x:     []float32{1, 1},
			y:     []float32{0, 0},
			ldA:   4,
			M:     2,
			N:     2,
			alpha: 1.0,
			beta:  0.0,
			// Column 0 of A: [1, 3]^T, column 1: [2, 4]^T
			// y[0] = 1*1 + 3*1 = 4
			// y[1] = 2*1 + 4*1 = 6
			wantY: []float32{4, 6},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Make a copy of y for testing
			yCopy := make([]float32, len(tt.y))
			copy(yCopy, tt.y)

			Gemv_T(yCopy, tt.a, tt.x, tt.ldA, tt.M, tt.N, tt.alpha, tt.beta)
			assert.InDeltaSlice(t, tt.wantY, yCopy, 1e-5)
		})
	}
}

func TestGer(t *testing.T) {
	tests := []struct {
		name      string
		a, x, y   []float32
		ldA, M, N int
		alpha     float32
		wantA     []float32
	}{
		{
			name: "simple rank-1 update",
			// A = [0 0]
			//     [0 0]
			// x = [1, 2]^T
			// y = [3, 4]^T
			// x*y^T = [1] * [3 4] = [3 4]
			//         [2]           [6 8]
			// A += x*y^T = [3 4]
			//              [6 8]
			a:     []float32{0, 0, 0, 0},
			x:     []float32{1, 2},
			y:     []float32{3, 4},
			ldA:   2,
			M:     2,
			N:     2,
			alpha: 1.0,
			wantA: []float32{3, 4, 6, 8}, // row-major: [3,4], [6,8]
		},
		{
			name: "with alpha",
			// A = [1 1]
			//     [1 1]
			// x = [1, 2]^T
			// y = [1, 1]^T
			// alpha*x*y^T = 2*[1] * [1 1] = 2*[1 1] = [2 2]
			//                 [2]           [2 2]   [4 4]
			// A += 2*x*y^T = [3 3]
			//                [5 5]
			a:     []float32{1, 1, 1, 1},
			x:     []float32{1, 2},
			y:     []float32{1, 1},
			ldA:   2,
			M:     2,
			N:     2,
			alpha: 2.0,
			wantA: []float32{3, 3, 5, 5}, // [1+2*1*1, 1+2*1*1], [1+2*2*1, 1+2*2*1]
		},
		{
			name:  "alpha=0 (no update)",
			a:     []float32{1, 2, 3, 4},
			x:     []float32{10, 20},
			y:     []float32{100, 200},
			ldA:   2,
			M:     2,
			N:     2,
			alpha: 0.0,
			wantA: []float32{1, 2, 3, 4}, // unchanged
		},
		{
			name: "with leading dimension padding",
			// A is 2x2 but stored with ldA=3
			// A = [1 2]
			//     [3 4]
			// stored as: [1, 2, _, 3, 4, _]
			// x = [1, 1]^T
			// y = [1, 1]^T
			// x*y^T = [1 1]
			//         [1 1]
			// A += x*y^T = [2 3]
			//              [4 5]
			a:     []float32{1, 2, 0, 3, 4, 0},
			x:     []float32{1, 1},
			y:     []float32{1, 1},
			ldA:   3,
			M:     2,
			N:     2,
			alpha: 1.0,
			wantA: []float32{2, 3, 0, 4, 5, 0}, // [1+1*1, 2+1*1], [3+1*1, 4+1*1]
		},
		{
			name: "3x2 matrix",
			// A = [0 0]
			//     [0 0]
			//     [0 0]
			// x = [1, 2, 3]^T
			// y = [4, 5]^T
			// x*y^T = [4  5]
			//         [8  10]
			//         [12 15]
			a:     []float32{0, 0, 0, 0, 0, 0},
			x:     []float32{1, 2, 3},
			y:     []float32{4, 5},
			ldA:   2,
			M:     3,
			N:     2,
			alpha: 1.0,
			wantA: []float32{4, 5, 8, 10, 12, 15},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Make a copy of a for testing
			aCopy := make([]float32, len(tt.a))
			copy(aCopy, tt.a)

			Ger(aCopy, tt.x, tt.y, tt.ldA, tt.M, tt.N, tt.alpha)
			assert.InDeltaSlice(t, tt.wantA, aCopy, 1e-5)
		})
	}
}

func TestLevel2Empty(t *testing.T) {
	a := []float32{1, 2, 3, 4, 5, 6}
	x := []float32{1, 2}
	y := []float32{0, 0}

	require.NotPanics(t, func() {
		// M=0 or N=0 should not panic
		Gemv_N(y, a, x, 3, 0, 3, 1.0, 0.0)
		Gemv_N(y, a, x, 3, 2, 0, 1.0, 0.0)
		Gemv_T(y, a, x, 3, 0, 3, 1.0, 0.0)
		Gemv_T(y, a, x, 3, 2, 0, 1.0, 0.0)
		Ger(a, x, x, 2, 0, 2, 1.0)
		Ger(a, x, x, 2, 2, 0, 1.0)
	})
}
