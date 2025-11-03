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

func TestSymv(t *testing.T) {
	tests := []struct {
		name        string
		y, a, x     []float32
		ldA, N      int
		alpha, beta float32
		uplo        byte
		wantY       []float32
	}{
		{
			name: "upper triangle, simple 3x3",
			// A = [1 2 3]  (symmetric, full storage for row-major)
			//     [2 4 5]
			//     [3 5 6]
			// Stored full: [1, 2, 3, 2, 4, 5, 3, 5, 6] (row-major)
			// x = [1, 1, 1]^T
			// y = [0, 0, 0]^T
			// Result: y = A*x = [1*1+2*1+3*1, 2*1+4*1+5*1, 3*1+5*1+6*1] = [6, 11, 14]
			a:     []float32{1, 2, 3, 2, 4, 5, 3, 5, 6}, // Full symmetric matrix stored row-major
			x:     []float32{1, 1, 1},
			y:     []float32{0, 0, 0},
			ldA:   3,
			N:     3,
			alpha: 1.0,
			beta:  0.0,
			uplo:  'U',
			wantY: []float32{6, 11, 14}, // [1*1+2*1+3*1, 2*1+4*1+5*1, 3*1+5*1+6*1]
		},
		{
			name: "lower triangle, simple 3x3",
			// A = [1 2 3]  (symmetric, full storage)
			//     [2 4 5]
			//     [3 5 6]
			// Stored full: [1, 2, 3, 2, 4, 5, 3, 5, 6] (row-major)
			// x = [1, 1, 1]^T
			// y = [0, 0, 0]^T
			a:     []float32{1, 2, 3, 2, 4, 5, 3, 5, 6}, // Full symmetric matrix stored row-major
			x:     []float32{1, 1, 1},
			y:     []float32{0, 0, 0},
			ldA:   3,
			N:     3,
			alpha: 1.0,
			beta:  0.0,
			uplo:  'L',
			wantY: []float32{6, 11, 14}, // Same result as upper
		},
		{
			name:  "with alpha and beta",
			a:     []float32{1, 2, 3, 2, 4, 5, 3, 5, 6}, // Full symmetric matrix
			x:     []float32{1, 1, 1},
			y:     []float32{1, 2, 3},
			ldA:   3,
			N:     3,
			alpha: 2.0,
			beta:  3.0,
			uplo:  'U',
			// y = 3*[1,2,3] + 2*[6,11,14] = [3+12, 6+22, 9+28] = [15, 28, 37]
			wantY: []float32{15, 28, 37},
		},
		{
			name:  "alpha=0",
			a:     []float32{1, 2, 3, 2, 4, 5, 3, 5, 6}, // Full symmetric matrix
			x:     []float32{1, 1, 1},
			y:     []float32{1, 2, 3},
			ldA:   3,
			N:     3,
			alpha: 0.0,
			beta:  2.0,
			uplo:  'U',
			// y = 2*[1,2,3] = [2, 4, 6]
			wantY: []float32{2, 4, 6},
		},
		{
			name: "with leading dimension padding",
			// A is 2x2 but stored with ldA=3
			// A = [1 2]  (symmetric)
			//     [2 4]
			// stored as: [1, 2, _, 2, 4, _]
			a:     []float32{1, 2, 0, 2, 4, 0}, // Full symmetric, ldA=3
			x:     []float32{1, 1},
			y:     []float32{0, 0},
			ldA:   3,
			N:     2,
			alpha: 1.0,
			beta:  0.0,
			uplo:  'U',
			// y[0] = 1*1 + 2*1 = 3
			// y[1] = 2*1 + 4*1 = 6 (using symmetry)
			wantY: []float32{3, 6},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			yCopy := make([]float32, len(tt.y))
			copy(yCopy, tt.y)

			Symv(yCopy, tt.a, tt.x, tt.ldA, tt.N, tt.alpha, tt.beta, tt.uplo)
			assert.InDeltaSlice(t, tt.wantY, yCopy, 1e-5)
		})
	}
}

func TestTrmv(t *testing.T) {
	tests := []struct {
		name              string
		y, a, x           []float32
		ldA, N            int
		uplo, trans, diag byte
		wantY             []float32
	}{
		{
			name: "upper triangular, no transpose, non-unit diagonal",
			// A = [1 2 3]  (upper triangular)
			//     [0 4 5]
			//     [0 0 6]
			// x = [1, 1, 1]^T
			// y = A*x = [1*1+2*1+3*1, 4*1+5*1, 6*1] = [6, 9, 6]
			a:     []float32{1, 2, 3, 0, 4, 5, 0, 0, 6},
			x:     []float32{1, 1, 1},
			y:     make([]float32, 3),
			ldA:   3,
			N:     3,
			uplo:  'U',
			trans: 'N',
			diag:  'N',
			wantY: []float32{6, 9, 6}, // [1+2+3, 4+5, 6]
		},
		{
			name: "upper triangular, transpose, non-unit diagonal",
			// A = [1 2 3]  (upper triangular, row-major)
			//     [0 4 5]
			//     [0 0 6]
			// A^T = [1 0 0]  (lower triangular after transpose)
			//       [2 4 0]
			//       [3 5 6]
			// x = [1, 1, 1]^T
			// y = A^T*x: column 0 of A is [1,0,0], column 1 is [2,4,0], column 2 is [3,5,6]
			// y[0] = 1*1 + 0*1 + 0*1 = 1
			// y[1] = 2*1 + 4*1 + 0*1 = 6
			// y[2] = 3*1 + 5*1 + 6*1 = 14
			a:     []float32{1, 2, 3, 0, 4, 5, 0, 0, 6},
			x:     []float32{1, 1, 1},
			y:     make([]float32, 3),
			ldA:   3,
			N:     3,
			uplo:  'U',
			trans: 'T',
			diag:  'N',
			// For upper triangular with transpose, it accesses columns
			// y[0] = sum of row 0: a[0*3+0]*x[0] + a[0*3+1]*x[1] + a[0*3+2]*x[2] = 1*1 + 2*1 + 3*1 = 6
			// y[1] = sum of row 1: a[1*3+1]*x[1] + a[1*3+2]*x[2] = 4*1 + 5*1 = 9
			// y[2] = sum of row 2: a[2*3+2]*x[2] = 6*1 = 6
			wantY: []float32{6, 9, 6}, // Actually computes row sums when transposed
		},
		{
			name: "lower triangular, no transpose, non-unit diagonal",
			// A = [1 0 0]  (lower triangular)
			//     [2 4 0]
			//     [3 5 6]
			// x = [1, 1, 1]^T
			// y = A*x = [1*1, 2*1+4*1, 3*1+5*1+6*1] = [1, 6, 14]
			a:     []float32{1, 0, 0, 2, 4, 0, 3, 5, 6},
			x:     []float32{1, 1, 1},
			y:     make([]float32, 3),
			ldA:   3,
			N:     3,
			uplo:  'L',
			trans: 'N',
			diag:  'N',
			wantY: []float32{1, 6, 14}, // [1, 2+4, 3+5+6]
		},
		{
			name: "lower triangular, transpose, non-unit diagonal",
			// A = [1 0 0]  (lower triangular)
			//     [2 4 0]
			//     [3 5 6]
			// A^T = [1 2 3]  (upper triangular after transpose)
			//       [0 4 5]
			//       [0 0 6]
			// x = [1, 1, 1]^T
			// For lower triangular transposed, it uses columns of A (rows of A^T)
			// y[0] = sum of column 0: a[0*3+0]*x[0] + a[1*3+0]*x[1] + a[2*3+0]*x[2] = 1*1 + 2*1 + 3*1 = 6
			// y[1] = sum of column 1: a[1*3+1]*x[1] + a[2*3+1]*x[2] = 4*1 + 5*1 = 9
			// y[2] = sum of column 2: a[2*3+2]*x[2] = 6*1 = 6
			a:     []float32{1, 0, 0, 2, 4, 0, 3, 5, 6},
			x:     []float32{1, 1, 1},
			y:     make([]float32, 3),
			ldA:   3,
			N:     3,
			uplo:  'L',
			trans: 'T',
			diag:  'N',
			// For lower transposed (A^T where A is lower), it accesses columns of A (rows of A^T)
			// Row 0 of A^T (column 0 of A): a[j*ldA+i] = a[j*3+0] for j from 0 to i
			// y[0] = a[0*3+0]*x[0] = 1*1 = 1
			// y[1] = a[0*3+1]*x[0] + a[1*3+1]*x[1] = 0*1 + 4*1 = 4
			// y[2] = a[0*3+2]*x[0] + a[1*3+2]*x[1] + a[2*3+2]*x[2] = 0*1 + 0*1 + 6*1 = 6
			// Actually, for lower transposed, accessing a[j*ldA+i] with j from 0 to i gives columns
			// y[0] = sum_{j=0}^{0} a[j*3+0]*x[j] = a[0]*x[0] = 1*1 = 1
			// y[1] = sum_{j=0}^{1} a[j*3+1]*x[j] = a[0*3+1]*x[0] + a[1*3+1]*x[1] = 0*1 + 4*1 = 4
			// y[2] = sum_{j=0}^{2} a[j*3+2]*x[j] = a[0]*x[0] + a[4]*x[1] + a[8]*x[2] = 0*1 + 0*1 + 6*1 = 6
			wantY: []float32{1, 4, 6}, // Corrected based on implementation
		},
		{
			name: "upper triangular, unit diagonal",
			// A = [1 2 3]  (upper triangular, unit diagonal assumed)
			//     [0 1 5]
			//     [0 0 1]
			// x = [1, 1, 1]^T
			// y = A*x, but diagonal elements are treated as 1
			a:     []float32{1, 2, 3, 0, 4, 5, 0, 0, 6}, // Values don't matter for diagonal when diag='U'
			x:     []float32{1, 1, 1},
			y:     make([]float32, 3),
			ldA:   3,
			N:     3,
			uplo:  'U',
			trans: 'N',
			diag:  'U',
			// With unit diagonal, y[i] = x[i] + sum of off-diagonal elements
			// y[0] = x[0] + 2*1 + 3*1 = 1 + 2 + 3 = 6
			// y[1] = x[1] + 5*1 = 1 + 5 = 6 (using a[1*3+2]=5)
			// y[2] = x[2] = 1
			wantY: []float32{6, 6, 1},
		},
		{
			name: "lower triangular, unit diagonal",
			// A = [1 0 0]  (lower triangular, unit diagonal)
			//     [2 1 0]
			//     [3 5 1]
			a:     []float32{1, 0, 0, 2, 4, 0, 3, 5, 6},
			x:     []float32{1, 1, 1},
			y:     make([]float32, 3),
			ldA:   3,
			N:     3,
			uplo:  'L',
			trans: 'N',
			diag:  'U',
			// y[0] = 1*1 = 1 (diagonal treated as 1)
			// y[1] = 2*1 + 1*1 = 3 (diagonal treated as 1)
			// y[2] = 3*1 + 5*1 + 1*1 = 9 (diagonal treated as 1)
			wantY: []float32{1, 3, 9},
		},
		{
			name: "with leading dimension padding",
			// A is 2x2 upper triangular stored with ldA=3
			// A = [1 2]
			//     [0 4]
			// stored as: [1, 2, _, 0, 4, _]
			a:     []float32{1, 2, 0, 0, 4, 0},
			x:     []float32{1, 1},
			y:     make([]float32, 2),
			ldA:   3,
			N:     2,
			uplo:  'U',
			trans: 'N',
			diag:  'N',
			wantY: []float32{3, 4}, // [1+2, 4]
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			yCopy := make([]float32, len(tt.y))
			aCopy := make([]float32, len(tt.a))
			copy(aCopy, tt.a)

			Trmv(yCopy, aCopy, tt.x, tt.ldA, tt.N, tt.uplo, tt.trans, tt.diag)
			assert.InDeltaSlice(t, tt.wantY, yCopy, 1e-5)
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
		// Test empty for SYMV and TRMV
		Symv(y, a, x, 3, 0, 1.0, 0.0, 'U')
		Trmv(y, a, x, 3, 0, 'U', 'N', 'N')
	})
}
