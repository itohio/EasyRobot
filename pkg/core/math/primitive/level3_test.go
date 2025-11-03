package primitive

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestGemm_NN(t *testing.T) {
	tests := []struct {
		name          string
		c, a, b       []float32
		ldC, ldA, ldB int
		M, N, K       int
		alpha, beta   float32
		wantC         []float32
	}{
		{
			name: "simple 2x2 matrices",
			// A = [1 2]  (2x2, M=2, K=2)
			//     [3 4]
			// B = [5 6]  (2x2, K=2, N=2)
			//     [7 8]
			// C = [0 0]  (2x2, M=2, N=2)
			//     [0 0]
			// A*B = [1*5+2*7  1*6+2*8] = [19 22]
			//       [3*5+4*7  3*6+4*8]   [43 50]
			a:     []float32{1, 2, 3, 4}, // row-major: row0, row1
			b:     []float32{5, 6, 7, 8}, // row-major: row0, row1
			c:     []float32{0, 0, 0, 0}, // row-major: row0, row1
			ldA:   2,
			ldB:   2,
			ldC:   2,
			M:     2,
			N:     2,
			K:     2,
			alpha: 1.0,
			beta:  0.0,
			wantC: []float32{19, 22, 43, 50}, // [1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]
		},
		{
			name:  "with alpha and beta",
			a:     []float32{1, 2, 3, 4},
			b:     []float32{1, 1, 1, 1},
			c:     []float32{10, 20, 30, 40},
			ldA:   2,
			ldB:   2,
			ldC:   2,
			M:     2,
			N:     2,
			K:     2,
			alpha: 2.0,
			beta:  3.0,
			// A*B = [3 3]  (2*A*B = [6 6])
			//       [7 7]            [14 14]
			// beta*C = [30 60]  (3*C)
			//          [90 120]
			// Result = [36 66]  (2*A*B + 3*C)
			//          [104 134]
			wantC: []float32{36, 66, 104, 134},
		},
		{
			name: "2x3 and 3x2 matrices",
			// A = [1 2 3]  (2x3, M=2, K=3)
			//     [4 5 6]
			// B = [1 1]    (3x2, K=3, N=2)
			//     [1 1]
			//     [1 1]
			// C = [0 0]    (2x2, M=2, N=2)
			//     [0 0]
			// A*B = [1*1+2*1+3*1  1*1+2*1+3*1] = [6 6]
			//       [4*1+5*1+6*1  4*1+5*1+6*1]   [15 15]
			a:     []float32{1, 2, 3, 4, 5, 6},
			b:     []float32{1, 1, 1, 1, 1, 1},
			c:     []float32{0, 0, 0, 0},
			ldA:   3,
			ldB:   2,
			ldC:   2,
			M:     2,
			N:     2,
			K:     3,
			alpha: 1.0,
			beta:  0.0,
			wantC: []float32{6, 6, 15, 15},
		},
		{
			name:  "alpha=0",
			a:     []float32{1, 2, 3, 4},
			b:     []float32{5, 6, 7, 8},
			c:     []float32{1, 2, 3, 4},
			ldA:   2,
			ldB:   2,
			ldC:   2,
			M:     2,
			N:     2,
			K:     2,
			alpha: 0.0,
			beta:  2.0,
			// C = 2*[1 2] = [2 4]
			//       [3 4]   [6 8]
			wantC: []float32{2, 4, 6, 8},
		},
		{
			name: "with leading dimension padding",
			// A is 2x2 but stored with ldA=3
			// A = [1 2]
			//     [3 4]
			// stored as: [1, 2, _, 3, 4, _]
			a:     []float32{1, 2, 0, 3, 4, 0},
			b:     []float32{1, 1, 1, 1},
			c:     []float32{0, 0, 0, 0},
			ldA:   3,
			ldB:   2,
			ldC:   2,
			M:     2,
			N:     2,
			K:     2,
			alpha: 1.0,
			beta:  0.0,
			// A*B = [1*1+2*1  1*1+2*1] = [3 3]
			//       [3*1+4*1  3*1+4*1]   [7 7]
			wantC: []float32{3, 3, 7, 7},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Make a copy of c for testing
			cCopy := make([]float32, len(tt.c))
			copy(cCopy, tt.c)

			Gemm_NN(cCopy, tt.a, tt.b, tt.ldC, tt.ldA, tt.ldB, tt.M, tt.N, tt.K, tt.alpha, tt.beta)
			assert.InDeltaSlice(t, tt.wantC, cCopy, 1e-5)
		})
	}
}

func TestGemm_NT(t *testing.T) {
	tests := []struct {
		name          string
		c, a, b       []float32
		ldC, ldA, ldB int
		M, N, K       int
		alpha, beta   float32
		wantC         []float32
	}{
		{
			name: "simple 2x2 matrices",
			// A = [1 2]  (2x2, M=2, K=2)
			//     [3 4]
			// B = [5 7]  (2x2, N=2, K=2) stored as B^T would be [5 6]
			//     [6 8]                                         [7 8]
			// But B is stored row-major as: [5, 7, 6, 8]
			// B^T = [5 6]  (2x2, K=2, N=2)
			//       [7 8]
			// C = [0 0]  (2x2, M=2, N=2)
			//     [0 0]
			// A*B^T = [1*5+2*7  1*6+2*8] = [19 22]
			//         [3*5+4*7  3*6+4*8]   [43 50]
			// B stored as N×K: [5 7] (row 0)
			//                  [6 8] (row 1)
			// B^T is K×N: [5 6] (row 0)
			//             [7 8] (row 1)
			a:     []float32{1, 2, 3, 4}, // A: M×K = 2×2
			b:     []float32{5, 7, 6, 8}, // B: N×K = 2×2 (row-major)
			c:     []float32{0, 0, 0, 0},
			ldA:   2,
			ldB:   2, // ldB = K = 2
			ldC:   2,
			M:     2,
			N:     2,
			K:     2,
			alpha: 1.0,
			beta:  0.0,
			// A*B^T: row0 of A dot row0 of B = [1 2]·[5 7] = 1*5+2*7 = 19
			//         row0 of A dot row1 of B = [1 2]·[6 8] = 1*6+2*8 = 22
			//         row1 of A dot row0 of B = [3 4]·[5 7] = 3*5+4*7 = 43
			//         row1 of A dot row1 of B = [3 4]·[6 8] = 3*6+4*8 = 50
			wantC: []float32{19, 22, 43, 50},
		},
		{
			name:  "with alpha and beta",
			a:     []float32{1, 2, 3, 4},
			b:     []float32{1, 1, 1, 1}, // B is N×K = 2×2
			c:     []float32{10, 20, 30, 40},
			ldA:   2,
			ldB:   2,
			ldC:   2,
			M:     2,
			N:     2,
			K:     2,
			alpha: 2.0,
			beta:  3.0,
			// A*B^T: [1 2]·[1 1] = [3 3]
			//        [3 4]·[1 1]   [7 7]
			// 2*A*B^T = [6 6]
			//            [14 14]
			// 3*C = [30 60]
			//       [90 120]
			// Result = [36 66]
			//          [104 134]
			wantC: []float32{36, 66, 104, 134},
		},
		{
			name: "2x3 and 2x3 matrices (B^T is 3x2)",
			// A = [1 2 3]  (2x3, M=2, K=3)
			//     [4 5 6]
			// B = [1 1 1]  (2x3, N=2, K=3) stored as row-major
			//     [2 2 2]
			// B^T = [1 2]  (3x2)
			//       [1 2]
			//       [1 2]
			// C = [0 0]    (2x2, M=2, N=2)
			//     [0 0]
			// A*B^T = [1*1+2*1+3*1  1*2+2*2+3*2] = [6 12]
			//         [4*1+5*1+6*1  4*2+5*2+6*2]   [15 30]
			a:     []float32{1, 2, 3, 4, 5, 6}, // A: M×K = 2×3
			b:     []float32{1, 1, 1, 2, 2, 2}, // B: N×K = 2×3
			c:     []float32{0, 0, 0, 0},
			ldA:   3,
			ldB:   3, // ldB = K = 3
			ldC:   2,
			M:     2,
			N:     2,
			K:     3,
			alpha: 1.0,
			beta:  0.0,
			wantC: []float32{6, 12, 15, 30},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Make a copy of c for testing
			cCopy := make([]float32, len(tt.c))
			copy(cCopy, tt.c)

			Gemm_NT(cCopy, tt.a, tt.b, tt.ldC, tt.ldA, tt.ldB, tt.M, tt.N, tt.K, tt.alpha, tt.beta)
			assert.InDeltaSlice(t, tt.wantC, cCopy, 1e-5)
		})
	}
}

func TestGemm_TN(t *testing.T) {
	tests := []struct {
		name          string
		c, a, b       []float32
		ldC, ldA, ldB int
		M, N, K       int
		alpha, beta   float32
		wantC         []float32
	}{
		{
			name: "simple 2x2 matrices",
			// A = [1 3]  (2x2, K=2, M=2) stored as row-major
			//     [2 4]
			// A^T = [1 2]  (2x2, M=2, K=2)
			//       [3 4]
			// B = [5 6]  (2x2, K=2, N=2)
			//     [7 8]
			// C = [0 0]  (2x2, M=2, N=2)
			//     [0 0]
			// A^T*B = [1*5+2*7  1*6+2*8] = [19 22]
			//         [3*5+4*7  3*6+4*8]   [43 50]
			a:     []float32{1, 3, 2, 4}, // A: K×M = 2×2 (stored as A would be)
			b:     []float32{5, 6, 7, 8}, // B: K×N = 2×2
			c:     []float32{0, 0, 0, 0},
			ldA:   2, // ldA = M = 2
			ldB:   2,
			ldC:   2,
			M:     2,
			N:     2,
			K:     2,
			alpha: 1.0,
			beta:  0.0,
			// A^T*B: column0 of A (row0 of A^T) = [1 2], column1 of A (row1 of A^T) = [3 4]
			// row0 of A^T dot column0 of B = [1 2]·[5 7] = 1*5+2*7 = 19
			// row0 of A^T dot column1 of B = [1 2]·[6 8] = 1*6+2*8 = 22
			// row1 of A^T dot column0 of B = [3 4]·[5 7] = 3*5+4*7 = 43
			// row1 of A^T dot column1 of B = [3 4]·[6 8] = 3*6+4*8 = 50
			wantC: []float32{19, 22, 43, 50},
		},
		{
			name:  "with alpha and beta",
			a:     []float32{1, 3, 2, 4}, // A: K×M = 2×2
			b:     []float32{1, 1, 1, 1}, // B: K×N = 2×2
			c:     []float32{10, 20, 30, 40},
			ldA:   2,
			ldB:   2,
			ldC:   2,
			M:     2,
			N:     2,
			K:     2,
			alpha: 2.0,
			beta:  3.0,
			// A^T*B = [3 3]
			//         [7 7]
			// 2*A^T*B = [6 6]
			//            [14 14]
			// 3*C = [30 60]
			//       [90 120]
			// Result = [36 66]
			//          [104 134]
			wantC: []float32{36, 66, 104, 134},
		},
		{
			name: "3x2 and 3x2 matrices",
			// A = [1 4]  (3x2, K=3, M=2)
			//     [2 5]
			//     [3 6]
			// A^T = [1 2 3]  (2x3, M=2, K=3)
			//       [4 5 6]
			// B = [1 1]    (3x2, K=3, N=2)
			//     [1 1]
			//     [1 1]
			// C = [0 0]    (2x2, M=2, N=2)
			//     [0 0]
			// A^T*B = [1*1+2*1+3*1  1*1+2*1+3*1] = [6 6]
			//         [4*1+5*1+6*1  4*1+5*1+6*1]   [15 15]
			a:     []float32{1, 4, 2, 5, 3, 6}, // A: K×M = 3×2
			b:     []float32{1, 1, 1, 1, 1, 1}, // B: K×N = 3×2
			c:     []float32{0, 0, 0, 0},
			ldA:   2, // ldA = M = 2
			ldB:   2,
			ldC:   2,
			M:     2,
			N:     2,
			K:     3,
			alpha: 1.0,
			beta:  0.0,
			wantC: []float32{6, 6, 15, 15},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Make a copy of c for testing
			cCopy := make([]float32, len(tt.c))
			copy(cCopy, tt.c)

			Gemm_TN(cCopy, tt.a, tt.b, tt.ldC, tt.ldA, tt.ldB, tt.M, tt.N, tt.K, tt.alpha, tt.beta)
			assert.InDeltaSlice(t, tt.wantC, cCopy, 1e-5)
		})
	}
}

func TestGemm_TT(t *testing.T) {
	tests := []struct {
		name          string
		c, a, b       []float32
		ldC, ldA, ldB int
		M, N, K       int
		alpha, beta   float32
		wantC         []float32
	}{
		{
			name: "simple 2x2 matrices",
			// A = [1 3]  (2x2, K=2, M=2)
			//     [2 4]
			// A^T = [1 2]  (2x2, M=2, K=2)
			//       [3 4]
			// B = [5 7]  (2x2, N=2, K=2)
			//     [6 8]
			// B^T = [5 6]  (2x2, K=2, N=2)
			//       [7 8]
			// C = [0 0]  (2x2, M=2, N=2)
			//     [0 0]
			// A^T*B^T = [1*5+2*7  1*6+2*8] = [19 22]
			//           [3*5+4*7  3*6+4*8]   [43 50]
			a:     []float32{1, 3, 2, 4}, // A: K×M = 2×2
			b:     []float32{5, 7, 6, 8}, // B: N×K = 2×2
			c:     []float32{0, 0, 0, 0},
			ldA:   2, // ldA = M = 2
			ldB:   2, // ldB = K = 2
			ldC:   2,
			M:     2,
			N:     2,
			K:     2,
			alpha: 1.0,
			beta:  0.0,
			// A^T*B^T: column0 of A (row0 of A^T) = [1 2]
			// row0 of A^T dot row0 of B = [1 2]·[5 7] = 1*5+2*7 = 19
			// row0 of A^T dot row1 of B = [1 2]·[6 8] = 1*6+2*8 = 22
			// column1 of A (row1 of A^T) = [3 4]
			// row1 of A^T dot row0 of B = [3 4]·[5 7] = 3*5+4*7 = 43
			// row1 of A^T dot row1 of B = [3 4]·[6 8] = 3*6+4*8 = 50
			wantC: []float32{19, 22, 43, 50},
		},
		{
			name:  "with alpha and beta",
			a:     []float32{1, 3, 2, 4}, // A: K×M = 2×2
			b:     []float32{1, 1, 1, 1}, // B: N×K = 2×2
			c:     []float32{10, 20, 30, 40},
			ldA:   2,
			ldB:   2,
			ldC:   2,
			M:     2,
			N:     2,
			K:     2,
			alpha: 2.0,
			beta:  3.0,
			// A^T*B^T = [3 3]
			//           [7 7]
			// 2*A^T*B^T = [6 6]
			//              [14 14]
			// 3*C = [30 60]
			//       [90 120]
			// Result = [36 66]
			//          [104 134]
			wantC: []float32{36, 66, 104, 134},
		},
		{
			name: "3x2 and 2x3 matrices",
			// A = [1 4]  (3x2, K=3, M=2)
			//     [2 5]
			//     [3 6]
			// A^T = [1 2 3]  (2x3, M=2, K=3)
			//       [4 5 6]
			// B = [1 1 1]  (2x3, N=2, K=3)
			//     [2 2 2]
			// B^T = [1 2]  (3x2, K=3, N=2)
			//       [1 2]
			//       [1 2]
			// C = [0 0]    (2x2, M=2, N=2)
			//     [0 0]
			// A^T*B^T = [1*1+2*1+3*1  1*2+2*2+3*2] = [6 12]
			//           [4*1+5*1+6*1  4*2+5*2+6*2]   [15 30]
			a:     []float32{1, 4, 2, 5, 3, 6}, // A: K×M = 3×2
			b:     []float32{1, 1, 1, 2, 2, 2}, // B: N×K = 2×3
			c:     []float32{0, 0, 0, 0},
			ldA:   2, // ldA = M = 2
			ldB:   3, // ldB = K = 3
			ldC:   2,
			M:     2,
			N:     2,
			K:     3,
			alpha: 1.0,
			beta:  0.0,
			wantC: []float32{6, 12, 15, 30},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Make a copy of c for testing
			cCopy := make([]float32, len(tt.c))
			copy(cCopy, tt.c)

			Gemm_TT(cCopy, tt.a, tt.b, tt.ldC, tt.ldA, tt.ldB, tt.M, tt.N, tt.K, tt.alpha, tt.beta)
			assert.InDeltaSlice(t, tt.wantC, cCopy, 1e-5)
		})
	}
}

func TestLevel3Empty(t *testing.T) {
	a := []float32{1, 2, 3, 4}
	b := []float32{5, 6, 7, 8}
	c := []float32{0, 0, 0, 0}

	require.NotPanics(t, func() {
		// M=0, N=0, or K=0 should not panic
		Gemm_NN(c, a, b, 2, 2, 2, 0, 2, 2, 1.0, 0.0)
		Gemm_NN(c, a, b, 2, 2, 2, 2, 0, 2, 1.0, 0.0)
		Gemm_NN(c, a, b, 2, 2, 2, 2, 2, 0, 1.0, 0.0)
		Gemm_NT(c, a, b, 2, 2, 2, 0, 2, 2, 1.0, 0.0)
		Gemm_TN(c, a, b, 2, 2, 2, 2, 0, 2, 1.0, 0.0)
		Gemm_TT(c, a, b, 2, 2, 2, 2, 2, 0, 1.0, 0.0)
	})
}
