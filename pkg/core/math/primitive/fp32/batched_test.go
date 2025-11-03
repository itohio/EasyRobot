package fp32

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestGemmBatched(t *testing.T) {
	tests := []struct {
		name                      string
		c, a, b                   []float32
		ldC, ldA, ldB             int
		M, N, K                   int
		alpha, beta               float32
		batchCount                int
		stridea, strideb, stridec int
		wantC                     []float32
	}{
		{
			name: "simple 2 batches, 2x2 matrices",
			// Batch 0: A[0] = [1 2], B[0] = [5 6], C[0] = [0 0]
			//          [3 4]       [7 8]       [0 0]
			// A[0]*B[0] = [19 22]
			//             [43 50]
			// Batch 1: A[1] = [2 3], B[1] = [1 1], C[1] = [0 0]
			//          [4 5]       [1 1]       [0 0]
			// A[1]*B[1] = [5 5]
			//             [9 9]
			a:          []float32{1, 2, 3, 4, 2, 3, 4, 5}, // Batch 0: row0, row1; Batch 1: row0, row1
			b:          []float32{5, 6, 7, 8, 1, 1, 1, 1}, // Batch 0: row0, row1; Batch 1: row0, row1
			c:          []float32{0, 0, 0, 0, 0, 0, 0, 0}, // Batch 0: row0, row1; Batch 1: row0, row1
			ldA:        2,
			ldB:        2,
			ldC:        2,
			M:          2,
			N:          2,
			K:          2,
			alpha:      1.0,
			beta:       0.0,
			batchCount: 2,
			stridea:    4, // 2 rows * 2 cols = 4 elements per matrix
			strideb:    4,
			stridec:    4,
			wantC:      []float32{19, 22, 43, 50, 5, 5, 9, 9}, // Batch 0 result, Batch 1 result
		},
		{
			name:       "with alpha and beta",
			a:          []float32{1, 2, 3, 4, 2, 3, 4, 5},
			b:          []float32{1, 1, 1, 1, 1, 1, 1, 1},
			c:          []float32{10, 20, 30, 40, 50, 60, 70, 80},
			ldA:        2,
			ldB:        2,
			ldC:        2,
			M:          2,
			N:          2,
			K:          2,
			alpha:      2.0,
			beta:       3.0,
			batchCount: 2,
			stridea:    4,
			strideb:    4,
			stridec:    4,
			// Batch 0: A[0]*B[0] = [3 3], 2*A[0]*B[0] = [6 6], 3*C[0] = [30 60]
			//                      [7 7]                  [14 14]         [90 120]
			// Result: [36 66]
			//         [104 134]
			// Batch 1: A[1]*B[1] = [5 5], 2*A[1]*B[1] = [10 10], 3*C[1] = [150 180]
			//                      [9 9]                  [18 18]           [210 240]
			// Result: [160 190]
			//         [228 258]
			wantC: []float32{36, 66, 104, 134, 160, 190, 228, 258},
		},
		{
			name: "with leading dimension padding",
			// Batch 0: A[0] = [1 2] stored with ldA=3: [1, 2, _, 3, 4, _]
			//          [3 4]      Row 0: [1,2,0], Row 1: [3,4,0]
			// Batch 1: A[1] = [2 3] stored with ldA=3: [2, 3, _, 4, 5, _]
			//          [4 5]      Row 0: [2,3,0], Row 1: [4,5,0]
			// B[0] = [1 1] (K=2, N=2, ldB=2): [1, 1, 1, 1]
			//        [1 1]      Row 0: [1,1], Row 1: [1,1]
			// B[1] = [1 1] (K=2, N=2, ldB=2): [1, 1, 1, 1]
			//        [1 1]
			a:          []float32{1, 2, 0, 3, 4, 0, 2, 3, 0, 4, 5, 0}, // Batch 0: 6 elems, Batch 1: 6 elems
			b:          []float32{1, 1, 1, 1, 1, 1, 1, 1},             // Batch 0: 4 elems, Batch 1: 4 elems
			c:          []float32{0, 0, 0, 0, 0, 0, 0, 0},             // Batch 0: 4 elems, Batch 1: 4 elems
			ldA:        3,
			ldB:        2,
			ldC:        2,
			M:          2,
			N:          2,
			K:          2,
			alpha:      1.0,
			beta:       0.0,
			batchCount: 2,
			stridea:    6, // 2 rows * 3 ldA = 6 elements per matrix
			strideb:    4, // 2 rows * 2 ldB = 4 elements per matrix
			stridec:    4, // 2 rows * 2 ldC = 4 elements per matrix
			// A[0]*B[0] = [1*1+2*1, 1*1+2*1] = [3 3]
			//             [3*1+4*1, 3*1+4*1]   [7 7]
			// A[1]*B[1] = [2*1+3*1, 2*1+3*1] = [5 5]
			//             [4*1+5*1, 4*1+5*1]   [9 9]
			wantC: []float32{3, 3, 7, 7, 5, 5, 9, 9},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Make a copy of c for testing
			cCopy := make([]float32, len(tt.c))
			copy(cCopy, tt.c)

			GemmBatched(cCopy, tt.a, tt.b, tt.ldC, tt.ldA, tt.ldB, tt.M, tt.N, tt.K, tt.alpha, tt.beta, tt.batchCount, tt.stridea, tt.strideb, tt.stridec)
			assert.InDeltaSlice(t, tt.wantC, cCopy, 1e-5)
		})
	}
}

func TestGemmStrided(t *testing.T) {
	tests := []struct {
		name                      string
		c, a, b                   []float32
		ldC, ldA, ldB             int
		M, N, K                   int
		alpha, beta               float32
		batchCount                int
		stridea, strideb, stridec int
		wantC                     []float32
	}{
		{
			name: "simple 2 batches, 2x2 matrices",
			// Same as GemmBatched test
			a:          []float32{1, 2, 3, 4, 2, 3, 4, 5},
			b:          []float32{5, 6, 7, 8, 1, 1, 1, 1},
			c:          []float32{0, 0, 0, 0, 0, 0, 0, 0},
			ldA:        2,
			ldB:        2,
			ldC:        2,
			M:          2,
			N:          2,
			K:          2,
			alpha:      1.0,
			beta:       0.0,
			batchCount: 2,
			stridea:    4,
			strideb:    4,
			stridec:    4,
			wantC:      []float32{19, 22, 43, 50, 5, 5, 9, 9},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Make a copy of c for testing
			cCopy := make([]float32, len(tt.c))
			copy(cCopy, tt.c)

			GemmStrided(cCopy, tt.a, tt.b, tt.ldC, tt.ldA, tt.ldB, tt.M, tt.N, tt.K, tt.alpha, tt.beta, tt.batchCount, tt.stridea, tt.strideb, tt.stridec)
			assert.InDeltaSlice(t, tt.wantC, cCopy, 1e-5)
		})
	}
}

func TestGemvBatched(t *testing.T) {
	tests := []struct {
		name                      string
		y, a, x                   []float32
		ldA                       int
		M, N                      int
		alpha, beta               float32
		batchCount                int
		strideA, strideX, strideY int
		wantY                     []float32
	}{
		{
			name: "simple 2 batches, 2x3 matrices",
			// Batch 0: A[0] = [1 2 3], x[0] = [1], y[0] = [0]
			//                [4 5 6]       [1]       [0]
			//                              [1]
			// A[0]*x[0] = [6]
			//             [15]
			// Batch 1: A[1] = [2 3 4], x[1] = [1], y[1] = [0]
			//                [5 6 7]       [0]       [0]
			//                              [1]
			// A[1]*x[1] = [6]
			//             [12]
			a:          []float32{1, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 7}, // Batch 0: row0, row1; Batch 1: row0, row1
			x:          []float32{1, 1, 1, 1, 0, 1},                   // Batch 0: [1,1,1]; Batch 1: [1,0,1]
			y:          []float32{0, 0, 0, 0},                         // Batch 0: [0,0]; Batch 1: [0,0]
			ldA:        3,
			M:          2,
			N:          3,
			alpha:      1.0,
			beta:       0.0,
			batchCount: 2,
			strideA:    6,                       // 2 rows * 3 cols = 6 elements per matrix
			strideX:    3,                       // 3 elements per vector
			strideY:    2,                       // 2 elements per vector
			wantY:      []float32{6, 15, 6, 12}, // Batch 0: [6,15]; Batch 1: [6,12]
		},
		{
			name: "with alpha and beta",
			// Batch 0: A[0] = [1 2], x[0] = [1], y[0] = [10]
			//          [3 4]       [1]       [20]
			// Batch 1: A[1] = [2 3], x[1] = [1], y[1] = [30]
			//          [4 5]       [1]       [40]
			a:          []float32{1, 2, 3, 4, 2, 3, 4, 5}, // Batch 0: [1,2,3,4], Batch 1: [2,3,4,5]
			x:          []float32{1, 1, 1, 1},             // Batch 0: [1,1], Batch 1: [1,1]
			y:          []float32{10, 20, 30, 40},         // Batch 0: [10,20], Batch 1: [30,40]
			ldA:        2,
			M:          2,
			N:          2,
			alpha:      2.0,
			beta:       3.0,
			batchCount: 2,
			strideA:    4, // 2 rows * 2 cols = 4
			strideX:    2, // 2 elements per vector
			strideY:    2, // 2 elements per vector
			// Batch 0: A[0]*x[0] = [1*1+2*1] = [3], 2*A[0]*x[0] = [6], 3*y[0] = [30]
			//                      [3*1+4*1]     [7]                 [14]         [60]
			// Result: [36]
			//         [74]
			// Batch 1: A[1]*x[1] = [2*1+3*1] = [5], 2*A[1]*x[1] = [10], 3*y[1] = [90]
			//                      [4*1+5*1]     [9]                 [18]         [120]
			// Result: [100]
			//         [138]
			wantY: []float32{36, 74, 100, 138},
		},
		{
			name: "with leading dimension padding",
			// Batch 0: A[0] = [1 2] stored with ldA=3: [1, 2, _, 3, 4, _]
			//          [3 4]
			a:          []float32{1, 2, 0, 3, 4, 0, 2, 3, 0, 4, 5, 0},
			x:          []float32{1, 1, 1, 1},
			y:          []float32{0, 0, 0, 0},
			ldA:        3,
			M:          2,
			N:          2,
			alpha:      1.0,
			beta:       0.0,
			batchCount: 2,
			strideA:    6,
			strideX:    2,
			strideY:    2,
			// A[0]*x[0] = [3]
			//             [7]
			// A[1]*x[1] = [5]
			//             [9]
			wantY: []float32{3, 7, 5, 9},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Make a copy of y for testing
			yCopy := make([]float32, len(tt.y))
			copy(yCopy, tt.y)

			GemvBatched(yCopy, tt.a, tt.x, tt.ldA, tt.M, tt.N, tt.alpha, tt.beta, tt.batchCount, tt.strideA, tt.strideX, tt.strideY)
			assert.InDeltaSlice(t, tt.wantY, yCopy, 1e-5)
		})
	}
}

func TestBatchedEmpty(t *testing.T) {
	a := []float32{1, 2, 3, 4}
	b := []float32{5, 6, 7, 8}
	c := []float32{0, 0, 0, 0}
	x := []float32{1, 2}
	y := []float32{0, 0}

	require.NotPanics(t, func() {
		// Empty batches should not panic
		GemmBatched(c, a, b, 2, 2, 2, 2, 2, 2, 1.0, 0.0, 0, 4, 4, 4)
		GemmStrided(c, a, b, 2, 2, 2, 2, 2, 2, 1.0, 0.0, 0, 4, 4, 4)
		GemvBatched(y, a, x, 2, 2, 2, 1.0, 0.0, 0, 4, 2, 2)
		// Zero dimensions should not panic
		GemmBatched(c, a, b, 2, 2, 2, 0, 2, 2, 1.0, 0.0, 1, 4, 4, 4)
		GemmBatched(c, a, b, 2, 2, 2, 2, 0, 2, 1.0, 0.0, 1, 4, 4, 4)
		GemvBatched(y, a, x, 2, 0, 2, 1.0, 0.0, 1, 4, 2, 2)
	})
}
