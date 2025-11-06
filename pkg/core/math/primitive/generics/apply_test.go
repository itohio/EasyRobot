package generics

import (
	"testing"
)

func TestElemApplyUnary(t *testing.T) {
	tests := []struct {
		name     string
		src      []float32
		shape    []int
		stridesD []int
		stridesS []int
		op       func(float32) float32
		want     []float32
	}{
		{
			name:     "negate contiguous",
			src:      []float32{1, 2, 3, 4},
			shape:    []int{4},
			stridesD: nil,
			stridesS: nil,
			op:       func(x float32) float32 { return -x },
			want:     []float32{-1, -2, -3, -4},
		},
		{
			name:     "square contiguous",
			src:      []float32{2, 3, 4},
			shape:    []int{3},
			stridesD: nil,
			stridesS: nil,
			op:       func(x float32) float32 { return x * x },
			want:     []float32{4, 9, 16},
		},
		{
			name:     "2D contiguous",
			src:      []float32{1, 2, 3, 4},
			shape:    []int{2, 2},
			stridesD: nil,
			stridesS: nil,
			op:       func(x float32) float32 { return x * 2 },
			want:     []float32{2, 4, 6, 8},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, len(tt.src))
			size := SizeFromShape(tt.shape)
			if tt.stridesD == nil && tt.stridesS == nil && IsContiguous(nil, tt.shape) {
				ElemApplyUnary(dst, tt.src, size, tt.op)
			} else {
				ElemApplyUnaryStrided(dst, tt.src, tt.shape, tt.stridesD, tt.stridesS, tt.op)
			}

			for i := 0; i < size; i++ {
				if dst[i] != tt.want[i] {
					t.Errorf("ElemApplyUnary() dst[%d] = %v, want %v", i, dst[i], tt.want[i])
				}
			}
		})
	}
}

func TestElemApplyBinary(t *testing.T) {
	tests := []struct {
		name     string
		a        []float32
		b        []float32
		shape    []int
		stridesD []int
		stridesA []int
		stridesB []int
		op       func(float32, float32) float32
		want     []float32
	}{
		{
			name:     "add contiguous",
			a:        []float32{1, 2, 3},
			b:        []float32{4, 5, 6},
			shape:    []int{3},
			stridesD: nil,
			stridesA: nil,
			stridesB: nil,
			op:       func(x, y float32) float32 { return x + y },
			want:     []float32{5, 7, 9},
		},
		{
			name:     "multiply contiguous",
			a:        []float32{2, 3, 4},
			b:        []float32{5, 6, 7},
			shape:    []int{3},
			stridesD: nil,
			stridesA: nil,
			stridesB: nil,
			op:       func(x, y float32) float32 { return x * y },
			want:     []float32{10, 18, 28},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, len(tt.want))
			size := SizeFromShape(tt.shape)
			if tt.stridesD == nil && tt.stridesA == nil && tt.stridesB == nil && IsContiguous(nil, tt.shape) {
				ElemApplyBinary(dst, tt.a, tt.b, size, tt.op)
			} else {
				ElemApplyBinaryStrided(dst, tt.a, tt.b, tt.shape, tt.stridesD, tt.stridesA, tt.stridesB, tt.op)
			}

			for i := 0; i < size; i++ {
				if dst[i] != tt.want[i] {
					t.Errorf("ElemApplyBinary() dst[%d] = %v, want %v", i, dst[i], tt.want[i])
				}
			}
		})
	}
}

func TestElemApplyTernary(t *testing.T) {
	tests := []struct {
		name      string
		condition []float32
		a         []float32
		b         []float32
		shape     []int
		op        func(float32, float32, float32) float32
		want      []float32
	}{
		{
			name:      "where contiguous",
			condition: []float32{1, 0, 1, 0},
			a:         []float32{10, 20, 30, 40},
			b:         []float32{100, 200, 300, 400},
			shape:     []int{4},
			op: func(c, x, y float32) float32 {
				if c > 0 {
					return x
				}
				return y
			},
			want: []float32{10, 200, 30, 400},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, len(tt.want))
			size := SizeFromShape(tt.shape)
			ElemApplyTernary(dst, tt.condition, tt.a, tt.b, size, tt.op)

			for i := 0; i < size; i++ {
				if dst[i] != tt.want[i] {
					t.Errorf("ElemApplyTernary() dst[%d] = %v, want %v", i, dst[i], tt.want[i])
				}
			}
		})
	}
}

func TestElemApplyUnaryScalar(t *testing.T) {
	tests := []struct {
		name     string
		src      []float32
		scalar   float32
		shape    []int
		op       func(float32, float32) float32
		want     []float32
	}{
		{
			name:   "add scalar",
			src:    []float32{1, 2, 3},
			scalar: 10,
			shape:  []int{3},
			op:     func(x, s float32) float32 { return x + s },
			want:   []float32{11, 12, 13},
		},
		{
			name:   "multiply scalar",
			src:    []float32{2, 3, 4},
			scalar: 5,
			shape:  []int{3},
			op:     func(x, s float32) float32 { return x * s },
			want:   []float32{10, 15, 20},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, len(tt.want))
			size := SizeFromShape(tt.shape)
			ElemApplyUnaryScalar(dst, tt.src, tt.scalar, size, tt.op)

			for i := 0; i < size; i++ {
				if dst[i] != tt.want[i] {
					t.Errorf("ElemApplyUnaryScalar() dst[%d] = %v, want %v", i, dst[i], tt.want[i])
				}
			}
		})
	}
}

func TestElemVecApply(t *testing.T) {
	tests := []struct {
		name     string
		src      []float32
		n        int
		strideD  int
		strideS  int
		op       func(float32) float32
		want     []float32
	}{
		{
			name:    "contiguous",
			src:     []float32{1, 2, 3, 4},
			n:       4,
			strideD: 1,
			strideS: 1,
			op:      func(x float32) float32 { return x * 2 },
			want:    []float32{2, 4, 6, 8},
		},
		{
			name:    "strided",
			src:     []float32{1, 0, 2, 0, 3, 0, 4},
			n:       4,
			strideD: 1,
			strideS: 2,
			op:      func(x float32) float32 { return x * 2 },
			want:    []float32{2, 4, 6, 8},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, len(tt.want))
			ElemVecApply(dst, tt.src, tt.n, tt.strideD, tt.strideS, tt.op)

			for i := 0; i < tt.n; i++ {
				if dst[i] != tt.want[i] {
					t.Errorf("ElemVecApply() dst[%d] = %v, want %v", i, dst[i], tt.want[i])
				}
			}
		})
	}
}

func TestElemMatApply(t *testing.T) {
	tests := []struct {
		name  string
		src   []float32
		rows  int
		cols  int
		ldDst int
		ldSrc int
		op    func(float32) float32
		want  []float32
	}{
		{
			name:  "contiguous 2x3",
			src:   []float32{1, 2, 3, 4, 5, 6},
			rows:  2,
			cols:  3,
			ldDst: 3,
			ldSrc: 3,
			op:    func(x float32) float32 { return x * 2 },
			want:  []float32{2, 4, 6, 8, 10, 12},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, len(tt.want))
			ElemMatApply(dst, tt.src, tt.rows, tt.cols, tt.ldDst, tt.ldSrc, tt.op)

			size := tt.rows * tt.cols
			for i := 0; i < size; i++ {
				if dst[i] != tt.want[i] {
					t.Errorf("ElemMatApply() dst[%d] = %v, want %v", i, dst[i], tt.want[i])
				}
			}
		})
	}
}

func TestElemApplyBinaryScalar(t *testing.T) {
	tests := []struct {
		name     string
		a        []float32
		scalar   float32
		shape    []int
		stridesD []int
		stridesA []int
		op       func(float32, float32) float32
		want     []float32
	}{
		{
			name:     "add scalar contiguous",
			a:        []float32{1, 2, 3},
			scalar:   10,
			shape:    []int{3},
			stridesD: nil,
			stridesA: nil,
			op:       func(x, s float32) float32 { return x + s },
			want:     []float32{11, 12, 13},
		},
		{
			name:     "multiply scalar contiguous",
			a:        []float32{2, 3, 4},
			scalar:   5,
			shape:    []int{3},
			stridesD: nil,
			stridesA: nil,
			op:       func(x, s float32) float32 { return x * s },
			want:     []float32{10, 15, 20},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, len(tt.want))
			size := SizeFromShape(tt.shape)
			if tt.stridesD == nil && tt.stridesA == nil && IsContiguous(nil, tt.shape) {
				ElemApplyBinaryScalar(dst, tt.a, tt.scalar, size, tt.op)
			} else {
				ElemApplyBinaryScalarStrided(dst, tt.a, tt.scalar, tt.shape, tt.stridesD, tt.stridesA, tt.op)
			}

			for i := 0; i < size; i++ {
				if dst[i] != tt.want[i] {
					t.Errorf("ElemApplyBinaryScalar() dst[%d] = %v, want %v", i, dst[i], tt.want[i])
				}
			}
		})
	}
}

func TestElemApplyTernaryScalar(t *testing.T) {
	tests := []struct {
		name      string
		condition []float32
		a         []float32
		scalar    float32
		shape     []int
		op        func(float32, float32, float32) float32
		want      []float32
	}{
		{
			name:      "where with scalar contiguous",
			condition: []float32{1, 0, 1, 0},
			a:         []float32{10, 20, 30, 40},
			scalar:    100,
			shape:     []int{4},
			op: func(c, x, s float32) float32 {
				if c > 0 {
					return x
				}
				return s
			},
			want: []float32{10, 100, 30, 100},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, len(tt.want))
			size := SizeFromShape(tt.shape)
			ElemApplyTernaryScalar(dst, tt.condition, tt.a, tt.scalar, size, tt.op)

			for i := 0; i < size; i++ {
				if dst[i] != tt.want[i] {
					t.Errorf("ElemApplyTernaryScalar() dst[%d] = %v, want %v", i, dst[i], tt.want[i])
				}
			}
		})
	}
}

func TestElemVecApplyUnary(t *testing.T) {
	tests := []struct {
		name     string
		src      []float32
		n        int
		strideD  int
		strideS  int
		op       func(float32) float32
		want     []float32
	}{
		{
			name:    "contiguous",
			src:     []float32{1, 2, 3, 4},
			n:       4,
			strideD: 1,
			strideS: 1,
			op:      func(x float32) float32 { return x * 2 },
			want:    []float32{2, 4, 6, 8},
		},
		{
			name:    "strided",
			src:     []float32{1, 0, 2, 0, 3, 0, 4},
			n:       4,
			strideD: 1,
			strideS: 2,
			op:      func(x float32) float32 { return x * 2 },
			want:    []float32{2, 4, 6, 8},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, len(tt.want))
			ElemVecApplyUnaryStrided(dst, tt.src, tt.n, tt.strideD, tt.strideS, tt.op)

			for i := 0; i < tt.n; i++ {
				if dst[i] != tt.want[i] {
					t.Errorf("ElemVecApplyUnaryStrided() dst[%d] = %v, want %v", i, dst[i], tt.want[i])
				}
			}
		})
	}
}

func TestElemVecApplyBinary(t *testing.T) {
	tests := []struct {
		name     string
		a        []float32
		b        []float32
		n        int
		strideD  int
		strideA  int
		strideB  int
		op       func(float32, float32) float32
		want     []float32
	}{
		{
			name:    "add contiguous",
			a:       []float32{1, 2, 3},
			b:       []float32{4, 5, 6},
			n:       3,
			strideD: 1,
			strideA: 1,
			strideB: 1,
			op:      func(x, y float32) float32 { return x + y },
			want:    []float32{5, 7, 9},
		},
		{
			name:    "multiply strided",
			a:       []float32{2, 0, 3, 0, 4},
			b:       []float32{5, 0, 6, 0, 7},
			n:       3,
			strideD: 1,
			strideA: 2,
			strideB: 2,
			op:      func(x, y float32) float32 { return x * y },
			want:    []float32{10, 18, 28},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, len(tt.want))
			ElemVecApplyBinaryStrided(dst, tt.a, tt.b, tt.n, tt.strideD, tt.strideA, tt.strideB, tt.op)

			for i := 0; i < tt.n; i++ {
				if dst[i] != tt.want[i] {
					t.Errorf("ElemVecApplyBinaryStrided() dst[%d] = %v, want %v", i, dst[i], tt.want[i])
				}
			}
		})
	}
}

func TestElemVecApplyTernary(t *testing.T) {
	tests := []struct {
		name      string
		condition []float32
		a         []float32
		b         []float32
		n         int
		strideD   int
		strideC   int
		strideA   int
		strideB   int
		op        func(float32, float32, float32) float32
		want      []float32
	}{
		{
			name:      "where contiguous",
			condition: []float32{1, 0, 1, 0},
			a:         []float32{10, 20, 30, 40},
			b:         []float32{100, 200, 300, 400},
			n:         4,
			strideD:   1,
			strideC:   1,
			strideA:   1,
			strideB:   1,
			op: func(c, x, y float32) float32 {
				if c > 0 {
					return x
				}
				return y
			},
			want: []float32{10, 200, 30, 400},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, len(tt.want))
			ElemVecApplyTernaryStrided(dst, tt.condition, tt.a, tt.b, tt.n, tt.strideD, tt.strideC, tt.strideA, tt.strideB, tt.op)

			for i := 0; i < tt.n; i++ {
				if dst[i] != tt.want[i] {
					t.Errorf("ElemVecApplyTernaryStrided() dst[%d] = %v, want %v", i, dst[i], tt.want[i])
				}
			}
		})
	}
}

func TestElemVecApplyUnaryScalar(t *testing.T) {
	tests := []struct {
		name     string
		src      []float32
		scalar   float32
		n        int
		strideD  int
		strideS  int
		op       func(float32, float32) float32
		want     []float32
	}{
		{
			name:    "add scalar contiguous",
			src:     []float32{1, 2, 3},
			scalar:  10,
			n:       3,
			strideD: 1,
			strideS: 1,
			op:      func(x, s float32) float32 { return x + s },
			want:    []float32{11, 12, 13},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, len(tt.want))
			ElemVecApplyUnaryScalarStrided(dst, tt.src, tt.scalar, tt.n, tt.strideD, tt.strideS, tt.op)

			for i := 0; i < tt.n; i++ {
				if dst[i] != tt.want[i] {
					t.Errorf("ElemVecApplyUnaryScalarStrided() dst[%d] = %v, want %v", i, dst[i], tt.want[i])
				}
			}
		})
	}
}

func TestElemVecApplyBinaryScalar(t *testing.T) {
	tests := []struct {
		name     string
		a        []float32
		scalar   float32
		n        int
		strideD  int
		strideA  int
		op       func(float32, float32) float32
		want     []float32
	}{
		{
			name:    "multiply scalar contiguous",
			a:       []float32{2, 3, 4},
			scalar:  5,
			n:       3,
			strideD: 1,
			strideA: 1,
			op:      func(x, s float32) float32 { return x * s },
			want:    []float32{10, 15, 20},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, len(tt.want))
			ElemVecApplyBinaryScalarStrided(dst, tt.a, tt.scalar, tt.n, tt.strideD, tt.strideA, tt.op)

			for i := 0; i < tt.n; i++ {
				if dst[i] != tt.want[i] {
					t.Errorf("ElemVecApplyBinaryScalarStrided() dst[%d] = %v, want %v", i, dst[i], tt.want[i])
				}
			}
		})
	}
}

func TestElemVecApplyTernaryScalar(t *testing.T) {
	tests := []struct {
		name      string
		condition []float32
		a         []float32
		scalar    float32
		n         int
		strideD   int
		strideC   int
		strideA   int
		op        func(float32, float32, float32) float32
		want      []float32
	}{
		{
			name:      "where with scalar contiguous",
			condition: []float32{1, 0, 1, 0},
			a:         []float32{10, 20, 30, 40},
			scalar:    100,
			n:         4,
			strideD:   1,
			strideC:   1,
			strideA:   1,
			op: func(c, x, s float32) float32 {
				if c > 0 {
					return x
				}
				return s
			},
			want: []float32{10, 100, 30, 100},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, len(tt.want))
			ElemVecApplyTernaryScalarStrided(dst, tt.condition, tt.a, tt.scalar, tt.n, tt.strideD, tt.strideC, tt.strideA, tt.op)

			for i := 0; i < tt.n; i++ {
				if dst[i] != tt.want[i] {
					t.Errorf("ElemVecApplyTernaryScalarStrided() dst[%d] = %v, want %v", i, dst[i], tt.want[i])
				}
			}
		})
	}
}

func TestElemMatApplyUnary(t *testing.T) {
	tests := []struct {
		name  string
		src   []float32
		rows  int
		cols  int
		ldDst int
		ldSrc int
		op    func(float32) float32
		want  []float32
	}{
		{
			name:  "contiguous 2x3",
			src:   []float32{1, 2, 3, 4, 5, 6},
			rows:  2,
			cols:  3,
			ldDst: 3,
			ldSrc: 3,
			op:    func(x float32) float32 { return x * 2 },
			want:  []float32{2, 4, 6, 8, 10, 12},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, len(tt.want))
			ElemMatApplyUnaryStrided(dst, tt.src, tt.rows, tt.cols, tt.ldDst, tt.ldSrc, tt.op)

			size := tt.rows * tt.cols
			for i := 0; i < size; i++ {
				if dst[i] != tt.want[i] {
					t.Errorf("ElemMatApplyUnaryStrided() dst[%d] = %v, want %v", i, dst[i], tt.want[i])
				}
			}
		})
	}
}

func TestElemMatApplyBinary(t *testing.T) {
	tests := []struct {
		name  string
		a     []float32
		b     []float32
		rows  int
		cols  int
		ldDst int
		ldA   int
		ldB   int
		op    func(float32, float32) float32
		want  []float32
	}{
		{
			name:  "add contiguous 2x2",
			a:     []float32{1, 2, 3, 4},
			b:     []float32{5, 6, 7, 8},
			rows:  2,
			cols:  2,
			ldDst: 2,
			ldA:   2,
			ldB:   2,
			op:    func(x, y float32) float32 { return x + y },
			want:  []float32{6, 8, 10, 12},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, len(tt.want))
			ElemMatApplyBinaryStrided(dst, tt.a, tt.b, tt.rows, tt.cols, tt.ldDst, tt.ldA, tt.ldB, tt.op)

			size := tt.rows * tt.cols
			for i := 0; i < size; i++ {
				if dst[i] != tt.want[i] {
					t.Errorf("ElemMatApplyBinaryStrided() dst[%d] = %v, want %v", i, dst[i], tt.want[i])
				}
			}
		})
	}
}

func TestElemMatApplyTernary(t *testing.T) {
	tests := []struct {
		name      string
		condition []float32
		a         []float32
		b         []float32
		rows      int
		cols      int
		ldDst     int
		ldCond    int
		ldA       int
		ldB       int
		op        func(float32, float32, float32) float32
		want      []float32
	}{
		{
			name:      "where contiguous 2x2",
			condition: []float32{1, 0, 1, 0},
			a:         []float32{10, 20, 30, 40},
			b:         []float32{100, 200, 300, 400},
			rows:      2,
			cols:      2,
			ldDst:     2,
			ldCond:    2,
			ldA:       2,
			ldB:       2,
			op: func(c, x, y float32) float32 {
				if c > 0 {
					return x
				}
				return y
			},
			want: []float32{10, 200, 30, 400},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, len(tt.want))
			ElemMatApplyTernaryStrided(dst, tt.condition, tt.a, tt.b, tt.rows, tt.cols, tt.ldDst, tt.ldCond, tt.ldA, tt.ldB, tt.op)

			size := tt.rows * tt.cols
			for i := 0; i < size; i++ {
				if dst[i] != tt.want[i] {
					t.Errorf("ElemMatApplyTernaryStrided() dst[%d] = %v, want %v", i, dst[i], tt.want[i])
				}
			}
		})
	}
}

func TestElemMatApplyUnaryScalar(t *testing.T) {
	tests := []struct {
		name   string
		src    []float32
		scalar float32
		rows   int
		cols   int
		ldDst  int
		ldSrc  int
		op     func(float32, float32) float32
		want   []float32
	}{
		{
			name:   "multiply scalar contiguous 2x2",
			src:    []float32{1, 2, 3, 4},
			scalar: 5,
			rows:   2,
			cols:   2,
			ldDst:  2,
			ldSrc:  2,
			op:     func(x, s float32) float32 { return x * s },
			want:   []float32{5, 10, 15, 20},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, len(tt.want))
			ElemMatApplyUnaryScalarStrided(dst, tt.src, tt.scalar, tt.rows, tt.cols, tt.ldDst, tt.ldSrc, tt.op)

			size := tt.rows * tt.cols
			for i := 0; i < size; i++ {
				if dst[i] != tt.want[i] {
					t.Errorf("ElemMatApplyUnaryScalarStrided() dst[%d] = %v, want %v", i, dst[i], tt.want[i])
				}
			}
		})
	}
}

func TestElemMatApplyBinaryScalar(t *testing.T) {
	tests := []struct {
		name   string
		a      []float32
		scalar float32
		rows   int
		cols   int
		ldDst  int
		ldA    int
		op     func(float32, float32) float32
		want   []float32
	}{
		{
			name:   "multiply scalar contiguous 2x2",
			a:      []float32{2, 3, 4, 5},
			scalar: 10,
			rows:   2,
			cols:   2,
			ldDst:  2,
			ldA:    2,
			op:     func(x, s float32) float32 { return x * s },
			want:   []float32{20, 30, 40, 50},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, len(tt.want))
			ElemMatApplyBinaryScalarStrided(dst, tt.a, tt.scalar, tt.rows, tt.cols, tt.ldDst, tt.ldA, tt.op)

			size := tt.rows * tt.cols
			for i := 0; i < size; i++ {
				if dst[i] != tt.want[i] {
					t.Errorf("ElemMatApplyBinaryScalarStrided() dst[%d] = %v, want %v", i, dst[i], tt.want[i])
				}
			}
		})
	}
}

func TestElemMatApplyTernaryScalar(t *testing.T) {
	tests := []struct {
		name      string
		condition []float32
		a         []float32
		scalar    float32
		rows      int
		cols      int
		ldDst     int
		ldCond    int
		ldA       int
		op        func(float32, float32, float32) float32
		want      []float32
	}{
		{
			name:      "where with scalar contiguous 2x2",
			condition: []float32{1, 0, 1, 0},
			a:         []float32{10, 20, 30, 40},
			scalar:    100,
			rows:      2,
			cols:      2,
			ldDst:     2,
			ldCond:    2,
			ldA:       2,
			op: func(c, x, s float32) float32 {
				if c > 0 {
					return x
				}
				return s
			},
			want: []float32{10, 100, 30, 100},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, len(tt.want))
			ElemMatApplyTernaryScalarStrided(dst, tt.condition, tt.a, tt.scalar, tt.rows, tt.cols, tt.ldDst, tt.ldCond, tt.ldA, tt.op)

			size := tt.rows * tt.cols
			for i := 0; i < size; i++ {
				if dst[i] != tt.want[i] {
					t.Errorf("ElemMatApplyTernaryScalarStrided() dst[%d] = %v, want %v", i, dst[i], tt.want[i])
				}
			}
		})
	}
}

