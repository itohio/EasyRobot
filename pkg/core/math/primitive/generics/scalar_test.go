package generics

import (
	"testing"

	. "github.com/itohio/EasyRobot/pkg/core/math/primitive/generics/helpers"
)

func TestElemFill(t *testing.T) {
	tests := []struct {
		name  string
		value float32
		n     int
		want  []float32
	}{
		{
			name:  "contiguous",
			value: 5.0,
			n:     4,
			want:  []float32{5, 5, 5, 5},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, tt.n)
			ElemFill(dst, tt.value, tt.n)
			for i := 0; i < tt.n; i++ {
				if dst[i] != tt.want[i] {
					t.Errorf("ElemFill() dst[%d] = %v, want %v", i, dst[i], tt.want[i])
				}
			}
		})
	}
}

func TestElemFillStrided(t *testing.T) {
	tests := []struct {
		name     string
		value    float32
		shape    []int
		stridesD []int
		want     []float32
	}{
		{
			name:     "contiguous",
			value:    5.0,
			shape:    []int{4},
			stridesD: nil,
			want:     []float32{5, 5, 5, 5},
		},
		{
			name:     "2D contiguous",
			value:    7.0,
			shape:    []int{2, 2},
			stridesD: nil,
			want:     []float32{7, 7, 7, 7},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			size := SizeFromShape(tt.shape)
			dst := make([]float32, size)
			ElemFillStrided(dst, tt.value, tt.shape, tt.stridesD)
			for i := 0; i < size; i++ {
				if dst[i] != tt.want[i] {
					t.Errorf("ElemFillStrided() dst[%d] = %v, want %v", i, dst[i], tt.want[i])
				}
			}
		})
	}
}

func TestElemEqualScalar(t *testing.T) {
	tests := []struct {
		name   string
		src    []float32
		scalar float32
		n      int
		want   []float32
	}{
		{
			name:   "contiguous",
			src:    []float32{1, 2, 2, 3},
			scalar: 2.0,
			n:      4,
			want:   []float32{0, 1, 1, 0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, tt.n)
			ElemEqualScalar(dst, tt.src, tt.scalar, tt.n)
			for i := 0; i < tt.n; i++ {
				if dst[i] != tt.want[i] {
					t.Errorf("ElemEqualScalar() dst[%d] = %v, want %v", i, dst[i], tt.want[i])
				}
			}
		})
	}
}

func TestElemGreaterScalar(t *testing.T) {
	tests := []struct {
		name   string
		src    []float32
		scalar float32
		n      int
		want   []float32
	}{
		{
			name:   "contiguous",
			src:    []float32{1, 2, 3, 4},
			scalar: 2.0,
			n:      4,
			want:   []float32{0, 0, 1, 1},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, tt.n)
			ElemGreaterScalar(dst, tt.src, tt.scalar, tt.n)
			for i := 0; i < tt.n; i++ {
				if dst[i] != tt.want[i] {
					t.Errorf("ElemGreaterScalar() dst[%d] = %v, want %v", i, dst[i], tt.want[i])
				}
			}
		})
	}
}

func TestElemLessScalar(t *testing.T) {
	tests := []struct {
		name   string
		src    []float32
		scalar float32
		n      int
		want   []float32
	}{
		{
			name:   "contiguous",
			src:    []float32{1, 2, 3, 4},
			scalar: 2.0,
			n:      4,
			want:   []float32{1, 0, 0, 0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, tt.n)
			ElemLessScalar(dst, tt.src, tt.scalar, tt.n)
			for i := 0; i < tt.n; i++ {
				if dst[i] != tt.want[i] {
					t.Errorf("ElemLessScalar() dst[%d] = %v, want %v", i, dst[i], tt.want[i])
				}
			}
		})
	}
}

func TestElemNotEqualScalar(t *testing.T) {
	tests := []struct {
		name   string
		src    []float32
		scalar float32
		n      int
		want   []float32
	}{
		{
			name:   "contiguous",
			src:    []float32{1, 2, 2, 3},
			scalar: 2.0,
			n:      4,
			want:   []float32{1, 0, 0, 1},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, tt.n)
			ElemNotEqualScalar(dst, tt.src, tt.scalar, tt.n)
			for i := 0; i < tt.n; i++ {
				if dst[i] != tt.want[i] {
					t.Errorf("ElemNotEqualScalar() dst[%d] = %v, want %v", i, dst[i], tt.want[i])
				}
			}
		})
	}
}

func TestElemLessEqualScalar(t *testing.T) {
	tests := []struct {
		name   string
		src    []float32
		scalar float32
		n      int
		want   []float32
	}{
		{
			name:   "contiguous",
			src:    []float32{1, 2, 3, 4},
			scalar: 2.0,
			n:      4,
			want:   []float32{1, 1, 0, 0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, tt.n)
			ElemLessEqualScalar(dst, tt.src, tt.scalar, tt.n)
			for i := 0; i < tt.n; i++ {
				if dst[i] != tt.want[i] {
					t.Errorf("ElemLessEqualScalar() dst[%d] = %v, want %v", i, dst[i], tt.want[i])
				}
			}
		})
	}
}

func TestElemGreaterEqualScalar(t *testing.T) {
	tests := []struct {
		name   string
		src    []float32
		scalar float32
		n      int
		want   []float32
	}{
		{
			name:   "contiguous",
			src:    []float32{1, 2, 3, 4},
			scalar: 2.0,
			n:      4,
			want:   []float32{0, 1, 1, 1},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, tt.n)
			ElemGreaterEqualScalar(dst, tt.src, tt.scalar, tt.n)
			for i := 0; i < tt.n; i++ {
				if dst[i] != tt.want[i] {
					t.Errorf("ElemGreaterEqualScalar() dst[%d] = %v, want %v", i, dst[i], tt.want[i])
				}
			}
		})
	}
}

func TestScalarComparisonMultipleTypes(t *testing.T) {
	// Test with different numeric types
	testScalarComparisonType[int32](t, []int32{1, 2, 3}, int32(2), []int{3})
	testScalarComparisonType[int64](t, []int64{10, 20, 30}, int64(20), []int{3})
	testScalarComparisonType[float64](t, []float64{1.5, 2.5, 3.5}, 2.0, []int{3})
}

func testScalarComparisonType[T Numeric](t *testing.T, src []T, scalar T, shape []int) {
	size := SizeFromShape(shape)
	dst := make([]T, size)

	ElemEqualScalar(dst, src, scalar, size)
	for i := 0; i < size; i++ {
		expected := T(0)
		if src[i] == scalar {
			expected = 1
		}
		if dst[i] != expected {
			t.Errorf("ElemEqualScalar() for type %T: dst[%d] = %v, want %v", *new(T), i, dst[i], expected)
		}
	}
}

