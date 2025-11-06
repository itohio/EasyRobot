package generics

import (
	"testing"

	. "github.com/itohio/EasyRobot/pkg/core/math/primitive/generics/helpers"
)

func TestElemSign(t *testing.T) {
	tests := []struct {
		name     string
		src      []float32
		shape    []int
		stridesD []int
		stridesS []int
		want     []float32
	}{
		{
			name:     "contiguous",
			src:      []float32{5, -3, 0, 7, -2},
			shape:    []int{5},
			stridesD: nil,
			stridesS: nil,
			want:     []float32{1, -1, 0, 1, -1},
		},
		{
			name:     "2D contiguous",
			src:      []float32{5, -3, 0, 7},
			shape:    []int{2, 2},
			stridesD: nil,
			stridesS: nil,
			want:     []float32{1, -1, 0, 1},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, len(tt.want))
			size := SizeFromShape(tt.shape)
			if tt.stridesD == nil && tt.stridesS == nil && IsContiguous(nil, tt.shape) {
				ElemSign(dst, tt.src, size)
			} else {
				ElemSignStrided(dst, tt.src, tt.shape, tt.stridesD, tt.stridesS)
			}

			for i := 0; i < size; i++ {
				if dst[i] != tt.want[i] {
					t.Errorf("ElemSign() dst[%d] = %v, want %v", i, dst[i], tt.want[i])
				}
			}
		})
	}
}

func TestElemNegative(t *testing.T) {
	tests := []struct {
		name     string
		src      []float32
		shape    []int
		stridesD []int
		stridesS []int
		want     []float32
	}{
		{
			name:     "contiguous",
			src:      []float32{5, -3, 0, 7, -2},
			shape:    []int{5},
			stridesD: nil,
			stridesS: nil,
			want:     []float32{-5, 3, 0, -7, 2},
		},
		{
			name:     "2D contiguous",
			src:      []float32{1, 2, 3, 4},
			shape:    []int{2, 2},
			stridesD: nil,
			stridesS: nil,
			want:     []float32{-1, -2, -3, -4},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, len(tt.want))
			size := SizeFromShape(tt.shape)
			if tt.stridesD == nil && tt.stridesS == nil && IsContiguous(nil, tt.shape) {
				ElemNegative(dst, tt.src, size)
			} else {
				ElemNegativeStrided(dst, tt.src, tt.shape, tt.stridesD, tt.stridesS)
			}

			for i := 0; i < size; i++ {
				if dst[i] != tt.want[i] {
					t.Errorf("ElemNegative() dst[%d] = %v, want %v", i, dst[i], tt.want[i])
				}
			}
		})
	}
}

func TestUnaryMultipleTypes(t *testing.T) {
	// Test Sign with different numeric types
	testSignType[int32](t, []int32{5, -3, 0}, []int{3})
	testSignType[int64](t, []int64{10, -20, 0}, []int{3})
	testSignType[float64](t, []float64{1.5, -2.5, 0.0}, []int{3})

	// Test Negative with different numeric types
	testNegativeType[int32](t, []int32{5, -3, 0}, []int{3})
	testNegativeType[int64](t, []int64{10, -20, 0}, []int{3})
	testNegativeType[float64](t, []float64{1.5, -2.5, 0.0}, []int{3})
}

func testSignType[T Numeric](t *testing.T, src []T, shape []int) {
	size := SizeFromShape(shape)
	dst := make([]T, size)

	ElemSign(dst, src, size)
	for i := 0; i < size; i++ {
		var expected T
		if src[i] > 0 {
			expected = 1
		} else if src[i] < 0 {
			expected = -1
		} else {
			expected = 0
		}
		if dst[i] != expected {
			t.Errorf("ElemSign() for type %T: dst[%d] = %v, want %v", *new(T), i, dst[i], expected)
		}
	}
}

func testNegativeType[T Numeric](t *testing.T, src []T, shape []int) {
	size := SizeFromShape(shape)
	dst := make([]T, size)

	ElemNegative(dst, src, size)
	for i := 0; i < size; i++ {
		expected := -src[i]
		if dst[i] != expected {
			t.Errorf("ElemNegative() for type %T: dst[%d] = %v, want %v", *new(T), i, dst[i], expected)
		}
	}
}

