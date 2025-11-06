package generics

import (
	"testing"
)

func TestElemGreaterThan(t *testing.T) {
	tests := []struct {
		name     string
		a        []float32
		b        []float32
		shape    []int
		stridesD []int
		stridesA []int
		stridesB []int
		want     []float32
	}{
		{
			name:     "contiguous",
			a:        []float32{5, 3, 7, 2},
			b:        []float32{4, 3, 6, 2},
			shape:    []int{4},
			stridesD: nil,
			stridesA: nil,
			stridesB: nil,
			want:     []float32{1, 0, 1, 0},
		},
		{
			name:     "2D contiguous",
			a:        []float32{5, 3, 7, 2},
			b:        []float32{4, 3, 6, 2},
			shape:    []int{2, 2},
			stridesD: nil,
			stridesA: nil,
			stridesB: nil,
			want:     []float32{1, 0, 1, 0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, len(tt.want))
			size := SizeFromShape(tt.shape)
			if tt.stridesD == nil && tt.stridesA == nil && tt.stridesB == nil && IsContiguous(nil, tt.shape) {
				ElemGreaterThan(dst, tt.a, tt.b, size)
			} else {
				ElemGreaterThanStrided(dst, tt.a, tt.b, tt.shape, tt.stridesD, tt.stridesA, tt.stridesB)
			}

			for i := 0; i < size; i++ {
				if dst[i] != tt.want[i] {
					t.Errorf("ElemGreaterThan() dst[%d] = %v, want %v", i, dst[i], tt.want[i])
				}
			}
		})
	}
}

func TestElemEqual(t *testing.T) {
	tests := []struct {
		name     string
		a        []float32
		b        []float32
		shape    []int
		stridesD []int
		stridesA []int
		stridesB []int
		want     []float32
	}{
		{
			name:     "contiguous",
			a:        []float32{5, 3, 7, 2},
			b:        []float32{5, 4, 7, 2},
			shape:    []int{4},
			stridesD: nil,
			stridesA: nil,
			stridesB: nil,
			want:     []float32{1, 0, 1, 1},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, len(tt.want))
			size := SizeFromShape(tt.shape)
			if tt.stridesD == nil && tt.stridesA == nil && tt.stridesB == nil && IsContiguous(nil, tt.shape) {
				ElemEqual(dst, tt.a, tt.b, size)
			} else {
				ElemEqualStrided(dst, tt.a, tt.b, tt.shape, tt.stridesD, tt.stridesA, tt.stridesB)
			}

			for i := 0; i < size; i++ {
				if dst[i] != tt.want[i] {
					t.Errorf("ElemEqual() dst[%d] = %v, want %v", i, dst[i], tt.want[i])
				}
			}
		})
	}
}

func TestElemLess(t *testing.T) {
	tests := []struct {
		name     string
		a        []float32
		b        []float32
		shape    []int
		stridesD []int
		stridesA []int
		stridesB []int
		want     []float32
	}{
		{
			name:     "contiguous",
			a:        []float32{3, 5, 2, 7},
			b:        []float32{4, 3, 6, 2},
			shape:    []int{4},
			stridesD: nil,
			stridesA: nil,
			stridesB: nil,
			want:     []float32{1, 0, 1, 0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, len(tt.want))
			size := SizeFromShape(tt.shape)
			if tt.stridesD == nil && tt.stridesA == nil && tt.stridesB == nil && IsContiguous(nil, tt.shape) {
				ElemLess(dst, tt.a, tt.b, size)
			} else {
				ElemLessStrided(dst, tt.a, tt.b, tt.shape, tt.stridesD, tt.stridesA, tt.stridesB)
			}

			for i := 0; i < size; i++ {
				if dst[i] != tt.want[i] {
					t.Errorf("ElemLess() dst[%d] = %v, want %v", i, dst[i], tt.want[i])
				}
			}
		})
	}
}

func TestElemNotEqual(t *testing.T) {
	tests := []struct {
		name     string
		a        []float32
		b        []float32
		shape    []int
		stridesD []int
		stridesA []int
		stridesB []int
		want     []float32
	}{
		{
			name:     "contiguous",
			a:        []float32{5, 3, 7, 2},
			b:        []float32{5, 4, 7, 2},
			shape:    []int{4},
			stridesD: nil,
			stridesA: nil,
			stridesB: nil,
			want:     []float32{0, 1, 0, 0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, len(tt.want))
			size := SizeFromShape(tt.shape)
			if tt.stridesD == nil && tt.stridesA == nil && tt.stridesB == nil && IsContiguous(nil, tt.shape) {
				ElemNotEqual(dst, tt.a, tt.b, size)
			} else {
				ElemNotEqualStrided(dst, tt.a, tt.b, tt.shape, tt.stridesD, tt.stridesA, tt.stridesB)
			}

			for i := 0; i < size; i++ {
				if dst[i] != tt.want[i] {
					t.Errorf("ElemNotEqual() dst[%d] = %v, want %v", i, dst[i], tt.want[i])
				}
			}
		})
	}
}

func TestElemLessEqual(t *testing.T) {
	tests := []struct {
		name     string
		a        []float32
		b        []float32
		shape    []int
		stridesD []int
		stridesA []int
		stridesB []int
		want     []float32
	}{
		{
			name:     "contiguous",
			a:        []float32{3, 5, 2, 7},
			b:        []float32{4, 3, 2, 7},
			shape:    []int{4},
			stridesD: nil,
			stridesA: nil,
			stridesB: nil,
			want:     []float32{1, 0, 1, 1},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, len(tt.want))
			size := SizeFromShape(tt.shape)
			if tt.stridesD == nil && tt.stridesA == nil && tt.stridesB == nil && IsContiguous(nil, tt.shape) {
				ElemLessEqual(dst, tt.a, tt.b, size)
			} else {
				ElemLessEqualStrided(dst, tt.a, tt.b, tt.shape, tt.stridesD, tt.stridesA, tt.stridesB)
			}

			for i := 0; i < size; i++ {
				if dst[i] != tt.want[i] {
					t.Errorf("ElemLessEqual() dst[%d] = %v, want %v", i, dst[i], tt.want[i])
				}
			}
		})
	}
}

func TestElemGreaterEqual(t *testing.T) {
	tests := []struct {
		name     string
		a        []float32
		b        []float32
		shape    []int
		stridesD []int
		stridesA []int
		stridesB []int
		want     []float32
	}{
		{
			name:     "contiguous",
			a:        []float32{5, 3, 7, 2},
			b:        []float32{4, 3, 6, 2},
			shape:    []int{4},
			stridesD: nil,
			stridesA: nil,
			stridesB: nil,
			want:     []float32{1, 1, 1, 1},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, len(tt.want))
			size := SizeFromShape(tt.shape)
			if tt.stridesD == nil && tt.stridesA == nil && tt.stridesB == nil && IsContiguous(nil, tt.shape) {
				ElemGreaterEqual(dst, tt.a, tt.b, size)
			} else {
				ElemGreaterEqualStrided(dst, tt.a, tt.b, tt.shape, tt.stridesD, tt.stridesA, tt.stridesB)
			}

			for i := 0; i < size; i++ {
				if dst[i] != tt.want[i] {
					t.Errorf("ElemGreaterEqual() dst[%d] = %v, want %v", i, dst[i], tt.want[i])
				}
			}
		})
	}
}

func TestComparisonMultipleTypes(t *testing.T) {
	// Test with different numeric types
	testComparisonType[int32](t, []int32{5, 3, 7}, []int32{4, 3, 6}, []int{3})
	testComparisonType[int64](t, []int64{10, 20, 30}, []int64{5, 20, 25}, []int{3})
	testComparisonType[float64](t, []float64{1.5, 2.5, 3.5}, []float64{1.0, 2.5, 4.0}, []int{3})
}

func testComparisonType[T Numeric](t *testing.T, a, b []T, shape []int) {
	size := SizeFromShape(shape)
	dst := make([]T, size)

	ElemGreaterThan(dst, a, b, size)
	for i := 0; i < size; i++ {
		expected := T(0)
		if a[i] > b[i] {
			expected = 1
		}
		if dst[i] != expected {
			t.Errorf("ElemGreaterThan() for type %T: dst[%d] = %v, want %v", *new(T), i, dst[i], expected)
		}
	}
}

