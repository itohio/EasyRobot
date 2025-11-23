package shape

import (
	"reflect"
	"testing"
)

func TestComputeStrides(t *testing.T) {
	tests := []struct {
		name    string
		shape   []int
		dst     []int
		want    []int
		wantNil bool
	}{
		{
			name:    "empty shape",
			shape:   []int{},
			dst:     nil,
			wantNil: true,
		},
		{
			name:  "1D shape",
			shape: []int{5},
			dst:   nil,
			want:  []int{1},
		},
		{
			name:  "2D shape",
			shape: []int{2, 3},
			dst:   nil,
			want:  []int{3, 1},
		},
		{
			name:  "3D shape",
			shape: []int{2, 3, 4},
			dst:   nil,
			want:  []int{12, 4, 1},
		},
		{
			name:  "4D shape",
			shape: []int{2, 3, 4, 5},
			dst:   nil,
			want:  []int{60, 20, 5, 1},
		},
		{
			name:  "with provided dst",
			shape: []int{2, 3},
			dst:   make([]int, 2),
			want:  []int{3, 1},
		},
		{
			name:  "with larger dst",
			shape: []int{2, 3},
			dst:   make([]int, 5),
			want:  []int{3, 1},
		},
		{
			name:  "MAX_DIMS shape",
			shape: []int{2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 1},
			dst:   nil,
			want:  computeStridesReference([]int{2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 1}),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ComputeStrides(tt.dst, tt.shape)
			if tt.wantNil {
				if got != nil {
					t.Errorf("ComputeStrides() = %v, want nil", got)
				}
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("ComputeStrides() = %v, want %v", got, tt.want)
			}
			// Verify it uses provided dst when appropriate
			if tt.dst != nil && len(tt.dst) >= len(tt.shape) {
				if len(got) != len(tt.want) {
					t.Errorf("ComputeStrides() returned slice of length %d, want %d", len(got), len(tt.want))
				}
			}
		})
	}
}

// computeStridesReference is a reference implementation for testing
func computeStridesReference(shape []int) []int {
	if len(shape) == 0 {
		return nil
	}
	strides := make([]int, len(shape))
	stride := 1
	for i := len(shape) - 1; i >= 0; i-- {
		strides[i] = stride
		stride *= shape[i]
	}
	return strides
}

func TestEnsureStrides(t *testing.T) {
	tests := []struct {
		name    string
		dst     []int
		strides []int
		shape   []int
		want    []int
	}{
		{
			name:    "empty shape",
			dst:     nil,
			strides: nil,
			shape:   []int{},
			want:    nil,
		},
		{
			name:    "valid strides match shape",
			dst:     nil,
			strides: []int{3, 1},
			shape:   []int{2, 3},
			want:    []int{3, 1},
		},
		{
			name:    "strides don't match rank - compute canonical",
			dst:     nil,
			strides: []int{1},
			shape:   []int{2, 3},
			want:    []int{3, 1},
		},
		{
			name:    "nil strides - compute canonical",
			dst:     nil,
			strides: nil,
			shape:   []int{2, 3},
			want:    []int{3, 1},
		},
		{
			name:    "with provided dst",
			dst:     make([]int, 2),
			strides: nil,
			shape:   []int{2, 3},
			want:    []int{3, 1},
		},
		{
			name:    "empty strides",
			dst:     nil,
			strides: []int{},
			shape:   []int{2, 3},
			want:    []int{3, 1},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := EnsureStrides(tt.dst, tt.strides, tt.shape)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("EnsureStrides() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestIsContiguous(t *testing.T) {
	tests := []struct {
		name    string
		strides []int
		shape   []int
		want    bool
	}{
		{
			name:    "empty shape",
			strides: nil,
			shape:   []int{},
			want:    true,
		},
		{
			name:    "1D contiguous",
			strides: []int{1},
			shape:   []int{5},
			want:    true,
		},
		{
			name:    "2D contiguous",
			strides: []int{3, 1},
			shape:   []int{2, 3},
			want:    true,
		},
		{
			name:    "3D contiguous",
			strides: []int{12, 4, 1},
			shape:   []int{2, 3, 4},
			want:    true,
		},
		{
			name:    "2D non-contiguous - wrong strides",
			strides: []int{4, 1},
			shape:   []int{2, 3},
			want:    false,
		},
		{
			name:    "2D non-contiguous - transposed",
			strides: []int{1, 2},
			shape:   []int{2, 3},
			want:    false,
		},
		{
			name:    "wrong rank",
			strides: []int{1},
			shape:   []int{2, 3},
			want:    false,
		},
		{
			name:    "empty strides",
			strides: []int{},
			shape:   []int{2, 3},
			want:    false,
		},
		{
			name:    "MAX_DIMS contiguous",
			strides: computeStridesReference([]int{2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 1}),
			shape:   []int{2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 1},
			want:    true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := IsContiguous(tt.strides, tt.shape)
			if got != tt.want {
				t.Errorf("IsContiguous() = %v, want %v", got, tt.want)
			}
		})
	}
}
