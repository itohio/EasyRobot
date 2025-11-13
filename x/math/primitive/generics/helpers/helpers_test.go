package helpers

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

func TestAdvanceOffsets(t *testing.T) {
	tests := []struct {
		name        string
		shape       []int
		indices     []int
		offsets     []int
		stridesDst  []int
		stridesSrc  []int
		want        bool
		wantIndices []int
		wantOffsets []int
	}{
		{
			name:       "empty shape",
			shape:      []int{},
			indices:    []int{},
			offsets:    []int{0, 0},
			stridesDst: []int{},
			stridesSrc: []int{},
			want:       false,
		},
		{
			name:        "1D - advance once",
			shape:       []int{3},
			indices:     []int{0},
			offsets:     []int{0, 0},
			stridesDst:  []int{1},
			stridesSrc:  []int{1},
			want:        true,
			wantIndices: []int{1},
			wantOffsets: []int{1, 1},
		},
		{
			name:        "1D - advance to end",
			shape:       []int{3},
			indices:     []int{2},
			offsets:     []int{2, 2},
			stridesDst:  []int{1},
			stridesSrc:  []int{1},
			want:        false,
			wantIndices: []int{0},
			wantOffsets: []int{0, 0},
		},
		{
			name:        "2D - advance column",
			shape:       []int{2, 3},
			indices:     []int{0, 0},
			offsets:     []int{0, 0},
			stridesDst:  []int{3, 1},
			stridesSrc:  []int{3, 1},
			want:        true,
			wantIndices: []int{0, 1},
			wantOffsets: []int{1, 1},
		},
		{
			name:        "2D - wrap column, advance row",
			shape:       []int{2, 3},
			indices:     []int{0, 2},
			offsets:     []int{2, 2},
			stridesDst:  []int{3, 1},
			stridesSrc:  []int{3, 1},
			want:        true,
			wantIndices: []int{1, 0},
			wantOffsets: []int{3, 3},
		},
		{
			name:        "2D - end of array",
			shape:       []int{2, 3},
			indices:     []int{1, 2},
			offsets:     []int{5, 5},
			stridesDst:  []int{3, 1},
			stridesSrc:  []int{3, 1},
			want:        false,
			wantIndices: []int{0, 0},
			wantOffsets: []int{0, 0},
		},
		{
			name:        "3D - advance through dimensions",
			shape:       []int{2, 2, 2},
			indices:     []int{0, 0, 0},
			offsets:     []int{0, 0},
			stridesDst:  []int{4, 2, 1},
			stridesSrc:  []int{4, 2, 1},
			want:        true,
			wantIndices: []int{0, 0, 1},
			wantOffsets: []int{1, 1},
		},
		{
			name:        "different strides",
			shape:       []int{2, 3},
			indices:     []int{0, 0},
			offsets:     []int{0, 0},
			stridesDst:  []int{3, 1},
			stridesSrc:  []int{6, 2}, // Different stride pattern
			want:        true,
			wantIndices: []int{0, 1},
			wantOffsets: []int{1, 2}, // Different offsets
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			indices := make([]int, len(tt.indices))
			offsets := make([]int, len(tt.offsets))
			copy(indices, tt.indices)
			copy(offsets, tt.offsets)

			got := AdvanceOffsets(tt.shape, indices, offsets, tt.stridesDst, tt.stridesSrc)
			if got != tt.want {
				t.Errorf("AdvanceOffsets() = %v, want %v", got, tt.want)
			}
			if tt.wantIndices != nil && !reflect.DeepEqual(indices, tt.wantIndices) {
				t.Errorf("AdvanceOffsets() indices = %v, want %v", indices, tt.wantIndices)
			}
			if tt.wantOffsets != nil && !reflect.DeepEqual(offsets, tt.wantOffsets) {
				t.Errorf("AdvanceOffsets() offsets = %v, want %v", offsets, tt.wantOffsets)
			}
		})
	}
}

func TestAdvanceOffsets_Iteration(t *testing.T) {
	// Test that AdvanceOffsets correctly iterates through all elements
	shape := []int{2, 3}
	stridesDst := []int{3, 1}
	stridesSrc := []int{3, 1}

	indices := make([]int, len(shape))
	offsets := make([]int, 2)
	visited := make(map[int]bool)
	count := 0

	for {
		// Compute linear index from offsets
		linearIdx := offsets[0]
		if visited[linearIdx] {
			t.Errorf("AdvanceOffsets() visited index %d twice", linearIdx)
		}
		visited[linearIdx] = true
		count++

		if !AdvanceOffsets(shape, indices, offsets, stridesDst, stridesSrc) {
			break
		}
	}

	expectedCount := 2 * 3 // shape[0] * shape[1]
	if count != expectedCount {
		t.Errorf("AdvanceOffsets() visited %d elements, want %d", count, expectedCount)
	}
}

func TestComputeStrideOffset(t *testing.T) {
	tests := []struct {
		name    string
		indices []int
		strides []int
		want    int
	}{
		{
			name:    "empty",
			indices: []int{},
			strides: []int{},
			want:    0,
		},
		{
			name:    "1D",
			indices: []int{3},
			strides: []int{1},
			want:    3,
		},
		{
			name:    "2D",
			indices: []int{1, 2},
			strides: []int{3, 1},
			want:    5, // 1*3 + 2*1
		},
		{
			name:    "3D",
			indices: []int{1, 2, 3},
			strides: []int{12, 4, 1},
			want:    23, // 1*12 + 2*4 + 3*1
		},
		{
			name:    "non-contiguous strides",
			indices: []int{1, 1},
			strides: []int{6, 2},
			want:    8, // 1*6 + 1*2
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ComputeStrideOffset(tt.indices, tt.strides)
			if got != tt.want {
				t.Errorf("ComputeStrideOffset() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestSizeFromShape(t *testing.T) {
	tests := []struct {
		name  string
		shape []int
		want  int
	}{
		{
			name:  "empty shape",
			shape: []int{},
			want:  0,
		},
		{
			name:  "1D",
			shape: []int{5},
			want:  5,
		},
		{
			name:  "2D",
			shape: []int{2, 3},
			want:  6,
		},
		{
			name:  "3D",
			shape: []int{2, 3, 4},
			want:  24,
		},
		{
			name:  "zero dimension",
			shape: []int{2, 0, 4},
			want:  0,
		},
		{
			name:  "negative dimension",
			shape: []int{2, -1, 4},
			want:  0,
		},
		{
			name:  "MAX_DIMS",
			shape: []int{2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 1},
			want:  computeSizeReference([]int{2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 1}),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := SizeFromShape(tt.shape)
			if got != tt.want {
				t.Errorf("SizeFromShape() = %v, want %v", got, tt.want)
			}
		})
	}
}

func computeSizeReference(shape []int) int {
	size := 1
	if len(shape) == 0 {
		return 0
	}
	for _, dim := range shape {
		if dim <= 0 {
			return 0
		}
		size *= dim
	}
	return size
}

func TestIterateOffsets(t *testing.T) {
	shape := []int{2, 3}
	stridesDst := []int{3, 1}
	stridesSrc := []int{3, 1}

	visited := make(map[int]bool)
	count := 0

	IterateOffsets(shape, stridesDst, stridesSrc, func(offsets []int) {
		linearIdx := offsets[0]
		if visited[linearIdx] {
			t.Errorf("IterateOffsets() visited index %d twice", linearIdx)
		}
		visited[linearIdx] = true
		count++
	})

	expectedCount := 2 * 3
	if count != expectedCount {
		t.Errorf("IterateOffsets() visited %d elements, want %d", count, expectedCount)
	}
}

func TestIterateOffsetsWithIndices(t *testing.T) {
	shape := []int{2, 3}
	stridesDst := []int{3, 1}
	stridesSrc := []int{3, 1}

	visited := make(map[string]bool)
	count := 0

	IterateOffsetsWithIndices(shape, stridesDst, stridesSrc, func(indices []int, offsets []int) {
		// Create a key from indices
		key := ""
		for _, idx := range indices {
			key += string(rune('0' + idx))
		}
		if visited[key] {
			t.Errorf("IterateOffsetsWithIndices() visited indices %v twice", indices)
		}
		visited[key] = true
		count++

		// Verify offset matches indices
		expectedOffset := ComputeStrideOffset(indices, stridesDst)
		if offsets[0] != expectedOffset {
			t.Errorf("IterateOffsetsWithIndices() offset[0] = %d, want %d (from indices %v)",
				offsets[0], expectedOffset, indices)
		}
	})

	expectedCount := 2 * 3
	if count != expectedCount {
		t.Errorf("IterateOffsetsWithIndices() visited %d elements, want %d", count, expectedCount)
	}
}
