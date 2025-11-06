package generics

import (
	"testing"
)

func TestElements(t *testing.T) {
	tests := []struct {
		name  string
		shape []int
		want  [][]int
	}{
		{
			name:  "1D",
			shape: []int{3},
			want:  [][]int{{0}, {1}, {2}},
		},
		{
			name:  "2D",
			shape: []int{2, 3},
			want: [][]int{
				{0, 0}, {0, 1}, {0, 2},
				{1, 0}, {1, 1}, {1, 2},
			},
		},
		{
			name:  "3D",
			shape: []int{2, 2, 2},
			want: [][]int{
				{0, 0, 0}, {0, 0, 1},
				{0, 1, 0}, {0, 1, 1},
				{1, 0, 0}, {1, 0, 1},
				{1, 1, 0}, {1, 1, 1},
			},
		},
		{
			name:  "empty",
			shape: []int{},
			want:  [][]int{{}},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var got [][]int
			for indices := range Elements(tt.shape) {
				// Copy indices since the slice is reused
				indicesCopy := make([]int, len(indices))
				copy(indicesCopy, indices)
				got = append(got, indicesCopy)
			}
			if len(got) != len(tt.want) {
				t.Errorf("Elements() count = %d, want %d", len(got), len(tt.want))
				return
			}
			for i := range got {
				if len(got[i]) != len(tt.want[i]) {
					t.Errorf("Elements() got[%d] length = %d, want %d", i, len(got[i]), len(tt.want[i]))
					continue
				}
				for j := range got[i] {
					if got[i][j] != tt.want[i][j] {
						t.Errorf("Elements() got[%d][%d] = %d, want %d", i, j, got[i][j], tt.want[i][j])
					}
				}
			}
		})
	}
}

func TestElementsStrided(t *testing.T) {
	tests := []struct {
		name    string
		shape   []int
		strides []int
		want    [][]int
	}{
		{
			name:    "2D contiguous",
			shape:   []int{2, 3},
			strides: nil,
			want: [][]int{
				{0, 0}, {0, 1}, {0, 2},
				{1, 0}, {1, 1}, {1, 2},
			},
		},
		{
			name:    "2D with strides",
			shape:   []int{2, 3},
			strides: []int{6, 2}, // Non-contiguous strides
			want: [][]int{
				{0, 0}, {0, 1}, {0, 2},
				{1, 0}, {1, 1}, {1, 2},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var got [][]int
			for indices := range ElementsStrided(tt.shape, tt.strides) {
				// Copy indices since the slice is reused
				indicesCopy := make([]int, len(indices))
				copy(indicesCopy, indices)
				got = append(got, indicesCopy)
			}
			if len(got) != len(tt.want) {
				t.Errorf("ElementsStrided() count = %d, want %d", len(got), len(tt.want))
				return
			}
			for i := range got {
				if len(got[i]) != len(tt.want[i]) {
					t.Errorf("ElementsStrided() got[%d] length = %d, want %d", i, len(got[i]), len(tt.want[i]))
					continue
				}
				for j := range got[i] {
					if got[i][j] != tt.want[i][j] {
						t.Errorf("ElementsStrided() got[%d][%d] = %d, want %d", i, j, got[i][j], tt.want[i][j])
					}
				}
			}
		})
	}
}

func TestElementsVec(t *testing.T) {
	tests := []struct {
		name string
		n    int
		want []int
	}{
		{
			name: "small",
			n:    3,
			want: []int{0, 1, 2},
		},
		{
			name: "empty",
			n:    0,
			want: []int{},
		},
		{
			name: "large",
			n:    10,
			want: []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var got []int
			for idx := range ElementsVec(tt.n) {
				got = append(got, idx)
			}
			if len(got) != len(tt.want) {
				t.Errorf("ElementsVec() count = %d, want %d", len(got), len(tt.want))
				return
			}
			for i := range got {
				if got[i] != tt.want[i] {
					t.Errorf("ElementsVec() got[%d] = %d, want %d", i, got[i], tt.want[i])
				}
			}
		})
	}
}

func TestElementsVecStrided(t *testing.T) {
	tests := []struct {
		name   string
		n      int
		stride int
		want   []int
	}{
		{
			name:   "stride 1",
			n:      3,
			stride: 1,
			want:   []int{0, 1, 2},
		},
		{
			name:   "stride 2",
			n:      3,
			stride: 2,
			want:   []int{0, 2, 4},
		},
		{
			name:   "stride 3",
			n:      3,
			stride: 3,
			want:   []int{0, 3, 6},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var got []int
			for idx := range ElementsVecStrided(tt.n, tt.stride) {
				got = append(got, idx)
			}
			if len(got) != len(tt.want) {
				t.Errorf("ElementsVecStrided() count = %d, want %d", len(got), len(tt.want))
				return
			}
			for i := range got {
				if got[i] != tt.want[i] {
					t.Errorf("ElementsVecStrided() got[%d] = %d, want %d", i, got[i], tt.want[i])
				}
			}
		})
	}
}

func TestElementsMat(t *testing.T) {
	tests := []struct {
		name  string
		rows  int
		cols  int
		want  [][2]int
	}{
		{
			name: "2x3",
			rows: 2,
			cols: 3,
			want: [][2]int{
				{0, 0}, {0, 1}, {0, 2},
				{1, 0}, {1, 1}, {1, 2},
			},
		},
		{
			name: "3x2",
			rows: 3,
			cols: 2,
			want: [][2]int{
				{0, 0}, {0, 1},
				{1, 0}, {1, 1},
				{2, 0}, {2, 1},
			},
		},
		{
			name: "empty",
			rows: 0,
			cols: 0,
			want: [][2]int{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var got [][2]int
			for idx := range ElementsMat(tt.rows, tt.cols) {
				got = append(got, idx)
			}
			if len(got) != len(tt.want) {
				t.Errorf("ElementsMat() count = %d, want %d", len(got), len(tt.want))
				return
			}
			for i := range got {
				if got[i] != tt.want[i] {
					t.Errorf("ElementsMat() got[%d] = %v, want %v", i, got[i], tt.want[i])
				}
			}
		})
	}
}

func TestElementsMatStrided(t *testing.T) {
	tests := []struct {
		name string
		rows int
		cols int
		ld   int
		want [][2]int
	}{
		{
			name: "2x3 contiguous",
			rows: 2,
			cols: 3,
			ld:   3,
			want: [][2]int{
				{0, 0}, {0, 1}, {0, 2},
				{1, 0}, {1, 1}, {1, 2},
			},
		},
		{
			name: "2x3 strided",
			rows: 2,
			cols: 3,
			ld:   5, // Leading dimension > cols
			want: [][2]int{
				{0, 0}, {0, 1}, {0, 2},
				{1, 0}, {1, 1}, {1, 2},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var got [][2]int
			for idx := range ElementsMatStrided(tt.rows, tt.cols, tt.ld) {
				got = append(got, idx)
			}
			if len(got) != len(tt.want) {
				t.Errorf("ElementsMatStrided() count = %d, want %d", len(got), len(tt.want))
				return
			}
			for i := range got {
				if got[i] != tt.want[i] {
					t.Errorf("ElementsMatStrided() got[%d] = %v, want %v", i, got[i], tt.want[i])
				}
			}
		})
	}
}

func TestElementsEarlyExit(t *testing.T) {
	// Test that early exit works (yield returns false)
	count := 0
	for idx := range ElementsVec(10) {
		if idx >= 5 {
			break
		}
		count++
	}
	if count != 5 {
		t.Errorf("ElementsVec() early exit: count = %d, want 5", count)
	}
}

func TestElementsMatEarlyExit(t *testing.T) {
	// Test that early exit works for matrix
	count := 0
	for idx := range ElementsMat(3, 3) {
		if idx[0] >= 1 {
			break
		}
		count++
	}
	if count != 3 {
		t.Errorf("ElementsMat() early exit: count = %d, want 3", count)
	}
}

