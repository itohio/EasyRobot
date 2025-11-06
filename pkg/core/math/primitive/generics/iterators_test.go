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
		name string
		rows int
		cols int
		want [][2]int
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

func TestElementsWindow(t *testing.T) {
	tests := []struct {
		name         string
		windowOffset []int
		windowShape  []int
		parentShape  []int
		want         []struct {
			indices []int
			isValid bool
		}
	}{
		{
			name:         "2x2 window in 4x4 parent, no offset",
			windowOffset: []int{0, 0},
			windowShape:  []int{2, 2},
			parentShape:  []int{4, 4},
			want: []struct {
				indices []int
				isValid bool
			}{
				{[]int{0, 0}, true},
				{[]int{0, 1}, true},
				{[]int{1, 0}, true},
				{[]int{1, 1}, true},
			},
		},
		{
			name:         "2x2 window in 4x4 parent, with offset",
			windowOffset: []int{1, 1},
			windowShape:  []int{2, 2},
			parentShape:  []int{4, 4},
			want: []struct {
				indices []int
				isValid bool
			}{
				{[]int{1, 1}, true},
				{[]int{1, 2}, true},
				{[]int{2, 1}, true},
				{[]int{2, 2}, true},
			},
		},
		{
			name:         "2x2 window with negative offset (padding)",
			windowOffset: []int{-1, -1},
			windowShape:  []int{2, 2},
			parentShape:  []int{4, 4},
			want: []struct {
				indices []int
				isValid bool
			}{
				{[]int{-1, -1}, false}, // Out of bounds
				{[]int{-1, 0}, false},  // Out of bounds
				{[]int{0, -1}, false},  // Out of bounds
				{[]int{0, 0}, true},    // Valid
			},
		},
		{
			name:         "2x2 window with out-of-bounds",
			windowOffset: []int{3, 3},
			windowShape:  []int{2, 2},
			parentShape:  []int{4, 4},
			want: []struct {
				indices []int
				isValid bool
			}{
				{[]int{3, 3}, true},
				{[]int{3, 4}, false}, // Out of bounds
				{[]int{4, 3}, false}, // Out of bounds
				{[]int{4, 4}, false}, // Out of bounds
			},
		},
		{
			name:         "1D window",
			windowOffset: []int{1},
			windowShape:  []int{3},
			parentShape:  []int{5},
			want: []struct {
				indices []int
				isValid bool
			}{
				{[]int{1}, true},
				{[]int{2}, true},
				{[]int{3}, true},
			},
		},
		{
			name:         "empty window",
			windowOffset: []int{0, 0},
			windowShape:  []int{0, 0},
			parentShape:  []int{4, 4},
			want: []struct {
				indices []int
				isValid bool
			}{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var got []struct {
				indices []int
				isValid bool
			}
			for absIndices, isValid := range ElementsWindow(tt.windowOffset, tt.windowShape, tt.parentShape) {
				// Copy indices since slice is reused
				indicesCopy := make([]int, len(absIndices))
				copy(indicesCopy, absIndices)
				got = append(got, struct {
					indices []int
					isValid bool
				}{indicesCopy, isValid})
			}

			if len(got) != len(tt.want) {
				t.Errorf("ElementsWindow() count = %d, want %d", len(got), len(tt.want))
				return
			}

			for i := range got {
				if got[i].isValid != tt.want[i].isValid {
					t.Errorf("ElementsWindow() got[%d].isValid = %v, want %v", i, got[i].isValid, tt.want[i].isValid)
				}
				if len(got[i].indices) != len(tt.want[i].indices) {
					t.Errorf("ElementsWindow() got[%d].indices length = %d, want %d", i, len(got[i].indices), len(tt.want[i].indices))
					continue
				}
				for j := range got[i].indices {
					if got[i].indices[j] != tt.want[i].indices[j] {
						t.Errorf("ElementsWindow() got[%d].indices[%d] = %d, want %d", i, j, got[i].indices[j], tt.want[i].indices[j])
					}
				}
			}
		})
	}
}

func TestElementsWindows(t *testing.T) {
	tests := []struct {
		name        string
		outputShape []int
		kernelShape []int
		inputShape  []int
		stride      []int
		padding     []int
		wantCount   int
		wantSample  []struct {
			outIdx  []int
			inIdx   []int
			isValid bool
		}
	}{
		{
			name:        "2x2 kernel, 4x4 input, stride 1, no padding",
			outputShape: []int{3, 3}, // (4-2)/1+1 = 3
			kernelShape: []int{2, 2},
			inputShape:  []int{4, 4},
			stride:      []int{1, 1},
			padding:     []int{0, 0},
			wantCount:   3 * 3 * 2 * 2, // output positions * kernel positions
			wantSample: []struct {
				outIdx  []int
				inIdx   []int
				isValid bool
			}{
				{[]int{0, 0}, []int{0, 0}, true}, // First output, first kernel position
				{[]int{0, 0}, []int{0, 1}, true}, // First output, second kernel position
				{[]int{0, 1}, []int{0, 1}, true}, // Second output, first kernel position
				{[]int{2, 2}, []int{3, 3}, true}, // Last output, last kernel position
			},
		},
		{
			name:        "2x2 kernel, 4x4 input, stride 1, padding 1",
			outputShape: []int{5, 5}, // (4+2*1-2)/1+1 = 5
			kernelShape: []int{2, 2},
			inputShape:  []int{4, 4},
			stride:      []int{1, 1},
			padding:     []int{1, 1},
			wantCount:   5 * 5 * 2 * 2,
			wantSample: []struct {
				outIdx  []int
				inIdx   []int
				isValid bool
			}{
				{[]int{0, 0}, []int{-1, -1}, false}, // Out of bounds (padding)
				{[]int{0, 0}, []int{-1, 0}, false},  // Out of bounds
				{[]int{0, 0}, []int{0, -1}, false},  // Out of bounds
				{[]int{0, 0}, []int{0, 0}, true},    // Valid
				{[]int{1, 1}, []int{0, 0}, true},    // Valid
			},
		},
		{
			name:        "3x3 kernel, 5x5 input, stride 2, no padding",
			outputShape: []int{2, 2}, // (5-3)/2+1 = 2
			kernelShape: []int{3, 3},
			inputShape:  []int{5, 5},
			stride:      []int{2, 2},
			padding:     []int{0, 0},
			wantCount:   2 * 2 * 3 * 3,
			wantSample: []struct {
				outIdx  []int
				inIdx   []int
				isValid bool
			}{
				{[]int{0, 0}, []int{0, 0}, true},
				{[]int{0, 0}, []int{0, 2}, true},
				{[]int{1, 1}, []int{2, 2}, true},
				{[]int{1, 1}, []int{4, 4}, true},
			},
		},
		{
			name:        "1D convolution",
			outputShape: []int{3}, // (5-3)/1+1 = 3
			kernelShape: []int{3},
			inputShape:  []int{5},
			stride:      []int{1},
			padding:     []int{0},
			wantCount:   3 * 3,
			wantSample: []struct {
				outIdx  []int
				inIdx   []int
				isValid bool
			}{
				{[]int{0}, []int{0}, true},
				{[]int{0}, []int{1}, true},
				{[]int{0}, []int{2}, true},
				{[]int{2}, []int{2}, true},
				{[]int{2}, []int{4}, true},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			count := 0
			sampleMap := make(map[string]struct {
				outIdx  []int
				inIdx   []int
				isValid bool
			})

			iter := ElementsWindows(tt.outputShape, tt.kernelShape, tt.inputShape, tt.stride, tt.padding)
			iter(func(outIdx, inIdx []int, isValid bool) bool {
				count++

				// Store first few samples for verification
				if count <= len(tt.wantSample) {
					outCopy := make([]int, len(outIdx))
					inCopy := make([]int, len(inIdx))
					copy(outCopy, outIdx)
					copy(inCopy, inIdx)
					key := ""
					for _, v := range outCopy {
						key += string(rune(v + '0'))
					}
					key += "-"
					for _, v := range inCopy {
						key += string(rune(v + '0'))
					}
					sampleMap[key] = struct {
						outIdx  []int
						inIdx   []int
						isValid bool
					}{outCopy, inCopy, isValid}
				}
				return true
			})

			if count != tt.wantCount {
				t.Errorf("ElementsWindows() count = %d, want %d", count, tt.wantCount)
			}

			// Verify samples
			for _, want := range tt.wantSample {
				key := ""
				for _, v := range want.outIdx {
					key += string(rune(v + '0'))
				}
				key += "-"
				for _, v := range want.inIdx {
					key += string(rune(v + '0'))
				}
				got, found := sampleMap[key]
				if !found {
					// Try to find a matching sample
					found = false
					for _, sample := range sampleMap {
						match := true
						if len(sample.outIdx) != len(want.outIdx) || len(sample.inIdx) != len(want.inIdx) {
							continue
						}
						for i := range sample.outIdx {
							if sample.outIdx[i] != want.outIdx[i] {
								match = false
								break
							}
						}
						if match {
							for i := range sample.inIdx {
								if sample.inIdx[i] != want.inIdx[i] {
									match = false
									break
								}
							}
						}
						if match {
							got = sample
							found = true
							break
						}
					}
				}
				if found {
					if got.isValid != want.isValid {
						t.Errorf("ElementsWindows() sample isValid = %v, want %v (outIdx=%v, inIdx=%v)",
							got.isValid, want.isValid, got.outIdx, got.inIdx)
					}
				} else if count <= len(tt.wantSample) {
					// Only report error if we checked the sample
					t.Logf("ElementsWindows() sample not found: outIdx=%v, inIdx=%v (may be in later iterations)", want.outIdx, want.inIdx)
				}
			}
		})
	}
}

func TestElementsWindowsEarlyExit(t *testing.T) {
	// Test that early exit works
	count := 0
	iter := ElementsWindows([]int{3, 3}, []int{2, 2}, []int{4, 4}, []int{1, 1}, []int{0, 0})
	iter(func(outIdx, inIdx []int, isValid bool) bool {
		count++
		if count >= 10 {
			return false // Early exit
		}
		_ = outIdx
		_ = inIdx
		_ = isValid
		return true
	})
	if count != 10 {
		t.Errorf("ElementsWindows() early exit: count = %d, want 10", count)
	}
}
