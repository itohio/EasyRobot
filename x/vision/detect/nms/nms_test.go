package nms

import (
	"math/rand/v2"
	"slices"
	"testing"
)

func TestIoU(t *testing.T) {
	a := Box{X0: 0, Y0: 0, X1: 10, Y1: 10}
	for _, tt := range []struct {
		b    Box
		want float32
	}{
		{Box{X0: 0, Y0: 0, X1: 10, Y1: 10}, 1},
		{Box{X0: 5, Y0: 0, X1: 15, Y1: 10}, 50.0 / 150},
		{Box{X0: 10, Y0: 0, X1: 20, Y1: 10}, 0}, // touching
		{Box{X0: 20, Y0: 20, X1: 30, Y1: 30}, 0},
		{Box{X0: 2, Y0: 2, X1: 8, Y1: 8}, 36.0 / 100}, // inside
		{Box{X0: 3, Y0: 3, X1: 3, Y1: 3}, 0},          // empty
	} {
		if got := IoU(a, tt.b); got < tt.want-1e-6 || got > tt.want+1e-6 {
			t.Errorf("IoU(%v) = %v, want %v", tt.b, got, tt.want)
		}
	}
}

// Greedy keeps the best box of each overlapping group, per class, in score
// order; boxes of other classes never suppress each other.
func TestGreedy(t *testing.T) {
	in := []Box{
		{X0: 0, Y0: 0, X1: 10, Y1: 10, Score: 0.6, Class: 0, Index: 1},
		{X0: 1, Y0: 1, X1: 11, Y1: 11, Score: 0.9, Class: 0, Index: 2},   // best of group A
		{X0: 1, Y0: 1, X1: 11, Y1: 11, Score: 0.8, Class: 5, Index: 3},   // same place, other class: kept
		{X0: 50, Y0: 50, X1: 60, Y1: 60, Score: 0.7, Class: 0, Index: 4}, // group B
		{X0: 52, Y0: 50, X1: 62, Y1: 60, Score: 0.7, Class: 0, Index: 5}, // tie with 4, IoU 0.67: dropped (4 has the lower index)
		{X0: 5, Y0: 0, X1: 15, Y1: 10, Score: 0.5, Class: 0, Index: 6},   // IoU with 2 = 0.39: kept at threshold 0.45
	}
	got := Greedy(slices.Clone(in), 0.45, true, make([]Box, 0, 8))
	var idx []int32
	for _, b := range got {
		idx = append(idx, b.Index)
	}
	if want := []int32{2, 3, 4, 6}; !slices.Equal(idx, want) {
		t.Errorf("kept %v, want %v", idx, want)
	}
	got = Greedy(slices.Clone(in), 0.45, false, make([]Box, 0, 8)) // class-agnostic: 3 goes too
	idx = idx[:0]
	for _, b := range got {
		idx = append(idx, b.Index)
	}
	if want := []int32{2, 4, 6}; !slices.Equal(idx, want) {
		t.Errorf("class-agnostic kept %v, want %v", idx, want)
	}
}

// out's capacity bounds the result: the best boxes win.
func TestGreedyCapacity(t *testing.T) {
	in := []Box{
		{X0: 0, Y0: 0, X1: 1, Y1: 1, Score: 0.1, Index: 0},
		{X0: 10, Y0: 0, X1: 11, Y1: 1, Score: 0.9, Index: 1},
		{X0: 20, Y0: 0, X1: 21, Y1: 1, Score: 0.5, Index: 2},
	}
	got := Greedy(in, 0.5, true, make([]Box, 0, 2))
	if len(got) != 2 || got[0].Index != 1 || got[1].Index != 2 {
		t.Errorf("got %v, want indices 1, 2", got)
	}
}

func TestGreedyNoAlloc(t *testing.T) {
	in := make([]Box, 200)
	for i := range in {
		f := float32(i % 20)
		in[i] = Box{X0: f * 5, Y0: 0, X1: f*5 + 10, Y1: 10, Score: float32(i%13) / 13, Class: int32(i % 3), Index: int32(i)}
	}
	work := make([]Box, len(in))
	out := make([]Box, 0, 64)
	if n := testing.AllocsPerRun(50, func() {
		copy(work, in)
		Greedy(work, 0.45, true, out[:0])
	}); n != 0 {
		t.Errorf("Greedy allocates %v times per run, want 0", n)
	}
}

// A crowded scene (verify note): many overlapping candidates of a few
// classes, random but seeded. Greedy must equal a plain reference that,
// for each candidate in score order, scans every box kept so far.
func TestGreedyCrowded(t *testing.T) {
	r := rand.New(rand.NewPCG(1, 2))
	for trial := range 50 {
		n := 50 + r.IntN(250)
		in := make([]Box, n)
		for i := range in {
			x, y := r.Float32()*100, r.Float32()*100
			w, h := 5+r.Float32()*40, 5+r.Float32()*40
			in[i] = Box{X0: x, Y0: y, X1: x + w, Y1: y + h, Score: float32(r.IntN(40)) / 40, Class: int32(r.IntN(3)), Index: int32(i)}
		}
		for _, perClass := range []bool{true, false} {
			iou := float32(0.3 + 0.4*r.Float64())
			ref := slices.Clone(in)
			slices.SortStableFunc(ref, func(a, b Box) int { // score desc, then index
				if a.Score != b.Score {
					if a.Score > b.Score {
						return -1
					}
					return 1
				}
				return int(a.Index - b.Index)
			})
			var want []int32
			var kept []Box
			for _, c := range ref {
				ok := true
				for _, k := range kept {
					if (!perClass || k.Class == c.Class) && IoU(k, c) > iou {
						ok = false
					}
				}
				if ok {
					kept = append(kept, c)
					want = append(want, c.Index)
				}
			}
			got := Greedy(slices.Clone(in), iou, perClass, make([]Box, 0, n))
			var idx []int32
			for _, b := range got {
				idx = append(idx, b.Index)
			}
			if !slices.Equal(idx, want) {
				t.Fatalf("trial %d perClass %v: kept %v, want %v", trial, perClass, idx, want)
			}
		}
	}
}
