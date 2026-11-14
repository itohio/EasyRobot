package yolox

import (
	"math"
	"testing"

	"github.com/itohio/EasyRobot/x/vision/detect/nms"
)

// The RTS YOLOX-nano 416 export: 52²+26²+13² = 3549 anchors, 80 classes.
var cfg416 = Config{W: 416, H: 416, Strides: []int{8, 16, 32}, Classes: 80,
	Scores: Quant{Scale: 0.0037369346246123314, ZP: -128}, Boxes: Quant{Scale: 0.02870197780430317, ZP: -27}}

func q(v float32, qt Quant) int8 {
	return int8(max(-128, min(127, int32(math.Round(float64(v/qt.Scale)))+qt.ZP)))
}

func tensors(d *Decoder) ([]int8, []int8) {
	s := make([]int8, d.Anchors()*80)
	for i := range s {
		s[i] = -128 // score 0
	}
	b := make([]int8, d.Anchors()*4)
	for i := range b {
		b[i] = int8(cfg416.Boxes.ZP)
	}
	return s, b
}

func TestAnchorsAndGrid(t *testing.T) {
	d, err := New(cfg416, 100)
	if err != nil {
		t.Fatal(err)
	}
	if d.Anchors() != 3549 {
		t.Fatalf("anchors %d, want 3549", d.Anchors())
	}
	// First anchor of each stride, and the last anchor overall.
	for _, tt := range []struct{ a, gx, gy, s int }{{0, 0, 0, 8}, {2703, 51, 51, 8}, {2704, 0, 0, 16}, {2704 + 27, 1, 1, 16}, {3380, 0, 0, 32}, {3548, 12, 12, 32}} {
		if int(d.gx[tt.a]) != tt.gx || int(d.gy[tt.a]) != tt.gy || int(d.stride[tt.a]) != tt.s {
			t.Errorf("anchor %d: cell %d,%d stride %d, want %d,%d %d", tt.a, d.gx[tt.a], d.gy[tt.a], d.stride[tt.a], tt.gx, tt.gy, tt.s)
		}
	}
}

// One anchor with a known regression decodes to the YOLOX box:
// centre (dx+gx)·s, size exp(dw)·s.
func TestDecodeBox(t *testing.T) {
	d, _ := New(cfg416, 100)
	s, b := tensors(d)
	a := 2704 + 3*26 + 5 // stride 16, cell (5, 3)
	s[a*80+17] = q(0.8, cfg416.Scores)
	reg := [4]float32{0.5, -0.25, 1.0, 0.5}
	for i, v := range reg {
		b[a*4+i] = q(v, cfg416.Boxes)
	}
	got, err := d.Decode(s, b, 0.3, 0.45, make([]nms.Box, 0, 10))
	if err != nil || len(got) != 1 {
		t.Fatalf("got %v, %v; want one box", got, err)
	}
	deq := func(v float32) float64 { // the value the int8 actually holds
		return float64(cfg416.Boxes.Scale) * float64(int32(q(v, cfg416.Boxes))-cfg416.Boxes.ZP)
	}
	cx, cy := (deq(reg[0])+5)*16, (deq(reg[1])+3)*16
	w, h := math.Exp(deq(reg[2]))*16, math.Exp(deq(reg[3]))*16
	want := [4]float64{cx - w/2, cy - h/2, cx + w/2, cy + h/2}
	g := got[0]
	for i, v := range [4]float32{g.X0, g.Y0, g.X1, g.Y1} {
		if math.Abs(float64(v)-want[i]) > 1e-3 {
			t.Errorf("corner %d = %v, want %v", i, v, want[i])
		}
	}
	if g.Class != 17 || g.Index != int32(a) || math.Abs(float64(g.Score)-0.8) > float64(cfg416.Scores.Scale) {
		t.Errorf("class %d index %d score %v, want 17 %d ~0.8", g.Class, g.Index, g.Score, a)
	}
}

// The threshold works in the quantised domain exactly like Scale·(q−ZP) ≥ conf.
func TestThreshold(t *testing.T) {
	d, _ := New(cfg416, 100)
	s, b := tensors(d)
	for qv := -128; qv <= 127; qv++ {
		s[0] = int8(qv)
		got, _ := d.Decode(s, b, 0.3, 0.45, make([]nms.Box, 0, 4))
		real := cfg416.Scores.Scale * float32(int32(qv)-cfg416.Scores.ZP)
		if (len(got) == 1) != (real >= 0.3) {
			t.Fatalf("q %d (score %v): kept %v", qv, real, len(got) == 1)
		}
	}
	got, _ := d.Decode(s, b, 1.5, 0.45, make([]nms.Box, 0, 4)) // above the scale's range
	if len(got) != 0 {
		t.Errorf("conf above the maximum score kept %d", len(got))
	}
}

// With more candidates than maxCand, the highest scores survive.
func TestCandidateCap(t *testing.T) {
	d, _ := New(cfg416, 3)
	s, b := tensors(d)
	for i, v := range []float32{0.4, 0.9, 0.5, 0.95, 0.35, 0.6} {
		a := i * 400 // far apart: no NMS overlap
		s[a*80] = q(v, cfg416.Scores)
	}
	got, _ := d.Decode(s, b, 0.3, 0.45, make([]nms.Box, 0, 10))
	var idx []int32
	for _, g := range got {
		idx = append(idx, g.Index)
	}
	if len(idx) != 3 || idx[0] != 1200 || idx[1] != 400 || idx[2] != 2000 {
		t.Errorf("kept anchors %v, want [1200 400 2000] (scores 0.95, 0.9, 0.6)", idx)
	}
}

func TestErrors(t *testing.T) {
	for _, c := range []Config{
		{W: 416, H: 416, Strides: []int{8, 16, 32}, Classes: 0, Scores: cfg416.Scores, Boxes: cfg416.Boxes},
		{W: 420, H: 416, Strides: []int{8, 16, 32}, Classes: 80, Scores: cfg416.Scores, Boxes: cfg416.Boxes},
		{W: 416, H: 416, Strides: nil, Classes: 80, Scores: cfg416.Scores, Boxes: cfg416.Boxes},
		{W: 416, H: 416, Strides: []int{8}, Classes: 80, Scores: Quant{}, Boxes: cfg416.Boxes},
	} {
		if _, err := New(c, 10); err != ErrConfig {
			t.Errorf("New(%+v): %v, want ErrConfig", c, err)
		}
	}
	d, _ := New(cfg416, 10)
	if _, err := d.Decode(make([]int8, 10), make([]int8, 3549*4), 0.3, 0.45, nil); err != ErrShape {
		t.Errorf("short scores: %v, want ErrShape", err)
	}
}

func TestDecodeNoAlloc(t *testing.T) {
	d, _ := New(cfg416, 200)
	s, b := tensors(d)
	for a := 0; a < d.Anchors(); a += 7 {
		s[a*80+a%80] = q(0.31+float32(a%60)/100, cfg416.Scores)
	}
	out := make([]nms.Box, 0, 100)
	if n := testing.AllocsPerRun(20, func() {
		if _, err := d.Decode(s, b, 0.3, 0.45, out[:0]); err != nil {
			t.Fatal(err)
		}
	}); n != 0 {
		t.Errorf("Decode allocates %v times per run, want 0", n)
	}
}
