package ssd

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"math"
	"testing"
)

// The anchors of TF's quantised SSD-MobileNetV1 COCO export (300×300; head
// maps 19, 10, 5, 3, 2, 1) are stored in the model as a uint8 [1917×4]
// tensor of scale 0.0064530605, zero point 0. Anchors with the export's
// generator settings must quantise to exactly those bytes: their SHA-256,
// and the rows where a layer's box list starts and ends.
func TestAnchorsReproduceSSDMobileNetV1(t *testing.T) {
	const scale = 0.0064530605
	a, err := Anchors(MobileNetV1Anchors([][2]int{{19, 19}, {10, 10}, {5, 5}, {3, 3}, {2, 2}, {1, 1}}))
	if err != nil {
		t.Fatal(err)
	}
	if len(a) != 1917*4 {
		t.Fatalf("%d values, want 1917 rows of 4", len(a))
	}
	q := make([]byte, len(a))
	for i, v := range a {
		q[i] = byte(min(math.Round(float64(v)/scale), 255))
	}
	rows := map[int][4]byte{
		0: {4, 4, 15, 15}, 1: {4, 4, 22, 44}, 2: {4, 4, 44, 22}, 3: {4, 12, 15, 15}, // layer 0: 3 boxes a cell
		1083: {8, 8, 54, 54}, 1084: {8, 8, 38, 77}, 1085: {8, 8, 77, 38}, 1086: {8, 8, 31, 94},
		1087: {8, 8, 94, 31}, 1088: {8, 8, 65, 65}, 1089: {8, 23, 54, 54}, // layer 1: 6 boxes a cell
		1914: {77, 77, 85, 255}, 1915: {77, 77, 255, 85}, 1916: {77, 77, 151, 151}, // layer 5
	}
	for r, want := range rows {
		if got := [4]byte(q[4*r : 4*r+4]); got != want {
			t.Errorf("anchor %d quantises to %v, want %v (ycenter, xcenter, h, w)", r, got, want)
		}
	}
	sum := sha256.Sum256(q)
	if got := hex.EncodeToString(sum[:]); got != "5a878983aa11ce9461f210bde947cb8f4a8dc11c61a0f9d56a01ac36ed10fd4a" {
		t.Errorf("quantised anchors hash %s, want the export's anchors tensor", got)
	}
}

// SSD-MobileNetV1 run at 256×256 with its sampling grid kept computes what
// the export computes on the top-left 256×256 pixels of a 300×300 input:
// head maps 16, 8, 4, 2, 1, 1 out of 19, 10, 5, 3, 2, 1. Its anchors are the
// export's for those cells, in pixels: brought back to the 300 scale they
// must quantise to the stored tensor's rows for the cells (their SHA-256).
// A grid generated for the new maps would centre the 2×2 map's cell (1,1)
// at 0.75 instead of 150/256.
func TestCropAnchorsSSDMobileNetV1At256(t *testing.T) {
	const scale = 0.0064530605
	cfg := MobileNetV1Anchors([][2]int{{19, 19}, {10, 10}, {5, 5}, {3, 3}, {2, 2}, {1, 1}})
	a, err := CropAnchors(cfg, [2]int{300, 300}, [2]int{256, 256}, [][2]int{{16, 16}, {8, 8}, {4, 4}, {2, 2}, {1, 1}, {1, 1}})
	if err != nil {
		t.Fatal(err)
	}
	if len(a) != 1284*4 {
		t.Fatalf("%d values, want 1284 rows of 4", len(a))
	}
	const k = 300.0 / 256
	sq2 := float32(math.Sqrt2)
	rows := map[int][4]float32{
		0:    {0.5 / 19 * k, 0.5 / 19 * k, 0.1 * k, 0.1 * k},                                           // layer 0, cell (0,0)
		767:  {15.5 / 19 * k, 15.5 / 19 * k, 0.2 * k * sq2, 0.2 * k / sq2},                             // layer 0, cell (15,15), ratio 1/2
		768:  {0.05 * k, 0.05 * k, 0.35 * k, 0.35 * k},                                                 // layer 1, cell (0,0)
		1151: {0.75 * k, 0.75 * k, float32(math.Sqrt(0.35*0.5)) * k, float32(math.Sqrt(0.35*0.5)) * k}, // layer 1, cell (7,7), interpolated
		1266: {150.0 / 256, 150.0 / 256, 0.65 * k, 0.65 * k},                                           // layer 3 (3×3 → 2×2), cell (1,1)
		1272: {75.0 / 256, 75.0 / 256, 0.8 * k, 0.8 * k},                                               // layer 4 (2×2 → 1×1)
		1278: {150.0 / 256, 150.0 / 256, 0.95 * k, 0.95 * k},                                           // layer 5
	}
	for r, want := range rows {
		for c, w := range want {
			if got := a[4*r+c]; math.Abs(float64(got-w)) > 1e-6 {
				t.Errorf("anchor %d value %d is %v, want %v (ycenter, xcenter, h, w)", r, c, got, w)
			}
		}
	}
	q := make([]byte, len(a))
	for i, v := range a {
		q[i] = byte(min(math.Round(float64(v)/k/scale), 255))
	}
	sum := sha256.Sum256(q)
	if got := hex.EncodeToString(sum[:]); got != "9a4685fc60c6ddcb3faa036728601d0c06fcf67f827194ae401ff3815f69a443" {
		t.Errorf("anchors at the 300 scale hash %s, want the export's anchors for the remaining cells", got)
	}
}

// A crop that is not square keeps rows and columns separately and scales
// each axis by its own ratio; the rows stay in the model's output order.
func TestCropAnchorsPerAxis(t *testing.T) {
	cfg := MobileNetV1Anchors([][2]int{{4, 4}, {2, 2}})
	a, err := CropAnchors(cfg, [2]int{64, 64}, [2]int{48, 32}, [][2]int{{3, 2}, {2, 1}})
	if err != nil {
		t.Fatal(err)
	}
	if len(a) != (3*2*3+2*1*6)*4 {
		t.Fatalf("%d values, want 30 rows of 4", len(a))
	}
	full, err := Anchors(cfg)
	if err != nil {
		t.Fatal(err)
	}
	// Layer 0, cell (2,1) is cell 9 of the 4×4 map and cell 5 of the kept
	// 3×2; layer 1, cell (1,0) is cell 2 of the 2×2 map and cell 1 of the
	// kept 2×1 (after the 18 rows of layer 0).
	for _, c := range []struct{ got, from int }{{5 * 3, 9 * 3}, {18 + 1*6 + 5, 48 + 2*6 + 5}} {
		g, f := a[4*c.got:4*c.got+4], full[4*c.from:4*c.from+4]
		want := [4]float32{f[0] * 64 / 48, f[1] * 64 / 32, f[2] * 64 / 48, f[3] * 64 / 32}
		for i, w := range want {
			if math.Abs(float64(g[i]-w)) > 1e-6 {
				t.Errorf("row %d value %d is %v, want %v", c.got, i, g[i], w)
			}
		}
	}
}

// The kept maps must be one per layer, non-empty and inside the trained
// maps, and the sizes positive.
func TestCropAnchorsRejectsInvalid(t *testing.T) {
	cfg := MobileNetV1Anchors([][2]int{{4, 4}, {2, 2}})
	size := [2]int{64, 64}
	for name, c := range map[string]struct {
		trained, cropped [2]int
		keep             [][2]int
	}{
		"one map short":   {size, size, [][2]int{{4, 4}}},
		"larger than map": {size, size, [][2]int{{4, 5}, {2, 2}}},
		"empty":           {size, size, [][2]int{{4, 4}, {0, 2}}},
		"zero size":       {size, [2]int{48, 0}, [][2]int{{3, 3}, {2, 2}}},
		"negative size":   {[2]int{-64, 64}, size, [][2]int{{3, 3}, {2, 2}}},
	} {
		if _, err := CropAnchors(cfg, c.trained, c.cropped, c.keep); !errors.Is(err, ErrAnchorConfig) {
			t.Errorf("%s: error %v, want ErrAnchorConfig", name, err)
		}
	}
	bad := cfg
	bad.MinScale = 0
	if _, err := CropAnchors(bad, size, size, cfg.Maps); !errors.Is(err, ErrAnchorConfig) {
		t.Errorf("invalid config: error %v, want ErrAnchorConfig", err)
	}
}

// One feature map leaves the scale step undefined, and a non-positive size,
// scale or ratio gives no box: all are refused.
func TestAnchorsRejectsInvalidConfig(t *testing.T) {
	ok := MobileNetV1Anchors([][2]int{{4, 4}, {2, 2}})
	if _, err := Anchors(ok); err != nil {
		t.Fatal(err)
	}
	bad := map[string]func(*AnchorConfig){
		"one map":        func(c *AnchorConfig) { c.Maps = c.Maps[:1] },
		"empty map":      func(c *AnchorConfig) { c.Maps = [][2]int{{4, 0}, {2, 2}} },
		"zero min scale": func(c *AnchorConfig) { c.MinScale = 0 },
		"max below min":  func(c *AnchorConfig) { c.MaxScale = c.MinScale / 2 },
		"no ratios":      func(c *AnchorConfig) { c.AspectRatios = nil },
		"negative ratio": func(c *AnchorConfig) { c.AspectRatios = []float32{1, -2} },
	}
	for name, mutate := range bad {
		c := ok
		c.AspectRatios = append([]float32(nil), ok.AspectRatios...)
		mutate(&c)
		if _, err := Anchors(c); !errors.Is(err, ErrAnchorConfig) {
			t.Errorf("%s: error %v, want ErrAnchorConfig", name, err)
		}
	}
}
