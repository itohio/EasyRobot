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
