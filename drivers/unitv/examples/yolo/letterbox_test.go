package main

import (
	"testing"

	"github.com/itohio/EasyRobot/x/vision/detect/nms"
)

// rgb565 packs 8-bit r, g, b into a little-endian RGB565 pixel.
func rgb565(r, g, b uint8) (lo, hi byte) {
	p := uint16(r>>3)<<11 | uint16(g>>2)<<5 | uint16(b>>3)
	return byte(p), byte(p >> 8)
}

func fill565(w, h int, r, g, b uint8) []byte {
	img := make([]byte, 2*w*h)
	lo, hi := rgb565(r, g, b)
	for i := 0; i < len(img); i += 2 {
		img[i], img[i+1] = lo, hi
	}
	return img
}

// pixel returns the int8 input value + 128 of channel c at (x, y).
func pixel(in []int8, x, y, c int) int { return int(in[(y*inputSize+x)*3+c]) + 128 }

func TestLetterboxConstantImage(t *testing.T) {
	lb := newLetterbox(320, 240)
	if lb.scale != 1.3 || lb.rw != 416 || lb.rh != 312 {
		t.Fatalf("scale %v, resized %d×%d; want 1.3, 416×312", lb.scale, lb.rw, lb.rh)
	}
	in := make([]int8, inputSize*inputSize*3)
	// 8-bit values that survive RGB565 exactly: r, b multiples of 8 with
	// the top bits replicated, g of 4.
	lb.run(in, fill565(320, 240, 0xff, 0x82, 0x00))
	for _, p := range [][2]int{{0, 0}, {415, 0}, {200, 150}, {0, 311}, {415, 311}} {
		if r, g, b := pixel(in, p[0], p[1], 0), pixel(in, p[0], p[1], 1), pixel(in, p[0], p[1], 2); r != 0xff || g != 0x82 || b != 0 {
			t.Errorf("image pixel %v = %d,%d,%d; want 255,130,0", p, r, g, b)
		}
	}
	for _, p := range [][2]int{{0, 312}, {415, 415}, {200, 360}} {
		for c := range 3 {
			if v := pixel(in, p[0], p[1], c); v != padValue {
				t.Errorf("pad pixel %v channel %d = %d, want %d", p, c, v, padValue)
			}
		}
	}
}

// A horizontal ramp stays monotonic and spans the source range.
func TestLetterboxRamp(t *testing.T) {
	const w, h = 320, 240
	img := make([]byte, 2*w*h)
	for y := range h {
		for x := range w {
			lo, hi := rgb565(0, uint8(x*255/(w-1)), 0)
			img[2*(y*w+x)], img[2*(y*w+x)+1] = lo, hi
		}
	}
	lb := newLetterbox(w, h)
	in := make([]int8, inputSize*inputSize*3)
	lb.run(in, img)
	prev := -1
	for x := range inputSize {
		g := pixel(in, x, 100, 1)
		if g < prev {
			t.Fatalf("green falls at x=%d: %d after %d", x, g, prev)
		}
		prev = g
	}
	if first, last := pixel(in, 0, 100, 1), pixel(in, inputSize-1, 100, 1); first > 4 || last < 251 {
		t.Errorf("ramp ends %d..%d, want ≈0..255", first, last)
	}
}

// A box in detector pixels maps back to the source image's pixels.
func TestLetterboxToImage(t *testing.T) {
	lb := newLetterbox(320, 240)
	d := lb.toImage(nms.Box{X0: -13, Y0: 13, X1: 403, Y1: 312, Score: 0.568, Class: 63})
	if d.X0 != -10 || d.Y0 != 10 || d.X1 != 310 || d.Y1 != 240 || d.Class != 63 || d.Score != 37224 {
		t.Errorf("got %+v", d)
	}
	if d := lb.toImage(nms.Box{Score: 1}); d.Score != 65535 {
		t.Errorf("score 1 → %d, want 65535", d.Score)
	}
}

func TestLetterboxAllocatesNothing(t *testing.T) {
	lb := newLetterbox(320, 240)
	in := make([]int8, inputSize*inputSize*3)
	img := fill565(320, 240, 10, 20, 30)
	if n := testing.AllocsPerRun(3, func() { lb.run(in, img) }); n != 0 {
		t.Errorf("%v allocations per run", n)
	}
}
