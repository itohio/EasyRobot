package main

import (
	"hash/fnv"
	"math/rand/v2"
	"testing"

	"github.com/itohio/EasyRobot/x/math/nn/qnet"
	"github.com/itohio/EasyRobot/x/vision/detect/nms"
)

// A 1×1 Conv2D is timed as pointwise; other convolutions keep their kind.
func TestBucket(t *testing.T) {
	cases := []struct {
		l    qnet.Layer
		want string
	}{
		{qnet.Layer{Kind: qnet.Conv2D, KH: 1, KW: 1}, "pointwise"},
		{qnet.Layer{Kind: qnet.Conv2D, KH: 3, KW: 3}, "conv2d"},
		{qnet.Layer{Kind: qnet.Conv2D, KH: 1, KW: 3}, "conv2d"},
		{qnet.Layer{Kind: qnet.DepthwiseConv2D, KH: 1, KW: 1}, "dwconv2d"},
		{qnet.Layer{Kind: qnet.Transpose}, "transpose"},
	}
	for _, c := range cases {
		b := bucket(&c.l)
		if b < 0 || b >= nBuckets {
			t.Fatalf("%v %dx%d: bucket %d out of range", c.l.Kind, c.l.KH, c.l.KW, b)
		}
		if got := bucketName(b); got != c.want {
			t.Errorf("%v %dx%d: bucket %q, want %q", c.l.Kind, c.l.KH, c.l.KW, got, c.want)
		}
	}
}

// The cut QUANTIZE maps uint8 (1, 0) onto int8 (1, −128).
func TestToInt8(t *testing.T) {
	src := []byte{0x00, 0x01, 0x7f, 0x80, 0xfe, 0xff}
	want := []int8{-128, -127, -1, 0, 126, 127}
	got := make([]int8, len(src))
	toInt8(got, src)
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("byte %#x → %d, want %d", src[i], got[i], want[i])
		}
	}
}

// fnv1a hashes the int8 values as their bytes, like hash/fnv's New32a.
func TestFNV1a(t *testing.T) {
	r := rand.New(rand.NewPCG(1, 2))
	for _, n := range []int{0, 1, 7, 4096} {
		b := make([]byte, n)
		for i := range b {
			b[i] = byte(r.Uint32())
		}
		i8 := make([]int8, n)
		for i, v := range b {
			i8[i] = int8(v)
		}
		h := fnv.New32a()
		h.Write(b)
		if got, want := fnv1a(i8), h.Sum32(); got != want {
			t.Errorf("n=%d: %#x, want %#x", n, got, want)
		}
	}
}

// A detection prints as a line of the LiteRT reference's dets.txt.
func TestAppendDet(t *testing.T) {
	cases := []struct {
		d    nms.Box
		want string
	}{
		{nms.Box{Index: 3478, Class: 63, Score: 0.568, X0: 72.34, Y0: 17.21, X1: 399.54, Y1: 373.84},
			"3478 laptop 0.5680 72.34 17.21 399.54 373.84"},
		{nms.Box{Index: 3464, Class: 62, Score: 0.497, X0: 76.1, Y0: 28.83, X1: 385.05, Y1: 375.37},
			"3464 tv 0.4970 76.10 28.83 385.05 375.37"},
		{nms.Box{Index: 0, Class: 0, Score: 1, X0: -3.5, Y0: 0, X1: 416, Y1: 416},
			"0 person 1.0000 -3.50 0.00 416.00 416.00"},
		{nms.Box{Index: 3548, Class: 79, Score: 0.3},
			"3548 toothbrush 0.3000 0.00 0.00 0.00 0.00"},
	}
	for _, c := range cases {
		if got := string(appendDet(nil, c.d)); got != c.want {
			t.Errorf("got  %q\nwant %q", got, c.want)
		}
	}
}
