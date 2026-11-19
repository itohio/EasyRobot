package kmodel

import (
	"bytes"
	"testing"
)

func TestCarryShift(t *testing.T) {
	// kpu_carry_shift: positive halves round up; a negative value whose last
	// shifted-out bit is 1 gets floor(v/2^s) − 1 (−2.5 → −4), as in the C
	// runtime and nncase's emulator.
	for _, c := range []struct {
		v     int64
		shift uint
		want  int64
	}{{5, 1, 3}, {4, 1, 2}, {-5, 1, -4}, {-4, 1, -2}, {6, 2, 2}, {-6, 2, -3}, {7, 0, 7}, {1023, 10, 1}, {511, 10, 0}} {
		if got := carryShift(c.v, c.shift); got != c.want {
			t.Errorf("carryShift(%d, %d) = %d, want %d", c.v, c.shift, got, c.want)
		}
	}
}

func TestSigned(t *testing.T) {
	if got := signed(0xC0_0000, 24); got != -4194304 {
		t.Errorf("signed 24: %d", got)
	}
	if got := signed(0x800000000, 36); got != -(1 << 35) {
		t.Errorf("signed 36: %d", got)
	}
	if got := signed(0x7f, 8); got != 127 {
		t.Errorf("signed 8: %d", got)
	}
}

// identityAct passes y through unchanged for y ≥ 0 (clamped to 255) and
// gives 0 below: segments 0–9 at the lowest start with mul 0, segment 10 at
// 0 with mul 1, segments 11–15 above any value (nncase's ReLU layout).
func identityAct() (a [16]ActSegment) {
	for i := range a {
		a[i] = ActSegment{StartX: -(1 << 35)}
		if i > 10 {
			a[i].StartX = 1<<35 - 16 + int64(i)
		}
	}
	a[10] = ActSegment{StartX: 0, Mul: 1}
	return a
}

func TestEmulate1x1(t *testing.T) {
	// 2×1 image, 2 input channels → 1 output channel, 1×1 kernel.
	var args Args
	args[2] = 1 | 0<<32                 // ic 2, oc 1
	args[3] = 1 | 0<<10 | 1<<32 | 0<<42 // 2×1 in and out
	p := ConvParams{
		Args: args, Kernel: 1,
		Weights: []byte{2, 3},
		ArgX:    4, ShrX: 1, // + (4·Σx) >> 1
		ArgW: 6, ShrW: 1, // + (6·Σw) >> 1 = 15
		ArgAdd: 1, // + 1·ic = 2
		BN:     []BN{{Mul: 3, Shift: 1, Add: -10}},
		Act:    identityAct(),
	}
	in := []byte{10, 20, 1, 2} // channel 0: 10 20, channel 1: 1 2
	out := make([]byte, 2)
	if err := p.Emulate(in, out); err != nil {
		t.Fatal(err)
	}
	// pixel 0: Σwx = 2·10 + 3·1 = 23; Σx = 11 → 22; acc = 23+22+15+2 = 62
	//          BN: 62·3>>1 − 10 = 83
	// pixel 1: Σwx = 40 + 6 = 46; Σx = 22 → 44; acc = 107; 107·3>>1 − 10 = 150
	if !bytes.Equal(out, []byte{83, 150}) {
		t.Errorf("got %v, want [83 150]", out)
	}
}

func TestEmulate3x3PadAndActivation(t *testing.T) {
	// 1-channel 3×3 depthwise of all-ones weights on a 2×2 image with pad
	// value 5: each output sums its four in-image pixels plus five pads.
	var args Args
	args[0] = 1 << 3 // depthwise
	args[3] = 1 | 1<<10 | 1<<32 | 1<<42
	act := identityAct()
	act[11] = ActSegment{StartX: 40, Mul: 0, Add: 255} // saturate from 40 up
	p := ConvParams{
		Args: args, Depthwise: true, Kernel: 3, PadValue: 5,
		Weights: bytes.Repeat([]byte{1}, 9),
		BN:      []BN{{Mul: 1}},
		Act:     act,
	}
	in := []byte{1, 2, 3, 4}
	out := make([]byte, 4)
	if err := p.Emulate(in, out); err != nil {
		t.Fatal(err)
	}
	// Every 3×3 window covers all four pixels (sum 10) and 5 pads (25): 35.
	if !bytes.Equal(out, []byte{35, 35, 35, 35}) {
		t.Errorf("got %v", out)
	}
	p.PadValue = 7 // 10 + 35 = 45 ≥ 40 → segment 11 → 255
	if err := p.Emulate(in, out); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(out, []byte{255, 255, 255, 255}) {
		t.Errorf("segment 11 not used: %v", out)
	}
}

func TestPool(t *testing.T) {
	src := []byte{
		1, 9, 2, 2,
		3, 4, 2, 6,
	} // 4×2, one channel
	for _, c := range []struct {
		typ  int
		want []byte
	}{{1, []byte{9, 6}}, {2, []byte{4, 3}}, {5, []byte{1, 2}}, {6, []byte{9, 2}}} {
		dst := make([]byte, 2)
		if err := pool(c.typ, src, 4, 2, 1, dst, 2, 1); err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(dst, c.want) {
			t.Errorf("pool %d: %v, want %v", c.typ, dst, c.want)
		}
	}
	// The average rounds half up, as observed on the K210 (nncase's
	// emulator floors): 18/4 = 4.5 → 5, 6/4 = 1.5 → 2, 13/4 = 3.25 → 3.
	ties := []byte{
		4, 5, 1, 1, 3, 3,
		4, 5, 2, 2, 3, 4,
	} // 6×2
	dst := make([]byte, 3)
	if err := pool(2, ties, 6, 2, 1, dst, 3, 1); err != nil || !bytes.Equal(dst, []byte{5, 2, 3}) {
		t.Errorf("pool 2 ties: %v %v, want [5 2 3]", dst, err)
	}
	if err := pool(7, src, 4, 2, 1, make([]byte, 2), 2, 1); err == nil {
		t.Error("pool type 7 accepted")
	}
}
