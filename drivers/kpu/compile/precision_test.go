package compile

import (
	"errors"
	"math"
	"testing"

	"github.com/itohio/EasyRobot/x/math/nn/qnet"
	"github.com/itohio/EasyRobot/x/math/primitive/qi8"
)

// conv1x1 is a 1×1 Conv2D on a 2×2×ic input with the given per-channel
// multipliers and biases; weights 127, input zero point −128.
func conv1x1(t *testing.T, ic int, ms []float64, bias []int32) qnet.Layer {
	t.Helper()
	oc := len(ms)
	l := qnet.Layer{Kind: qnet.Conv2D, In: qi8.Shape{H: 2, W: 2, C: ic}, Out: qi8.Shape{H: 2, W: 2, C: oc},
		InQ: qnet.Quant{Scale: 1, ZP: -128}, OutQ: qnet.Quant{Scale: 1, ZP: -100},
		KH: 1, KW: 1, StrideH: 1, StrideW: 1, Padding: qi8.PaddingValid, ActMin: -128, ActMax: 127,
		Weights: make([]int8, ic*oc), Bias: bias, Requant: make([]qi8.Requant, oc)}
	for i := range l.Weights {
		l.Weights[i] = 127
	}
	for c, m := range ms {
		rq, err := qi8.QuantizeMultiplier(m)
		if err != nil {
			t.Fatal(err)
		}
		l.Requant[c] = rq
	}
	return l
}

// TestLowerConvPrecisionFloor: P is one per layer. A channel whose bias·M
// forces a small P leaves another channel with a tiny multiplier too few
// BN bits to stay within 1 LSB: such a layer is refused. With both
// multipliers ordinary, the same small P lowers.
func TestLowerConvPrecisionFloor(t *testing.T) {
	big := int32(1 << 30) // bias·M·2^P must stay inside int32: P = 1
	l := conv1x1(t, 4, []float64{0.5, 1.1e-5}, []int32{big, 0})
	if _, err := LowerConv(&l, 0, 1024); !errors.Is(err, ErrUnsupported) {
		t.Errorf("tiny multiplier at P = 1: err %v, want ErrUnsupported", err)
	}
	ok := conv1x1(t, 4, []float64{0.5, 0.25}, []int32{big, 0})
	if _, err := LowerConv(&ok, 0, 1024); err != nil {
		t.Errorf("ordinary multipliers at P = 1: %v", err)
	}
}

// TestLowerConvYBound: 1024 input channels of extreme weights and inputs
// give an accumulator near its worst case; the post-shift P must keep the
// BN output inside the 2^34 design bound (the 36-bit activation compare
// with a factor-2 margin), and the KPU output must match qnet's.
func TestLowerConvYBound(t *testing.T) {
	const ic = 1024
	l := conv1x1(t, ic, []float64{math.Ldexp(1, -10)}, []int32{0})
	bn, _, err := batchNorm(&l, ic)
	if err != nil {
		t.Fatal(err)
	}
	acc := float64(ic) * 127 * 255 // weights 127, inputs 127 − (−128)
	if y := math.Ldexp(acc*float64(bn[0].Mul), -int(bn[0].Shift)) + float64(bn[0].Add); math.Abs(y) >= 1<<34 {
		t.Fatalf("BN output %.4g for the largest accumulator, beyond 2^34", y)
	}
	c, err := LowerConv(&l, 0, 1024)
	if err != nil {
		t.Fatal(err)
	}
	in := make([]int8, l.In.Len())
	for i := range in {
		in[i] = 127
	}
	got := emulate(t, c, in, l.In, l.Out)
	want := make([]int8, l.Out.Len())
	if err := qnet.RunLayer(&l, want, in); err != nil {
		t.Fatal(err)
	}
	for i := range want {
		if d := int(got[i]) - int(want[i]); d > 1 || d < -1 {
			t.Fatalf("output %d: KPU %d, qnet %d", i, got[i], want[i])
		}
	}
}

// TestLowerConvSpreadMultipliers (review R-K210-K9b L1): a 3×3×1024 conv
// with M = 1 on one channel and M = 2⁻²² on another. The first forces a
// small P, which would round the second channel's BN multiplier to 0 (its
// output stuck at zp + bias·M, 71 LSB off): the layer is refused.
func TestLowerConvSpreadMultipliers(t *testing.T) {
	const ic = 1024
	l := conv1x1(t, ic, []float64{0.999, math.Ldexp(1, -22)}, []int32{0, 0})
	l.KH, l.KW, l.Padding = 3, 3, qi8.PaddingSame
	l.Weights = make([]int8, 9*ic*2)
	if _, err := LowerConv(&l, 0, 1024); !errors.Is(err, ErrUnsupported) {
		t.Errorf("M = 1 and M = 2⁻²² in one layer: err %v, want ErrUnsupported", err)
	}
}

// TestLowerConvYBoundBias (review R-K210-K9b L2): the bias counts in the
// y bound. IC 1024, extreme weights and inputs, M chosen so the
// accumulator alone fits at P = 20 but accumulator + bias does not: P must
// drop, keeping the BN output for the largest accumulator plus the bias
// inside 2^34.
func TestLowerConvYBoundBias(t *testing.T) {
	const ic, bias = 1024, 3_000_000
	accMax := float64(ic) * 128 * 255
	m := 0.95 * math.Ldexp(1, 14) / accMax // accMax·M·2^20 = 0.95·2^34
	l := conv1x1(t, ic, []float64{m}, []int32{bias})
	bn, _, err := batchNorm(&l, ic)
	if err != nil {
		t.Fatal(err)
	}
	acc := float64(ic) * 127 * 255
	if y := math.Ldexp(acc*float64(bn[0].Mul), -int(bn[0].Shift)) + float64(bn[0].Add); math.Abs(y) >= 1<<34 {
		t.Fatalf("BN output %.4g for the largest accumulator plus the bias, beyond 2^34", y)
	}
}

// TestUnclampedWindow (review R-K210-K9b recheck): the precision floor
// only counts accumulators whose output (acc + b)·m lies within 256 LSB
// of zero; the window follows the bias and is empty when every output
// clamps.
func TestUnclampedWindow(t *testing.T) {
	const m = 1.0 / 256 // 256 LSB = 65,536 accumulator units
	for _, c := range []struct {
		accMax, b, want float64
	}{
		{1e6, 0, 65536},         // the window, inside the range
		{1000, 0, 1000},         // the whole range
		{1000, 1e6, 0},          // every output clamps high
		{1000, -1e6, 0},         // every output clamps low
		{1e6, 30000, 95536},     // shifted window: [−95,536, 35,536], the far edge counts
		{1e6, -30000, 95536},    // mirrored
		{50000, 30000, 50000},   // clipped to the range on one side
		{1e6, 65536 + 1e6, 1e6}, // touches −accMax only: (−1e6 + b)·m = 256
		{1e6, 65537 + 1e6, 0},   // just past it
		{100, 65536 + 100.5, 0}, // all clamp although the range is tiny
	} {
		if got := unclamped(c.accMax, c.b, m); got != c.want {
			t.Errorf("unclamped(accMax %g, b %g): %g, want %g", c.accMax, c.b, got, c.want)
		}
	}
}

// TestLowerConvClampedChannel: a channel with a tiny multiplier is refused
// at a small P (TestLowerConvPrecisionFloor) unless its bias clamps every
// output, which its BN precision cannot change: then the layer lowers and
// matches qnet. A bias that leaves outputs up to 200 LSB from zero keeps
// them inside the window, so the layer is still refused.
func TestLowerConvClampedChannel(t *testing.T) {
	const m = 1.1e-5
	big := int32(1 << 30) // forces P = 1 through channel 0
	accMax := 4.0 * 128 * 255
	for _, c := range []struct {
		b  int32
		ok bool
	}{
		{30_000_000, true},             // (acc + b)·m > 256 for every acc
		{-30_000_000, true},            // < −256 for every acc
		{int32(accMax + 200/m), false}, // reaches 200 LSB at acc = −accMax
		{-int32(accMax + 200/m), false},
	} {
		l := conv1x1(t, 4, []float64{0.5, m}, []int32{big, c.b})
		conv, err := LowerConv(&l, 0, 1024)
		if !c.ok {
			if !errors.Is(err, ErrUnsupported) {
				t.Errorf("bias %d: err %v, want ErrUnsupported", c.b, err)
			}
			continue
		}
		if err != nil {
			t.Errorf("bias %d: %v", c.b, err)
			continue
		}
		for _, v := range []int8{127, -128, 0} {
			in := make([]int8, l.In.Len())
			for i := range in {
				in[i] = v
			}
			got := emulate(t, conv, in, l.In, l.Out)
			want := make([]int8, l.Out.Len())
			if err := qnet.RunLayer(&l, want, in); err != nil {
				t.Fatal(err)
			}
			for i := range want {
				if d := int(got[i]) - int(want[i]); d > 1 || d < -1 {
					t.Errorf("bias %d, input %d, output %d: KPU %d, qnet %d", c.b, v, i, got[i], want[i])
				}
			}
		}
	}
}
