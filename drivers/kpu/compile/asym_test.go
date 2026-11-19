package compile

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/itohio/EasyRobot/x/math/nn/qnet"
	"github.com/itohio/EasyRobot/x/math/primitive/qi8"
)

// TestLowerAsymmetricWeights: layers with a weight zero point (TFLite uint8
// "quant" models, shifted to int8 by qnet/tfl: Layer.WZP), lowered and run
// on the KPU model, against qnet's kernels on the same layer (Σ (q_w −
// WZP)(q_x − z_x)), over the full weight range: within 1 LSB, 2 with
// average pool.
func TestLowerAsymmetricWeights(t *testing.T) {
	r := rand.New(rand.NewPCG(8, 2026))
	layers := 600 // ≈ 1.3 M outputs; 75 s under -race at 1,200
	if testing.Short() {
		layers = 200
	}
	var hist [3]int
	for n := range layers {
		var c chain
		if n%4 == 0 { // stride-1 depthwise (randomChain draws regular convs)
			l := randomLayer(r)
			for l.Kind != qnet.DepthwiseConv2D {
				l = randomLayer(r)
			}
			c = chain{conv: l, in: l.In, out: l.Out}
		} else {
			c = randomChain(r)
		}
		for i := range c.conv.Weights {
			c.conv.Weights[i] = int8(r.IntN(256) - 128)
		}
		c.conv.WZP = int8(r.IntN(255) - 127)
		if c.conv.WZP == 0 {
			c.conv.WZP = 1
		}
		in := make([]int8, c.in.Len())
		for i := range in {
			in[i] = int8(r.IntN(256) - 128)
		}
		want := c.run(t, in)
		lc, err := Lower(&c.conv, c.fuse, 0, 16384)
		if err != nil {
			t.Fatalf("layer %d (%v k%d s%d, WZP %d): %v", n, c.conv.Kind, c.conv.KH, c.conv.StrideH, c.conv.WZP, err)
		}
		got := emulate(t, lc, in, c.in, c.out)
		for i := range want {
			d := int(got[i]) - int(want[i])
			d = max(d, -d)
			if d > 1 {
				t.Fatalf("layer %d (%v k%d s%d, z_x %d, WZP %d): output %d: KPU %d, qnet %d", n, c.conv.Kind, c.conv.KH, c.conv.StrideH, c.conv.InQ.ZP, c.conv.WZP, i, got[i], want[i])
			}
			hist[d]++
		}
	}
	t.Logf("%d asymmetric layers, %d outputs: %d equal, %d at 1 LSB", layers, hist[0]+hist[1], hist[0], hist[1])
}

// TestAsymmetricExtremeAccumulator: q_w − WZP = 255 on every tap and
// q_x − z_x = 255 on every input give the largest accumulator, 255²·taps;
// the BN scaling must keep it inside the 2^34 design bound, and the output
// saturates as qnet's does.
func TestAsymmetricExtremeAccumulator(t *testing.T) {
	for _, dw := range []bool{false, true} {
		const c = 16
		l := qnet.Layer{Kind: qnet.Conv2D, In: qi8.Shape{H: 4, W: 4, C: c}, Out: qi8.Shape{H: 4, W: 4, C: c},
			InQ: qnet.Quant{Scale: 1, ZP: -128}, OutQ: qnet.Quant{Scale: 1, ZP: -100},
			KH: 1, KW: 1, StrideH: 1, StrideW: 1, Padding: qi8.PaddingValid, ActMin: -128, ActMax: 127, WZP: -128}
		taps := c
		if dw {
			l.Kind, l.KH, l.KW, l.Padding = qnet.DepthwiseConv2D, 3, 3, qi8.PaddingSame
			taps = 9
			l.In.H, l.In.W, l.Out.H, l.Out.W = 3, 3, 3, 3 // the centre pixel sees no padding
		}
		nw := taps * c
		if dw {
			nw = 9 * c
		}
		l.Weights = make([]int8, nw)
		for i := range l.Weights {
			l.Weights[i] = 127 // q_w − WZP = 255
		}
		acc := float64(taps) * 255 * 255
		m := 3000 / acc // y ≈ 3000·2^P: the 2^34 bound decides P; the output clamps
		rq, err := qi8.QuantizeMultiplier(m)
		if err != nil {
			t.Fatal(err)
		}
		l.Requant = make([]qi8.Requant, c)
		for i := range l.Requant {
			l.Requant[i] = rq
		}
		bn, _, err := batchNorm(&l, int64(taps))
		if err != nil {
			t.Fatal(err)
		}
		for ch, e := range bn {
			if y := math.Ldexp(acc*float64(e.Mul), -int(e.Shift)) + float64(e.Add); math.Abs(y) >= 1<<34 {
				t.Fatalf("depthwise %v: channel %d: BN output %.4g for the largest accumulator, beyond 2^34", dw, ch, y)
			}
		}
		lc, err := Lower(&l, Fuse{}, 0, 16384)
		if err != nil {
			t.Fatal(err)
		}
		in := make([]int8, l.In.Len())
		for i := range in {
			in[i] = 127 // q_x − z_x = 255
		}
		got := emulate(t, lc, in, l.In, l.Out)
		want := min(int(math.Round(acc*multiplier(rq)))+int(l.OutQ.ZP), 127)
		centre := (l.Out.H/2*l.Out.W + l.Out.W/2) * c
		for ch := range c {
			if d := int(got[centre+ch]) - want; d > 1 || d < -1 {
				t.Fatalf("depthwise %v: channel %d: KPU %d, want %d", dw, ch, got[centre+ch], want)
			}
		}
	}
}

// TestTiesUpAveragePoolFused: TFLite's uint8 average pool rounds ties up
// (Layer.TiesUp), as the KPU's fused average does on its uint8 values
// (observed on the K210), so it is fused. Where the conv output is exact the
// pooled output equals qnet's; overall within 1 LSB.
func TestTiesUpAveragePoolFused(t *testing.T) {
	r := rand.New(rand.NewPCG(3, 9))
	var hist [2]int
	for n := 0; n < 150; {
		c := randomChain(r)
		if c.fuse.Pool == nil || c.fuse.Pool.Kind != qnet.AvgPool2D {
			continue
		}
		n++
		c.fuse.Pool.TiesUp = true
		in := make([]int8, c.in.Len())
		for i := range in {
			in[i] = int8(r.IntN(256) - 128)
		}
		want := c.run(t, in)
		lc, err := Lower(&c.conv, c.fuse, 0, 16384)
		if err != nil {
			t.Fatalf("chain %d: ties-up average pool: %v", n, err)
		}
		got := emulate(t, lc, in, c.in, c.out)
		for i := range want {
			d := int(got[i]) - int(want[i])
			d = max(d, -d)
			if d > 1 {
				t.Fatalf("chain %d output %d: KPU %d, qnet %d", n, i, got[i], want[i])
			}
			hist[d]++
		}
	}
	t.Logf("150 ties-up average-pool chains: %d equal, %d at 1 LSB", hist[0], hist[1])
}

// TestLowerAsymmetricSAMEStride2: TF SAME stride-2 3×3 layers (regular and
// depthwise, as in the official uint8 MobileNet) with a weight zero point,
// lowered at full resolution (Fuse.Full) and sampled at the odd pixels as
// the OddSample reader does, against qnet's stride-2 SAME layer: within
// 1 LSB.
func TestLowerAsymmetricSAMEStride2(t *testing.T) {
	r := rand.New(rand.NewPCG(5, 2026))
	var hist [2]int
	for n := range 200 {
		l := randomLayer(r)
		for l.KH != 3 {
			l = randomLayer(r)
		}
		h, w := 2*(1+r.IntN(12)), 2*(1+r.IntN(12))
		l.In.H, l.In.W = h, w
		l.StrideH, l.StrideW, l.Padding = 2, 2, qi8.PaddingSame
		l.Out.H, l.Out.W = h/2, w/2
		for i := range l.Weights {
			l.Weights[i] = int8(r.IntN(256) - 128)
		}
		l.WZP = int8(r.IntN(255) - 127)
		if l.WZP == 0 {
			l.WZP = 1
		}
		in := make([]int8, l.In.Len())
		for i := range in {
			in[i] = int8(r.IntN(256) - 128)
		}
		want := make([]int8, l.Out.Len())
		if err := qnet.RunLayer(&l, want, in); err != nil {
			t.Fatal(err)
		}
		lc, err := Lower(&l, Fuse{Full: true}, 0, 16384)
		if err != nil {
			t.Fatalf("layer %d (%v, WZP %d): %v", n, l.Kind, l.WZP, err)
		}
		full := emulate(t, lc, in, l.In, qi8.Shape{H: h, W: w, C: l.Out.C})
		got := sampleAt(full, l.Out, 1)
		for i := range want {
			d := int(got[i]) - int(want[i])
			d = max(d, -d)
			if d > 1 {
				t.Fatalf("layer %d (%v %v, z_x %d, WZP %d): output %d: KPU %d, qnet %d", n, l.Kind, l.In, l.InQ.ZP, l.WZP, i, got[i], want[i])
			}
			hist[d]++
		}
	}
	t.Logf("200 SAME stride-2 asymmetric layers: %d equal, %d at 1 LSB", hist[0], hist[1])
}
