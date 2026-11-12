package compile

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/itohio/EasyRobot/x/math/nn/qnet"
	"github.com/itohio/EasyRobot/x/math/primitive/qi8"
)

// asymmetric splits the chain's conv weights into weights q_w and a zero
// point z_w ≠ 0 with q_w − z_w = the reference weights (clamped so q_w
// fits int8). It returns the asymmetric conv; c.conv keeps the reference.
func asymmetric(r *rand.Rand, c *chain) (qnet.Layer, int8) {
	zw := int8(r.IntN(201) - 100)
	if zw == 0 {
		zw = 1
	}
	a := c.conv
	a.Weights = make([]int8, len(c.conv.Weights))
	for i, v := range c.conv.Weights {
		ref := min(max(int(v), -128-int(zw)), 127-int(zw))
		c.conv.Weights[i] = int8(ref)
		a.Weights[i] = int8(ref + int(zw))
	}
	return a, zw
}

// TestLowerAsymmetricWeights: layers with a weight zero point (TFLite uint8
// "quant" models), lowered with z_w and run on the KPU model, against qnet
// on the equivalent symmetric layer (weights q_w − z_w), which is exactly
// TFLite's Σ (q_w − z_w)(q_x − z_x): within 1 LSB, 2 with average pool.
func TestLowerAsymmetricWeights(t *testing.T) {
	r := rand.New(rand.NewPCG(8, 2026))
	layers := 1200
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
		a, zw := asymmetric(r, &c)
		in := make([]int8, c.in.Len())
		for i := range in {
			in[i] = int8(r.IntN(256) - 128)
		}
		want := c.run(t, in)
		lc, err := lowerZW(&a, c.fuse, 0, 16384, zw)
		if err != nil {
			t.Fatalf("layer %d (%v k%d s%d, z_w %d): %v", n, a.Kind, a.KH, a.StrideH, zw, err)
		}
		got := emulate(t, lc, in, c.in, c.out)
		tol := 1
		if c.fuse.Pool != nil && c.fuse.Pool.Kind == qnet.AvgPool2D {
			tol = 2
		}
		for i := range want {
			d := int(got[i]) - int(want[i])
			d = max(d, -d)
			if d > tol {
				t.Fatalf("layer %d (%v k%d s%d, z_x %d, z_w %d): output %d: KPU %d, qnet %d", n, a.Kind, a.KH, a.StrideH, a.InQ.ZP, zw, i, got[i], want[i])
			}
			hist[d]++
		}
	}
	t.Logf("%d asymmetric layers, %d outputs: %d equal, %d at 1 LSB, %d at 2 (average pool only)", layers, hist[0]+hist[1]+hist[2], hist[0], hist[1], hist[2])
}

// TestAsymmetricExtremeAccumulator: q_w − z_w = 255 on every tap and
// q_x − z_x = 255 on every input give the largest accumulator, 255²·taps;
// the BN scaling must keep it inside the 2^34 design bound, and the output
// saturates as qnet's does.
func TestAsymmetricExtremeAccumulator(t *testing.T) {
	for _, dw := range []bool{false, true} {
		const c = 16
		l := qnet.Layer{Kind: qnet.Conv2D, In: qi8.Shape{H: 4, W: 4, C: c}, Out: qi8.Shape{H: 4, W: 4, C: c},
			InQ: qnet.Quant{Scale: 1, ZP: -128}, OutQ: qnet.Quant{Scale: 1, ZP: -100},
			KH: 1, KW: 1, StrideH: 1, StrideW: 1, Padding: qi8.PaddingValid, ActMin: -128, ActMax: 127}
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
			l.Weights[i] = 127 // with z_w = −128: q_w − z_w = 255
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
		// The scaling keeps the BN output inside the 2^34 design bound (the
		// 36-bit activation compare with a factor-2 margin).
		bn, _, err := batchNorm(&l, int64(taps), -128)
		if err != nil {
			t.Fatal(err)
		}
		for ch, e := range bn {
			if y := math.Ldexp(acc*float64(e.Mul), -int(e.Shift)) + float64(e.Add); math.Abs(y) >= 1<<34 {
				t.Fatalf("depthwise %v: channel %d: BN output %.4g for the largest accumulator, beyond 2^34", dw, ch, y)
			}
		}
		lc, err := lowerZW(&l, Fuse{}, 0, 16384, -128)
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
