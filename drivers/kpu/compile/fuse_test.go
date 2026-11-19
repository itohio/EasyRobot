package compile

import (
	"errors"
	"math/rand/v2"
	"testing"

	"github.com/itohio/EasyRobot/x/math/nn/qnet"
	"github.com/itohio/EasyRobot/x/math/primitive/qi8"
)

// chain is a conv with its fused layers, as qnet runs them one by one.
type chain struct {
	conv qnet.Layer
	fuse Fuse
	in   qi8.Shape // the chain's input (before a Pad)
	out  qi8.Shape // after the pool or sampler
}

// randomChain draws a lowerable conv with stride 2 (Pad + 3×3 VALID, or
// 1×1 VALID), or stride 1 with an optional fused ReLU/ReLU6 and pool.
func randomChain(r *rand.Rand) chain {
	l := randomLayer(r)
	// A depthwise conv takes neither a pool nor the stride-2 sampler on the
	// KPU (pairs are tested in TestStride2DepthwisePair): use a regular conv.
	if l.Kind == qnet.DepthwiseConv2D {
		l.Kind = qnet.Conv2D
		l.Weights = make([]int8, l.KH*l.KW*l.In.C*l.Out.C)
		for i := range l.Weights {
			l.Weights[i] = int8(r.IntN(255) - 127)
		}
	}
	c := chain{in: l.In}
	evenUp := func(v int) int { return v + v%2 }
	switch r.IntN(3) {
	case 0: // stride 2
		in := qi8.Shape{H: evenUp(l.In.H), W: evenUp(l.In.W), C: l.In.C}
		c.in = in
		out := qi8.Shape{H: in.H / 2, W: in.W / 2, C: l.Out.C}
		l.StrideH, l.StrideW, l.Padding = 2, 2, qi8.PaddingValid
		if l.KH == 3 {
			pv := l.InQ.ZP
			if r.IntN(2) == 0 {
				pv = int8(r.IntN(256) - 128)
			}
			c.fuse.Pad = &qnet.Layer{Kind: qnet.Pad, In: in, Out: qi8.Shape{H: in.H + 2, W: in.W + 2, C: in.C},
				InQ: l.InQ, OutQ: l.InQ, Pads: [6]int{1, 1, 1, 1, 0, 0}, PadValue: pv}
			l.In = c.fuse.Pad.Out
		} else {
			l.In = in
		}
		l.Out = out
		c.out = out
	default: // stride 1, optional activation and pool
		c.out = l.Out
		l.OutQ.Scale = 0.01 + 0.1*r.Float32()
		switch r.IntN(3) {
		case 1:
			c.fuse.Act = &qnet.Layer{Kind: qnet.ReLU, In: l.Out, Out: l.Out, InQ: l.OutQ, OutQ: l.OutQ}
		case 2:
			c.fuse.Act = &qnet.Layer{Kind: qnet.ReLU6, In: l.Out, Out: l.Out, InQ: l.OutQ, OutQ: l.OutQ}
		}
		if r.IntN(2) == 0 {
			pk := 2 + 2*r.IntN(2)
			h, w := pk*(1+r.IntN(6)), pk*(1+r.IntN(6))
			l.In.H, l.In.W, l.Out.H, l.Out.W = h, w, h, w
			if c.fuse.Act != nil {
				c.fuse.Act.In, c.fuse.Act.Out = l.Out, l.Out
			}
			kind := qnet.MaxPool2D
			if r.IntN(2) == 0 {
				kind = qnet.AvgPool2D
			}
			c.out = qi8.Shape{H: h / pk, W: w / pk, C: l.Out.C}
			c.fuse.Pool = &qnet.Layer{Kind: kind, In: l.Out, Out: c.out, InQ: l.OutQ, OutQ: l.OutQ,
				KH: pk, KW: pk, StrideH: pk, StrideW: pk, Padding: qi8.PaddingValid}
			c.in = l.In
		}
	}
	c.conv = l
	return c
}

// run executes the chain with qnet's CPU kernels, layer by layer.
func (c chain) run(t *testing.T, in []int8) []int8 {
	t.Helper()
	step := func(l *qnet.Layer, src []int8) []int8 {
		dst := make([]int8, l.Out.H*l.Out.W*l.Out.C)
		if err := qnet.RunLayer(l, dst, src); err != nil {
			t.Fatalf("qnet %v: %v", l.Kind, err)
		}
		return dst
	}
	x := in
	if c.fuse.Pad != nil {
		x = step(c.fuse.Pad, x)
	}
	x = step(&c.conv, x)
	if c.fuse.Act != nil {
		x = step(c.fuse.Act, x)
	}
	if c.fuse.Pool != nil {
		x = step(c.fuse.Pool, x)
	}
	return x
}

// TestLowerFusedMatchesQnet: random convs with stride 2 (Pad + VALID),
// fused ReLU/ReLU6 and fused pools, lowered and run on the KPU model,
// against qnet's layer-by-layer result: within 1 LSB (the average pool
// rounds half up, as the silicon does).
func TestLowerFusedMatchesQnet(t *testing.T) {
	r := rand.New(rand.NewPCG(9, 211))
	layers := 1500
	if testing.Short() {
		layers = 200
	}
	var hist [3]int
	kinds := map[string]int{}
	off := map[string][3]int{} // per kind: outputs at 0, 1, 2 LSB
	for n := range layers {
		c := randomChain(r)
		in := make([]int8, c.in.H*c.in.W*c.in.C)
		for i := range in {
			in[i] = int8(r.IntN(256) - 128)
		}
		want := c.run(t, in)
		lc, err := Lower(&c.conv, c.fuse, 0, 16384)
		if err != nil {
			t.Fatalf("chain %d (%v k%d s%d pad %v act %v pool %v, in %v): %v", n, c.conv.Kind, c.conv.KH, c.conv.StrideH,
				c.fuse.Pad != nil, c.fuse.Act != nil, c.fuse.Pool != nil, c.in, err)
		}
		got := emulate(t, lc, in, c.in, c.out)
		tol := 1
		key := "stride 1"
		switch {
		case c.conv.StrideH == 2 && c.fuse.Pad != nil:
			key = "stride 2, Pad + 3×3"
		case c.conv.StrideH == 2:
			key = "stride 2, 1×1"
		case c.fuse.Pool != nil && c.fuse.Pool.Kind == qnet.AvgPool2D:
			key = "avg pool"
		case c.fuse.Pool != nil:
			key = "max pool"
		}
		if c.fuse.Act != nil {
			key += " + " + c.fuse.Act.Kind.String()
		}
		kinds[key]++
		for i := range want {
			d := int(got[i]) - int(want[i])
			d = max(d, -d)
			if d > tol {
				t.Fatalf("chain %d (%s, %v k%d, in %v → %v, zp %d→%d, clamp [%d,%d]): output %d: KPU %d, qnet %d",
					n, key, c.conv.Kind, c.conv.KH, c.in, c.out, c.conv.InQ.ZP, c.conv.OutQ.ZP, c.conv.ActMin, c.conv.ActMax, i, got[i], want[i])
			}
			hist[d]++
			h := off[key]
			h[d]++
			off[key] = h
		}
	}
	for k, h := range off {
		t.Logf("  %-28s %8d outputs: %8d equal, %6d at 1, %d at 2", k, h[0]+h[1]+h[2], h[0], h[1], h[2])
	}
	t.Logf("%d chains %v, %d outputs: %d equal, %d at 1 LSB", layers, kinds, hist[0]+hist[1], hist[0], hist[1])
}

func TestLowerFusedRejects(t *testing.T) {
	r := rand.New(rand.NewPCG(3, 4))
	var s2, pool chain
	for s2.fuse.Pad == nil || pool.fuse.Pool == nil {
		c := randomChain(r)
		if c.fuse.Pad != nil {
			s2 = c
		}
		if c.fuse.Pool != nil && c.fuse.Act == nil {
			pool = c
		}
	}
	clone := func(c chain) chain {
		if c.fuse.Pad != nil {
			p := *c.fuse.Pad
			c.fuse.Pad = &p
		}
		if c.fuse.Pool != nil {
			p := *c.fuse.Pool
			c.fuse.Pool = &p
		}
		return c
	}
	cases := map[string]struct {
		base chain
		mod  func(c *chain)
	}{
		"TF SAME stride 2 without Full": {s2, func(c *chain) { c.fuse.Pad = nil; c.conv.Padding = qi8.PaddingSame; c.conv.In = c.in }},
		"TF SAME stride 2 with Pad":     {s2, func(c *chain) { c.conv.Padding = qi8.PaddingSame; c.fuse.Full = true }},
		"TF SAME stride 2 odd size": {s2, func(c *chain) {
			c.fuse.Pad, c.fuse.Full = nil, true
			c.conv.Padding, c.conv.In = qi8.PaddingSame, c.in
			c.conv.In.H++
			c.conv.Out.H = (c.conv.In.H + 1) / 2
		}},
		"257-row input": {pool, func(c *chain) {
			c.fuse.Pool = nil
			c.conv.In.H, c.conv.Out.H = 257, 257
		}},
		"OddSample without Subsample": {s2, func(c *chain) { c.fuse.OddSample = true }},
		"Full on a stride-1 conv":     {pool, func(c *chain) { c.fuse.Pool = nil; c.fuse.Full = true }},
		"stride 2 without Pad":        {s2, func(c *chain) { c.fuse.Pad = nil; c.conv.In = c.in }},
		"Pad (0,2,0,2)":               {s2, func(c *chain) { c.fuse.Pad.Pads = [6]int{0, 2, 0, 2, 0, 0} }},
		"Pad with a channel pad":      {s2, func(c *chain) { c.fuse.Pad.Pads = [6]int{1, 1, 1, 1, 0, 1} }},
		"Pad not chaining":            {s2, func(c *chain) { c.fuse.Pad.In.C++ }},
		"stride 2 odd height":         {s2, func(c *chain) { c.fuse.Pad.In.H++; c.fuse.Pad.Out.H++; c.conv.In.H++ }},
		"pool after stride 2":         {s2, func(c *chain) { c.fuse.Pool = &qnet.Layer{Kind: qnet.MaxPool2D, KH: 2, KW: 2, StrideH: 2, StrideW: 2} }},
		"Pad before stride 1":         {pool, func(c *chain) { c.fuse.Pad = &qnet.Layer{Kind: qnet.Pad, Pads: [6]int{1, 1, 1, 1, 0, 0}} }},
		"pool kernel ≠ stride":        {pool, func(c *chain) { c.fuse.Pool.StrideH, c.fuse.Pool.StrideW = 1, 1 }},
		"3×3 pool":                    {pool, func(c *chain) { c.fuse.Pool.KH, c.fuse.Pool.KW, c.fuse.Pool.StrideH, c.fuse.Pool.StrideW = 3, 3, 3, 3 }},
		"pool SAME":                   {pool, func(c *chain) { c.fuse.Pool.Padding = qi8.PaddingSame }},
		"pool not dividing":           {pool, func(c *chain) { c.conv.In.W++; c.conv.Out.W++; c.fuse.Pool.In.W++ }},
		"pool quantisation":           {pool, func(c *chain) { c.fuse.Pool.OutQ.ZP++ }},
		"fused LeakyReLU": {pool, func(c *chain) {
			c.fuse.Pool = nil
			c.fuse.Act = &qnet.Layer{Kind: qnet.LeakyReLU, In: c.conv.Out, Out: c.conv.Out, InQ: c.conv.OutQ, OutQ: c.conv.OutQ}
		}},
		"activation quantisation": {pool, func(c *chain) {
			c.fuse.Pool = nil
			c.fuse.Act = &qnet.Layer{Kind: qnet.ReLU, In: c.conv.Out, Out: c.conv.Out, InQ: c.conv.OutQ, OutQ: qnet.Quant{Scale: 1, ZP: 3}}
		}},
		"pool on a depthwise conv": {pool, func(c *chain) {
			// A valid depthwise in every other respect, with the pool.
			ch := c.conv.In.C
			c.conv.Kind, c.conv.KH, c.conv.KW, c.conv.Padding = qnet.DepthwiseConv2D, 3, 3, qi8.PaddingSame
			c.conv.Out.C = ch
			c.conv.Weights = make([]int8, 9*ch)
			c.conv.Requant = make([]qi8.Requant, ch)
			c.conv.Bias = make([]int32, ch)
			for i := range c.conv.Requant {
				c.conv.Requant[i] = qi8.Requant{Mult: 1 << 30, Shift: -8}
			}
			c.fuse.Pool.In.C, c.fuse.Pool.Out.C = ch, ch
		}},
		"Full on a Pad + VALID regular conv": {s2, func(c *chain) { c.fuse.Full = true }},
		"Subsample on a 3×3 conv":            {pool, func(c *chain) { c.fuse.Pool = nil; c.conv.KH, c.conv.KW = 3, 3; c.fuse.Subsample = true }},
		"stride-2 depthwise without Full": {s2, func(c *chain) {
			ch := c.conv.In.C
			c.conv.Kind, c.conv.Out.C = qnet.DepthwiseConv2D, ch
			c.conv.Weights = make([]int8, 9*ch)
			c.conv.Requant = make([]qi8.Requant, ch)
			c.conv.Bias = make([]int32, ch)
			for i := range c.conv.Requant {
				c.conv.Requant[i] = qi8.Requant{Mult: 1 << 30, Shift: -8}
			}
		}},
	}
	for name, tc := range cases {
		c := clone(tc.base)
		tc.mod(&c)
		if _, err := Lower(&c.conv, c.fuse, 0, 0); !errors.Is(err, ErrUnsupported) {
			t.Errorf("%s: err %v, want ErrUnsupported", name, err)
		}
	}
}

// TestFusedClamp: a ReLU/ReLU6 after a conv clamp composes as two clamps,
// including when the ranges do not overlap.
func TestFusedClamp(t *testing.T) {
	q := qnet.Quant{Scale: 0.05, ZP: 10} // 6.0 → 10 + 120 = 130 → 127
	l := &qnet.Layer{Out: qi8.Shape{H: 1, W: 1, C: 1}, OutQ: q}
	act := func(k qnet.Kind) *qnet.Layer {
		return &qnet.Layer{Kind: k, In: l.Out, Out: l.Out, InQ: q, OutQ: q}
	}
	q6 := &qnet.Layer{Out: l.Out, OutQ: qnet.Quant{Scale: 0.1, ZP: 10}}
	for _, c := range []struct {
		conv           *qnet.Layer
		act            qnet.Kind
		min, max       int8
		wantLo, wantHi int64
	}{
		{l, qnet.ReLU, -128, 127, 10, 127},
		{l, qnet.ReLU, -50, 5, 10, 10},      // conv range below the ReLU: constant zp
		{l, qnet.ReLU, 20, 40, 20, 40},      // inside
		{q6, qnet.ReLU6, -128, 127, 10, 70}, // 6/0.1 = 60 above zp 10
		{q6, qnet.ReLU6, 80, 100, 70, 70},   // conv range above 6.0: constant
	} {
		c.conv.ActMin, c.conv.ActMax = c.min, c.max
		a := act(c.act)
		a.InQ, a.OutQ = c.conv.OutQ, c.conv.OutQ
		lo, hi, err := clamp(c.conv, a)
		if err != nil || lo != c.wantLo || hi != c.wantHi {
			t.Errorf("%v after [%d,%d] (zp %d): [%d,%d] %v, want [%d,%d]", c.act, c.min, c.max, c.conv.OutQ.ZP, lo, hi, err, c.wantLo, c.wantHi)
		}
	}
}

// pair is a stride-2 depthwise (Pad + 3×3 VALID) and the 1×1 conv after
// it, lowered as Full + Subsample.
func randomPair(r *rand.Rand) (pad, dw, pw qnet.Layer, in qi8.Shape) {
	for {
		l := randomLayer(r)
		if l.Kind == qnet.DepthwiseConv2D {
			dw = l
			break
		}
	}
	in = qi8.Shape{H: 2 * (1 + r.IntN(12)), W: 2 * (1 + r.IntN(12)), C: dw.In.C}
	pad = qnet.Layer{Kind: qnet.Pad, In: in, Out: qi8.Shape{H: in.H + 2, W: in.W + 2, C: in.C},
		InQ: dw.InQ, OutQ: dw.InQ, Pads: [6]int{1, 1, 1, 1, 0, 0}, PadValue: dw.InQ.ZP}
	dw.In, dw.Out = pad.Out, qi8.Shape{H: in.H / 2, W: in.W / 2, C: in.C}
	dw.StrideH, dw.StrideW, dw.Padding = 2, 2, qi8.PaddingValid
	for {
		pw = randomLayer(r)
		if pw.Kind == qnet.Conv2D && pw.KH == 1 {
			break
		}
	}
	oc := pw.Out.C
	pw.In, pw.Out, pw.InQ = dw.Out, qi8.Shape{H: dw.Out.H, W: dw.Out.W, C: oc}, dw.OutQ
	pw.Weights = make([]int8, in.C*oc)
	for i := range pw.Weights {
		pw.Weights[i] = int8(r.IntN(255) - 127)
	}
	return pad, dw, pw, in
}

// TestStride2DepthwisePair: a stride-2 depthwise lowered at full resolution
// (Full) and the 1×1 conv after it with the left-top sampler (Subsample):
// the depthwise's full-resolution output at even pixels and the 1×1's
// output (from the KPU's sampled input) are each within 1 LSB of qnet.
func TestStride2DepthwisePair(t *testing.T) {
	r := rand.New(rand.NewPCG(9, 212))
	pairs := 600
	if testing.Short() {
		pairs = 100
	}
	var hist [2]int
	for n := range pairs {
		pad, dw, pw, in := randomPair(r)
		x := make([]int8, in.Len())
		for i := range x {
			x[i] = int8(r.IntN(256) - 128)
		}
		run := func(l *qnet.Layer, src []int8) []int8 {
			dst := make([]int8, l.Out.Len())
			if err := qnet.RunLayer(l, dst, src); err != nil {
				t.Fatal(err)
			}
			return dst
		}
		wantDW := run(&dw, run(&pad, x))
		ldw, err := Lower(&dw, Fuse{Pad: &pad, Full: true}, 0, 16384)
		if err != nil {
			t.Fatalf("pair %d: depthwise: %v", n, err)
		}
		full := emulate(t, ldw, x, in, in) // full resolution
		half := dw.Out
		sampled := make([]int8, half.Len())
		for y := range half.H {
			for xx := range half.W {
				copy(sampled[(y*half.W+xx)*half.C:(y*half.W+xx+1)*half.C], full[(2*y*in.W+2*xx)*in.C:])
			}
		}
		check := func(got, want []int8, what string) {
			for i := range want {
				d := int(got[i]) - int(want[i])
				d = max(d, -d)
				if d > 1 {
					t.Fatalf("pair %d %s: output %d: KPU %d, qnet %d", n, what, i, got[i], want[i])
				}
				hist[d]++
			}
		}
		check(sampled, wantDW, "depthwise (even pixels)")
		lpw, err := Lower(&pw, Fuse{Subsample: true}, 16384, 0)
		if err != nil {
			t.Fatalf("pair %d: 1×1: %v", n, err)
		}
		gotPW := emulate(t, lpw, full, in, pw.Out) // input at full resolution
		check(gotPW, run(&pw, sampled), "1×1 with sampler")
	}
	t.Logf("%d pairs, %d outputs: %d equal, %d at 1 LSB", pairs, hist[0]+hist[1], hist[0], hist[1])
}
