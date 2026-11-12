package qi8

import (
	"fmt"
	"math"
	"math/big"
	"math/rand/v2"
	"testing"
)

// This file holds an independent "float-then-quantise" reference: inputs are
// dequantised to exact real numbers (big.Float with 512-bit mantissa; every
// value is dyadic so all products and sums are exact, which is asserted), the
// op is evaluated on reals, and the result is quantised with exact rational
// arithmetic, rounding half away from zero, then clamped. Kernel outputs must
// be bit-identical.

const refPrec = 512

// exactF tracks that every big.Float operation was exact.
type exactF struct{ inexact bool }

func (e *exactF) chk(f *big.Float) *big.Float {
	if f.Acc() != big.Exact {
		e.inexact = true
	}
	return f
}

func newF() *big.Float { return new(big.Float).SetPrec(refPrec) }

func (e *exactF) fromInt(v int64) *big.Float { return e.chk(newF().SetInt64(v)) }

func (e *exactF) mul(a, b *big.Float) *big.Float { return e.chk(newF().Mul(a, b)) }

// dyadic returns m * 2^exp exactly.
func (e *exactF) dyadic(m int64, exp int) *big.Float {
	f := e.fromInt(m)
	return e.chk(f.SetMantExp(f, exp))
}

// refScales are the real scales of one quantised layer. sIn is a power of
// two, sOut = b*2^-k is an arbitrary dyadic, and the per-channel weight
// scales are chosen so that sIn*sW[o]/sOut == M'_o exactly.
type refScales struct {
	sIn  *big.Float
	sOut *big.Rat
	sW   []*big.Float
}

func newRefScales(r *rand.Rand, e *exactF, rq []Requant) refScales {
	ein := r.IntN(13)
	b := int64(2*r.IntN(128) + 1) // odd, 1..255
	eout := r.IntN(13)
	s := refScales{
		sIn:  e.dyadic(1, -ein),
		sOut: new(big.Rat).SetFrac(big.NewInt(b), new(big.Int).Lsh(big.NewInt(1), uint(eout))),
		sW:   make([]*big.Float, len(rq)),
	}
	for o, q := range rq {
		// sW = Mult*2^(Shift-31) * b*2^-eout / 2^-ein
		s.sW[o] = e.dyadic(int64(q.Mult)*b, int(q.Shift)-31-eout+ein)
	}
	return s
}

func (s refScales) outScale() float32 {
	f, _ := s.sOut.Float64()
	return float32(f)
}

// roundHalfAway rounds a rational half away from zero.
func roundHalfAway(r *big.Rat) *big.Int {
	n := new(big.Int).Abs(r.Num())
	n.Lsh(n, 1).Add(n, r.Denom())
	d := new(big.Int).Lsh(r.Denom(), 1)
	n.Quo(n, d)
	if r.Sign() < 0 {
		n.Neg(n)
	}
	return n
}

func clampBig(q *big.Int, lo, hi int8) int8 {
	if q.Cmp(big.NewInt(int64(lo))) < 0 {
		return lo
	}
	if q.Cmp(big.NewInt(int64(hi))) > 0 {
		return hi
	}
	return int8(q.Int64())
}

// quantizeReal maps a real value to int8 with scale sOut and zero point zp:
// clamp(round(y/sOut) + zp, lo, hi).
func quantizeReal(y *big.Float, sOut *big.Rat, zp, lo, hi int8) int8 {
	r, _ := y.Rat(nil)
	r.Quo(r, sOut)
	q := roundHalfAway(r)
	q.Add(q, big.NewInt(int64(zp)))
	return clampBig(q, lo, hi)
}

// refWindow is an independent computation of TF SAME/VALID geometry.
func refWindow(in, k, s int, pad Padding) (out, before int) {
	if pad == PaddingSame {
		out = int(math.Ceil(float64(in) / float64(s)))
		total := (out-1)*s + k - in
		if total < 0 {
			total = 0
		}
		return out, total / 2
	}
	return int(math.Floor(float64(in-k)/float64(s))) + 1, 0
}

func dequant(e *exactF, q []int8, zp int8, s *big.Float) []*big.Float {
	out := make([]*big.Float, len(q))
	for i, v := range q {
		out[i] = e.mul(e.fromInt(int64(v)-int64(zp)), s)
	}
	return out
}

// refConv is the float-then-quantise reference for Conv2D (depthwise when
// dw is true: weights [KH][KW][C], OutC == In.C).
func refConv(t *testing.T, src, w []int8, bias []int32, sc refScales, p Conv2DParams, dw bool) []int8 {
	e := &exactF{}
	in := p.In
	oh, pt := refWindow(in.H, p.KH, p.StrideH, p.Padding)
	ow, pl := refWindow(in.W, p.KW, p.StrideW, p.Padding)
	xr := dequant(e, src, p.InZP, sc.sIn)
	outC := p.OutC
	wr := make([]*big.Float, len(w))
	for i, v := range w {
		var o int
		if dw {
			o = i % in.C
		} else {
			o = i / (p.KH * p.KW * in.C)
		}
		wr[i] = e.mul(e.fromInt(int64(v)-int64(p.WZP)), sc.sW[o])
	}
	out := make([]int8, 0, oh*ow*outC)
	for oy := range oh {
		for ox := range ow {
			for oc := range outC {
				y := e.fromInt(0)
				if len(bias) != 0 {
					y = e.mul(e.mul(e.fromInt(int64(bias[oc])), sc.sIn), sc.sW[oc])
				}
				for ky := range p.KH {
					for kx := range p.KW {
						iy, ix := oy*p.StrideH-pt+ky, ox*p.StrideW-pl+kx
						if iy < 0 || iy >= in.H || ix < 0 || ix >= in.W {
							continue // padding: real value 0 (quantised zpIn)
						}
						if dw {
							xv := xr[(iy*in.W+ix)*in.C+oc]
							wv := wr[(ky*p.KW+kx)*in.C+oc]
							e.chk(y.Add(y, e.mul(xv, wv)))
							continue
						}
						for ic := range in.C {
							xv := xr[(iy*in.W+ix)*in.C+ic]
							wv := wr[((oc*p.KH+ky)*p.KW+kx)*in.C+ic]
							e.chk(y.Add(y, e.mul(xv, wv)))
						}
					}
				}
				out = append(out, quantizeReal(y, sc.sOut, p.OutZP, p.ActMin, p.ActMax))
			}
		}
	}
	if e.inexact {
		t.Fatal("reference lost exactness")
	}
	return out
}

func refFC(t *testing.T, src, w []int8, bias []int32, sc refScales, p FCParams) []int8 {
	e := &exactF{}
	xr := dequant(e, src, p.InZP, sc.sIn)
	out := make([]int8, p.Out)
	for o := range p.Out {
		y := e.fromInt(0)
		if len(bias) != 0 {
			y = e.mul(e.mul(e.fromInt(int64(bias[o])), sc.sIn), sc.sW[o])
		}
		for i := range p.In {
			wv := e.mul(e.fromInt(int64(w[o*p.In+i])-int64(p.WZP)), sc.sW[o])
			e.chk(y.Add(y, e.mul(xr[i], wv)))
		}
		out[o] = quantizeReal(y, sc.sOut, p.OutZP, p.ActMin, p.ActMax)
	}
	if e.inexact {
		t.Fatal("reference lost exactness")
	}
	return out
}

// refPool: max of the valid cells' reals, or their exact mean, mapped back to
// the shared quantised domain (scale 1 relative, zero point irrelevant because
// in and out share it: q = mean(q_i), rounded half away from zero, or half
// up with p.TiesUp).
func refPool(src []int8, p Pool2DParams, avg bool) []int8 {
	in := p.In
	oh, pt := refWindow(in.H, p.KH, p.StrideH, p.Padding)
	ow, pl := refWindow(in.W, p.KW, p.StrideW, p.Padding)
	out := make([]int8, 0, oh*ow*in.C)
	for oy := range oh {
		for ox := range ow {
			for c := range in.C {
				sum := new(big.Rat)
				n := 0
				m := math.Inf(-1)
				for ky := range p.KH {
					for kx := range p.KW {
						iy, ix := oy*p.StrideH-pt+ky, ox*p.StrideW-pl+kx
						if iy < 0 || iy >= in.H || ix < 0 || ix >= in.W {
							continue
						}
						v := src[(iy*in.W+ix)*in.C+c]
						sum.Add(sum, new(big.Rat).SetInt64(int64(v)))
						m = math.Max(m, float64(v))
						n++
					}
				}
				if avg {
					sum.Quo(sum, new(big.Rat).SetInt64(int64(n)))
					if p.TiesUp {
						sum.Add(sum, big.NewRat(1, 2))
						out = append(out, clampBig(new(big.Int).Div(sum.Num(), sum.Denom()), -128, 127)) // floor
						continue
					}
					out = append(out, clampBig(roundHalfAway(sum), -128, 127))
				} else {
					out = append(out, int8(m))
				}
			}
		}
	}
	return out
}

// ---- random case generation ----

type gen struct{ r *rand.Rand }

func newGen(seed uint64) gen { return gen{rand.New(rand.NewPCG(seed, 0x9e3779b97f4a7c15))} }

func (g gen) zp() int8 {
	switch g.r.IntN(6) {
	case 0:
		return -128
	case 1:
		return 127
	case 2:
		return 0
	}
	return int8(g.r.IntN(256) - 128)
}

func (g gen) int8s(n int) []int8 {
	s := make([]int8, n)
	switch g.r.IntN(8) {
	case 0: // saturating extremes
		for i := range s {
			s[i] = []int8{-128, 127}[g.r.IntN(2)]
		}
	case 1: // constant extreme
		v := []int8{-128, 127}[g.r.IntN(2)]
		for i := range s {
			s[i] = v
		}
	default:
		for i := range s {
			s[i] = int8(g.r.IntN(256) - 128)
		}
	}
	return s
}

func (g gen) bias(n int) []int32 {
	switch g.r.IntN(5) {
	case 0:
		return nil
	case 1:
		b := make([]int32, n)
		for i := range b {
			b[i] = int32(g.r.IntN(1<<25) - 1<<24)
		}
		return b
	}
	b := make([]int32, n)
	for i := range b {
		b[i] = int32(g.r.IntN(1<<17) - 1<<16)
	}
	return b
}

func (g gen) requant() Requant {
	switch g.r.IntN(12) {
	case 0:
		return Requant{Mult: 1 << 30, Shift: MinShift}
	case 1:
		return Requant{Mult: math.MaxInt32, Shift: MaxShift}
	case 2:
		return Requant{Mult: math.MaxInt32, Shift: MinShift}
	case 3:
		return Requant{Mult: 1 << 30, Shift: int32(MinShift + g.r.IntN(MaxShift-MinShift+1))}
	case 4, 5: // power-of-two-ish multipliers make exact .5 ties frequent
		return Requant{Mult: []int32{1 << 30, 3 << 29, 5 << 28}[g.r.IntN(3)], Shift: int32(-g.r.IntN(9))}
	}
	return Requant{Mult: 1<<30 + g.r.Int32N(1<<30), Shift: int32(-14 + g.r.IntN(15))}
}

func (g gen) requants(n int) []Requant {
	rq := make([]Requant, n)
	for i := range rq {
		rq[i] = g.requant()
	}
	return rq
}

func (g gen) act(scale float32, zp int8) (int8, int8) {
	switch g.r.IntN(5) {
	case 0:
		return ActReLU(scale, zp)
	case 1:
		return ActReLU6(scale, zp)
	case 2:
		a, b := int8(g.r.IntN(256)-128), int8(g.r.IntN(256)-128)
		if a > b {
			a, b = b, a
		}
		return a, b
	}
	return ActNone()
}

func (g gen) padding(in, k int) Padding {
	if in >= k && g.r.IntN(3) == 0 {
		return PaddingValid
	}
	return PaddingSame
}

type mismatchLog struct {
	t          *testing.T
	cases, out int
	bad        int
}

func (m *mismatchLog) compare(desc func() string, got, want []int8) {
	m.cases++
	m.out += len(want)
	if len(got) != len(want) {
		m.t.Fatalf("%s: len %d want %d", desc(), len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			if m.bad < 5 {
				m.t.Errorf("%s: out[%d]=%d want %d", desc(), i, got[i], want[i])
			}
			m.bad++
		}
	}
}

func (m *mismatchLog) done() {
	m.t.Logf("cases=%d outputs=%d mismatches=%d", m.cases, m.out, m.bad)
	if m.bad != 0 {
		m.t.Fatalf("%d mismatches", m.bad)
	}
}

// ---- reference comparisons ----

func TestRequantApplyReference(t *testing.T) {
	g := newGen(1)
	m := &mismatchLog{t: t}
	accs := []int32{0, 1, -1, 2, -2, 3, -3, math.MaxInt32, math.MinInt32, math.MinInt32 + 1, 1 << 30, -(1 << 30)}
	for range 6000 {
		accs = append(accs, int32(g.r.Uint32()))
	}
	for shift := int32(MinShift); shift <= MaxShift; shift++ {
		for _, mult := range []int32{1 << 30, 1<<30 + 1, 3 << 29, math.MaxInt32, 1<<30 + g.r.Int32N(1<<30)} {
			rq := Requant{Mult: mult, Shift: shift}
			den := new(big.Int).Lsh(big.NewInt(1), uint(31-shift))
			got := make([]int8, 0, len(accs))
			want := make([]int8, 0, len(accs))
			for _, acc := range accs {
				v := rq.Apply(acc)
				r := new(big.Rat).SetFrac(new(big.Int).Mul(big.NewInt(int64(acc)), big.NewInt(int64(mult))), den)
				exp := roundHalfAway(r)
				// compare full-width results, encoded as equal/unequal bytes
				if exp.IsInt64() && exp.Int64() == v {
					got, want = append(got, 0), append(want, 0)
				} else {
					got, want = append(got, 1), append(want, 0)
				}
			}
			m.compare(func() string { return fmt.Sprintf("rq=%+v", rq) }, got, want)
		}
	}
	m.done()
}

func TestFullyConnectedReference(t *testing.T) {
	g := newGen(2)
	m := &mismatchLog{t: t}
	for c := range 2000 {
		in := 1 + g.r.IntN(70)
		if c%50 == 0 {
			in = 1000 + g.r.IntN(2000)
		}
		p := FCParams{In: in, Out: 1 + g.r.IntN(12), InZP: g.zp(), OutZP: g.zp()}
		src, w := g.int8s(p.In), g.int8s(p.In*p.Out)
		bias, rq := g.bias(p.Out), g.requants(p.Out)
		e := &exactF{}
		sc := newRefScales(g.r, e, rq)
		p.ActMin, p.ActMax = g.act(sc.outScale(), p.OutZP)
		dst := make([]int8, p.Out)
		if err := FullyConnected(dst, src, w, bias, rq, p); err != nil {
			t.Fatalf("case %d: %v", c, err)
		}
		m.compare(func() string { return fmt.Sprintf("case %d %+v", c, p) }, dst, refFC(t, src, w, bias, sc, p))
	}
	m.done()
}

func randConvParams(g gen, depthwise bool) Conv2DParams {
	ks := []int{1, 1, 2, 3, 3, 3, 5}
	p := Conv2DParams{
		In:      Shape{H: 1 + g.r.IntN(9), W: 1 + g.r.IntN(9), C: 1 + g.r.IntN(7)},
		KH:      ks[g.r.IntN(len(ks))],
		StrideH: 1 + g.r.IntN(2),
		InZP:    g.zp(),
		OutZP:   g.zp(),
	}
	if g.r.IntN(10) == 0 {
		p.StrideH = 3
	}
	p.KW, p.StrideW = p.KH, p.StrideH
	if g.r.IntN(4) == 0 { // non-square kernel/stride
		p.KW = ks[g.r.IntN(len(ks))]
		p.StrideW = 1 + g.r.IntN(2)
	}
	if depthwise {
		p.OutC = p.In.C
	} else {
		p.OutC = 1 + g.r.IntN(6)
	}
	p.Padding = g.padding(min(p.In.H, p.In.W), max(p.KH, p.KW))
	return p
}

func TestConv2DReference(t *testing.T) {
	g := newGen(3)
	m := &mismatchLog{t: t}
	for c := range 3000 {
		p := randConvParams(g, false)
		src := g.int8s(p.In.Len())
		w := g.int8s(p.OutC * p.KH * p.KW * p.In.C)
		bias, rq := g.bias(p.OutC), g.requants(p.OutC)
		e := &exactF{}
		sc := newRefScales(g.r, e, rq)
		p.ActMin, p.ActMax = g.act(sc.outScale(), p.OutZP)
		os, err := p.OutShape()
		if err != nil {
			t.Fatalf("case %d %+v: %v", c, p, err)
		}
		dst := make([]int8, os.Len())
		if err := Conv2D(dst, src, w, bias, rq, p); err != nil {
			t.Fatalf("case %d: %v", c, err)
		}
		m.compare(func() string { return fmt.Sprintf("case %d %+v", c, p) }, dst, refConv(t, src, w, bias, sc, p, false))
	}
	m.done()
}

func TestDepthwiseConv2DReference(t *testing.T) {
	g := newGen(4)
	m := &mismatchLog{t: t}
	// The last 120 cases have 33…128 channels: the generic loop accumulates
	// runs of dwChunk = 32 channels, so these cross one or more run edges.
	wide := []int{33, 40, 64, 65, 96, 128}
	for c := range 3000 + 120 {
		cp := randConvParams(g, true)
		if c%2 == 0 { // bias towards the 3x3 stride 1/2 hot path
			cp.KH, cp.KW = 3, 3
			cp.StrideW = cp.StrideH
			if cp.In.H < 3 || cp.In.W < 3 {
				cp.Padding = PaddingSame
			}
		}
		if c >= 3000 {
			cp.In.C = wide[(c-3000)%len(wide)]
			cp.OutC = cp.In.C
		}
		p := DepthwiseConv2DParams{
			In: cp.In, KH: cp.KH, KW: cp.KW, StrideH: cp.StrideH, StrideW: cp.StrideW,
			Padding: cp.Padding, InZP: cp.InZP, OutZP: cp.OutZP,
		}
		src := g.int8s(p.In.Len())
		w := g.int8s(p.KH * p.KW * p.In.C)
		bias, rq := g.bias(p.In.C), g.requants(p.In.C)
		e := &exactF{}
		sc := newRefScales(g.r, e, rq)
		p.ActMin, p.ActMax = g.act(sc.outScale(), p.OutZP)
		cp.ActMin, cp.ActMax = p.ActMin, p.ActMax
		os, err := p.OutShape()
		if err != nil {
			t.Fatalf("case %d %+v: %v", c, p, err)
		}
		dst := make([]int8, os.Len())
		if err := DepthwiseConv2D(dst, src, w, bias, rq, p); err != nil {
			t.Fatalf("case %d: %v", c, err)
		}
		m.compare(func() string { return fmt.Sprintf("case %d %+v", c, p) }, dst, refConv(t, src, w, bias, sc, cp, true))
	}
	m.done()
}

// TestWeightZeroPointReference: asymmetric weights (a per-tensor weight zero
// point, as in TFLite's uint8 models; ER-QU8) for Conv2D, DepthwiseConv2D
// and FullyConnected against the exact reference.
func TestWeightZeroPointReference(t *testing.T) {
	g := newGen(5)
	m := &mismatchLog{t: t}
	for c := range 3000 {
		wzp := g.zp()
		if wzp == 0 {
			wzp = 1 + int8(g.r.IntN(127))
		}
		e := &exactF{}
		switch c % 3 {
		case 0:
			p := randConvParams(g, false)
			p.WZP = wzp
			src, w := g.int8s(p.In.Len()), g.int8s(p.OutC*p.KH*p.KW*p.In.C)
			bias, rq := g.bias(p.OutC), g.requants(p.OutC)
			sc := newRefScales(g.r, e, rq)
			p.ActMin, p.ActMax = g.act(sc.outScale(), p.OutZP)
			os, err := p.OutShape()
			if err != nil {
				t.Fatalf("case %d %+v: %v", c, p, err)
			}
			dst := make([]int8, os.Len())
			if err := Conv2D(dst, src, w, bias, rq, p); err != nil {
				t.Fatalf("case %d: %v", c, err)
			}
			m.compare(func() string { return fmt.Sprintf("conv case %d %+v", c, p) }, dst, refConv(t, src, w, bias, sc, p, false))
		case 1:
			cp := randConvParams(g, true)
			cp.WZP = wzp
			p := DepthwiseConv2DParams{
				In: cp.In, KH: cp.KH, KW: cp.KW, StrideH: cp.StrideH, StrideW: cp.StrideW,
				Padding: cp.Padding, InZP: cp.InZP, OutZP: cp.OutZP, WZP: wzp,
			}
			src, w := g.int8s(p.In.Len()), g.int8s(p.KH*p.KW*p.In.C)
			bias, rq := g.bias(p.In.C), g.requants(p.In.C)
			sc := newRefScales(g.r, e, rq)
			p.ActMin, p.ActMax = g.act(sc.outScale(), p.OutZP)
			cp.ActMin, cp.ActMax = p.ActMin, p.ActMax
			os, err := p.OutShape()
			if err != nil {
				t.Fatalf("case %d %+v: %v", c, p, err)
			}
			dst := make([]int8, os.Len())
			if err := DepthwiseConv2D(dst, src, w, bias, rq, p); err != nil {
				t.Fatalf("case %d: %v", c, err)
			}
			m.compare(func() string { return fmt.Sprintf("dw case %d %+v", c, p) }, dst, refConv(t, src, w, bias, sc, cp, true))
		default:
			p := FCParams{In: 1 + g.r.IntN(70), Out: 1 + g.r.IntN(12), InZP: g.zp(), OutZP: g.zp(), WZP: wzp}
			src, w := g.int8s(p.In), g.int8s(p.In*p.Out)
			bias, rq := g.bias(p.Out), g.requants(p.Out)
			sc := newRefScales(g.r, e, rq)
			p.ActMin, p.ActMax = g.act(sc.outScale(), p.OutZP)
			dst := make([]int8, p.Out)
			if err := FullyConnected(dst, src, w, bias, rq, p); err != nil {
				t.Fatalf("case %d: %v", c, err)
			}
			m.compare(func() string { return fmt.Sprintf("fc case %d %+v", c, p) }, dst, refFC(t, src, w, bias, sc, p))
		}
	}
	m.done()
}

func TestPool2DReference(t *testing.T) {
	for _, avg := range []bool{false, true} {
		name := "Max"
		if avg {
			name = "Avg"
		}
		t.Run(name, func(t *testing.T) {
			g := newGen(5)
			m := &mismatchLog{t: t}
			for c := range 3000 {
				cp := randConvParams(g, true)
				p := Pool2DParams{In: cp.In, KH: cp.KH, KW: cp.KW, StrideH: cp.StrideH, StrideW: cp.StrideW, Padding: cp.Padding}
				src := g.int8s(p.In.Len())
				os, err := p.OutShape()
				if err != nil {
					t.Fatalf("case %d %+v: %v", c, p, err)
				}
				dst := make([]int8, os.Len())
				f := MaxPool2D
				if avg {
					f = AvgPool2D
				}
				if err := f(dst, src, p); err != nil {
					t.Fatalf("case %d: %v", c, err)
				}
				m.compare(func() string { return fmt.Sprintf("case %d %+v", c, p) }, dst, refPool(src, p, avg))
			}
			m.done()
		})
	}
}

// TestAvgPoolTiesUpReference: with TiesUp, AvgPool2D rounds the exact mean
// half up (TFLite's uint8 AVERAGE_POOL_2D on the value − 128 form, ER-QU8);
// it differs from the default only on negative .5 ties.
func TestAvgPoolTiesUpReference(t *testing.T) {
	g := newGen(6)
	m := &mismatchLog{t: t}
	differ := 0
	for c := range 3000 {
		cp := randConvParams(g, true)
		p := Pool2DParams{In: cp.In, KH: cp.KH, KW: cp.KW, StrideH: cp.StrideH, StrideW: cp.StrideW, Padding: cp.Padding, TiesUp: true}
		src := g.int8s(p.In.Len())
		os, err := p.OutShape()
		if err != nil {
			t.Fatalf("case %d %+v: %v", c, p, err)
		}
		dst := make([]int8, os.Len())
		if err := AvgPool2D(dst, src, p); err != nil {
			t.Fatalf("case %d: %v", c, err)
		}
		m.compare(func() string { return fmt.Sprintf("case %d %+v", c, p) }, dst, refPool(src, p, true))
		p.TiesUp = false
		away := refPool(src, p, true)
		for i := range dst {
			if dst[i] != away[i] {
				differ++
			}
		}
	}
	m.done()
	if differ == 0 {
		t.Fatal("TiesUp never differed from half-away rounding: the cases have no negative ties")
	}
	t.Logf("%d outputs differ from half-away rounding", differ)
}

func TestReLUReference(t *testing.T) {
	g := newGen(6)
	m := &mismatchLog{t: t}
	for range 500 {
		zp := g.zp()
		scale := float32(g.r.Float64()*0.2 + 1e-3)
		src := g.int8s(1 + g.r.IntN(100))
		q6 := Quantize6(scale, zp)
		want, want6 := make([]int8, len(src)), make([]int8, len(src))
		for i, v := range src {
			// Dequantise, apply max(0,x)/min(max(0,x),6) on reals, requantise.
			real := (float64(v) - float64(zp)) * float64(scale)
			r := math.Max(real, 0)
			want[i] = int8(math.Round(r/float64(scale)) + float64(zp))
			r6 := math.Min(r, 6)
			q := math.Round(r6/float64(scale)) + float64(zp)
			want6[i] = int8(math.Min(q, 127))
		}
		got := make([]int8, len(src))
		if err := ReLU(got, src, zp); err != nil {
			t.Fatal(err)
		}
		m.compare(func() string { return "relu" }, got, want)
		if err := ReLU6(got, src, zp, q6); err != nil {
			t.Fatal(err)
		}
		m.compare(func() string { return fmt.Sprintf("relu6 zp=%d scale=%g", zp, scale) }, got, want6)
	}
	m.done()
}
