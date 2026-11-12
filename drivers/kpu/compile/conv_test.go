package compile

import (
	"errors"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/itohio/EasyRobot/drivers/kpu/kmodel"
	"github.com/itohio/EasyRobot/x/math/nn/qnet"
	"github.com/itohio/EasyRobot/x/math/primitive/qi8"
)

// emulate runs a lowered conv with kmodel's integer model of the KPU on an
// int8 NHWC input of shape s and returns the int8 NHWC output of shape o.
func emulate(t *testing.T, c Conv, in []int8, s, o qi8.Shape) []int8 {
	t.Helper()
	data := append(append(append([]byte{}, c.Weights...), c.BN...), c.Act...)
	m := &kmodel.Model{Data: data}
	p, err := m.ParamsAt(c.Args, 0, uint32(len(c.Weights)), uint32(len(c.Weights)+len(c.BN)))
	if err != nil {
		t.Fatal(err)
	}
	chw := make([]byte, len(in))
	for y := range s.H {
		for x := range s.W {
			for ch := range s.C {
				chw[(ch*s.H+y)*s.W+x] = byte(int16(in[(y*s.W+x)*s.C+ch]) + 128)
			}
		}
	}
	out := make([]byte, o.H*o.W*o.C)
	if err := p.Emulate(chw, out); err != nil {
		t.Fatal(err)
	}
	nhwc := make([]int8, len(out))
	for y := range o.H {
		for x := range o.W {
			for ch := range o.C {
				nhwc[(y*o.W+x)*o.C+ch] = int8(int16(out[(ch*o.H+y)*o.W+x]) - 128)
			}
		}
	}
	return nhwc
}

// randomLayer draws a KPU-lowerable qnet conv layer.
func randomLayer(r *rand.Rand) qnet.Layer {
	kinds := []struct {
		kind qnet.Kind
		k    int
	}{{qnet.Conv2D, 1}, {qnet.Conv2D, 3}, {qnet.DepthwiseConv2D, 3}}
	kk := kinds[r.IntN(len(kinds))]
	in := qi8.Shape{H: 1 + r.IntN(24), W: 1 + r.IntN(24), C: 1 + r.IntN(48)}
	oc := 1 + r.IntN(48)
	if kk.kind == qnet.DepthwiseConv2D {
		oc = in.C
	}
	l := qnet.Layer{
		Kind: kk.kind, In: in, Out: qi8.Shape{H: in.H, W: in.W, C: oc},
		InQ:  qnet.Quant{Scale: 0.05, ZP: int8(r.IntN(256) - 128)},
		OutQ: qnet.Quant{Scale: 0.05, ZP: int8(r.IntN(256) - 128)},
		KH:   kk.k, KW: kk.k, StrideH: 1, StrideW: 1, Padding: qi8.PaddingSame,
		Round: qi8.Rounding(r.IntN(2)),
	}
	switch r.IntN(4) {
	case 0:
		l.ActMin, l.ActMax = -128, 127
	case 1:
		l.ActMin, l.ActMax = l.OutQ.ZP, 127 // ReLU
	case 2:
		l.ActMin, l.ActMax = l.OutQ.ZP, int8(min(127, int(l.OutQ.ZP)+r.IntN(128))) // ReLU6-like
	default:
		a, b := int8(r.IntN(256)-128), int8(r.IntN(256)-128)
		l.ActMin, l.ActMax = min(a, b), max(a, b)
	}
	nw := kk.k * kk.k * in.C * oc
	if kk.kind == qnet.DepthwiseConv2D {
		nw = kk.k * kk.k * in.C
	}
	l.Weights = make([]int8, nw)
	for i := range l.Weights {
		l.Weights[i] = int8(r.IntN(255) - 127)
	}
	l.Bias = make([]int32, oc)
	l.Requant = make([]qi8.Requant, oc)
	for c := range oc {
		l.Bias[c] = int32(r.IntN(1<<16) - 1<<15)
		m := math.Ldexp(1+r.Float64(), -1-r.IntN(16)) // 2^-16 … 2^0
		rq, err := qi8.QuantizeMultiplier(m / 2)
		if err != nil {
			panic(err)
		}
		l.Requant[c] = rq
	}
	return l
}

// TestLowerConvMatchesQnet: on random lowerable layers, the KPU arithmetic
// (kmodel.Emulate, bit-identical to nncase's simulator) of the lowered
// layer agrees with qnet's CPU kernel within 1 LSB on every output.
func TestLowerConvMatchesQnet(t *testing.T) {
	r := rand.New(rand.NewPCG(9, 210))
	layers := 1200
	if testing.Short() {
		layers = 200
	}
	var hist [2]int
	for n := range layers {
		l := randomLayer(r)
		in := make([]int8, l.In.H*l.In.W*l.In.C)
		for i := range in {
			in[i] = int8(r.IntN(256) - 128)
		}
		want := make([]int8, l.Out.H*l.Out.W*l.Out.C)
		if err := qnet.RunLayer(&l, want, in); err != nil {
			t.Fatalf("layer %d: qnet: %v", n, err)
		}
		c, err := LowerConv(&l, 0, 16384)
		if err != nil {
			t.Fatalf("layer %d (%v k%d %v→%d): %v", n, l.Kind, l.KH, l.In, l.Out.C, err)
		}
		got := emulate(t, c, in, l.In, l.Out)
		for i := range want {
			d := int(got[i]) - int(want[i])
			if d < -1 || d > 1 {
				t.Fatalf("layer %d (%v k%d %v→%d, zp %d→%d, clamp [%d,%d], round %d): output %d: KPU %d, qnet %d",
					n, l.Kind, l.KH, l.In, l.Out.C, l.InQ.ZP, l.OutQ.ZP, l.ActMin, l.ActMax, l.Round, i, got[i], want[i])
			}
			hist[max(d, -d)]++
		}
	}
	t.Logf("%d layers, %d outputs: %d equal, %d at 1 LSB", layers, hist[0]+hist[1], hist[0], hist[1])
}

func TestLowerConvRejects(t *testing.T) {
	r := rand.New(rand.NewPCG(1, 2))
	base := randomLayer(r)
	for base.Kind != qnet.Conv2D || base.KH != 3 {
		base = randomLayer(r)
	}
	cases := map[string]func(l *qnet.Layer){
		"stride 2":                           func(l *qnet.Layer) { l.StrideH, l.StrideW = 2, 2 },
		"5×5 kernel":                         func(l *qnet.Layer) { l.KH, l.KW = 5, 5 },
		"3×3 VALID":                          func(l *qnet.Layer) { l.Padding = qi8.PaddingValid },
		"fully connected":                    func(l *qnet.Layer) { l.Kind = qnet.FullyConnected },
		"depthwise ×2":                       func(l *qnet.Layer) { l.Kind = qnet.DepthwiseConv2D; l.Out.C = 2 * l.In.C },
		"depthwise 1×1":                      func(l *qnet.Layer) { l.Kind = qnet.DepthwiseConv2D; l.KH, l.KW = 1, 1; l.Out.C = l.In.C },
		"multiplier 2^29 (y beyond 36 bits)": func(l *qnet.Layer) { l.Requant[0] = qi8.Requant{Mult: 1 << 30, Shift: 30} },
	}
	for name, mod := range cases {
		l := base
		l.Requant = append([]qi8.Requant(nil), base.Requant...)
		mod(&l)
		if _, err := LowerConv(&l, 0, 0); !errors.Is(err, ErrUnsupported) {
			t.Errorf("%s: err %v, want ErrUnsupported", name, err)
		}
	}
}

// evalAct applies a KPU activation table to a BN output y as the hardware
// does (k210-kpu.md): the last segment whose start ≤ y, carry-shifted
// product, plus the bias byte, clamped to 0..255.
func evalAct(t [16]kmodel.ActSegment, y int64) int64 {
	s := t[0]
	for _, seg := range t {
		if seg.StartX <= y {
			s = seg
		}
	}
	v := (y - s.StartX) * s.Mul
	if s.Shift > 0 {
		v >>= s.Shift - 1
		if v&1 != 0 {
			if v < 0 {
				v = v>>1 - 1
			} else {
				v = v>>1 + 1
			}
		} else {
			v >>= 1
		}
	}
	return max(0, min(255, v+s.Add))
}

// TestActivationTable: the table computes clamp(zp + round(y/2^p), lo, hi)
// exactly (positive halves round up), including at both clamp edges.
func TestActivationTable(t *testing.T) {
	for _, c := range []struct{ zp, lo, hi int64 }{{128, 0, 255}, {100, 100, 255}, {0, 0, 177}, {200, 30, 220}, {77, 77, 77}} {
		for _, p := range []int{1, 5, 17, 24} {
			tab := activation(c.zp, c.lo, c.hi, p)
			for y := (c.lo - c.zp - 3) << p; y <= (c.hi-c.zp+3)<<p; y += max(1, int64(1)<<p/7) {
				half := int64(1) << (p - 1)
				r := (y + half) >> p // round half up
				want := max(c.lo, min(c.hi, c.zp+r))
				if got := evalAct(tab, y); got != want {
					t.Fatalf("zp %d [%d,%d] p %d y %d: got %d, want %d", c.zp, c.lo, c.hi, p, y, got, want)
				}
			}
		}
	}
}

// TestLoadPlanMatchesNcc: the weight load plan equals ncc's on MobileNet's
// layers (ER-K210-K8 evidence: layer 24 1×1 128→256: 30,720 B × 2 loads,
// 240 channels per load; layer 26 1×1 256→256: 30,720 B × 3, 120; layer 25
// depthwise 3×3 ×256: 2,304 B × 1, 256).
func TestLoadPlanMatchesNcc(t *testing.T) {
	for _, c := range []struct {
		k, ic, oc        int
		dw               bool
		para, loads, per uint64
	}{{1, 128, 256, false, 30720, 2, 240}, {1, 256, 256, false, 30720, 3, 120}, {3, 256, 256, true, 2304, 1, 256}, {3, 3, 8, false, 216, 1, 8}} {
		a := kmodel.NewArgs()
		if err := setLoad(&a, c.k, c.ic, c.oc, c.dw); err != nil {
			t.Fatal(err)
		}
		para, loads, per := a[5]>>15&0x1ffff, a[5]>>1&0x3f+1, a[2]>>48&0x3ff+1
		if para != c.para || loads != c.loads || per != c.per {
			t.Errorf("%+v: para %d loads %d per %d", c, para, loads, per)
		}
	}
}
