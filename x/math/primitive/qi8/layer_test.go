package qi8

import (
	"fmt"
	"testing"
	"unsafe"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// alignedAt returns n bytes starting off bytes past a 16-byte boundary.
func alignedAt(n, off int) []int8 {
	b := make([]int8, n+32)
	a := int((16 - uintptr(unsafe.Pointer(&b[0]))&15) & 15)
	return b[a+off : a+off+n : a+off+n]
}

// TestLayerDepthwiseWideChannels covers channel counts the random cases do
// not reach (several 16-lane groups, multiples of 16) with 16-byte aligned
// and misaligned inputs; on the PIE backend these select the direct and the
// staged paths.
func TestLayerDepthwiseWideChannels(t *testing.T) {
	g := newGen(46)
	m := &mismatchLog{t: t}
	for _, c := range []int{16, 17, 32, 40} {
		for _, off := range []int{0, 1, 4} {
			for _, pad := range []Padding{PaddingSame, PaddingValid} {
				p := DepthwiseConv2DParams{In: Shape{H: 5, W: 6, C: c}, KH: 3, KW: 3, StrideH: 1 + g.r.IntN(2), StrideW: 1,
					Padding: pad, InZP: g.zp(), OutZP: g.zp()}
				p.ActMin, p.ActMax = g.act(randScale(g), p.OutZP)
				w, bias, rq := g.int8s(9*c), g.bias(c), g.requants(c)
				l := withScratch(&DepthwiseConv2DLayer{P: p, W: w, Bias: bias, RQ: rq}, setDWScratch)
				require.NoError(t, l.Prepare())
				os, _ := p.OutShape()
				src := alignedAt(p.In.Len(), off)
				copy(src, g.int8s(len(src)))
				want, got := make([]int8, os.Len()), make([]int8, os.Len())
				require.NoError(t, DepthwiseConv2D(want, src, w, bias, rq, p))
				require.NoError(t, l.Run(got, src))
				m.compare(func() string { return fmt.Sprintf("C=%d off=%d %+v", c, off, p) }, got, want)
			}
		}
	}
	m.done()
}

// preparedOp is the method set every prepared layer has.
type preparedOp interface {
	Op
	Prepare() error
}

// TestBackend checks that the build selected a known backend and that the
// weighted layers ask for Scratch exactly when the backend repacks weights.
func TestBackend(t *testing.T) {
	assert.Contains(t, []string{"generic", "esp32s3-pie"}, Backend())
	fc := &FCLayer{P: FCParams{In: 4, Out: 2, ActMax: 127}}
	assert.Equal(t, Backend() != "generic", fc.ScratchLen() > 0, "ScratchLen on %s", Backend())
}

// withScratch gives a weighted layer the Scratch its backend needs (none on
// the generic backend), so the suite runs on every backend. The scratch
// starts 3 bytes past a 16-byte boundary and holds garbage, so a backend
// must align its layout and initialise every byte it reads.
func withScratch[L interface{ ScratchLen() int }](l L, set func(L, []int8)) L {
	s := alignedAt(l.ScratchLen(), 3)
	for i := range s {
		s[i] = 0x5a
	}
	set(l, s)
	return l
}

func setConvScratch(l *Conv2DLayer, s []int8)        { l.Scratch = s }
func setDWScratch(l *DepthwiseConv2DLayer, s []int8) { l.Scratch = s }
func setFCScratch(l *FCLayer, s []int8)              { l.Scratch = s }

// layerCase pairs a prepared layer with the free function it must match.
type layerCase struct {
	name     string
	layer    preparedOp
	free     func(dst, src []int8) error
	in, out  Shape
	srcLen   int
	dstLen   int
	describe string
}

func randScale(g gen) float32 { return float32(0.005 + g.r.Float64()*0.2) }

// wzp draws a weight zero point: 0 (symmetric) three times in four.
func (g gen) wzp() int8 {
	if g.r.IntN(4) != 0 {
		return 0
	}
	return g.zp()
}

func randLayerCase(g gen, kind int) layerCase {
	switch kind {
	case 0:
		p := randConvParams(g, false)
		p.Round, p.WZP = g.rounding(), g.wzp()
		p.ActMin, p.ActMax = g.act(randScale(g), p.OutZP)
		w := g.int8s(p.OutC * p.KH * p.KW * p.In.C)
		bias, rq := g.bias(p.OutC), g.requants(p.OutC)
		os, _ := p.OutShape()
		return layerCase{
			name:  "Conv2D",
			layer: withScratch(&Conv2DLayer{P: p, W: w, Bias: bias, RQ: rq}, setConvScratch),
			free:  func(dst, src []int8) error { return Conv2D(dst, src, w, bias, rq, p) },
			in:    p.In, out: os, describe: fmt.Sprintf("%+v", p),
		}
	case 1:
		cp := randConvParams(g, true)
		p := DepthwiseConv2DParams{
			In: cp.In, KH: cp.KH, KW: cp.KW, StrideH: cp.StrideH, StrideW: cp.StrideW,
			Padding: cp.Padding, InZP: cp.InZP, OutZP: cp.OutZP, Round: g.rounding(), WZP: g.wzp(),
		}
		p.ActMin, p.ActMax = g.act(randScale(g), p.OutZP)
		w := g.int8s(p.KH * p.KW * p.In.C)
		bias, rq := g.bias(p.In.C), g.requants(p.In.C)
		os, _ := p.OutShape()
		return layerCase{
			name:  "DepthwiseConv2D",
			layer: withScratch(&DepthwiseConv2DLayer{P: p, W: w, Bias: bias, RQ: rq}, setDWScratch),
			free:  func(dst, src []int8) error { return DepthwiseConv2D(dst, src, w, bias, rq, p) },
			in:    p.In, out: os, describe: fmt.Sprintf("%+v", p),
		}
	case 2:
		p := FCParams{In: 1 + g.r.IntN(200), Out: 1 + g.r.IntN(12), InZP: g.zp(), OutZP: g.zp(), Round: g.rounding(), WZP: g.wzp()}
		p.ActMin, p.ActMax = g.act(randScale(g), p.OutZP)
		w := g.int8s(p.In * p.Out)
		bias, rq := g.bias(p.Out), g.requants(p.Out)
		return layerCase{
			name:  "FC",
			layer: withScratch(&FCLayer{P: p, W: w, Bias: bias, RQ: rq}, setFCScratch),
			free:  func(dst, src []int8) error { return FullyConnected(dst, src, w, bias, rq, p) },
			in:    Shape{1, 1, p.In}, out: Shape{1, 1, p.Out}, describe: fmt.Sprintf("%+v", p),
		}
	case 3, 4:
		cp := randConvParams(g, true)
		p := Pool2DParams{In: cp.In, KH: cp.KH, KW: cp.KW, StrideH: cp.StrideH, StrideW: cp.StrideW, Padding: cp.Padding}
		os, _ := p.OutShape()
		lc := layerCase{in: p.In, out: os, describe: fmt.Sprintf("%+v", p)}
		if kind == 3 {
			lc.name, lc.layer = "MaxPool2D", &MaxPool2DLayer{P: p}
			lc.free = func(dst, src []int8) error { return MaxPool2D(dst, src, p) }
		} else {
			lc.name, lc.layer = "AvgPool2D", &AvgPool2DLayer{P: p}
			lc.free = func(dst, src []int8) error { return AvgPool2D(dst, src, p) }
		}
		return lc
	case 5:
		n, zp := 1+g.r.IntN(300), g.zp()
		return layerCase{
			name:  "ReLU",
			layer: &ReLULayer{Len: n, ZP: zp},
			free:  func(dst, src []int8) error { return ReLU(dst, src, zp) },
			in:    Shape{1, 1, n}, out: Shape{1, 1, n}, describe: fmt.Sprintf("n=%d zp=%d", n, zp),
		}
	}
	n, zp := 1+g.r.IntN(300), g.zp()
	q6 := Quantize6(randScale(g), zp)
	return layerCase{
		name:  "ReLU6",
		layer: &ReLU6Layer{Len: n, ZP: zp, Q6: q6},
		free:  func(dst, src []int8) error { return ReLU6(dst, src, zp, q6) },
		in:    Shape{1, 1, n}, out: Shape{1, 1, n}, describe: fmt.Sprintf("n=%d zp=%d q6=%d", n, zp, q6),
	}
}

const layerKinds = 7

func TestLayersMatchFreeFunctions(t *testing.T) {
	g := newGen(42)
	m := &mismatchLog{t: t}
	for c := range 7000 {
		lc := randLayerCase(g, c%layerKinds)
		desc := func() string { return fmt.Sprintf("case %d %s %s", c, lc.name, lc.describe) }
		require.NoError(t, lc.layer.Prepare(), desc())
		require.Equal(t, lc.in, lc.layer.InShape(), desc())
		require.Equal(t, lc.out, lc.layer.OutShape(), desc())
		want, got := make([]int8, lc.out.Len()), make([]int8, lc.out.Len())
		// Run twice per Prepare with different inputs: Run is reusable.
		for range 2 {
			src := g.int8s(lc.in.Len())
			require.NoError(t, lc.free(want, src), desc())
			require.NoError(t, lc.layer.Run(got, src), desc())
			m.compare(desc, got, want)
		}
	}
	m.done()
}

func TestLayerRunLengthAndPrepared(t *testing.T) {
	g := newGen(43)
	for kind := range layerKinds {
		lc := randLayerCase(g, kind)
		t.Run(lc.name, func(t *testing.T) {
			src, dst := make([]int8, lc.in.Len()+1), make([]int8, lc.out.Len()+1)
			okSrc, okDst := src[:lc.in.Len()], dst[:lc.out.Len()]
			assert.Equal(t, ErrNotPrepared, lc.layer.Run(okDst, okSrc), "before Prepare")
			assert.Equal(t, Shape{}, lc.layer.InShape(), "before Prepare")
			require.NoError(t, lc.layer.Prepare())
			require.NoError(t, lc.layer.Run(okDst, okSrc))
			assert.Equal(t, ErrLength, lc.layer.Run(okDst, src), "long src")
			assert.Equal(t, ErrLength, lc.layer.Run(dst, okSrc), "long dst")
			if len(okSrc) > 1 {
				assert.Equal(t, ErrLength, lc.layer.Run(okDst, okSrc[:len(okSrc)-1]), "short src")
			}
			if len(okDst) > 1 {
				assert.Equal(t, ErrLength, lc.layer.Run(okDst[:len(okDst)-1], okSrc), "short dst")
			}
		})
	}
}

// mutation changes one exported field; when Prepare then fails, fix
// restores a valid configuration.
type mutation struct {
	name   string
	mutate func()
	fix    func() // non-nil: Prepare fails after mutate until fix is applied
}

func TestLayerNotPreparedAfterMutation(t *testing.T) {
	g := newGen(44)
	conv := randLayerCase(g, 0).layer.(*Conv2DLayer)
	dw := randLayerCase(g, 1).layer.(*DepthwiseConv2DLayer)
	fc := randLayerCase(g, 2).layer.(*FCLayer)
	mp := randLayerCase(g, 3).layer.(*MaxPool2DLayer)
	ap := randLayerCase(g, 4).layer.(*AvgPool2DLayer)
	relu := &ReLULayer{Len: 7, ZP: -3}
	relu6 := &ReLU6Layer{Len: 7, ZP: -3, Q6: 40}

	type entry struct {
		layer preparedOp
		muts  []mutation
	}
	cloneI8 := func(s *[]int8) func() { return func() { *s = append([]int8(nil), *s...) } }
	cloneRQ := func(s *[]Requant) func() { return func() { *s = append([]Requant(nil), *s...) } }
	entries := map[string]entry{
		"Conv2D": {conv, []mutation{
			{name: "P.OutZP", mutate: func() { conv.P.OutZP++ }},
			{name: "W copy", mutate: cloneI8(&conv.W)},
			{name: "RQ copy", mutate: cloneRQ(&conv.RQ)},
			{name: "Bias set", mutate: func() { conv.Bias = make([]int32, conv.P.OutC) }},
			{name: "Scratch set", mutate: func() { conv.Scratch = make([]int8, conv.ScratchLen()+3) }},
			{name: "P.In.C", mutate: func() { conv.P.In.C++ }, fix: func() { conv.P.In.C-- }},
		}},
		"DepthwiseConv2D": {dw, []mutation{
			{name: "P.OutZP", mutate: func() { dw.P.OutZP++ }},
			{name: "W copy", mutate: cloneI8(&dw.W)},
			{name: "Scratch set", mutate: func() { dw.Scratch = make([]int8, dw.ScratchLen()+3) }},
			{name: "RQ nil", mutate: func() { dw.RQ = nil }, fix: func() { dw.RQ = g.requants(dw.P.In.C) }},
		}},
		"FC": {fc, []mutation{
			{name: "P.InZP", mutate: func() { fc.P.InZP++ }},
			{name: "W reslice", mutate: func() { fc.W = fc.W[:len(fc.W)-1] }, fix: func() { fc.W = fc.W[:len(fc.W)+1] }},
			{name: "Bias set", mutate: func() { fc.Bias = make([]int32, fc.P.Out) }},
			{name: "Scratch set", mutate: func() { fc.Scratch = make([]int8, fc.ScratchLen()+3) }},
		}},
		"MaxPool2D": {mp, []mutation{
			{name: "P.Padding", mutate: func() { mp.P.Padding = 9 }, fix: func() { mp.P.Padding = PaddingSame }},
			{name: "P.StrideH", mutate: func() { mp.P.StrideH++ }},
		}},
		"AvgPool2D": {ap, []mutation{
			{name: "P.StrideW", mutate: func() { ap.P.StrideW++ }},
		}},
		"ReLU": {relu, []mutation{
			{name: "ZP", mutate: func() { relu.ZP = 4 }},
			{name: "Len", mutate: func() { relu.Len = 0 }, fix: func() { relu.Len = 5 }},
		}},
		"ReLU6": {relu6, []mutation{
			{name: "Q6", mutate: func() { relu6.Q6 = 41 }},
			{name: "ZP", mutate: func() { relu6.ZP = 50 }, fix: func() { relu6.Q6 = 60 }},
			{name: "Len", mutate: func() { relu6.Len++ }},
		}},
	}
	for name, e := range entries {
		t.Run(name, func(t *testing.T) {
			for _, mu := range e.muts {
				require.NoError(t, e.layer.Prepare(), mu.name)
				dst, src := make([]int8, e.layer.OutShape().Len()), make([]int8, e.layer.InShape().Len())
				require.NoError(t, e.layer.Run(dst, src), mu.name)
				mu.mutate()
				assert.Equal(t, ErrNotPrepared, e.layer.Run(dst, src), "%s: after mutation", mu.name)
				if mu.fix != nil {
					assert.Error(t, e.layer.Prepare(), mu.name)
					assert.Equal(t, ErrNotPrepared, e.layer.Run(dst, src), "%s: after failed Prepare", mu.name)
					assert.Equal(t, Shape{}, e.layer.OutShape(), "%s: after failed Prepare", mu.name)
					mu.fix()
				}
				require.NoError(t, e.layer.Prepare(), "%s: re-Prepare", mu.name)
				dst, src = make([]int8, e.layer.OutShape().Len()), make([]int8, e.layer.InShape().Len())
				assert.NoError(t, e.layer.Run(dst, src), "%s: after re-Prepare", mu.name)
			}
		})
	}
}

func TestLayerScratch(t *testing.T) {
	l := &Conv2DLayer{P: Conv2DParams{In: Shape{3, 3, 1}, OutC: 1, KH: 3, KW: 3, StrideH: 1, StrideW: 1, ActMax: 127},
		W: make([]int8, 9), RQ: []Requant{half}, Scratch: make([]int8, 1)}
	dw := &DepthwiseConv2DLayer{P: DepthwiseConv2DParams{In: Shape{3, 3, 1}, KH: 3, KW: 3, StrideH: 1, StrideW: 1, ActMax: 127}}
	fc := &FCLayer{P: FCParams{In: 4, Out: 2}}
	if Backend() == "generic" {
		assert.Zero(t, l.ScratchLen())
		assert.Zero(t, dw.ScratchLen())
		assert.Zero(t, fc.ScratchLen())
		require.NoError(t, l.Prepare(), "short Scratch is ignored when ScratchLen is 0")
		assert.NoError(t, l.Run(make([]int8, 9), make([]int8, 9)))
		return
	}
	assert.Positive(t, l.ScratchLen())
	assert.Positive(t, dw.ScratchLen())
	assert.Positive(t, fc.ScratchLen())
	assert.Equal(t, ErrLength, l.Prepare(), "Scratch shorter than ScratchLen")
	l.Scratch = make([]int8, l.ScratchLen()-1)
	assert.Equal(t, ErrLength, l.Prepare(), "Scratch one byte short")
	l.Scratch = make([]int8, l.ScratchLen())
	require.NoError(t, l.Prepare())
	assert.NoError(t, l.Run(make([]int8, 9), make([]int8, 9)))
}

// sized returns slices of the lengths implied by a valid shape, else a dummy.
func sized(s Shape, err error) []int8 {
	if err != nil {
		return make([]int8, 1)
	}
	return make([]int8, s.Len())
}

func TestLayerPrepareErrors(t *testing.T) {
	rq1 := []Requant{half}
	rq3 := []Requant{half, half, half}
	okC := Conv2DParams{In: Shape{3, 3, 1}, OutC: 1, KH: 3, KW: 3, StrideH: 1, StrideW: 1, ActMax: 127}
	okD := DepthwiseConv2DParams{In: Shape{3, 3, 1}, KH: 3, KW: 3, StrideH: 1, StrideW: 1, ActMax: 127}
	okF := FCParams{In: 3, Out: 3, ActMax: 127}
	okP := Pool2DParams{In: Shape{3, 3, 1}, KH: 2, KW: 2, StrideH: 2, StrideW: 2}
	w9 := make([]int8, 9)
	withC := func(f func(*Conv2DParams)) Conv2DParams { p := okC; f(&p); return p }
	withD := func(f func(*DepthwiseConv2DParams)) DepthwiseConv2DParams { p := okD; f(&p); return p }
	withF := func(f func(*FCParams)) FCParams { p := okF; f(&p); return p }
	withP := func(f func(*Pool2DParams)) Pool2DParams { p := okP; f(&p); return p }

	type row struct {
		name  string
		want  error
		layer preparedOp
		free  func() error
	}
	conv := func(name string, want error, p Conv2DParams, w []int8, bias []int32, rq []Requant) row {
		os, err := p.OutShape()
		src, dst := sized(p.In, err), sized(os, err)
		return row{name, want, withScratch(&Conv2DLayer{P: p, W: w, Bias: bias, RQ: rq}, setConvScratch),
			func() error { return Conv2D(dst, src, w, bias, rq, p) }}
	}
	dwc := func(name string, want error, p DepthwiseConv2DParams, w []int8, bias []int32, rq []Requant) row {
		os, err := p.OutShape()
		src, dst := sized(p.In, err), sized(os, err)
		return row{name, want, withScratch(&DepthwiseConv2DLayer{P: p, W: w, Bias: bias, RQ: rq}, setDWScratch),
			func() error { return DepthwiseConv2D(dst, src, w, bias, rq, p) }}
	}
	fcr := func(name string, want error, p FCParams, w []int8, bias []int32, rq []Requant) row {
		src, dst := make([]int8, max(p.In, 1)), make([]int8, max(p.Out, 1))
		return row{name, want, withScratch(&FCLayer{P: p, W: w, Bias: bias, RQ: rq}, setFCScratch),
			func() error { return FullyConnected(dst, src, w, bias, rq, p) }}
	}
	pool := func(name string, want error, p Pool2DParams, avg bool) row {
		os, err := p.OutShape()
		src, dst := sized(p.In, err), sized(os, err)
		if avg {
			return row{name, want, &AvgPool2DLayer{P: p}, func() error { return AvgPool2D(dst, src, p) }}
		}
		return row{name, want, &MaxPool2DLayer{P: p}, func() error { return MaxPool2D(dst, src, p) }}
	}
	rows := []row{
		conv("conv ok", nil, okC, w9, nil, rq1),
		conv("conv stride 0", ErrShape, withC(func(p *Conv2DParams) { p.StrideH = 0 }), w9, nil, rq1),
		conv("conv OutC 0", ErrShape, withC(func(p *Conv2DParams) { p.OutC = 0 }), w9, nil, rq1),
		conv("conv VALID k>in", ErrShape, withC(func(p *Conv2DParams) { p.In.H, p.Padding = 2, PaddingValid }), w9, nil, rq1),
		conv("conv bad padding", ErrShape, withC(func(p *Conv2DParams) { p.Padding = 7 }), w9, nil, rq1),
		conv("conv K > MaxTaps", ErrShape, withC(func(p *Conv2DParams) { p.In = Shape{1, 1, MaxTaps + 1}; p.KH, p.KW = 1, 1 }), w9, nil, rq1),
		conv("conv huge", ErrShape, withC(func(p *Conv2DParams) { p.In = Shape{1 << 16, 1 << 16, 1} }), w9, nil, rq1),
		conv("conv W short", ErrLength, okC, w9[:8], nil, rq1),
		conv("conv W long", ErrLength, okC, append(w9[:9:9], 0), nil, rq1),
		conv("conv bias len", ErrLength, okC, w9, []int32{1, 2}, rq1),
		conv("conv rq nil", ErrLength, okC, w9, nil, nil),
		conv("conv rq mult", ErrRequant, okC, w9, nil, []Requant{{1<<30 - 1, 0}}),
		conv("conv rq neg", ErrRequant, okC, w9, nil, []Requant{{-1, 0}}),
		conv("conv rq shift lo", ErrRequant, okC, w9, nil, []Requant{{1 << 30, MinShift - 1}}),
		conv("conv rq shift hi", ErrRequant, okC, w9, nil, []Requant{{1 << 30, MaxShift + 1}}),
		conv("conv act", ErrActivation, withC(func(p *Conv2DParams) { p.ActMin, p.ActMax = 1, 0 }), w9, nil, rq1),
		dwc("dw ok", nil, okD, w9, nil, rq1),
		dwc("dw KW 0", ErrShape, withD(func(p *DepthwiseConv2DParams) { p.KW = 0 }), w9, nil, rq1),
		dwc("dw W short", ErrLength, okD, w9[:8], nil, rq1),
		dwc("dw W long", ErrLength, okD, append(w9[:9:9], 0), nil, rq1),
		dwc("dw bias", ErrLength, okD, w9, []int32{1, 2}, rq1),
		dwc("dw rq", ErrRequant, okD, w9, nil, []Requant{{Mult: 5}}),
		dwc("dw act", ErrActivation, withD(func(p *DepthwiseConv2DParams) { p.ActMin, p.ActMax = 5, 4 }), w9, nil, rq1),
		fcr("fc ok", nil, okF, w9, nil, rq3),
		fcr("fc In 0", ErrShape, withF(func(p *FCParams) { p.In = 0 }), w9, nil, rq3),
		fcr("fc In > MaxTaps", ErrShape, withF(func(p *FCParams) { p.In = MaxTaps + 1 }), w9, nil, rq3),
		fcr("fc W short", ErrLength, okF, w9[:8], nil, rq3),
		fcr("fc W long", ErrLength, okF, append(w9[:9:9], 0), nil, rq3),
		fcr("fc rq len", ErrLength, okF, w9, nil, rq1),
		fcr("fc bias len", ErrLength, okF, w9, []int32{1}, rq3),
		fcr("fc rq", ErrRequant, okF, w9, nil, []Requant{half, {Mult: 5}, half}),
		fcr("fc act", ErrActivation, withF(func(p *FCParams) { p.ActMin, p.ActMax = 127, 0 }), w9, nil, rq3),
		pool("max ok", nil, okP, false),
		pool("avg ok", nil, okP, true),
		pool("max C 0", ErrShape, withP(func(p *Pool2DParams) { p.In.C = 0 }), false),
		pool("avg stride 0", ErrShape, withP(func(p *Pool2DParams) { p.StrideW = 0 }), true),
		pool("avg VALID k>in", ErrShape, withP(func(p *Pool2DParams) { p.KH, p.Padding = 4, PaddingValid }), true),
	}
	for _, r := range rows {
		assert.Equal(t, r.want, r.free(), "%s: free function", r.name)
		assert.Equal(t, r.want, r.layer.Prepare(), "%s: Prepare", r.name)
	}

	// Element-wise layers: same errors as the free functions where defined;
	// Len outside [1, MaxElements] is ErrShape.
	src, dst := make([]int8, 4), make([]int8, 4)
	assert.Equal(t, ErrActivation, ReLU6(dst, src, 3, 2))
	assert.Equal(t, ErrActivation, (&ReLU6Layer{Len: 4, ZP: 3, Q6: 2}).Prepare())
	assert.NoError(t, (&ReLU6Layer{Len: 4, ZP: 3, Q6: 3}).Prepare())
	for _, n := range []int{0, -1, MaxElements + 1} {
		assert.Equal(t, ErrShape, (&ReLULayer{Len: n}).Prepare(), "relu len %d", n)
		assert.Equal(t, ErrShape, (&ReLU6Layer{Len: n, Q6: 1}).Prepare(), "relu6 len %d", n)
	}
	assert.NoError(t, (&ReLULayer{Len: MaxElements}).Prepare())
}

func TestLayerNoAllocs(t *testing.T) {
	rq := make([]Requant, 8)
	for i := range rq {
		rq[i] = half
	}
	bias := make([]int32, 8)
	src, w := make([]int8, 8*8*8), make([]int8, 8*3*3*8)
	dst := make([]int8, 8*8*8)
	pp := Pool2DParams{In: Shape{8, 8, 8}, KH: 2, KW: 2, StrideH: 2, StrideW: 2}
	layers := map[string]struct {
		l        preparedOp
		dst, src []int8
	}{
		"Conv2D": {withScratch(&Conv2DLayer{P: Conv2DParams{In: Shape{8, 8, 8}, OutC: 8, KH: 3, KW: 3, StrideH: 2, StrideW: 2, ActMin: -128, ActMax: 127},
			W: w, Bias: bias, RQ: rq}, setConvScratch), dst[:4*4*8], src},
		"DepthwiseConv2D": {withScratch(&DepthwiseConv2DLayer{P: DepthwiseConv2DParams{In: Shape{8, 8, 8}, KH: 3, KW: 3, StrideH: 1, StrideW: 1, ActMin: -128, ActMax: 127},
			W: w[:72], Bias: bias, RQ: rq}, setDWScratch), dst, src},
		"FC":        {withScratch(&FCLayer{P: FCParams{In: 64, Out: 8, ActMin: -128, ActMax: 127}, W: w[:512], Bias: bias, RQ: rq}, setFCScratch), dst[:8], src[:64]},
		"MaxPool2D": {&MaxPool2DLayer{P: pp}, dst[:128], src},
		"AvgPool2D": {&AvgPool2DLayer{P: pp}, dst[:128], src},
		"ReLU":      {&ReLULayer{Len: 512, ZP: 3}, dst, src},
		"ReLU6":     {&ReLU6Layer{Len: 512, ZP: 3, Q6: 60}, dst, src},
	}
	for name, c := range layers {
		require.NoError(t, c.l.Prepare(), name)
		require.NoError(t, c.l.Run(c.dst, c.src), name)
		assert.Zero(t, testing.AllocsPerRun(100, func() { _ = c.l.Prepare() }), "%s Prepare", name)
		assert.Zero(t, testing.AllocsPerRun(100, func() { _ = c.l.Run(c.dst, c.src) }), "%s Run", name)
		assert.Zero(t, testing.AllocsPerRun(100, func() { _ = c.l.Run(c.dst[:1], c.src) }), "%s Run error", name)
	}
}
