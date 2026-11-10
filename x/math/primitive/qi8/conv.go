package qi8

// Conv2DParams describes a 2-D convolution over an NHWC (batch 1) input.
type Conv2DParams struct {
	In   Shape // input H, W, C
	OutC int   // output channels
	// KH, KW is the kernel size; StrideH, StrideW the strides (>= 1).
	KH, KW           int
	StrideH, StrideW int
	Padding          Padding
	// InZP and OutZP are the activation zero points (weights are symmetric).
	InZP, OutZP int8
	// ActMin/ActMax is the fused activation clamp (see ActNone/ActReLU/ActReLU6).
	// Note the zero value [0, 0] clamps everything to 0.
	ActMin, ActMax int8
	// Round is the requantisation rounding (zero value RoundExact).
	Round Rounding
}

// OutShape returns the output shape for p.
func (p Conv2DParams) OutShape() (Shape, error) {
	if p.OutC <= 0 || p.OutC > MaxElements {
		return Shape{}, ErrShape
	}
	g, err := newGeometry(p.In, p.KH, p.KW, p.StrideH, p.StrideW, p.Padding)
	if err != nil {
		return Shape{}, err
	}
	if _, err := g.outLen(p.OutC); err != nil {
		return Shape{}, err
	}
	return Shape{H: g.outH, W: g.outW, C: p.OutC}, nil
}

// Conv2D computes a quantised 2-D convolution.
//
// src is NHWC [In.H][In.W][In.C]; weights are OHWI [OutC][KH][KW][In.C];
// bias is nil or len OutC; rq has len OutC; dst is [outH][outW][OutC] (see
// OutShape). For every output element
//
//	acc = bias[o] + sum (x - InZP) * w      (int32; padded taps contribute 0)
//	y   = clamp(rq[o].Apply(acc) + OutZP, ActMin, ActMax)   (ApplyTFLite with RoundTFLite)
func Conv2D(dst, src, weights []int8, bias []int32, rq []Requant, p Conv2DParams) error {
	pl, err := p.plan()
	if err != nil {
		return err
	}
	if len(src) != pl.inLen || len(weights) != pl.wLen || len(dst) != pl.outLen {
		return ErrLength
	}
	if err := pl.setQuant(bias, rq, p.InZP, p.OutZP, p.ActMin, p.ActMax, p.Round); err != nil {
		return err
	}
	conv2DGeneric(dst, src, weights, bias, rq, &pl)
	return nil
}

// convPlan is the validated, resolved form of Conv2DParams or
// DepthwiseConv2DParams: geometry, checked sizes and the quantisation
// constants the inner loops need.
type convPlan struct {
	geometry
	outC    int // output channels (== ch for depthwise)
	wPerOut int // weights per output channel: taps·inC (conv) or taps (depthwise)
	wLen    int // total weight count
	outLen  int // outH·outW·outC

	inZP          int32
	outZP, lo, hi int64 // output zero point and activation clamp
	round         Rounding
}

// plan validates the shape part of p (everything except slices, bias,
// requant and activation range) in the order Conv2D reports errors.
func (p Conv2DParams) plan() (convPlan, error) {
	if p.OutC <= 0 || p.OutC > MaxElements {
		return convPlan{}, ErrShape
	}
	g, err := newGeometry(p.In, p.KH, p.KW, p.StrideH, p.StrideW, p.Padding)
	if err != nil {
		return convPlan{}, err
	}
	wPerOut, ok := mul2(g.taps, g.ch)
	if !ok || wPerOut > MaxTaps {
		return convPlan{}, ErrShape
	}
	wLen, ok := mul2(p.OutC, wPerOut)
	if !ok {
		return convPlan{}, ErrShape
	}
	outLen, err := g.outLen(p.OutC)
	if err != nil {
		return convPlan{}, err
	}
	return convPlan{geometry: g, outC: p.OutC, wPerOut: wPerOut, wLen: wLen, outLen: outLen}, nil
}

// setQuant validates bias, requant (multipliers, then the rounding mode) and
// the activation range against pl.outC (in that order) and stores the
// quantisation constants in pl.
func (pl *convPlan) setQuant(bias []int32, rq []Requant, inZP, outZP, actMin, actMax int8, round Rounding) error {
	if err := validBias(bias, pl.outC); err != nil {
		return err
	}
	if err := validRequants(rq, pl.outC); err != nil {
		return err
	}
	if !round.valid() {
		return ErrRequant
	}
	lo, hi, err := clampRange(actMin, actMax)
	if err != nil {
		return err
	}
	pl.inZP, pl.outZP, pl.lo, pl.hi, pl.round = int32(inZP), int64(outZP), lo, hi, round
	return nil
}

// conv2DGeneric is the Conv2D inner loop. All arguments are validated
// against pl.
func conv2DGeneric(dst, src, weights []int8, bias []int32, rq []Requant, pl *convPlan) {
	g := &pl.geometry
	inC, outC, wPerOut := g.ch, pl.outC, pl.wPerOut
	zp, zpOut, lo, hi, tfl := pl.inZP, pl.outZP, pl.lo, pl.hi, pl.round == RoundTFLite
	rowStride := g.inW * inC
	di := 0
	for oy := range g.outH {
		iy, ky0, ky1 := span(oy, g.strH, g.padT, g.kh, g.inH)
		for ox := range g.outW {
			ix, kx0, kx1 := span(ox, g.strW, g.padL, g.kw, g.inW)
			n := (kx1 - kx0) * inC // contiguous run per kernel row (NHWC/OHWI)
			xBase := iy*rowStride + (ix+kx0)*inC
			for oc := range outC {
				var acc int32
				if len(bias) != 0 {
					acc = bias[oc]
				}
				wBase := oc*wPerOut + kx0*inC
				for ky := ky0; ky < ky1; ky++ {
					xo := xBase + ky*rowStride
					wo := wBase + ky*g.kw*inC
					acc += dotZP(src[xo:xo+n], weights[wo:wo+n], zp)
				}
				if tfl {
					dst[di] = requantizeTFLite(acc, rq[oc], zpOut, lo, hi)
				} else {
					dst[di] = requantize(acc, rq[oc], zpOut, lo, hi)
				}
				di++
			}
		}
	}
}

// DepthwiseConv2DParams describes a depthwise 2-D convolution with channel
// multiplier 1 (output channels == In.C).
type DepthwiseConv2DParams struct {
	In               Shape
	KH, KW           int
	StrideH, StrideW int
	Padding          Padding
	InZP, OutZP      int8
	// ActMin/ActMax is the fused activation clamp. The zero value [0, 0]
	// clamps everything to 0.
	ActMin, ActMax int8
	// Round is the requantisation rounding (zero value RoundExact).
	Round Rounding
}

// OutShape returns the output shape for p.
func (p DepthwiseConv2DParams) OutShape() (Shape, error) {
	g, err := newGeometry(p.In, p.KH, p.KW, p.StrideH, p.StrideW, p.Padding)
	if err != nil {
		return Shape{}, err
	}
	return Shape{H: g.outH, W: g.outW, C: p.In.C}, nil
}

// DepthwiseConv2D computes a quantised depthwise convolution.
//
// src is NHWC [In.H][In.W][C]; weights are [KH][KW][C]; bias is nil or len C;
// rq has len C; dst is [outH][outW][C]. Per channel c:
//
//	acc = bias[c] + sum_{ky,kx} (x[iy+ky][ix+kx][c] - InZP) * w[ky][kx][c]
//	y   = clamp(rq[c].Apply(acc) + OutZP, ActMin, ActMax)   (ApplyTFLite with RoundTFLite)
func DepthwiseConv2D(dst, src, weights []int8, bias []int32, rq []Requant, p DepthwiseConv2DParams) error {
	pl, err := p.plan()
	if err != nil {
		return err
	}
	if len(src) != pl.inLen || len(weights) != pl.wLen || len(dst) != pl.outLen {
		return ErrLength
	}
	if err := pl.setQuant(bias, rq, p.InZP, p.OutZP, p.ActMin, p.ActMax, p.Round); err != nil {
		return err
	}
	depthwiseConv2DGeneric(dst, src, weights, bias, rq, &pl)
	return nil
}

// plan validates the shape part of p in the order DepthwiseConv2D reports
// errors.
func (p DepthwiseConv2DParams) plan() (convPlan, error) {
	g, err := newGeometry(p.In, p.KH, p.KW, p.StrideH, p.StrideW, p.Padding)
	if err != nil {
		return convPlan{}, err
	}
	wLen, ok := mul2(g.taps, g.ch)
	if !ok {
		return convPlan{}, ErrShape
	}
	outLen, err := g.outLen(g.ch)
	if err != nil {
		return convPlan{}, err
	}
	return convPlan{geometry: g, outC: g.ch, wPerOut: g.taps, wLen: wLen, outLen: outLen}, nil
}

// depthwiseConv2DGeneric is the DepthwiseConv2D inner loop. All arguments
// are validated against pl.
func depthwiseConv2DGeneric(dst, src, weights []int8, bias []int32, rq []Requant, pl *convPlan) {
	g := &pl.geometry
	c := g.ch
	zp, zpOut, lo, hi, tfl := pl.inZP, pl.outZP, pl.lo, pl.hi, pl.round == RoundTFLite
	rowStride := g.inW * c
	wRow := g.kw * c
	di := 0
	for oy := range g.outH {
		iy, ky0, ky1 := span(oy, g.strH, g.padT, g.kh, g.inH)
		for ox := range g.outW {
			ix, kx0, kx1 := span(ox, g.strW, g.padL, g.kw, g.inW)
			base := iy*rowStride + ix*c
			for ch := range c {
				var acc int32
				if len(bias) != 0 {
					acc = bias[ch]
				}
				for ky := ky0; ky < ky1; ky++ {
					xo := base + ky*rowStride + ch
					wo := ky*wRow + ch
					for kx := kx0; kx < kx1; kx++ {
						acc += (int32(src[xo+kx*c]) - zp) * int32(weights[wo+kx*c])
					}
				}
				if tfl {
					dst[di] = requantizeTFLite(acc, rq[ch], zpOut, lo, hi)
				} else {
					dst[di] = requantize(acc, rq[ch], zpOut, lo, hi)
				}
				di++
			}
		}
	}
}
