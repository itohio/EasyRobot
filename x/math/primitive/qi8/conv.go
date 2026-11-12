package qi8

// Conv2DParams describes a 2-D convolution over an NHWC (batch 1) input.
type Conv2DParams struct {
	In   Shape // input H, W, C
	OutC int   // output channels
	// KH, KW is the kernel size; StrideH, StrideW the strides (>= 1).
	KH, KW           int
	StrideH, StrideW int
	Padding          Padding
	// InZP and OutZP are the activation zero points.
	InZP, OutZP int8
	// WZP is the per-tensor weight zero point: 0 for symmetric weights (the
	// usual int8 case); TFLite's uint8 models have asymmetric weights.
	WZP int8
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
//	acc = bias[o] + sum (x - InZP) * (w - WZP)      (int32; padded taps contribute 0)
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
	wZP           int32 // weight zero point (0: symmetric)
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
	if !ok || wPerOut > maxTaps(p.WZP) {
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
	return convPlan{geometry: g, outC: p.OutC, wPerOut: wPerOut, wLen: wLen, outLen: outLen, wZP: int32(p.WZP)}, nil
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
// against pl. Output channels are computed four at a time (dotZP4: each
// input byte is loaded once for four weight rows), then a pair (dotZP2) and
// a single channel for the remainder. A weight zero point enters as
// Σ (x−zp)(w−WZP) = Σ (x−zp)·w − WZP·Σ (x−zp): the window sum is computed
// once per output pixel (not per channel) and only when WZP ≠ 0; int32
// addition wraps, so the decomposition gives the same int32.
func conv2DGeneric(dst, src, weights []int8, bias []int32, rq []Requant, pl *convPlan) {
	g := &pl.geometry
	inC, outC, wPerOut := g.ch, pl.outC, pl.wPerOut
	zp, zpOut, lo, hi := pl.inZP, pl.outZP, pl.lo, pl.hi
	rowStride := g.inW * inC
	wRow := g.kw * inC
	rnd, wzp := pl.round, pl.wZP
	di := 0
	for oy := range g.outH {
		iy, ky0, ky1 := span(oy, g.strH, g.padT, g.kh, g.inH)
		for ox := range g.outW {
			ix, kx0, kx1 := span(ox, g.strW, g.padL, g.kw, g.inW)
			n := (kx1 - kx0) * inC // contiguous run per kernel row (NHWC/OHWI)
			xBase := iy*rowStride + (ix+kx0)*inC
			var corr int32 // WZP·Σ (x−zp) over the valid window
			if wzp != 0 {
				var s int32
				for ky := ky0; ky < ky1; ky++ {
					xo := xBase + ky*rowStride
					s += sumZP(src[xo:xo+n], zp)
				}
				corr = wzp * s
			}
			oc := 0
			for ; oc+3 < outC; oc += 4 {
				var a [4]int32
				if len(bias) != 0 {
					a = [4]int32{bias[oc], bias[oc+1], bias[oc+2], bias[oc+3]}
				}
				wBase := oc*wPerOut + kx0*inC
				for ky := ky0; ky < ky1; ky++ {
					xo := xBase + ky*rowStride
					wo := wBase + ky*wRow
					d := dotZP4(src[xo:xo+n], weights[wo:], wPerOut, zp)
					a[0] += d[0]
					a[1] += d[1]
					a[2] += d[2]
					a[3] += d[3]
				}
				for j := range a {
					dst[di+j] = requantizeMode(a[j]-corr, rq[oc+j], zpOut, lo, hi, rnd)
				}
				di += 4
			}
			for ; oc+1 < outC; oc += 2 {
				var a0, a1 int32
				if len(bias) != 0 {
					a0, a1 = bias[oc], bias[oc+1]
				}
				wBase := oc*wPerOut + kx0*inC
				for ky := ky0; ky < ky1; ky++ {
					xo := xBase + ky*rowStride
					wo := wBase + ky*wRow
					d0, d1 := dotZP2(src[xo:xo+n], weights[wo:wo+n], weights[wo+wPerOut:wo+wPerOut+n], zp)
					a0 += d0
					a1 += d1
				}
				dst[di] = requantizeMode(a0-corr, rq[oc], zpOut, lo, hi, rnd)
				dst[di+1] = requantizeMode(a1-corr, rq[oc+1], zpOut, lo, hi, rnd)
				di += 2
			}
			if oc < outC {
				var acc int32
				if len(bias) != 0 {
					acc = bias[oc]
				}
				wBase := oc*wPerOut + kx0*inC
				for ky := ky0; ky < ky1; ky++ {
					xo := xBase + ky*rowStride
					wo := wBase + ky*wRow
					acc += dotZP(src[xo:xo+n], weights[wo:wo+n], zp)
				}
				dst[di] = requantizeMode(acc-corr, rq[oc], zpOut, lo, hi, rnd)
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
	// WZP is the per-tensor weight zero point (0: symmetric; see Conv2DParams).
	WZP int8
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
//	acc = bias[c] + sum_{ky,kx} (x[iy+ky][ix+kx][c] - InZP) * (w[ky][kx][c] - WZP)
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
	if p.WZP != 0 && g.taps > MaxTapsWZP {
		return convPlan{}, ErrShape
	}
	wLen, ok := mul2(g.taps, g.ch)
	if !ok {
		return convPlan{}, ErrShape
	}
	outLen, err := g.outLen(g.ch)
	if err != nil {
		return convPlan{}, err
	}
	return convPlan{geometry: g, outC: g.ch, wPerOut: g.taps, wLen: wLen, outLen: outLen, wZP: int32(p.WZP)}, nil
}

// dwChunk is how many channels the generic depthwise loop accumulates at
// once (a stack array; no scratch needed).
const dwChunk = 32

// depthwiseConv2DGeneric is the DepthwiseConv2D inner loop. All arguments
// are validated against pl. Per output pixel it walks the taps outermost and
// a contiguous run of up to dwChunk channels innermost (sequential loads,
// no per-element stride arithmetic); int32 addition wraps, so the sum order
// does not change the result. A weight zero point selects a second copy of
// the tap loops (w−WZP) once per channel run, so the symmetric loops are
// the same code as without it.
func depthwiseConv2DGeneric(dst, src, weights []int8, bias []int32, rq []Requant, pl *convPlan) {
	g := &pl.geometry
	c := g.ch
	zp, zpOut, lo, hi, tfl := pl.inZP, pl.outZP, pl.lo, pl.hi, pl.round == RoundTFLite
	rowStride := g.inW * c
	wRow := g.kw * c
	var accBuf [dwChunk]int32
	di := 0
	for oy := range g.outH {
		iy, ky0, ky1 := span(oy, g.strH, g.padT, g.kh, g.inH)
		for ox := range g.outW {
			ix, kx0, kx1 := span(ox, g.strW, g.padL, g.kw, g.inW)
			base := iy*rowStride + ix*c
			for c0 := 0; c0 < c; c0 += dwChunk {
				acc := accBuf[:min(dwChunk, c-c0)]
				n := len(acc)
				if len(bias) != 0 {
					copy(acc, bias[c0:c0+n])
				} else {
					clear(acc)
				}
				if pl.wZP == 0 {
					for ky := ky0; ky < ky1; ky++ {
						xo := base + ky*rowStride + c0
						wo := ky*wRow + c0
						for kx := kx0; kx < kx1; kx++ {
							x := src[xo+kx*c:][:n]
							w := weights[wo+kx*c:][:n]
							for i := range acc {
								acc[i] += (int32(x[i]) - zp) * int32(w[i])
							}
						}
					}
				} else {
					wzp := pl.wZP
					for ky := ky0; ky < ky1; ky++ {
						xo := base + ky*rowStride + c0
						wo := ky*wRow + c0
						for kx := kx0; kx < kx1; kx++ {
							x := src[xo+kx*c:][:n]
							w := weights[wo+kx*c:][:n]
							for i := range acc {
								acc[i] += (int32(x[i]) - zp) * (int32(w[i]) - wzp)
							}
						}
					}
				}
				out := dst[di : di+n]
				r := rq[c0 : c0+n]
				if tfl {
					for i, a := range acc {
						out[i] = requantizeTFLite(a, r[i], zpOut, lo, hi)
					}
				} else {
					for i, a := range acc {
						out[i] = requantize(a, r[i], zpOut, lo, hi)
					}
				}
				di += n
			}
		}
	}
}
