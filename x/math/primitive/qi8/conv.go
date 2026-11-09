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
//	y   = clamp(rq[o].Apply(acc) + OutZP, ActMin, ActMax)
func Conv2D(dst, src, weights []int8, bias []int32, rq []Requant, p Conv2DParams) error {
	if p.OutC <= 0 || p.OutC > MaxElements {
		return ErrShape
	}
	g, err := newGeometry(p.In, p.KH, p.KW, p.StrideH, p.StrideW, p.Padding)
	if err != nil {
		return err
	}
	inC, outC := g.ch, p.OutC
	wPerOut, ok := mul2(g.taps, inC)
	if !ok || wPerOut > MaxTaps {
		return ErrShape
	}
	wLen, ok := mul2(outC, wPerOut)
	if !ok {
		return ErrShape
	}
	outLen, err := g.outLen(outC)
	if err != nil {
		return err
	}
	if len(src) != g.inLen || len(weights) != wLen || len(dst) != outLen {
		return ErrLength
	}
	if err := validBias(bias, outC); err != nil {
		return err
	}
	if err := validRequants(rq, outC); err != nil {
		return err
	}
	lo, hi, err := clampRange(p.ActMin, p.ActMax)
	if err != nil {
		return err
	}
	zp, zpOut := int32(p.InZP), int64(p.OutZP)
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
				dst[di] = requantize(acc, rq[oc], zpOut, lo, hi)
				di++
			}
		}
	}
	return nil
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
//	y   = clamp(rq[c].Apply(acc) + OutZP, ActMin, ActMax)
func DepthwiseConv2D(dst, src, weights []int8, bias []int32, rq []Requant, p DepthwiseConv2DParams) error {
	g, err := newGeometry(p.In, p.KH, p.KW, p.StrideH, p.StrideW, p.Padding)
	if err != nil {
		return err
	}
	c := g.ch
	wLen, ok := mul2(g.taps, c)
	if !ok {
		return ErrShape
	}
	outLen, err := g.outLen(c)
	if err != nil {
		return err
	}
	if len(src) != g.inLen || len(weights) != wLen || len(dst) != outLen {
		return ErrLength
	}
	if err := validBias(bias, c); err != nil {
		return err
	}
	if err := validRequants(rq, c); err != nil {
		return err
	}
	lo, hi, err := clampRange(p.ActMin, p.ActMax)
	if err != nil {
		return err
	}
	zp, zpOut := int32(p.InZP), int64(p.OutZP)
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
				dst[di] = requantize(acc, rq[ch], zpOut, lo, hi)
				di++
			}
		}
	}
	return nil
}
