package qi8

// Pool2DParams describes a spatial pooling window over an NHWC (batch 1)
// input. Input and output share scale and zero point.
type Pool2DParams struct {
	In               Shape
	KH, KW           int
	StrideH, StrideW int
	Padding          Padding
}

// OutShape returns the output shape for p.
func (p Pool2DParams) OutShape() (Shape, error) {
	g, err := newGeometry(p.In, p.KH, p.KW, p.StrideH, p.StrideW, p.Padding)
	if err != nil {
		return Shape{}, err
	}
	if _, err := g.outLen(g.ch); err != nil {
		return Shape{}, err
	}
	return Shape{H: g.outH, W: g.outW, C: p.In.C}, nil
}

// poolPlan is the validated, resolved form of Pool2DParams.
type poolPlan struct {
	geometry
	outLen int // outH·outW·ch
}

// plan validates p (everything except the slices).
func (p Pool2DParams) plan() (poolPlan, error) {
	g, err := newGeometry(p.In, p.KH, p.KW, p.StrideH, p.StrideW, p.Padding)
	if err != nil {
		return poolPlan{}, err
	}
	outLen, err := g.outLen(g.ch)
	if err != nil {
		return poolPlan{}, err
	}
	return poolPlan{geometry: g, outLen: outLen}, nil
}

func (p Pool2DParams) planFor(dst, src []int8) (poolPlan, error) {
	pl, err := p.plan()
	if err != nil {
		return pl, err
	}
	if len(src) != pl.inLen || len(dst) != pl.outLen {
		return pl, ErrLength
	}
	return pl, nil
}

// MaxPool2D computes per-channel maxima over each window; padded cells are
// ignored (only valid cells take part).
func MaxPool2D(dst, src []int8, p Pool2DParams) error {
	pl, err := p.planFor(dst, src)
	if err != nil {
		return err
	}
	maxPool2DGeneric(dst, src, &pl)
	return nil
}

// maxPool2DGeneric is the MaxPool2D inner loop; dst and src are validated
// against pl.
func maxPool2DGeneric(dst, src []int8, pl *poolPlan) {
	g := &pl.geometry
	c := g.ch
	rowStride := g.inW * c
	di := 0
	for oy := range g.outH {
		iy, ky0, ky1 := span(oy, g.strH, g.padT, g.kh, g.inH)
		for ox := range g.outW {
			ix, kx0, kx1 := span(ox, g.strW, g.padL, g.kw, g.inW)
			base := iy*rowStride + ix*c
			for ch := range c {
				m := int8(-128)
				for ky := ky0; ky < ky1; ky++ {
					xo := base + ky*rowStride + ch
					for kx := kx0; kx < kx1; kx++ {
						if v := src[xo+kx*c]; v > m {
							m = v
						}
					}
				}
				dst[di] = m
				di++
			}
		}
	}
}

// AvgPool2D computes per-channel averages over each window, counting only
// valid (non-padded) cells: y = roundHalfAwayFromZero(sum(q)/count), exact
// integer arithmetic on the raw int8 values (TFLite semantics).
func AvgPool2D(dst, src []int8, p Pool2DParams) error {
	pl, err := p.planFor(dst, src)
	if err != nil {
		return err
	}
	avgPool2DGeneric(dst, src, &pl)
	return nil
}

// avgPool2DGeneric is the AvgPool2D inner loop; dst and src are validated
// against pl.
func avgPool2DGeneric(dst, src []int8, pl *poolPlan) {
	g := &pl.geometry
	c := g.ch
	rowStride := g.inW * c
	di := 0
	for oy := range g.outH {
		iy, ky0, ky1 := span(oy, g.strH, g.padT, g.kh, g.inH)
		for ox := range g.outW {
			ix, kx0, kx1 := span(ox, g.strW, g.padL, g.kw, g.inW)
			base := iy*rowStride + ix*c
			count := int32((ky1 - ky0) * (kx1 - kx0)) // >= 1 for SAME and VALID
			half := count / 2
			for ch := range c {
				var sum int32
				for ky := ky0; ky < ky1; ky++ {
					xo := base + ky*rowStride + ch
					for kx := kx0; kx < kx1; kx++ {
						sum += int32(src[xo+kx*c])
					}
				}
				// Rounds half away from zero; the result lies in [-128, 127].
				if sum >= 0 {
					sum = (sum + half) / count
				} else {
					sum = -((-sum + half) / count)
				}
				dst[di] = int8(sum)
				di++
			}
		}
	}
}
