package qi8

// PadParams describes constant padding of an NHWC tensor (TFLite PAD and
// PADV2, batch 1, no batch padding): Top/Bottom rows, Left/Right columns and
// Front/Back channels of Value. TFLite's int8 PAD pads with the output zero
// point; PADV2 with its constant input (whose quantisation must equal the
// output's). Both copy input values unchanged (no requantisation, even when
// the input and output quantisation differ).
type PadParams struct {
	In                       Shape
	Top, Bottom, Left, Right int // H and W padding
	Front, Back              int // C padding
	Value                    int8
}

// addDim returns d + a + b, or false when a padding is negative or the sum
// exceeds MaxElements. d must be in (0, MaxElements].
func addDim(d, a, b int) (int, bool) {
	if a < 0 || b < 0 || a > MaxElements-d || b > MaxElements-d-a {
		return 0, false
	}
	return d + a + b, true
}

// OutShape returns the padded shape, or ErrShape for an invalid input
// shape, a negative padding, or an output larger than MaxElements.
func (p PadParams) OutShape() (Shape, error) {
	if !p.In.valid() {
		return Shape{}, ErrShape
	}
	h, okH := addDim(p.In.H, p.Top, p.Bottom)
	w, okW := addDim(p.In.W, p.Left, p.Right)
	c, okC := addDim(p.In.C, p.Front, p.Back)
	out := Shape{H: h, W: w, C: c}
	if !okH || !okW || !okC || !out.valid() {
		return Shape{}, ErrShape
	}
	return out, nil
}

// Pad writes src (shape p.In) padded with p.Value into dst (shape
// p.OutShape()). ErrShape from OutShape; ErrLength when len(src) or len(dst)
// does not match. dst must not overlap src.
func Pad(dst, src []int8, p PadParams) error {
	out, err := p.OutShape()
	if err != nil {
		return err
	}
	if len(src) != p.In.Len() || len(dst) != out.Len() {
		return ErrLength
	}
	padGeneric(dst, src, &p, out)
	return nil
}

// padFill sets every element of s to v.
func padFill(s []int8, v int8) {
	for i := range s {
		s[i] = v
	}
}

// padGeneric is the Pad loop; lengths checked.
func padGeneric(dst, src []int8, p *PadParams, out Shape) {
	in := p.In
	oc := out.C
	orow := out.W * oc
	irow := in.W * in.C
	top := p.Top * orow
	padFill(dst[:top], p.Value)
	d := dst[top:]
	for y := range in.H {
		row, s := d[:orow], src[y*irow:y*irow+irow]
		padFill(row[:p.Left*oc], p.Value)
		mid := row[p.Left*oc : (p.Left+in.W)*oc]
		if oc == in.C {
			copy(mid, s)
		} else {
			for x := range in.W {
				px := mid[x*oc : x*oc+oc]
				padFill(px[:p.Front], p.Value)
				copy(px[p.Front:], s[x*in.C:x*in.C+in.C])
				padFill(px[p.Front+in.C:], p.Value)
			}
		}
		padFill(row[(p.Left+in.W)*oc:], p.Value)
		d = d[orow:]
	}
	padFill(d, p.Value)
}
