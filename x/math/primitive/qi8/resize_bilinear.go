package qi8

// maxBilinearDim bounds every spatial dimension of ResizeBilinear so that
// TFLite's int32 index arithmetic (1024·size, value·scale) cannot overflow.
const maxBilinearDim = 1 << 20

// bilinearScale10 is TFLite's 10-bit fixed-point axis scale:
// round(1024·in/out), or round(1024·(in−1)/(out−1)) with align corners and
// out > 1 (integer division with +divisor/2).
func bilinearScale10(in, out int, align bool) int32 {
	if align && out > 1 {
		return int32((1<<10*(in-1) + (out-1)/2) / (out - 1))
	}
	return int32((1<<10*in + out/2) / out)
}

// bilinearInterp is TFLite's ComputeInterpolationValuesInteger: the source
// position s in 1/1024 units (with the half-pixel shift), its lower cell
// max(s/1024, 0) and upper cell min(⌈s/1024⌉, in−1), divisions truncating.
func bilinearInterp(v, scale10 int32, half bool, in int32) (s, lo, hi int32) {
	s = v * scale10
	if half {
		s += scale10/2 - 1<<9
	}
	lo = max(s/(1<<10), 0)
	hi = min((s+1<<10-1)/(1<<10), in-1)
	return s, lo, hi
}

// bilinearAxisOK reports whether the last output index's lower cell lies
// inside the input (TFLite does not clamp it and would read out of bounds
// for extreme upscaling, e.g. 2 → 3000).
func bilinearAxisOK(in, out int, align, half bool) bool {
	_, lo, _ := bilinearInterp(int32(out-1), bilinearScale10(in, out, align), half, int32(in))
	return int(lo) <= in-1
}

// ResizeBilinear is TFLite's int8 RESIZE_BILINEAR (reference
// ResizeBilinearInteger) on src (shape p.In) into dst (p.OutH×p.OutW×p.In.C).
// It works on the raw int8 values (no zero-point arithmetic; TFLite does not
// require equal input and output quantisation and does not requantise). Per
// axis, scale10 and the cells (y0, y1), (x0, x1) come from bilinearScale10
// and bilinearInterp; with dy = sy − 1024·y0 and dx = sx − 1024·x0:
//
//	o = q[y0,x0]·(1024−dy)(1024−dx) + q[y1,x0]·dy(1024−dx) + q[y0,x1]·(1024−dy)dx + q[y1,x1]·dy·dx
//	dst = (o ± 2^19) / 2^20   (sign of o; truncating division)
//
// ErrShape for an invalid shape or output size, AlignCorners together with
// HalfPixel (TFLite rejects it), a dimension above 2^20, or an upscale so
// extreme that TFLite would read outside the input; ErrLength when len(src)
// or len(dst) does not match. dst must not overlap src.
func ResizeBilinear(dst, src []int8, p ResizeParams) error {
	n, ok := p.In.size()
	if !ok {
		return ErrShape
	}
	m, ok := mul3(p.OutH, p.OutW, p.In.C)
	if !ok || p.AlignCorners && p.HalfPixel {
		return ErrShape
	}
	if p.In.H > maxBilinearDim || p.In.W > maxBilinearDim || p.OutH > maxBilinearDim || p.OutW > maxBilinearDim {
		return ErrShape
	}
	if !bilinearAxisOK(p.In.H, p.OutH, p.AlignCorners, p.HalfPixel) ||
		!bilinearAxisOK(p.In.W, p.OutW, p.AlignCorners, p.HalfPixel) {
		return ErrShape
	}
	if len(src) != n || len(dst) != m {
		return ErrLength
	}
	resizeBilinearGeneric(dst, src, &p)
	return nil
}

// resizeBilinearGeneric is the ResizeBilinear loop; p and lengths checked.
//
// int32 is exact: |dy|, |dx| ≤ 512 below zero (half-pixel, s ≥ −512) and
// < 1024 otherwise, so every weight magnitude is below 1536² and the sum of
// |q|·|w| over the four terms is below 128·2048² = 2^29.
func resizeBilinearGeneric(dst, src []int8, p *ResizeParams) {
	c := p.In.C
	inH, inW := int32(p.In.H), int32(p.In.W)
	hs := bilinearScale10(p.In.H, p.OutH, p.AlignCorners)
	ws := bilinearScale10(p.In.W, p.OutW, p.AlignCorners)
	irow := p.In.W * c
	o := 0
	for oy := range p.OutH {
		sy, y0, y1 := bilinearInterp(int32(oy), hs, p.HalfPixel, inH)
		dy := sy - y0<<10
		r0, r1 := src[int(y0)*irow:], src[int(y1)*irow:]
		for ox := range p.OutW {
			sx, x0, x1 := bilinearInterp(int32(ox), ws, p.HalfPixel, inW)
			dx := sx - x0<<10
			w00, w10 := (1<<10-dy)*(1<<10-dx), dy*(1<<10-dx)
			w01, w11 := (1<<10-dy)*dx, dy*dx
			q00, q10 := r0[int(x0)*c:int(x0)*c+c], r1[int(x0)*c:int(x0)*c+c]
			q01, q11 := r0[int(x1)*c:int(x1)*c+c], r1[int(x1)*c:int(x1)*c+c]
			d := dst[o : o+c]
			for ch := range d {
				v := int32(q00[ch])*w00 + int32(q10[ch])*w10 + int32(q01[ch])*w01 + int32(q11[ch])*w11
				if v > 0 {
					v += 1 << 19
				} else {
					v -= 1 << 19
				}
				d[ch] = int8(v / (1 << 20))
			}
			o += c
		}
	}
}
