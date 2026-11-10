package qi8

import "math"

// ResizeParams describes TFLite RESIZE_NEAREST_NEIGHBOR on an NHWC tensor
// (batch 1): output OutH×OutW×In.C, each output pixel a copy of one input
// pixel (no requantisation; TFLite copies values even when the input and
// output quantisation differ).
type ResizeParams struct {
	In                      Shape
	OutH, OutW              int
	AlignCorners, HalfPixel bool
}

// nearestIndex is TFLite's reference GetNearestNeighbor, in float32:
//
//	scale = AlignCorners && out > 1 ? (in−1)/(out−1) : in/out
//	p     = (o + (HalfPixel ? 0.5 : 0)) · scale
//	i     = min(AlignCorners ? roundf(p) : floor(p), in−1), and ≥ 0 if HalfPixel
//
// roundf rounds half away from zero. All flag combinations are accepted
// (TFLite does too).
func nearestIndex(o, in, out int, align, half bool) int {
	var scale float32
	if align && out > 1 {
		scale = float32(in-1) / float32(out-1)
	} else {
		scale = float32(in) / float32(out)
	}
	var off float32
	if half {
		off = 0.5
	}
	// Explicit conversions round each step to float32 (no fusing).
	p := float64(float32(float32(float32(o)+off) * scale))
	var i int
	if align {
		i = int(math.Round(p))
	} else {
		i = int(math.Floor(p))
	}
	i = min(i, in-1)
	if half {
		i = max(i, 0)
	}
	return i
}

// ResizeNearest resizes src (shape p.In) into dst (p.OutH×p.OutW×p.In.C).
// ErrShape for an invalid input shape or output size; ErrLength when
// len(src) or len(dst) does not match. dst must not overlap src.
func ResizeNearest(dst, src []int8, p ResizeParams) error {
	n, ok := p.In.size()
	if !ok {
		return ErrShape
	}
	m, ok := mul3(p.OutH, p.OutW, p.In.C)
	if !ok {
		return ErrShape
	}
	if len(src) != n || len(dst) != m {
		return ErrLength
	}
	resizeGeneric(dst, src, &p)
	return nil
}

// resizeGeneric is the ResizeNearest loop; lengths checked. An output row
// that maps to the same input row as the previous one is copied from it.
func resizeGeneric(dst, src []int8, p *ResizeParams) {
	in, c := p.In, p.In.C
	orow := p.OutW * c
	prev := -1
	for oy := range p.OutH {
		row := dst[oy*orow : oy*orow+orow]
		iy := nearestIndex(oy, in.H, p.OutH, p.AlignCorners, p.HalfPixel)
		if iy == prev {
			copy(row, dst[(oy-1)*orow:oy*orow])
			continue
		}
		prev = iy
		srow := src[iy*in.W*c : (iy+1)*in.W*c]
		for ox := range p.OutW {
			ix := nearestIndex(ox, in.W, p.OutW, p.AlignCorners, p.HalfPixel)
			copy(row[ox*c:ox*c+c], srow[ix*c:ix*c+c])
		}
	}
}
