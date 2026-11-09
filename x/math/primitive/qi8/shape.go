package qi8

// Shape is an NHWC activation shape with batch 1.
type Shape struct {
	H, W, C int
}

// Len returns H*W*C.
func (s Shape) Len() int { return s.H * s.W * s.C }

func (s Shape) valid() bool { return s.H > 0 && s.W > 0 && s.C > 0 }

// Padding selects the spatial padding scheme.
type Padding uint8

const (
	// PaddingSame is TensorFlow "SAME": out = ceil(in/stride),
	// padTotal = max((out-1)*stride + k - in, 0), padBefore = padTotal/2
	// (the extra cell, if any, goes after).
	PaddingSame Padding = iota
	// PaddingValid is TensorFlow "VALID": out = (in-k)/stride + 1, no padding;
	// requires in >= k.
	PaddingValid
)

// window computes the output size and leading padding of one spatial axis.
func window(in, k, stride int, pad Padding) (out, before int, err error) {
	if in <= 0 || k <= 0 || stride <= 0 {
		return 0, 0, ErrShape
	}
	switch pad {
	case PaddingSame:
		out = (in + stride - 1) / stride
		total := (out-1)*stride + k - in
		if total < 0 {
			total = 0
		}
		return out, total / 2, nil
	case PaddingValid:
		if in < k {
			return 0, 0, ErrShape
		}
		return (in-k)/stride + 1, 0, nil
	}
	return 0, 0, ErrShape
}

// geometry is the resolved spatial geometry shared by conv and pool kernels.
type geometry struct {
	outH, outW   int
	padT, padL   int
	kh, kw       int
	strH, strW   int
	inH, inW, ch int
}

func newGeometry(in Shape, kh, kw, sh, sw int, pad Padding) (geometry, error) {
	if !in.valid() {
		return geometry{}, ErrShape
	}
	oh, pt, err := window(in.H, kh, sh, pad)
	if err != nil {
		return geometry{}, err
	}
	ow, pl, err := window(in.W, kw, sw, pad)
	if err != nil {
		return geometry{}, err
	}
	return geometry{
		outH: oh, outW: ow, padT: pt, padL: pl,
		kh: kh, kw: kw, strH: sh, strW: sw,
		inH: in.H, inW: in.W, ch: in.C,
	}, nil
}

// span returns the input origin of output index o along one axis and the
// valid kernel tap range [k0, k1) that falls inside [0, in).
func span(o, stride, padBefore, k, in int) (origin, k0, k1 int) {
	origin = o*stride - padBefore
	k0, k1 = 0, k
	if origin < 0 {
		k0 = -origin
	}
	if in-origin < k1 {
		k1 = in - origin
	}
	return origin, k0, k1
}

func clampRange(lo, hi int8) (int64, int64, error) {
	if lo > hi {
		return 0, 0, ErrActivation
	}
	return int64(lo), int64(hi), nil
}
