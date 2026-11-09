package qi8

// Size limits, enforced by every kernel (ErrShape when exceeded).
const (
	// MaxElements bounds every element count (activations, weights, kernel
	// windows, dimensions) so that size arithmetic cannot overflow a 32-bit
	// int (ESP32-S3, TinyGo targets).
	MaxElements = 1 << 30
	// MaxTaps bounds a reduction length K (kernel taps × input channels, or
	// FC inputs) so the int32 accumulator cannot wrap: 255·128·K < 2^31,
	// since |x−zp| ≤ 255 and |w| ≤ 128. The bias term is the caller's
	// responsibility.
	MaxTaps = (1<<31 - 1) / (255 * 128)
)

// Shape is an NHWC activation shape with batch 1.
type Shape struct {
	H, W, C int
}

// Len returns H*W*C. It does not check for overflow; kernels validate shapes
// with the checked size instead (see MaxElements).
func (s Shape) Len() int { return s.H * s.W * s.C }

// size returns H*W*C, or false when a dimension is not positive or the count
// exceeds MaxElements.
func (s Shape) size() (int, bool) { return mul3(s.H, s.W, s.C) }

func (s Shape) valid() bool { _, ok := s.size(); return ok }

// mul2 returns a*b, or false when an operand is not positive or the product
// exceeds MaxElements. It never overflows, on 32- or 64-bit int.
func mul2(a, b int) (int, bool) {
	if a <= 0 || b <= 0 || a > MaxElements || b > MaxElements/a {
		return 0, false
	}
	return a * b, true
}

// mul3 is mul2 for three factors.
func mul3(a, b, c int) (int, bool) {
	ab, ok := mul2(a, b)
	if !ok {
		return 0, false
	}
	return mul2(ab, c)
}

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
	if in <= 0 || k <= 0 || stride <= 0 || in > MaxElements || k > MaxElements || stride > MaxElements {
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
// All sizes in it are checked against MaxElements.
type geometry struct {
	outH, outW   int
	padT, padL   int
	kh, kw       int
	strH, strW   int
	inH, inW, ch int
	inLen        int // inH·inW·ch
	outHW        int // outH·outW
	taps         int // kh·kw
}

func newGeometry(in Shape, kh, kw, sh, sw int, pad Padding) (geometry, error) {
	inLen, ok := in.size()
	if !ok {
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
	outHW, ok := mul2(oh, ow)
	if !ok {
		return geometry{}, ErrShape
	}
	taps, ok := mul2(kh, kw)
	if !ok || taps > MaxTaps {
		return geometry{}, ErrShape
	}
	return geometry{
		outH: oh, outW: ow, padT: pt, padL: pl,
		kh: kh, kw: kw, strH: sh, strW: sw,
		inH: in.H, inW: in.W, ch: in.C,
		inLen: inLen, outHW: outHW, taps: taps,
	}, nil
}

// outLen returns outH·outW·c, checked.
func (g geometry) outLen(c int) (int, error) {
	n, ok := mul2(g.outHW, c)
	if !ok {
		return 0, ErrShape
	}
	return n, nil
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
