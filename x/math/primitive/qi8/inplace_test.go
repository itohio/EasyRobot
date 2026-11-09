package qi8

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// inPlaceCase is a weighted prepared layer whose K (conv, FC) or C
// (depthwise) is not a multiple of the PIE row padding (32 or 16), so a
// packed row is longer than an unpacked one.
type inPlaceCase struct {
	name     string
	layer    preparedOp
	scratch  func() int              // ScratchLen
	inPlaceW func() []int8           // InPlaceW
	set      func(w, scratch []int8) // assigns W and Scratch
	free     func(dst, src, w []int8) error
	wLen     int
	in, out  Shape
}

func inPlaceCases(g gen) []inPlaceCase {
	cp := Conv2DParams{In: Shape{5, 4, 3}, OutC: 8, KH: 3, KW: 3, StrideH: 1, StrideW: 1, Padding: PaddingSame, InZP: 7, OutZP: -3, ActMin: -128, ActMax: 127}
	dp := DepthwiseConv2DParams{In: Shape{5, 4, 20}, KH: 3, KW: 3, StrideH: 1, StrideW: 1, Padding: PaddingSame, InZP: -5, OutZP: 2, ActMin: -128, ActMax: 127}
	fp := FCParams{In: 45, Out: 7, InZP: 3, OutZP: 1, ActMin: -128, ActMax: 127}
	conv := &Conv2DLayer{P: cp, Bias: g.bias(8), RQ: g.requants(8)}
	dw := &DepthwiseConv2DLayer{P: dp, Bias: g.bias(20), RQ: g.requants(20)}
	fc := &FCLayer{P: fp, Bias: g.bias(7), RQ: g.requants(7)}
	cos, _ := cp.OutShape()
	dos, _ := dp.OutShape()
	return []inPlaceCase{
		{"Conv2D", conv, conv.ScratchLen, conv.InPlaceW, func(w, s []int8) { conv.W, conv.Scratch = w, s },
			func(dst, src, w []int8) error { return Conv2D(dst, src, w, conv.Bias, conv.RQ, conv.P) },
			8 * 9 * 3, cp.In, cos},
		{"DepthwiseConv2D", dw, dw.ScratchLen, dw.InPlaceW, func(w, s []int8) { dw.W, dw.Scratch = w, s },
			func(dst, src, w []int8) error { return DepthwiseConv2D(dst, src, w, dw.Bias, dw.RQ, dw.P) },
			9 * 20, dp.In, dos},
		{"FC", fc, fc.ScratchLen, fc.InPlaceW, func(w, s []int8) { fc.W, fc.Scratch = w, s },
			func(dst, src, w []int8) error { return FullyConnected(dst, src, w, fc.Bias, fc.RQ, fc.P) },
			45 * 7, Shape{1, 1, 45}, Shape{1, 1, 7}},
	}
}

// TestInPlaceWPosition: W that overlaps the backend's packed region of
// Scratch anywhere but at its aligned start is rejected with ErrLength
// instead of being packed wrongly; W elsewhere is accepted.
func TestInPlaceWPosition(t *testing.T) {
	for _, c := range inPlaceCases(newGen(47)) {
		t.Run(c.name, func(t *testing.T) {
			n := c.scratch()
			if n == 0 {
				t.Skipf("backend %s packs no weights", Backend())
			}
			s := alignedAt(n+c.wLen, 3) // aligned start at s[13]
			for _, off := range []int{13 + 16, 13 + 1, 5} {
				c.set(s[off:off+c.wLen], s)
				assert.Equal(t, ErrLength, c.layer.Prepare(), "W at Scratch[%d:]", off)
			}
			// Right after Scratch (adjacent, not overlapping) or a separate
			// buffer: accepted.
			c.set(s[n:n+c.wLen:n+c.wLen], s[:n:n])
			require.NoError(t, c.layer.Prepare(), "W right after Scratch")
			c.set(make([]int8, c.wLen), s)
			require.NoError(t, c.layer.Prepare(), "separate W")
		})
	}
}

// TestInPlaceW: weights copied into InPlaceW() are packed in place and the
// layer computes what the free function computes with the original weights.
// InPlaceW is nil when the backend packs nothing or Scratch is short.
func TestInPlaceW(t *testing.T) {
	g := newGen(48)
	m := &mismatchLog{t: t}
	for _, c := range inPlaceCases(g) {
		n := c.scratch()
		c.set(nil, alignedAt(max(n-1, 0), 3))
		assert.Nil(t, c.inPlaceW(), "%s: Scratch shorter than ScratchLen", c.name)
		s := alignedAt(n, 3)
		c.set(nil, s)
		w := c.inPlaceW()
		if n == 0 {
			assert.Nil(t, w, "%s: backend %s packs nothing", c.name, Backend())
			continue
		}
		require.Len(t, w, c.wLen, c.name)
		orig := g.int8s(c.wLen)
		copy(w, orig)
		c.set(w, s)
		require.NoError(t, c.layer.Prepare(), c.name)
		src := g.int8s(c.in.Len())
		want, got := make([]int8, c.out.Len()), make([]int8, c.out.Len())
		require.NoError(t, c.free(want, src, orig), c.name)
		require.NoError(t, c.layer.Run(got, src), c.name)
		m.compare(func() string { return c.name }, got, want)
	}
	m.done()
}
