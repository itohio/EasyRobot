package qi8

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

var half = Requant{Mult: 1 << 30, Shift: 0} // M' = 0.5

func TestQuantizeMultiplier(t *testing.T) {
	cases := []struct {
		real float64
		want Requant
	}{
		{0.5, Requant{1 << 30, 0}},
		{1, Requant{1 << 30, 1}},
		{0.75, Requant{3 << 29, 0}},
		{0.25, Requant{1 << 30, -1}},
		{1.0 / 3, Requant{1431655765, -1}}, // round(2^31 * 2/3)
		{math.Ldexp(1, -32), Requant{1 << 30, MinShift}},
	}
	for _, c := range cases {
		got, err := QuantizeMultiplier(c.real)
		require.NoError(t, err, c.real)
		assert.Equal(t, c.want, got, "%g", c.real)
		assert.True(t, got.Valid())
	}
	// Just below 2^30: frac rounds to 2^31 and renormalises to Shift 31 → out of range.
	_, err := QuantizeMultiplier(math.Nextafter(math.Ldexp(1, 30), 0))
	assert.Equal(t, ErrRequant, err)
	// Largest representable: (2^31-1)/2^31 * 2^30.
	got, err := QuantizeMultiplier(math.Ldexp(float64(math.MaxInt32), -1))
	require.NoError(t, err)
	assert.Equal(t, Requant{math.MaxInt32, MaxShift}, got)

	for _, bad := range []float64{0, -1, math.NaN(), math.Inf(1), math.Inf(-1), math.Ldexp(1, 30), math.Ldexp(1, -33)} {
		_, err := QuantizeMultiplier(bad)
		assert.Equal(t, ErrRequant, err, "%g", bad)
	}
	// Round trip: |M' - real| <= 2^(Shift-32).
	g := newGen(7)
	for range 10000 {
		real := math.Ldexp(0.5+g.r.Float64()/2, -31+g.r.IntN(61))
		rq, err := QuantizeMultiplier(real)
		require.NoError(t, err, real)
		back := math.Ldexp(float64(rq.Mult), int(rq.Shift)-31)
		assert.LessOrEqual(t, math.Abs(back-real), math.Ldexp(1, int(rq.Shift)-32), real)
	}
}

func TestRequantApplyHand(t *testing.T) {
	for _, c := range []struct{ acc, want int32 }{
		{0, 0}, {1, 1}, {-1, -1}, {2, 1}, {3, 2}, {-3, -2}, {5, 3}, {-5, -3}, {4, 2},
	} {
		assert.Equal(t, int64(c.want), half.Apply(c.acc), "acc=%d", c.acc)
	}
	quarter := Requant{1 << 30, -1}
	assert.Equal(t, int64(1), quarter.Apply(2))   // 0.5 → 1
	assert.Equal(t, int64(-1), quarter.Apply(-2)) // -0.5 → -1
	assert.Equal(t, int64(0), quarter.Apply(1))   // 0.25 → 0
	assert.Equal(t, int64(2), quarter.Apply(6))   // 1.5 → 2
	// Extreme: acc=MinInt32, Mult=MaxInt32, shift 62.
	r := Requant{math.MaxInt32, MinShift}
	assert.Equal(t, int64(-1), r.Apply(math.MinInt32)) // -(2^31-1)/2^31 → -1
	assert.Equal(t, int64(0), r.Apply(1))
}

func TestActHelpers(t *testing.T) {
	lo, hi := ActNone()
	assert.Equal(t, [2]int8{-128, 127}, [2]int8{lo, hi})
	lo, hi = ActReLU(0.1, -5)
	assert.Equal(t, [2]int8{-5, 127}, [2]int8{lo, hi})
	lo, hi = ActReLU6(0.05, -10) // 6/0.05 = 120 → 110
	assert.Equal(t, [2]int8{-10, 110}, [2]int8{lo, hi})
	lo, hi = ActReLU6(0.01, 0) // 600 → saturate
	assert.Equal(t, [2]int8{0, 127}, [2]int8{lo, hi})
	assert.Equal(t, int8(-128+24), Quantize6(0.25, -128))
	assert.Equal(t, int8(127), Quantize6(0, 0))
	assert.Equal(t, int8(-128+2), Quantize6(4, -128)) // 1.5 → 2 (half away)
}

func TestReLUHand(t *testing.T) {
	src := []int8{-128, -3, -2, 0, 5, 127}
	dst := make([]int8, len(src))
	require.NoError(t, ReLU(dst, src, -2))
	assert.Equal(t, []int8{-2, -2, -2, 0, 5, 127}, dst)
	require.NoError(t, ReLU6(dst, src, -2, 4))
	assert.Equal(t, []int8{-2, -2, -2, 0, 4, 4}, dst)
	// In place.
	require.NoError(t, ReLU(src, src, 0))
	assert.Equal(t, []int8{0, 0, 0, 0, 5, 127}, src)
	assert.Equal(t, ErrLength, ReLU(dst[:1], src, 0))
	assert.Equal(t, ErrLength, ReLU6(dst[:1], src, 0, 1))
	assert.Equal(t, ErrActivation, ReLU6(dst, src, 3, 2))
}

func TestDotHand(t *testing.T) {
	assert.Equal(t, int32(0), Dot(nil, nil))
	assert.Equal(t, int32(1*4-2*5+3*-6), Dot([]int8{1, -2, 3}, []int8{4, 5, -6, 99}))
	a := []int8{-128, -128, -128, -128, -128}
	assert.Equal(t, int32(5*16384), Dot(a, a))
}

func TestFullyConnectedHand(t *testing.T) {
	// acc = 7 + (10-2)*3 + (-5-2)*(-4) = 59; *0.5 = 29.5 → 30; +(-3) = 27.
	// Row 2: acc = 0 + 8*1 + (-7)*1 = 1 → 0.5 → 1; -3 → -2, clamped by ReLU(zp=-3) to -2.
	dst := make([]int8, 2)
	p := FCParams{In: 2, Out: 2, InZP: 2, OutZP: -3, ActMin: -3, ActMax: 127}
	require.NoError(t, FullyConnected(dst, []int8{10, -5}, []int8{3, -4, 1, 1}, []int32{7, 0}, []Requant{half, half}, p))
	assert.Equal(t, []int8{27, -2}, dst)
	// Saturation.
	p.ActMax = 20
	require.NoError(t, FullyConnected(dst, []int8{10, -5}, []int8{3, -4, 1, 1}, []int32{7, 0}, []Requant{half, half}, p))
	assert.Equal(t, []int8{20, -2}, dst)
}

func seq(n, start, step int) []int8 {
	s := make([]int8, n)
	for i := range s {
		s[i] = int8(start + i*step)
	}
	return s
}

func TestConv2DHand(t *testing.T) {
	ones := []int8{1, 1, 1, 1, 1, 1, 1, 1, 1}
	src := seq(9, 1, 1) // 3x3x1: 1..9
	p := Conv2DParams{In: Shape{3, 3, 1}, OutC: 1, KH: 3, KW: 3, StrideH: 1, StrideW: 1, ActMin: -128, ActMax: 127}
	dst := make([]int8, 9)
	require.NoError(t, Conv2D(dst, src, ones, nil, []Requant{half}, p))
	// window sums: 12 21 16 / 27 45 33 / 24 39 28, halved, rounding half away.
	assert.Equal(t, []int8{6, 11, 8, 14, 23, 17, 12, 20, 14}, dst)

	p.StrideH, p.StrideW = 2, 2 // out 2x2, padBefore 1: sums 12 16 24 28
	dst = dst[:4]
	require.NoError(t, Conv2D(dst, src, ones, nil, []Requant{half}, p))
	assert.Equal(t, []int8{6, 8, 12, 14}, dst)

	p.Padding = PaddingValid // 1x1: 45 → 23 (+ bias 1 → 46 → 23)
	require.NoError(t, Conv2D(dst[:1], src, ones, []int32{1}, []Requant{half}, p))
	assert.Equal(t, int8(23), dst[0])

	// Padding uses zpIn: with InZP=1 the padded taps are (zp-zp)=0, valid taps x-1.
	p = Conv2DParams{In: Shape{3, 3, 1}, OutC: 1, KH: 3, KW: 3, StrideH: 2, StrideW: 2, InZP: 1, OutZP: 0, ActMin: -128, ActMax: 127}
	require.NoError(t, Conv2D(dst, src, ones, nil, []Requant{half}, p))
	assert.Equal(t, []int8{4, 6, 10, 12}, dst) // (12-4)/2, (16-4)/2, (24-4)/2, (28-4)/2

	// 1x1 conv, 2 in channels → 2 out channels (pure per-pixel GEMV).
	p = Conv2DParams{In: Shape{1, 2, 2}, OutC: 2, KH: 1, KW: 1, StrideH: 1, StrideW: 1, ActMin: -128, ActMax: 127}
	one := Requant{1 << 30, 1}
	dst = make([]int8, 4)
	require.NoError(t, Conv2D(dst, []int8{1, 2, 3, 4}, []int8{1, 0, 1, -1}, nil, []Requant{one, one}, p))
	assert.Equal(t, []int8{1, -1, 3, -1}, dst)
}

func TestDepthwiseConv2DHand(t *testing.T) {
	// 3x3x2: channel 0 = 1..9, channel 1 = -1..-9; weights: ch0 all 1, ch1 all -1.
	src := make([]int8, 18)
	w := make([]int8, 18)
	for i := range 9 {
		src[2*i], src[2*i+1] = int8(i+1), int8(-(i + 1))
		w[2*i], w[2*i+1] = 1, -1
	}
	p := DepthwiseConv2DParams{In: Shape{3, 3, 2}, KH: 3, KW: 3, StrideH: 2, StrideW: 2, ActMin: -128, ActMax: 127}
	dst := make([]int8, 8)
	require.NoError(t, DepthwiseConv2D(dst, src, w, []int32{0, 2}, []Requant{half, half}, p))
	// sums 12 16 24 28 for both; ch1 gets bias 2 → 14 18 26 30.
	assert.Equal(t, []int8{6, 7, 8, 9, 12, 13, 14, 15}, dst)
}

func TestPoolHand(t *testing.T) {
	p := Pool2DParams{In: Shape{3, 3, 1}, KH: 2, KW: 2, StrideH: 2, StrideW: 2}
	dst := make([]int8, 4)
	require.NoError(t, MaxPool2D(dst, seq(9, 1, 1), p))
	assert.Equal(t, []int8{5, 6, 8, 9}, dst)
	require.NoError(t, AvgPool2D(dst, seq(9, 1, 1), p))
	assert.Equal(t, []int8{3, 5, 8, 9}, dst) // 3, 4.5→5, 7.5→8, 9
	require.NoError(t, AvgPool2D(dst, seq(9, -1, -1), p))
	assert.Equal(t, []int8{-3, -5, -8, -9}, dst)
	require.NoError(t, MaxPool2D(dst, seq(9, -1, -1), p))
	assert.Equal(t, []int8{-1, -3, -7, -9}, dst)
	all := []int8{-128, -128, -128, -128, -128, -128, -128, -128, -128}
	require.NoError(t, MaxPool2D(dst, all, p))
	assert.Equal(t, []int8{-128, -128, -128, -128}, dst)
	require.NoError(t, AvgPool2D(dst, all, p))
	assert.Equal(t, []int8{-128, -128, -128, -128}, dst)
}

func TestOutShapeSame(t *testing.T) {
	for _, c := range []struct{ in, k, s, out, before int }{
		{7, 3, 2, 4, 1}, {8, 3, 2, 4, 0}, {8, 3, 1, 8, 1}, {5, 1, 3, 2, 0}, {1, 5, 1, 1, 2}, {6, 2, 2, 3, 0}, {7, 5, 2, 4, 2},
	} {
		out, before, err := window(c.in, c.k, c.s, PaddingSame)
		require.NoError(t, err)
		assert.Equal(t, [2]int{c.out, c.before}, [2]int{out, before}, "%+v", c)
	}
	out, before, err := window(7, 3, 2, PaddingValid)
	require.NoError(t, err)
	assert.Equal(t, [2]int{3, 0}, [2]int{out, before})
}

func TestErrors(t *testing.T) {
	rq := []Requant{half}
	ok := Conv2DParams{In: Shape{3, 3, 1}, OutC: 1, KH: 3, KW: 3, StrideH: 1, StrideW: 1, ActMax: 127}
	dst, src, w := make([]int8, 9), make([]int8, 9), make([]int8, 9)
	require.NoError(t, Conv2D(dst, src, w, nil, rq, ok))

	bad := ok
	bad.StrideH = 0
	assert.Equal(t, ErrShape, Conv2D(dst, src, w, nil, rq, bad))
	bad = ok
	bad.OutC = 0
	assert.Equal(t, ErrShape, Conv2D(dst, src, w, nil, rq, bad))
	bad = ok
	bad.In.H = 2
	bad.Padding = PaddingValid
	assert.Equal(t, ErrShape, Conv2D(dst, src, w, nil, rq, bad))
	bad = ok
	bad.Padding = 7
	assert.Equal(t, ErrShape, Conv2D(dst, src, w, nil, rq, bad))
	assert.Equal(t, ErrLength, Conv2D(dst[:8], src, w, nil, rq, ok))
	assert.Equal(t, ErrLength, Conv2D(dst, src[:8], w, nil, rq, ok))
	assert.Equal(t, ErrLength, Conv2D(dst, src, w[:8], nil, rq, ok))
	assert.Equal(t, ErrLength, Conv2D(dst, src, w, []int32{1, 2}, rq, ok))
	assert.Equal(t, ErrLength, Conv2D(dst, src, w, nil, nil, ok))
	for _, r := range []Requant{{1<<30 - 1, 0}, {-1, 0}, {1 << 30, MinShift - 1}, {1 << 30, MaxShift + 1}} {
		assert.Equal(t, ErrRequant, Conv2D(dst, src, w, nil, []Requant{r}, ok), "%+v", r)
	}
	bad = ok
	bad.ActMin, bad.ActMax = 1, 0
	assert.Equal(t, ErrActivation, Conv2D(dst, src, w, nil, rq, bad))

	dp := DepthwiseConv2DParams{In: Shape{3, 3, 1}, KH: 3, KW: 3, StrideH: 1, StrideW: 1, ActMax: 127}
	require.NoError(t, DepthwiseConv2D(dst, src, w, nil, rq, dp))
	assert.Equal(t, ErrLength, DepthwiseConv2D(dst, src, w[:8], nil, rq, dp))
	assert.Equal(t, ErrRequant, DepthwiseConv2D(dst, src, w, nil, []Requant{{}}, dp))
	dp.KW = 0
	assert.Equal(t, ErrShape, DepthwiseConv2D(dst, src, w, nil, rq, dp))

	fp := FCParams{In: 3, Out: 3, ActMax: 127}
	require.NoError(t, FullyConnected(dst[:3], src[:3], w, nil, []Requant{half, half, half}, fp))
	assert.Equal(t, ErrLength, FullyConnected(dst[:3], src[:3], w[:8], nil, []Requant{half, half, half}, fp))
	assert.Equal(t, ErrLength, FullyConnected(dst[:3], src[:3], w, nil, rq, fp))
	assert.Equal(t, ErrShape, FullyConnected(dst[:3], src[:3], w, nil, rq, FCParams{In: 0, Out: 3}))
	fp.ActMin = 127
	fp.ActMax = 0
	assert.Equal(t, ErrActivation, FullyConnected(dst[:3], src[:3], w, nil, []Requant{half, half, half}, fp))

	pp := Pool2DParams{In: Shape{3, 3, 1}, KH: 2, KW: 2, StrideH: 2, StrideW: 2}
	assert.Equal(t, ErrLength, MaxPool2D(dst, src, pp))
	assert.Equal(t, ErrLength, AvgPool2D(dst[:4], src[:8], pp))
	pp.In.C = 0
	assert.Equal(t, ErrShape, AvgPool2D(dst[:4], src, pp))
}

func TestNoAllocs(t *testing.T) {
	cp := Conv2DParams{In: Shape{8, 8, 8}, OutC: 8, KH: 3, KW: 3, StrideH: 2, StrideW: 2, ActMin: -128, ActMax: 127}
	src, w := make([]int8, 8*8*8), make([]int8, 8*3*3*8)
	dst := make([]int8, 4*4*8)
	bias := make([]int32, 8)
	rq := make([]Requant, 8)
	for i := range rq {
		rq[i] = half
	}
	dp := DepthwiseConv2DParams{In: Shape{8, 8, 8}, KH: 3, KW: 3, StrideH: 1, StrideW: 1, ActMin: -128, ActMax: 127}
	ddst := make([]int8, 8*8*8)
	pp := Pool2DParams{In: Shape{8, 8, 8}, KH: 2, KW: 2, StrideH: 2, StrideW: 2}
	fp := FCParams{In: 64, Out: 8, ActMin: -128, ActMax: 127}
	var sink int32
	kernels := map[string]func(){
		"Dot":             func() { sink += Dot(src, w) },
		"FullyConnected":  func() { _ = FullyConnected(dst[:8], src[:64], w[:512], bias, rq, fp) },
		"Conv2D":          func() { _ = Conv2D(dst, src, w, bias, rq, cp) },
		"DepthwiseConv2D": func() { _ = DepthwiseConv2D(ddst, src, w[:72], bias, rq, dp) },
		"MaxPool2D":       func() { _ = MaxPool2D(dst, src, pp) },
		"AvgPool2D":       func() { _ = AvgPool2D(dst, src, pp) },
		"ReLU":            func() { _ = ReLU(ddst, src, 3) },
		"ReLU6":           func() { _ = ReLU6(ddst, src, 3, 60) },
		"ConvError":       func() { _ = Conv2D(dst[:1], src, w, bias, rq, cp) },
		"QuantizeMult":    func() { _, _ = QuantizeMultiplier(0.123) },
	}
	for name, f := range kernels {
		f()
		assert.Zero(t, testing.AllocsPerRun(100, f), name)
	}
	_ = sink
}
