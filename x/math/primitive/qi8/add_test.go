package qi8

import (
	"fmt"
	"math"
	"math/big"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestQuantizeAddMultipliers(t *testing.T) {
	cases := []struct {
		a, b float64
		want AddRequant
	}{
		{1, 1, AddRequant{MultA: 1 << 30, MultB: 1 << 30, Shift: 1}},
		{0.5, 0.25, AddRequant{MultA: 1 << 30, MultB: 1 << 29, Shift: 0}},
		{0.25, 0.75, AddRequant{MultA: 1 << 29, MultB: 3 << 29, Shift: 0}},
		{3, 1.0 / 3, AddRequant{MultA: 3 << 29, MultB: int32(math.Round(1.0 / 3 * (1 << 29))), Shift: 2}},
		// rounds up to 2^31: renormalised, the other input follows the new shift
		{1 - 0x1p-40, 0.5, AddRequant{MultA: 1 << 30, MultB: 1 << 29, Shift: 1}},
		{0x1p-32, 0x1p-33, AddRequant{MultA: 1 << 30, MultB: 1 << 29, Shift: -31}},
	}
	for _, c := range cases {
		got, err := QuantizeAddMultipliers(c.a, c.b)
		require.NoError(t, err, "%v %v", c.a, c.b)
		assert.Equal(t, c.want, got, "%v %v", c.a, c.b)
		assert.True(t, got.Valid(), "%v %v", c.a, c.b)
		swapped, err := QuantizeAddMultipliers(c.b, c.a)
		require.NoError(t, err)
		assert.Equal(t, AddRequant{MultA: got.MultB, MultB: got.MultA, Shift: got.Shift}, swapped)
	}
	bad := [][2]float64{
		{0, 1}, {1, 0}, {-1, 1}, {1, math.NaN()}, {math.Inf(1), 1},
		{0x1p30, 1},        // too large
		{0x1p-33, 0x1p-34}, // too small
		{1, 0x1p-33},       // the smaller multiplier rounds to 0
	}
	for _, c := range bad {
		_, err := QuantizeAddMultipliers(c[0], c[1])
		assert.ErrorIs(t, err, ErrRequant, "%v", c)
	}
}

func TestAddRequantValid(t *testing.T) {
	assert.True(t, AddRequant{MultA: 1 << 30, MultB: 1, Shift: 0}.Valid())
	assert.True(t, AddRequant{MultA: 5, MultB: math.MaxInt32, Shift: MaxShift}.Valid())
	assert.False(t, AddRequant{MultA: 1<<30 - 1, MultB: 1<<30 - 1, Shift: 0}.Valid(), "neither in [2^30, 2^31)")
	assert.False(t, AddRequant{MultA: 1 << 30, MultB: 0, Shift: 0}.Valid())
	assert.False(t, AddRequant{MultA: -1, MultB: 1 << 30, Shift: 0}.Valid())
	assert.False(t, AddRequant{MultA: 1 << 30, MultB: 1, Shift: MinShift - 1}.Valid())
	assert.False(t, AddRequant{MultA: 1 << 30, MultB: 1, Shift: MaxShift + 1}.Valid())
}

func TestAddHand(t *testing.T) {
	// real multipliers 1/2 and 1/4: out = round(a/2 + b/4) with ties away
	// from zero, then + zpOut and the clamp.
	rq := AddRequant{MultA: 1 << 30, MultB: 1 << 29, Shift: 0}
	p := AddParams{ZPA: 1, ZPB: -2, ZPOut: 3, RQ: rq, Min: -128, Max: 127}
	a := []int8{1, 2, 0, 1, 1, 127, -128, 3}
	b := []int8{-2, -2, -2, 0, -4, 127, -128, 0}
	// (a−1)/2 + (b+2)/4: 0, .5→1, −.5→−1, .5→1, −.5→−1, 63+32.25→95,
	// −64.5−31.5 = −96, 1+.5→2
	want := []int8{3, 4, 2, 4, 2, 98, -93, 5}
	dst := make([]int8, len(a))
	require.NoError(t, Add(dst, a, b, p))
	assert.Equal(t, want, dst)

	p.Min, p.Max = 3, 50 // fused ReLU-style clamp
	require.NoError(t, Add(dst, a, b, p))
	assert.Equal(t, []int8{3, 4, 3, 4, 3, 50, 3, 5}, dst)

	// dst may alias either input
	c := append([]int8(nil), a...)
	p.Min, p.Max = -128, 127
	require.NoError(t, Add(c, c, b, p))
	assert.Equal(t, want, c)
	c = append([]int8(nil), b...)
	require.NoError(t, Add(c, a, c, p))
	assert.Equal(t, want, c)
}

func TestAddErrors(t *testing.T) {
	ok := AddParams{RQ: AddRequant{MultA: 1 << 30, MultB: 1 << 30, Shift: 0}, Min: -128, Max: 127}
	x := make([]int8, 4)
	assert.NoError(t, Add(nil, nil, nil, ok), "empty is allowed")
	assert.Equal(t, ErrLength, Add(x[:3], x, x, ok))
	assert.Equal(t, ErrLength, Add(x, x[:3], x, ok))
	assert.Equal(t, ErrLength, Add(x, x, x[:3], ok))
	badRQ := ok
	badRQ.RQ.MultB = 0
	assert.Equal(t, ErrRequant, Add(x, x, x, badRQ))
	badAct := ok
	badAct.Min, badAct.Max = 5, 4
	assert.Equal(t, ErrActivation, Add(x, x, x, badAct))
}

// addRequant draws a valid AddRequant, biased towards ties and extremes.
func (g gen) addRequant() AddRequant {
	var hi, lo int32
	var shift int32
	switch g.r.IntN(8) {
	case 0: // powers of two: exact .5 ties are frequent
		hi, lo = 1<<30, int32(1)<<(20+g.r.IntN(11))
		shift = int32(-g.r.IntN(9))
	case 1: // extremes
		hi, lo = math.MaxInt32, []int32{1, math.MaxInt32, 1 << 30}[g.r.IntN(3)]
		shift = []int32{MinShift, MaxShift, 0}[g.r.IntN(3)]
	case 2: // any shift, incl. the boundaries around a total shift of 16/17 and 41/42
		hi = 1<<30 + g.r.Int32N(1<<30)
		lo = 1 + g.r.Int32N(hi)
		shift = []int32{14, 15, 16, -10, -11, int32(MinShift + g.r.IntN(MaxShift-MinShift+1))}[g.r.IntN(6)]
	default:
		hi = 1<<30 + g.r.Int32N(1<<30)
		lo = 1 + g.r.Int32N(hi)
		shift = int32(-10 + g.r.IntN(12))
	}
	if g.r.IntN(2) == 0 {
		return AddRequant{MultA: hi, MultB: lo, Shift: shift}
	}
	return AddRequant{MultA: lo, MultB: hi, Shift: shift}
}

// refAdd is the float-then-quantise reference: with sOut = b·2^-eout the
// input scales sA = MultA·2^(Shift−31)·sOut and sB likewise are dyadic, so
// the real sum is exact.
func refAdd(t *testing.T, g gen, a, b []int8, p AddParams) []int8 {
	e := &exactF{}
	bb := int64(2*g.r.IntN(128) + 1)
	eout := g.r.IntN(13)
	sOut := new(big.Rat).SetFrac(big.NewInt(bb), new(big.Int).Lsh(big.NewInt(1), uint(eout)))
	sA := e.dyadic(int64(p.RQ.MultA)*bb, int(p.RQ.Shift)-31-eout)
	sB := e.dyadic(int64(p.RQ.MultB)*bb, int(p.RQ.Shift)-31-eout)
	out := make([]int8, len(a))
	for i := range a {
		x := e.mul(e.fromInt(int64(a[i])-int64(p.ZPA)), sA)
		y := e.mul(e.fromInt(int64(b[i])-int64(p.ZPB)), sB)
		out[i] = quantizeReal(e.chk(newF().Add(x, y)), sOut, p.ZPOut, p.Min, p.Max)
	}
	if e.inexact {
		t.Fatal("reference arithmetic was inexact")
	}
	return out
}

func TestAddReference(t *testing.T) {
	g := newGen(11)
	m := &mismatchLog{t: t}
	for c := range 3000 {
		n := 1 + g.r.IntN(64)
		p := AddParams{ZPA: g.zp(), ZPB: g.zp(), ZPOut: g.zp(), RQ: g.addRequant()}
		p.Min, p.Max = g.act(randScale(g), p.ZPOut)
		a, b := g.int8s(n), g.int8s(n)
		if p.RQ.Shift > 2 || g.r.IntN(8) == 0 {
			// large multipliers: inputs near the zero points, so that
			// outputs are not all clamped
			for i := range a {
				a[i] = nearZP(g, p.ZPA)
				b[i] = nearZP(g, p.ZPB)
			}
		}
		dst := make([]int8, n)
		if err := Add(dst, a, b, p); err != nil {
			t.Fatalf("case %d: %v", c, err)
		}
		m.compare(func() string { return fmt.Sprintf("case %d %+v", c, p) }, dst, refAdd(t, g, a, b, p))
	}
	m.done()
}

// nearZP returns zp + d for a small d, saturated to int8.
func nearZP(g gen, zp int8) int8 {
	return int8(min(max(int(zp)+g.r.IntN(7)-3, -128), 127))
}

func TestAddLayer(t *testing.T) {
	g := newGen(12)
	for c := range 500 {
		n := 1 + g.r.IntN(300)
		p := AddParams{ZPA: g.zp(), ZPB: g.zp(), ZPOut: g.zp(), RQ: g.addRequant()}
		p.Min, p.Max = g.act(randScale(g), p.ZPOut)
		a, b := g.int8s(n), g.int8s(n)
		want, got := make([]int8, n), make([]int8, n)
		require.NoError(t, Add(want, a, b, p))
		l := &AddLayer{Len: n, P: p}
		require.NoError(t, l.Prepare(), "case %d", c)
		assert.Equal(t, Shape{H: 1, W: 1, C: n}, l.InShape())
		assert.Equal(t, l.InShape(), l.OutShape())
		require.NoError(t, l.Run(got, a, b))
		require.Equal(t, want, got, "case %d %+v", c, p)
	}
}

func TestAddLayerErrors(t *testing.T) {
	ok := AddParams{RQ: AddRequant{MultA: 1 << 30, MultB: 1 << 30, Shift: 0}, Min: -128, Max: 127}
	x := make([]int8, 4)

	var l AddLayer
	assert.Equal(t, ErrNotPrepared, l.Run(x, x, x), "never prepared")
	assert.Equal(t, Shape{}, l.InShape())

	for _, c := range []struct {
		name string
		l    AddLayer
		want error
	}{
		{"len 0", AddLayer{Len: 0, P: ok}, ErrShape},
		{"len too big", AddLayer{Len: MaxElements + 1, P: ok}, ErrShape},
		{"bad requant", AddLayer{Len: 4, P: AddParams{RQ: AddRequant{MultA: 1, MultB: 1}, Max: 127}}, ErrRequant},
		{"bad clamp", AddLayer{Len: 4, P: AddParams{RQ: ok.RQ, Min: 1, Max: 0}}, ErrActivation},
	} {
		l := c.l
		assert.Equal(t, c.want, l.Prepare(), c.name)
		assert.Equal(t, ErrNotPrepared, l.Run(x, x, x), c.name)
	}

	l = AddLayer{Len: 4, P: ok}
	require.NoError(t, l.Prepare())
	assert.Equal(t, ErrLength, l.Run(x[:3], x, x))
	assert.Equal(t, ErrLength, l.Run(x, x[:3], x))
	assert.Equal(t, ErrLength, l.Run(x, x, x[:3]))
	l.P.ZPOut = 1
	assert.Equal(t, ErrNotPrepared, l.Run(x, x, x), "params changed after Prepare")
	require.NoError(t, l.Prepare())
	l.Len = 3
	assert.Equal(t, ErrNotPrepared, l.Run(x[:3], x[:3], x[:3]), "Len changed after Prepare")
}

func TestAddNoAllocs(t *testing.T) {
	p := AddParams{ZPA: 3, ZPB: -7, ZPOut: -128, RQ: AddRequant{MultA: 3 << 29, MultB: 1 << 30, Shift: -1}, Min: -128, Max: 127}
	a, b, dst := make([]int8, 512), make([]int8, 512), make([]int8, 512)
	l := &AddLayer{Len: 512, P: p}
	require.NoError(t, l.Prepare())
	assert.Zero(t, testing.AllocsPerRun(100, func() { _ = Add(dst, a, b, p) }), "Add")
	assert.Zero(t, testing.AllocsPerRun(100, func() { _ = Add(dst[:1], a, b, p) }), "Add error")
	assert.Zero(t, testing.AllocsPerRun(100, func() { _ = l.Prepare() }), "Prepare")
	assert.Zero(t, testing.AllocsPerRun(100, func() { _ = l.Run(dst, a, b) }), "Run")
	assert.Zero(t, testing.AllocsPerRun(100, func() { _ = l.Run(dst[:1], a, b) }), "Run error")
}
