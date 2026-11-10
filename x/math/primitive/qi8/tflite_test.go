package qi8

import (
	"compress/gzip"
	"encoding/json"
	"math"
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// tfliteFixtures are TensorFlow Lite int8 reference-kernel results computed
// offline (see testdata/README.md for the generator and its sources).
type tfliteFixtures struct {
	Source  string `json:"source"`
	Requant struct {
		Acc  []int32 `json:"acc"`
		Rows []struct {
			Mult  int32    `json:"mult"`
			Shift int32    `json:"shift"`
			Out   []*int32 `json:"out"`
		} `json:"rows"`
	} `json:"requant"`
	Conv      []tfliteConvCase `json:"conv"`
	Depthwise []tfliteConvCase `json:"depthwise"`
	FC        []tfliteConvCase `json:"fc"`
	Add       []struct {
		N     int      `json:"n"`
		SA    float64  `json:"scaleA"`
		SB    float64  `json:"scaleB"`
		SO    float64  `json:"scaleOut"`
		ZA    int8     `json:"zpA"`
		ZB    int8     `json:"zpB"`
		ZO    int8     `json:"zpOut"`
		AMin  int8     `json:"actMin"`
		AMax  int8     `json:"actMax"`
		Mult  [3]int32 `json:"mult"`
		Shift [3]int32 `json:"shift"`
		AI    []int8   `json:"a"`
		BI    []int8   `json:"b"`
		OutI  []int8   `json:"out"`
	} `json:"add"`
	AvgPool []tfliteConvCase `json:"avgpool"`
	Act     []struct {
		Scale float64 `json:"scale"`
		ZP    int8    `json:"zp"`
		ReLU6 [2]int8 `json:"relu6"`
	} `json:"act"`
}

type tfliteConvCase struct {
	In       json.RawMessage `json:"in"` // [H, W, C]
	FCIn     int             `json:"inLen"`
	FCOut    int             `json:"outLen"`
	OutC     int             `json:"outC"`
	KH       int             `json:"kh"`
	KW       int             `json:"kw"`
	SH       int             `json:"sh"`
	SW       int             `json:"sw"`
	Pad      string          `json:"pad"`
	InZP     int8            `json:"inZP"`
	OutZP    int8            `json:"outZP"`
	InScale  float64         `json:"inScale"`
	OutScale float64         `json:"outScale"`
	WScale   []float64       `json:"wScale"`
	ActMin   int8            `json:"actMin"`
	ActMax   int8            `json:"actMax"`
	Mult     []int32         `json:"mult"`
	Shift    []int32         `json:"shift"`
	Src      []int8          `json:"src"`
	W        []int8          `json:"w"`
	Bias     []int32         `json:"bias"`
	Out      []int8          `json:"out"`
}

func (c *tfliteConvCase) shape(t *testing.T) Shape {
	var s [3]int
	require.NoError(t, json.Unmarshal(c.In, &s))
	return Shape{H: s[0], W: s[1], C: s[2]}
}

func (c *tfliteConvCase) padding() Padding {
	if c.Pad == "valid" {
		return PaddingValid
	}
	return PaddingSame
}

func (c *tfliteConvCase) requants() []Requant {
	rq := make([]Requant, len(c.Mult))
	for i := range rq {
		rq[i] = Requant{Mult: c.Mult[i], Shift: c.Shift[i]}
	}
	return rq
}

// checkMultipliers asserts that QuantizeMultiplier(inScale·wScale/outScale)
// in float64 from the float32 scales reproduces TFLite's per-channel
// multipliers (when the case was built from scales).
func (c *tfliteConvCase) checkMultipliers(t *testing.T, name string, i int) {
	if c.WScale == nil {
		return
	}
	for o, ws := range c.WScale {
		real := float64(float32(c.InScale)) * float64(float32(ws)) / float64(float32(c.OutScale))
		r, err := QuantizeMultiplier(real)
		require.NoError(t, err, "%s %d", name, i)
		require.Equal(t, Requant{Mult: c.Mult[o], Shift: c.Shift[o]}, r, "%s %d channel %d", name, i, o)
	}
}

func loadTFLiteFixtures(t *testing.T) *tfliteFixtures {
	t.Helper()
	f, err := os.Open("testdata/tflite_ref.json.gz")
	require.NoError(t, err)
	defer f.Close()
	z, err := gzip.NewReader(f)
	require.NoError(t, err)
	var fx tfliteFixtures
	require.NoError(t, json.NewDecoder(z).Decode(&fx))
	return &fx
}

func TestRequantApplyTFLiteHand(t *testing.T) {
	half := Requant{Mult: 1 << 30, Shift: 0} // 0.5
	// SaturatingRoundingDoublingHighMul rounds ties up (toward +inf); the
	// exact mode rounds them away from zero.
	assert.Equal(t, int32(1), half.ApplyTFLite(1))
	assert.Equal(t, int32(0), half.ApplyTFLite(-1))
	assert.Equal(t, int64(-1), half.Apply(-1))
	assert.Equal(t, int32(-1), half.ApplyTFLite(-2))
	// RoundingDivideByPOT rounds ties away from zero: 0.25·(−6) = −1.5 → −2
	// (SRDHM gives −3 exactly, then −3/2 → −2).
	q := Requant{Mult: 1 << 30, Shift: -1}
	assert.Equal(t, int32(-2), q.ApplyTFLite(-6))
	assert.Equal(t, int32(2), q.ApplyTFLite(6))
	// Double rounding: 0.25·(−3) = −0.75. SRDHM(−3, 0.5) = −1.5 → −1, then
	// −1/2 = −0.5 → −1 (away). Exact: −0.75 → −1. And 0.25·(−2) = −0.5:
	// SRDHM → −1 (exact), −1/2 → −1; exact → −1.
	assert.Equal(t, int32(-1), q.ApplyTFLite(-3))
	assert.Equal(t, int32(-1), q.ApplyTFLite(-2))
	// 0.25·5 = 1.25: SRDHM(5, 0.5) = 2.5 → 3, 3/2 = 1.5 → 2; exact gives 1.
	assert.Equal(t, int32(2), q.ApplyTFLite(5))
	assert.Equal(t, int64(1), q.Apply(5))
	// Positive shift: x·2^shift first (int32), then the high multiply.
	assert.Equal(t, int32(3), Requant{Mult: 3 << 29, Shift: 1}.ApplyTFLite(2)) // 1.5·2
	// Largest right shift.
	assert.Equal(t, int32(-1), Requant{Mult: math.MaxInt32, Shift: MinShift}.ApplyTFLite(math.MinInt32))
	assert.Equal(t, int32(1), Requant{Mult: math.MaxInt32, Shift: MinShift}.ApplyTFLite(math.MaxInt32)) // ≈ 0.99999…
}

func TestTFLiteRequantFixtures(t *testing.T) {
	fx := loadTFLiteFixtures(t)
	n := 0
	for _, row := range fx.Requant.Rows {
		r := Requant{Mult: row.Mult, Shift: row.Shift}
		require.True(t, r.Valid())
		for i, want := range row.Out {
			if want == nil {
				continue
			}
			require.Equal(t, *want, r.ApplyTFLite(fx.Requant.Acc[i]), "rq %+v acc %d", r, fx.Requant.Acc[i])
			n++
		}
	}
	t.Logf("%s: %d requant outputs", fx.Source, n)
}

// TestTFLiteOpFixtures runs every fixture case through the free function and
// the prepared layer in the rounding mode TFLite uses for the op —
// RoundTFLite for CONV_2D and DEPTHWISE_CONV_2D, RoundExact for
// FULLY_CONNECTED, whose TFLite kernel rounds once, half away from zero —
// and counts the outputs the other mode would get different (it must be > 0,
// or the fixtures do not tell the modes apart).
func TestTFLiteOpFixtures(t *testing.T) {
	fx := loadTFLiteFixtures(t)
	type op struct {
		name        string
		cases       []tfliteConvCase
		mode, other Rounding
	}
	for _, o := range []op{
		{"conv", fx.Conv, RoundTFLite, RoundExact},
		{"depthwise", fx.Depthwise, RoundTFLite, RoundExact},
		{"fc", fx.FC, RoundExact, RoundTFLite},
	} {
		t.Run(o.name, func(t *testing.T) {
			outputs, differ := 0, 0
			for i := range o.cases {
				c := &o.cases[i]
				c.checkMultipliers(t, o.name, i)
				rq := c.requants()
				got, prepared, alt := make([]int8, len(c.Out)), make([]int8, len(c.Out)), make([]int8, len(c.Out))
				var free func(dst []int8, round Rounding) error
				var layer interface {
					Prepare() error
					Run(dst, src []int8) error
				}
				switch o.name {
				case "conv":
					p := Conv2DParams{In: c.shape(t), OutC: c.OutC, KH: c.KH, KW: c.KW, StrideH: c.SH, StrideW: c.SW,
						Padding: c.padding(), InZP: c.InZP, OutZP: c.OutZP, ActMin: c.ActMin, ActMax: c.ActMax, Round: o.mode}
					free = func(dst []int8, r Rounding) error { q := p; q.Round = r; return Conv2D(dst, c.Src, c.W, c.Bias, rq, q) }
					layer = withScratch(&Conv2DLayer{P: p, W: c.W, Bias: c.Bias, RQ: rq}, setConvScratch)
				case "depthwise":
					p := DepthwiseConv2DParams{In: c.shape(t), KH: c.KH, KW: c.KW, StrideH: c.SH, StrideW: c.SW,
						Padding: c.padding(), InZP: c.InZP, OutZP: c.OutZP, ActMin: c.ActMin, ActMax: c.ActMax, Round: o.mode}
					free = func(dst []int8, r Rounding) error {
						q := p
						q.Round = r
						return DepthwiseConv2D(dst, c.Src, c.W, c.Bias, rq, q)
					}
					layer = withScratch(&DepthwiseConv2DLayer{P: p, W: c.W, Bias: c.Bias, RQ: rq}, setDWScratch)
				default:
					p := FCParams{In: c.FCIn, Out: c.FCOut, InZP: c.InZP, OutZP: c.OutZP, ActMin: c.ActMin, ActMax: c.ActMax, Round: o.mode}
					free = func(dst []int8, r Rounding) error {
						q := p
						q.Round = r
						return FullyConnected(dst, c.Src, c.W, c.Bias, rq, q)
					}
					layer = withScratch(&FCLayer{P: p, W: c.W, Bias: c.Bias, RQ: rq}, setFCScratch)
				}
				require.NoError(t, free(got, o.mode), "%s %d", o.name, i)
				require.Equal(t, c.Out, got, "%s %d", o.name, i)
				require.NoError(t, layer.Prepare(), "%s %d", o.name, i)
				require.NoError(t, layer.Run(prepared, c.Src), "%s %d", o.name, i)
				require.Equal(t, c.Out, prepared, "%s %d prepared", o.name, i)
				require.NoError(t, free(alt, o.other))
				for j := range alt {
					if alt[j] != c.Out[j] {
						differ++
					}
				}
				outputs += len(c.Out)
			}
			t.Logf("%d cases, %d outputs, the other mode differs on %d", len(o.cases), outputs, differ)
			assert.Positive(t, differ, "fixtures must distinguish the rounding modes")
		})
	}
}

func TestTFLiteAddFixtures(t *testing.T) {
	fx := loadTFLiteFixtures(t)
	outputs, differ := 0, 0
	for i, c := range fx.Add {
		rq, err := QuantizeAddTFLite(float32(c.SA), float32(c.SB), float32(c.SO))
		require.NoError(t, err, "add %d", i)
		want := AddRequantTFLite{
			A:   Requant{Mult: c.Mult[0], Shift: c.Shift[0]},
			B:   Requant{Mult: c.Mult[1], Shift: c.Shift[1]},
			Out: Requant{Mult: c.Mult[2], Shift: c.Shift[2]},
		}
		require.Equal(t, want, rq, "add %d", i)
		p := AddParams{ZPA: c.ZA, ZPB: c.ZB, ZPOut: c.ZO, Min: c.AMin, Max: c.AMax, Round: RoundTFLite, TFLite: rq}
		got := make([]int8, c.N)
		require.NoError(t, Add(got, c.AI, c.BI, p), "add %d", i)
		require.Equal(t, c.OutI, got, "add %d %+v", i, p)
		l := &AddLayer{Len: c.N, P: p}
		require.NoError(t, l.Prepare())
		clear(got)
		require.NoError(t, l.Run(got, c.AI, c.BI))
		require.Equal(t, c.OutI, got, "add %d prepared", i)

		// For information: the exact mode with the same real multipliers.
		// Both modes round ties away from zero at the end and TFLite's
		// intermediates carry 20 extra bits, so they rarely differ; p.RQ is
		// zero above, so a TFLite case taking the exact path would fail.
		so := float64(float32(c.SO))
		ex, err := QuantizeAddMultipliers(float64(float32(c.SA))/so, float64(float32(c.SB))/so)
		if err == nil {
			e := AddParams{ZPA: c.ZA, ZPB: c.ZB, ZPOut: c.ZO, Min: c.AMin, Max: c.AMax, RQ: ex}
			require.NoError(t, Add(got, c.AI, c.BI, e))
			for j := range got {
				if got[j] != c.OutI[j] {
					differ++
				}
			}
		}
		outputs += c.N
	}
	t.Logf("%d cases, %d outputs, exact mode differs on %d", len(fx.Add), outputs, differ)
}

// TFLite's average pooling already rounds like qi8's (one mode for both).
func TestTFLiteAvgPoolFixtures(t *testing.T) {
	fx := loadTFLiteFixtures(t)
	for i := range fx.AvgPool {
		c := &fx.AvgPool[i]
		p := Pool2DParams{In: c.shape(t), KH: c.KH, KW: c.KW, StrideH: c.SH, StrideW: c.SW, Padding: c.padding()}
		got := make([]int8, len(c.Out))
		require.NoError(t, AvgPool2D(got, c.Src, p), "avgpool %d", i)
		require.Equal(t, c.Out, got, "avgpool %d %+v", i, p)
	}
}

func TestTFLiteActivationFixtures(t *testing.T) {
	fx := loadTFLiteFixtures(t)
	differ := 0
	for _, c := range fx.Act {
		lo, hi := ActReLU6TFLite(float32(c.Scale), c.ZP)
		require.Equal(t, c.ReLU6, [2]int8{lo, hi}, "scale %v zp %d", c.Scale, c.ZP)
		if _, h := ActReLU6(float32(c.Scale), c.ZP); h != hi {
			differ++
		}
	}
	t.Logf("%d cases; float64 Quantize6 differs on %d", len(fx.Act), differ)
}

func TestRoundingValidation(t *testing.T) {
	rq := []Requant{{Mult: 1 << 30}}
	x := make([]int8, 1)
	bad := Rounding(2)
	assert.Equal(t, ErrRequant, FullyConnected(x, x, x, nil, rq, FCParams{In: 1, Out: 1, ActMax: 127, Round: bad}))
	assert.Equal(t, ErrRequant, Conv2D(x, x, x, nil, rq, Conv2DParams{In: Shape{1, 1, 1}, OutC: 1, KH: 1, KW: 1, StrideH: 1, StrideW: 1, ActMax: 127, Round: bad}))
	assert.Equal(t, ErrRequant, DepthwiseConv2D(x, x, x, nil, rq, DepthwiseConv2DParams{In: Shape{1, 1, 1}, KH: 1, KW: 1, StrideH: 1, StrideW: 1, ActMax: 127, Round: bad}))
	l := &FCLayer{P: FCParams{In: 1, Out: 1, ActMax: 127, Round: bad}, W: x, RQ: rq}
	assert.Equal(t, ErrRequant, l.Prepare())

	okT := AddRequantTFLite{A: Requant{Mult: 1 << 30}, B: Requant{Mult: 1 << 30}, Out: Requant{Mult: 1 << 30, Shift: -10}}
	assert.NoError(t, Add(x, x, x, AddParams{Max: 127, Round: RoundTFLite, TFLite: okT}))
	assert.Equal(t, ErrRequant, Add(x, x, x, AddParams{Max: 127, Round: bad, RQ: AddRequant{MultA: 1 << 30, MultB: 1 << 30}}))
	for _, mut := range []func(*AddRequantTFLite){
		func(r *AddRequantTFLite) { r.A.Shift = 1 }, // multipliers must be < 1 (shift ≤ 0)
		func(r *AddRequantTFLite) { r.B.Mult = 1 },
		func(r *AddRequantTFLite) { r.Out.Shift = 1 },
	} {
		r := okT
		mut(&r)
		assert.Equal(t, ErrRequant, Add(x, x, x, AddParams{Max: 127, Round: RoundTFLite, TFLite: r}), "%+v", r)
	}
	_, err := QuantizeAddTFLite(0, 1, 1)
	assert.Equal(t, ErrRequant, err)
	_, err = QuantizeAddTFLite(1, 1, float32(math.Inf(1)))
	assert.Equal(t, ErrRequant, err)
	_, err = QuantizeAddTFLite(1, 1, 1e-9) // output multiplier ≥ 1
	assert.Equal(t, ErrRequant, err)
}

func (g gen) rounding() Rounding {
	if g.r.IntN(2) == 0 {
		return RoundTFLite
	}
	return RoundExact
}
