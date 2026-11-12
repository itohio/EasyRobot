package kmodel

import (
	"bytes"
	"errors"
	"testing"

	"github.com/itohio/EasyRobot/drivers/kpu"
	"github.com/itohio/EasyRobot/drivers/kpu/compile"
	km "github.com/itohio/EasyRobot/drivers/kpu/kmodel"
	"github.com/itohio/EasyRobot/x/math/nn/qnet"
	"github.com/itohio/EasyRobot/x/math/primitive/qi8"
)

// mobileNet is qnet's MobileNetV1-0.25 (TF SAME stride 2, as the official
// TFLite model) at size×size with synthetic parameters, global pool
// included (layer index 27).
func mobileNet(t *testing.T, size int) *qnet.Graph {
	t.Helper()
	ls := make([]qnet.Layer, qnet.MobileNetV1Layers(0))
	g, err := qnet.MobileNetV1(0.25, size, size, 0, ls)
	if err != nil {
		t.Fatal(err)
	}
	for i := range g.Layers {
		l := &g.Layers[i]
		nw, nb, nr := l.ParamCounts()
		l.Weights, l.Bias, l.Requant = make([]int8, nw), make([]int32, nb), make([]qi8.Requant, nr)
	}
	if err := qnet.SyntheticParams(&g, 210); err != nil {
		t.Fatal(err)
	}
	return &g
}

// TestMarshalRoundTrip: a 32×32 MobileNet, cut before its global pool and
// resized to 64×64, marshals to a kmodel; the Unmarshaller reads it back,
// and drivers/kpu runs it (emulated) to qnet's CPU result for the same cut
// and size.
func TestMarshalRoundTrip(t *testing.T) {
	g := mobileNet(t, 32)
	m := NewMarshaller(WithCut(26))
	var buf bytes.Buffer
	if err := m.Marshal(&buf, g, WithInputSize(64, 64)); err != nil {
		t.Fatal(err)
	}
	if m.Info.Input != (qi8.Shape{H: 64, W: 64, C: 3}) || m.Info.Output != (qi8.Shape{H: 2, W: 2, C: 256}) || m.Info.KPULayers != 28 {
		t.Fatalf("info %+v", m.Info)
	}
	var model km.Model
	if err := NewUnmarshaller().Unmarshal(&buf, &model); err != nil {
		t.Fatal(err)
	}
	r, err := kpu.NewEmulated(&model, make([]byte, model.MainMemUsage))
	if err != nil {
		t.Fatal(err)
	}
	info := m.Info
	in := make([]int8, info.Input.Len())
	for i := range in {
		in[i] = int8(i*37 + i/97)
	}
	s := info.Input
	chw := make([]byte, s.Len())
	for y := range s.H {
		for x := range s.W {
			for c := range s.C {
				chw[(c*s.H+y)*s.W+x] = byte(in[(y*s.W+x)*s.C+c]) ^ 0x80
			}
		}
	}
	if err := r.Run(chw); err != nil {
		t.Fatal(err)
	}
	out := r.Output(0)

	ref, err := Prepare(g, map[string]string{keyCut: "26", keyInput: "64x64"})
	if err != nil {
		t.Fatal(err)
	}
	p, err := qnet.PlanMemory(ref)
	if err != nil {
		t.Fatal(err)
	}
	cr, err := qnet.NewRunner(ref, make([]int8, p.Size()))
	if err != nil {
		t.Fatal(err)
	}
	cpu, err := cr.Run(in)
	if err != nil {
		t.Fatal(err)
	}
	o := info.Output
	for y := range o.H {
		for x := range o.W {
			for c := range o.C {
				if got, want := int8(out[(c*o.H+y)*o.W+x]^0x80), cpu[(y*o.W+x)*o.C+c]; got != want {
					t.Fatalf("output (%d,%d,%d): kmodel %d, CPU %d", y, x, c, got, want)
				}
			}
		}
	}
}

func TestMarshalRefuses(t *testing.T) {
	g := mobileNet(t, 32) // ends in a global average pool, which the KPU cannot run
	err := NewMarshaller().Marshal(&bytes.Buffer{}, g)
	if !errors.Is(err, compile.ErrUnsupported) {
		t.Errorf("graph with a global pool: %v, want ErrUnsupported", err)
	}
	if err := NewMarshaller().Marshal(&bytes.Buffer{}, 42); err == nil {
		t.Error("an int marshalled")
	}
	if err := NewMarshaller(WithCut(26)).Marshal(&bytes.Buffer{}, g, WithInputSize(0, 5)); err == nil {
		t.Error("input size 0×5 accepted")
	}
	if err := NewMarshaller().Marshal(&bytes.Buffer{}, []byte("not a flatbuffer")); err == nil {
		t.Error("garbage TFLite accepted")
	}
	var m km.Model
	if err := NewUnmarshaller().Unmarshal(bytes.NewReader([]byte{1, 2, 3}), &m); err == nil {
		t.Error("garbage kmodel unmarshalled")
	}
}
