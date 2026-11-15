package kpu

import (
	"encoding/binary"
	"testing"

	"github.com/itohio/EasyRobot/drivers/kpu/kmodel"
)

// TestEmulatedSharedWeights: two convs that share
// one weight blob but have their own BN and activation tables (legal in a
// kmodel; kmodel.Builder never writes it) each run with their own tables.
func TestEmulatedSharedWeights(t *testing.T) {
	const w, h, c = 8, 8, 4
	l := kmodel.UploadLayout(w, h)
	conv := func(src, dst uint32) kmodel.Args {
		a := kmodel.NewArgs()
		for _, err := range []error{
			a.SetSrcAddr(src), a.SetDstAddr(dst), a.SetInShape(w, h, c), a.SetOutShape(w, h, c),
			a.SetInLayout(l), a.SetOutLayout(l), a.SetKernel(1, false), a.SetPool(0),
			a.SetConvValue(0, 0, 0, 0), a.SetArgAdd(0), a.SetLoad(c*c, 1, c),
		} {
			if err != nil {
				t.Fatal(err)
			}
		}
		return a
	}
	weights, err := kmodel.EncodeWeights([]uint16{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}, 1, c, c, false, true)
	if err != nil {
		t.Fatal(err)
	}
	table := func(add int64) ([]byte, []byte) {
		bn := make([]kmodel.BN, c)
		for i := range bn {
			bn[i] = kmodel.BN{Mul: 1, Add: add}
		}
		bnb, err := kmodel.EncodeBN(bn)
		if err != nil {
			t.Fatal(err)
		}
		var act [16]kmodel.ActSegment
		act[0] = kmodel.ActSegment{StartX: -1 << 35}
		act[1] = kmodel.ActSegment{StartX: 0, Mul: 1}
		for i := 2; i < 16; i++ {
			act[i] = kmodel.ActSegment{StartX: 1<<35 - 16 + int64(i), Add: 255}
		}
		actb, err := kmodel.EncodeAct(act)
		if err != nil {
			t.Fatal(err)
		}
		return bnb, actb
	}
	bn0, act0 := table(0)  // identity
	bn1, act1 := table(50) // +50
	n := uint32(w * h * c)
	b := kmodel.NewBuilder(true, 0)
	b.SetMainMem(n)
	b.AddOutput(kmodel.Output{Size: n})
	b.AddConv(conv(1000, 0), weights, bn0, act0)
	b.AddConvToMain(0, conv(0, 1000), weights, bn1, act1)
	data, err := b.Build()
	if err != nil {
		t.Fatal(err)
	}
	m, err := kmodel.Parse(AlignedCopy(data))
	if err != nil {
		t.Fatal(err)
	}
	// Point the second conv's weights at the first conv's blob.
	c0, _ := m.Layers[0].Conv()
	binary.LittleEndian.PutUint32(m.Layers[1].Body[12:], c0.WeightsOffset)
	if c1, _ := m.Layers[1].Conv(); c1.WeightsOffset != c0.WeightsOffset || c1.BNOffset == c0.BNOffset {
		t.Fatalf("patch failed: %+v vs %+v", c1, c0)
	}
	r, err := NewEmulated(m, make([]byte, m.MainMemUsage))
	if err != nil {
		t.Fatal(err)
	}
	in := make([]byte, n)
	for i := range in {
		in[i] = byte(i % 100)
	}
	if err := r.Run(in); err != nil {
		t.Fatal(err)
	}
	out := r.Output(0)
	for i := range in {
		if want := in[i] + 50; out[i] != want { // identity, then +50
			t.Fatalf("output %d: %d, want %d (the second conv ran with the first conv's tables?)", i, out[i], want)
		}
	}
}
