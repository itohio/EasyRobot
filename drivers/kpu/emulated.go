package kpu

import (
	"github.com/itohio/EasyRobot/drivers/kpu/kmodel"
)

// emuBase is the model address an emulated Runner patches into the layer
// arguments (any 128-byte-aligned value below 4 GiB); the engine subtracts
// it again to find the weight, batch-norm and activation tables.
const emuBase = 0x8000_0000

// emulated is an engine that runs each conv with kmodel's integer model of
// the KPU (ConvParams.EmulateInto) on a 2 MiB slice standing in for KPU
// memory. Each conv's parameters are decoded on its first run and its
// buffers grow once, so later runs allocate nothing.
type emulated struct {
	m            *kmodel.Model
	mem          []byte
	params       map[tables]*kmodel.ConvParams // by table offsets
	in, out, tmp []byte
}

// tables identifies a conv's parameters: its weight, BN and activation
// table offsets (convs may share a weight blob and differ in the others).
type tables struct{ weights, bn, act uint32 }

func (e *emulated) ram() []byte { return e.mem }
func (e *emulated) now() uint64 { return 0 }

func grow(b []byte, n int) []byte {
	if cap(b) < n {
		return make([]byte, n)
	}
	return b[:n]
}

func (e *emulated) conv(a *kmodel.Args) error {
	key := tables{uint32(a[5]>>32) - emuBase, uint32(a[4]>>32) - emuBase, uint32(a[7]>>32) - emuBase}
	p := e.params[key]
	if p == nil {
		q, err := e.m.ParamsAt(*a, key.weights, key.bn, key.act)
		if err != nil {
			return err
		}
		p = &q
		e.params[key] = p
	}
	p.Args = *a // the addresses and flags of this run
	w, h, c := a.InShape()
	e.in = grow(e.in, w*h*c)
	if err := a.InLayout().DownloadFrom(e.mem, a.SrcAddr(), w, h, c, e.in); err != nil {
		return err
	}
	_, _, oc := a.OutShape()
	e.tmp = grow(e.tmp, w*h*oc)
	ow, oh, _ := a.OutShape()
	e.out = grow(e.out, ow*oh*oc)
	if err := p.EmulateInto(e.in, e.out, e.tmp); err != nil {
		return err
	}
	return a.OutLayout().UploadTo(e.mem, a.DstAddr(), ow, oh, oc, e.out)
}

// NewEmulated prepares m to run on any host without a KPU: each conv is
// computed by kmodel's integer model of the KPU, which is bit-identical to
// nncase v0.1.0-rc5's simulator and within about 1 LSB per layer of the
// silicon (observed on the K210). The CPU layers are the same code as on the
// K210. The first Run decodes each conv's parameters; later runs allocate
// nothing. It is for checking models and compilers on the host, not for
// speed. Times reports zeros.
func NewEmulated(m *kmodel.Model, main []byte) (*Runner, error) {
	return newRunner(m, main, emuBase, &emulated{m: m, mem: make([]byte, kmodel.RAMSize), params: map[tables]*kmodel.ConvParams{}})
}
