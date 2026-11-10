package kpu

import (
	"github.com/itohio/EasyRobot/drivers/kpu/kmodel"
)

// emuBase is the model address an emulated Runner patches into the layer
// arguments (any 128-byte-aligned value below 4 GiB); the engine subtracts
// it again to find the weight, batch-norm and activation tables.
const emuBase = 0x8000_0000

// emulated is an engine that runs each conv with kmodel's integer model of
// the KPU (ConvParams.Emulate) on a 2 MiB slice standing in for KPU memory.
type emulated struct {
	m   *kmodel.Model
	mem []byte
}

func (e *emulated) ram() []byte { return e.mem }
func (e *emulated) now() uint64 { return 0 }

func (e *emulated) conv(a *kmodel.Args) error {
	p, err := e.m.ParamsAt(*a, uint32(a[5]>>32)-emuBase, uint32(a[4]>>32)-emuBase, uint32(a[7]>>32)-emuBase)
	if err != nil {
		return err
	}
	w, h, c := a.InShape()
	in := make([]byte, w*h*c)
	if err := a.InLayout().DownloadFrom(e.mem, a.SrcAddr(), w, h, c, in); err != nil {
		return err
	}
	w, h, c = a.OutShape()
	out := make([]byte, w*h*c)
	if err := p.Emulate(in, out); err != nil {
		return err
	}
	return a.OutLayout().UploadTo(e.mem, a.DstAddr(), w, h, c, out)
}

// NewEmulated prepares m to run on any host without a KPU: each conv is
// computed by kmodel's integer model of the KPU, which is bit-identical to
// nncase v0.1.0-rc5's simulator and within about 1 LSB per layer of the
// silicon (ER-K210-K8 evidence). The CPU layers are the same code as on the
// K210. It allocates per conv; it is for checking models and compilers on
// the host, not for speed. Times reports zeros.
func NewEmulated(m *kmodel.Model, main []byte) (*Runner, error) {
	return newRunner(m, main, emuBase, &emulated{m: m, mem: make([]byte, kmodel.RAMSize)})
}
