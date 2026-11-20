package kgraph

import (
	"fmt"

	"github.com/itohio/EasyRobot/drivers/kpu"
	"github.com/itohio/EasyRobot/drivers/kpu/kmodel"
	"github.com/itohio/EasyRobot/x/math/nn/qnet"
	"github.com/itohio/EasyRobot/x/math/primitive/qi8"
)

// RunnerFunc builds the kpu.Runner of a segment: kpu.New on the K210,
// kpu.NewEmulated on any host.
type RunnerFunc func(m *kmodel.Model, main []byte) (*kpu.Runner, error)

// Backend runs a File's KPU segments; it implements qnet.Backend. Pass it
// with &File.Graph to qnet.NewRunner: Claim recognises the segments by
// their layers' addresses. All segments share one upload buffer (the
// largest segment input) and one main-memory output buffer (the largest
// output): a segment's output is copied into the runner's arena before the
// next segment runs.
type Backend struct {
	segs      map[*qnet.Layer]*segment
	chw, main []byte
}

type segment struct {
	count   int
	in, out qi8.Shape
	r       *kpu.Runner
}

var _ qnet.Backend = (*Backend)(nil)

// NewBackend parses every segment's kmodel in place and builds its runner
// with newRunner. A kmodel whose input or output size is not its segment's
// is refused (ErrFormat). It allocates the two shared buffers and the
// runners; the kmodels stay where the file is.
func NewBackend(f *File, newRunner RunnerFunc) (*Backend, error) {
	b := &Backend{segs: make(map[*qnet.Layer]*segment, len(f.Segments))}
	models := make([]*kmodel.Model, len(f.Segments))
	var maxIn, maxMain int
	for k, s := range f.Segments {
		m, err := kmodel.Parse(s.Kmodel)
		if err != nil {
			return nil, fmt.Errorf("kgraph: segment %d: %w", k, err)
		}
		in, out := f.Graph.Layers[s.First].In, f.Graph.Layers[s.First+s.Count-1].Out
		if len(m.Outputs) != 1 || int(m.Outputs[0].Size) != out.Len() {
			return nil, fmt.Errorf("%w: segment %d: the kmodel's output is not the %d bytes of layer %d", ErrFormat, k, out.Len(), s.First+s.Count-1)
		}
		models[k] = m
		maxIn, maxMain = max(maxIn, in.Len()), max(maxMain, int(m.MainMemUsage))
	}
	b.chw, b.main = make([]byte, maxIn), make([]byte, maxMain)
	for k, s := range f.Segments {
		r, err := newRunner(models[k], b.main)
		if err != nil {
			return nil, fmt.Errorf("kgraph: segment %d: %w", k, err)
		}
		in, out := f.Graph.Layers[s.First].In, f.Graph.Layers[s.First+s.Count-1].Out
		if r.InputSize() != in.Len() {
			return nil, fmt.Errorf("%w: segment %d: the kmodel reads %d bytes, layer %d has %d", ErrFormat, k, r.InputSize(), s.First, in.Len())
		}
		b.segs[&f.Graph.Layers[s.First]] = &segment{count: s.Count, in: in, out: out, r: r}
	}
	return b, nil
}

// Name identifies the backend in plans.
func (b *Backend) Name() string { return "k210-kpu" }

// BufferBytes returns the sizes of the shared upload and output buffers.
func (b *Backend) BufferBytes() (upload, output int) { return len(b.chw), len(b.main) }

// Claim returns the length of the segment starting at layers[0], or 0.
func (b *Backend) Claim(layers []qnet.Layer) int {
	if len(layers) == 0 {
		return 0
	}
	if s := b.segs[&layers[0]]; s != nil && len(layers) >= s.count {
		return s.count
	}
	return 0
}

// Run executes the segment: src (int8 NHWC) is converted to uint8 CHW in
// the shared upload buffer, the KPU runs the segment's kmodel, and its
// output is converted from the shared output buffer into dst (int8 NHWC).
// No allocation.
func (b *Backend) Run(layers []qnet.Layer, dst, src []int8) error {
	if len(layers) == 0 {
		return qnet.ErrSegment
	}
	s := b.segs[&layers[0]]
	if s == nil || len(layers) != s.count || len(src) != s.in.Len() || len(dst) != s.out.Len() {
		return qnet.ErrSegment
	}
	chw := b.chw[:len(src)]
	ToCHW(chw, src, s.in.H*s.in.W, s.in.C)
	if err := s.r.Run(chw); err != nil {
		return err
	}
	FromCHW(dst, s.r.Output(0), s.out.H*s.out.W, s.out.C)
	return nil
}

// ToCHW converts int8 NHWC (hw pixels × c channels) to the KPU's uint8 CHW
// (q + 128), one channel plane at a time: the plane is a hoisted slice, so
// the inner loop is a strided load and a sequential store with no index
// arithmetic beyond the stride.
func ToCHW(chw []byte, src []int8, hw, c int) {
	for ch := range c {
		d := chw[ch*hw : (ch+1)*hw]
		s := src[ch:]
		for i := range d {
			d[i] = byte(s[i*c]) ^ 0x80
		}
	}
}

// FromCHW converts the KPU's uint8 CHW back to int8 NHWC (− 128).
func FromCHW(dst []int8, chw []byte, hw, c int) {
	for ch := range c {
		p := chw[ch*hw : (ch+1)*hw]
		d := dst[ch:]
		for i, v := range p {
			d[i*c] = int8(v ^ 0x80)
		}
	}
}
