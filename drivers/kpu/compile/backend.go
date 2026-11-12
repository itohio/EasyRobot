package compile

import (
	"errors"
	"fmt"

	"github.com/itohio/EasyRobot/drivers/kpu"
	"github.com/itohio/EasyRobot/drivers/kpu/kmodel"
	"github.com/itohio/EasyRobot/x/math/nn/qnet"
	"github.com/itohio/EasyRobot/x/math/primitive/qi8"
)

// RunnerFunc builds the kpu.Runner for a compiled segment: kpu.New on the
// K210, kpu.NewEmulated on any host.
type RunnerFunc func(m *kmodel.Model, main []byte) (*kpu.Runner, error)

// Group is one KPU layer of a segment: graph layers [First, First+Count)
// ([Pad] conv [ReLU/ReLU6] [pool]) lowered together. Full marks a stride-2
// depthwise lowered at full resolution, Subsample the 1×1 conv after it
// that samples (see Fuse).
type Group struct {
	First, Count    int
	Full, Subsample bool
}

// Segment is a run of graph layers the KPU executes as one kmodel.
type Segment struct {
	First, Count int
	Groups       []Group
	Model        *kmodel.Model
	Runner       *kpu.Runner
	in, out      qi8.Shape
	chw          []byte // the input, converted to uint8 CHW
}

// Backend runs a graph's KPU segments; it implements qnet.Backend. Build it
// with Compile for one graph and pass the same *qnet.Graph to
// qnet.NewRunner: Claim recognises the segments by their layers' addresses.
// The graph must not change afterwards.
type Backend struct {
	segs  map[*qnet.Layer]*Segment
	order []*Segment
}

var _ qnet.Backend = (*Backend)(nil)

// Name identifies the backend in plans.
func (b *Backend) Name() string { return "k210-kpu" }

// Segments lists the compiled segments in graph order.
func (b *Backend) Segments() []*Segment { return b.order }

// Claim returns the length of the segment starting at layers[0], or 0.
func (b *Backend) Claim(layers []qnet.Layer) int {
	if len(layers) == 0 {
		return 0
	}
	if s := b.segs[&layers[0]]; s != nil && len(layers) >= s.Count {
		return s.Count
	}
	return 0
}

// ErrSegment is returned by Run for layers that are not a compiled segment.
var ErrSegment = errors.New("compile: not a compiled KPU segment")

// Run executes the segment: src (int8 NHWC, the first layer's input) is
// converted to uint8 CHW, the KPU runs the segment's kmodel, and its uint8
// CHW output is converted into dst (int8 NHWC). No allocation.
func (b *Backend) Run(layers []qnet.Layer, dst, src []int8) error {
	if len(layers) == 0 {
		return ErrSegment
	}
	s := b.segs[&layers[0]]
	if s == nil || len(layers) != s.Count || len(src) != s.in.Len() || len(dst) != s.out.Len() {
		return ErrSegment
	}
	h, w, c := s.in.H, s.in.W, s.in.C
	for y := range h {
		for x := range w {
			row := src[(y*w+x)*c:]
			for ch := range c {
				s.chw[(ch*h+y)*w+x] = byte(row[ch]) ^ 0x80 // q + 128
			}
		}
	}
	if err := s.Runner.Run(s.chw); err != nil {
		return err
	}
	out := s.Runner.Output(0)
	h, w, c = s.out.H, s.out.W, s.out.C
	for y := range h {
		for x := range w {
			row := dst[(y*w+x)*c:]
			for ch := range c {
				row[ch] = int8(out[(ch*h+y)*w+x] ^ 0x80) // − 128
			}
		}
	}
	return nil
}

// graphInfo is the reader structure Compile needs.
type graphInfo struct {
	g       *qnet.Graph
	readers []int // how many layers read each layer's output
	out     []bool
}

func producer(g *qnet.Graph, i int) int {
	if f := g.Layers[i].From; f != nil {
		if len(f) != 1 {
			return -2 // not single-input
		}
		return f[0]
	}
	return i - 1
}

func newGraphInfo(g *qnet.Graph) graphInfo {
	n := len(g.Layers)
	gi := graphInfo{g: g, readers: make([]int, n), out: make([]bool, n)}
	for i, l := range g.Layers {
		from := l.From
		if from == nil && i > 0 {
			from = []int{i - 1}
		}
		for _, p := range from {
			if p >= 0 && p < n {
				gi.readers[p]++
			}
		}
	}
	if g.Outputs == nil {
		gi.out[n-1] = true
	}
	for _, o := range g.Outputs {
		if o >= 0 && o < n {
			gi.out[o] = true
		}
	}
	return gi
}

// follows reports whether layer j reads only layer j−1, and j−1 has no
// other reader and is not a graph output: j−1's output can stay inside a
// segment.
func (gi graphInfo) follows(j int) bool {
	return j > 0 && j < len(gi.g.Layers) && producer(gi.g, j) == j-1 && gi.readers[j-1] == 1 && !gi.out[j-1]
}

// group finds the KPU group starting at layer i: [Pad] conv [act] [pool],
// with the longest fusion Lower accepts. ok is false if layer i does not
// start one.
func (gi graphInfo) group(i int, full, subsample bool) (Group, Fuse, *qnet.Layer, bool) {
	ls := gi.g.Layers
	j := i
	f := Fuse{Full: full, Subsample: subsample}
	if ls[j].Kind == qnet.Pad {
		if !gi.follows(j + 1) {
			return Group{}, Fuse{}, nil, false
		}
		f.Pad = &ls[j]
		j++
	}
	if j >= len(ls) || (ls[j].Kind != qnet.Conv2D && ls[j].Kind != qnet.DepthwiseConv2D) {
		return Group{}, Fuse{}, nil, false
	}
	conv := &ls[j]
	j++
	var act, pool *qnet.Layer
	if gi.follows(j) && (ls[j].Kind == qnet.ReLU || ls[j].Kind == qnet.ReLU6) {
		act = &ls[j]
		j++
	}
	if gi.follows(j) && (ls[j].Kind == qnet.MaxPool2D || ls[j].Kind == qnet.AvgPool2D) {
		pool = &ls[j]
	}
	// The longest fusion Lower accepts: conv + act + pool, conv + act, conv.
	base := j - i // [Pad] conv [act]
	if act != nil {
		base--
	}
	cands := []struct {
		f Fuse
		n int
	}{
		{Fuse{Pad: f.Pad, Act: act, Pool: pool, Full: full, Subsample: subsample}, base + b2i(act != nil) + b2i(pool != nil)},
		{Fuse{Pad: f.Pad, Act: act, Full: full, Subsample: subsample}, base + b2i(act != nil)},
		{Fuse{Pad: f.Pad, Full: full, Subsample: subsample}, base},
	}
	for _, c := range cands {
		if _, err := Lower(conv, c.f, 0, 0); err == nil {
			return Group{First: i, Count: c.n, Full: full, Subsample: subsample}, c.f, conv, true
		}
	}
	return Group{}, Fuse{}, nil, false
}

// lines is how many 64-byte KPU memory lines a tensor takes in the
// standard layout.
func lines(s qi8.Shape) int {
	l := kmodel.UploadLayout(s.W, s.H)
	return l.ChanLines * ((s.C + l.Group - 1) / l.Group)
}

const ramLines = kmodel.RAMSize / 64

// Compile finds the graph's KPU segments, compiles each into a kmodel and
// builds its runner with newRunner. Layers outside segments stay on qnet's
// CPU kernels. g must be validated by qnet (shapes, parameters).
func Compile(g *qnet.Graph, newRunner RunnerFunc) (*Backend, error) {
	gi := newGraphInfo(g)
	b := &Backend{segs: map[*qnet.Layer]*Segment{}}
	for i := 0; i < len(g.Layers); {
		s, err := gi.segment(i)
		if err != nil {
			return nil, err
		}
		if s == nil {
			i++
			continue
		}
		if err := s.build(g, newRunner); err != nil {
			return nil, fmt.Errorf("compile: segment at layer %d: %w", i, err)
		}
		b.segs[&g.Layers[s.First]] = s
		b.order = append(b.order, s)
		i = s.First + s.Count
	}
	return b, nil
}

// segment collects the longest chain of groups from layer i that fits KPU
// memory layer by layer; nil if layer i starts no group. A stride-2
// depthwise joins only together with the 1×1 conv after it (Full +
// Subsample).
func (gi graphInfo) segment(i int) (*Segment, error) {
	var s *Segment
	j := i
	push := func(gr Group, conv *qnet.Layer, f Fuse) {
		if s == nil {
			in := conv.In
			if f.Pad != nil {
				in = f.Pad.In
			}
			s = &Segment{First: i, in: in}
		}
		s.Groups = append(s.Groups, gr)
		s.out = gi.g.Layers[gr.First+gr.Count-1].Out
		j = gr.First + gr.Count
	}
	for j < len(gi.g.Layers) {
		if s != nil && !gi.follows(j) {
			break // a graph output or a second reader ends the segment
		}
		if gr, f, conv, ok := gi.group(j, false, false); ok {
			if !gi.inner(gr) || !fits(conv, f) {
				break
			}
			push(gr, conv, f)
			continue
		}
		// A stride-2 depthwise at full resolution, then the sampling 1×1;
		// the depthwise joins only together with it.
		gr, f, conv, ok := gi.group(j, true, false)
		if !ok || !gi.inner(gr) || !fits(conv, f) || !gi.follows(gr.First+gr.Count) {
			break
		}
		gr2, f2, conv2, ok := gi.group(gr.First+gr.Count, false, true)
		if !ok || !gi.inner(gr2) || !fits(conv2, f2) {
			break
		}
		push(gr, conv, f)
		push(gr2, conv2, f2)
	}
	if s != nil {
		s.Count = j - i
	}
	return s, nil
}

// fits reports whether a lowered layer's input and output fit KPU memory
// together (else it needs row-band tiling, K9g).
func fits(conv *qnet.Layer, f Fuse) bool {
	c, err := Lower(conv, f, 0, 0)
	if err != nil {
		return false
	}
	iw, ih, ic := c.Args.InShape()
	ow, oh, oc := c.Args.OutShape()
	return lines(qi8.Shape{H: ih, W: iw, C: ic})+lines(qi8.Shape{H: oh, W: ow, C: oc}) <= ramLines
}

// inner reports whether every output inside the group but the last has a
// single reader in the group (so it can stay in the KPU layer).
func (gi graphInfo) inner(gr Group) bool {
	for k := gr.First + 1; k < gr.First+gr.Count; k++ {
		if !gi.follows(k) {
			return false
		}
	}
	return true
}

// build lowers the segment's groups with ping-pong KPU addresses (the input
// at the top of memory, outputs alternating bottom and top), assembles the
// kmodel and its runner.
func (s *Segment) build(g *qnet.Graph, newRunner RunnerFunc) error {
	gi := newGraphInfo(g)
	b := kmodel.NewBuilder(true, 0)
	outLen := uint32(s.out.Len())
	b.SetMainMem(outLen)
	b.AddOutput(kmodel.Output{Address: 0, Size: outLen})
	src := uint32(ramLines - lines(s.in))
	top := true // where the current input sits
	for k, gr := range s.Groups {
		_, f, conv, ok := gi.group(gr.First, gr.Full, gr.Subsample)
		if !ok {
			return fmt.Errorf("group at layer %d vanished", gr.First)
		}
		probe, err := Lower(conv, f, 0, 0) // the KPU output shape (full resolution for Full)
		if err != nil {
			return err
		}
		ow, oh, oc := probe.Args.OutShape()
		dst := uint32(0)
		if !top {
			dst = uint32(ramLines - lines(qi8.Shape{H: oh, W: ow, C: oc}))
		}
		c, err := Lower(conv, f, src, dst)
		if err != nil {
			return err
		}
		bn, act := c.BN, c.Act
		if k == len(s.Groups)-1 {
			b.AddConvToMain(0, c.Args, c.Weights, bn, act)
		} else {
			b.AddConv(c.Args, c.Weights, bn, act)
		}
		src, top = dst, !top
	}
	data, err := b.Build()
	if err != nil {
		return err
	}
	m, err := kmodel.Parse(kpu.AlignedCopy(data))
	if err != nil {
		return err
	}
	r, err := newRunner(m, make([]byte, m.MainMemUsage))
	if err != nil {
		return err
	}
	s.Model, s.Runner, s.chw = m, r, make([]byte, s.in.Len())
	return nil
}
