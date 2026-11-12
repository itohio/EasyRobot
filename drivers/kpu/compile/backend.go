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
// ([Pad] conv [ReLU/ReLU6] [pool]) lowered together, with the lowering
// mode (Fuse's Full, Subsample, OddSample). A Sampler group has no graph
// layers (Count 0): it is the identity conv (Sampler) on layer First−1's
// output, sampling the Full layer before it.
type Group struct {
	First, Count int
	Mode         Mode
	c            Conv // lowered at addresses 0 (the planner's one Lower per group)
}

// Mode is how a group is lowered beyond plain fusion (see Fuse).
type Mode struct{ Full, Subsample, OddSample, Sampler bool }

func (m Mode) fuse(pad, act, pool *qnet.Layer) Fuse {
	return Fuse{Pad: pad, Act: act, Pool: pool, Full: m.Full, Subsample: m.Subsample, OddSample: m.OddSample}
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
	toCHW(s.chw, src, s.in.H*s.in.W, s.in.C)
	if err := s.Runner.Run(s.chw); err != nil {
		return err
	}
	fromCHW(dst, s.Runner.Output(0), s.out.H*s.out.W, s.out.C)
	return nil
}

// toCHW converts int8 NHWC (hw pixels × c channels) to the KPU's uint8 CHW
// (q + 128), one channel plane at a time: the plane is a hoisted slice, so
// the inner loop is a strided load and a sequential store with no index
// arithmetic beyond the stride (review R-K210-K9de D1).
func toCHW(chw []byte, src []int8, hw, c int) {
	for ch := range c {
		d := chw[ch*hw : (ch+1)*hw]
		s := src[ch:]
		for i := range d {
			d[i] = byte(s[i*c]) ^ 0x80
		}
	}
}

// fromCHW converts the KPU's uint8 CHW back to int8 NHWC (− 128).
func fromCHW(dst []int8, chw []byte, hw, c int) {
	for ch := range c {
		p := chw[ch*hw : (ch+1)*hw]
		d := dst[ch:]
		for i, v := range p {
			d[i*c] = int8(v ^ 0x80)
		}
	}
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
	// Explicit outputs end a segment. The last layer's output never sits
	// inside one, so the default output (Outputs nil) needs no mark.
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
// start one. Every layer it fuses after the first reads only its
// predecessor, whose output has no other reader and is not a graph output
// (follows), so the group's inner tensors can stay inside the KPU layer.
// A Sampler mode returns the sampler on layer i−1's output.
func (gi graphInfo) group(i int, mode Mode) (Group, Fuse, *qnet.Layer, bool) {
	ls := gi.g.Layers
	if mode.Sampler {
		sl := Sampler(&ls[i-1])
		f := mode.fuse(nil, nil, nil)
		c, err := Lower(&sl, f, 0, 0)
		if err != nil {
			return Group{}, Fuse{}, nil, false
		}
		return Group{First: i, Mode: mode, c: c}, f, &sl, true
	}
	j := i
	var f Fuse
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
		{mode.fuse(f.Pad, act, pool), base + b2i(act != nil) + b2i(pool != nil)},
		{mode.fuse(f.Pad, act, nil), base + b2i(act != nil)},
		{mode.fuse(f.Pad, nil, nil), base},
	}
	for _, cd := range cands {
		if c, err := Lower(conv, cd.f, 0, 0); err == nil {
			return Group{First: i, Count: cd.n, Mode: mode, c: c}, cd.f, conv, true
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
		s := gi.segment(i)
		if s == nil {
			i++
			continue
		}
		if err := s.build(newRunner); err != nil {
			return nil, fmt.Errorf("compile: segment at layer %d: %w", i, err)
		}
		b.segs[&g.Layers[s.First]] = s
		b.order = append(b.order, s)
		i = s.First + s.Count
	}
	return b, nil
}

// segment collects the longest chain of groups from layer i that fits KPU
// memory layer by layer; nil if layer i starts no group. A Full layer (a
// stride-2 depthwise, or a TF SAME stride-2 conv) joins only together with
// its sampling reader: the 1×1 conv after it, else a Sampler.
func (gi graphInfo) segment(i int) *Segment {
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
		if gr, f, conv, ok := gi.group(j, Mode{}); ok {
			if !fits(gr, false) {
				break
			}
			push(gr, conv, f)
			continue
		}
		// A layer at full resolution, then its sampling reader.
		gr, f, conv, ok := gi.group(j, Mode{Full: true})
		if !ok {
			break
		}
		next := gr.First + gr.Count
		odd := f.Pad == nil // TF SAME: odd pixels; Pad(1,1)+VALID: even
		sample := Mode{Subsample: true, OddSample: odd}
		var gr2 Group
		var f2 Fuse
		var conv2 *qnet.Layer
		ok = gi.follows(next) // follows is false past the last layer
		if ok {
			gr2, f2, conv2, ok = gi.group(next, sample)
		}
		if !ok {
			sample.Sampler = true
			gr2, f2, conv2, ok = gi.group(next, sample)
		}
		if !ok || !fits(gr, odd) || !fits(gr2, false) {
			break
		}
		push(gr, conv, f)
		push(gr2, conv2, f2)
	}
	if s != nil {
		s.Count = j - i
	}
	return s
}

// fits reports whether a lowered layer's input and output fit KPU memory
// together (else it needs row-band tiling, K9g). An OddSample reader reads
// one row past its input, which counts too (review R-K210-K9same S3);
// readPast reserves that row after the output, for the layer such a reader
// follows.
func fits(gr Group, readPast bool) bool {
	a := gr.c.Args
	iw, ih, ic := a.InShape()
	ow, oh, oc := a.OutShape()
	n := lines(qi8.Shape{H: ih, W: iw, C: ic}) + lines(qi8.Shape{H: oh, W: ow, C: oc})
	if gr.Mode.OddSample {
		n += kmodel.UploadLayout(iw, ih).RowLines
	}
	if readPast {
		n += kmodel.UploadLayout(ow, oh).RowLines
	}
	return n <= ramLines
}

// build lowers the segment's groups with ping-pong KPU addresses (the input
// at the top of memory, outputs alternating bottom and top), assembles the
// kmodel and its runner.
func (s *Segment) build(newRunner RunnerFunc) error {
	data, kin, err := s.kmodel()
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
	s.Model, s.Runner, s.chw = m, r, make([]byte, kin.Len())
	return nil
}

// kmodel assembles the segment's kmodel file and returns it with the KPU
// input shape.
func (s *Segment) kmodel() ([]byte, qi8.Shape, error) {
	b := kmodel.NewBuilder(true, 0)
	outLen := uint32(s.out.Len())
	b.SetMainMem(outLen)
	b.AddOutput(kmodel.Output{Address: 0, Size: outLen})
	kin := s.in
	src := uint32(ramLines - lines(kin))
	top := true // where the current input sits
	for k, gr := range s.Groups {
		a := gr.c.Args // lowered at addresses 0; the source holds only the row skip
		ow, oh, oc := a.OutShape()
		dst := uint32(0)
		if !top {
			n := lines(qi8.Shape{H: oh, W: ow, C: oc})
			if k+1 < len(s.Groups) && s.Groups[k+1].Mode.OddSample {
				n += kmodel.UploadLayout(ow, oh).RowLines // the reader reads one row past
			}
			dst = uint32(ramLines - n)
		}
		if err := a.SetSrcAddr(src + a.SrcAddr()); err != nil {
			return nil, kin, err
		}
		if err := a.SetDstAddr(dst); err != nil {
			return nil, kin, err
		}
		c := gr.c
		if k == len(s.Groups)-1 {
			b.AddConvToMain(0, a, c.Weights, c.BN, c.Act)
		} else {
			b.AddConv(a, c.Weights, c.BN, c.Act)
		}
		src, top = dst, !top
	}
	data, err := b.Build()
	return data, kin, err
}
