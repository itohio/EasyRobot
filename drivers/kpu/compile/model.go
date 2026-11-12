package compile

import (
	"errors"
	"fmt"

	"github.com/itohio/EasyRobot/x/math/nn/qnet"
	"github.com/itohio/EasyRobot/x/math/primitive/qi8"
)

// Info describes a kmodel compiled by Kmodel.
type Info struct {
	// Input and InputQ are the graph's input (int8 NHWC); the kmodel's
	// input is it as uint8 CHW, value q + 128.
	Input  qi8.Shape
	InputQ qnet.Quant
	// Output and OutputQ are the graph's output; the kmodel's output 0 is it
	// as uint8 CHW, value q + 128.
	Output  qi8.Shape
	OutputQ qnet.Quant
	// KPULayers is the number of KPU layers (convs with their fused layers).
	KPULayers int
}

// Kmodel compiles the whole graph into one kmodel for drivers/kpu. Every
// layer must lower into a single KPU segment, the graph's only output being
// its last layer; otherwise the error names the first layer left out.
func Kmodel(g *qnet.Graph) ([]byte, Info, error) {
	if err := g.Validate(); err != nil {
		return nil, Info{}, err
	}
	if g.Outputs != nil && (len(g.Outputs) != 1 || g.Outputs[0] != len(g.Layers)-1) {
		return nil, Info{}, fmt.Errorf("%w: a kmodel has one output, the last layer (graph outputs %v)", ErrUnsupported, g.Outputs)
	}
	gi := newGraphInfo(g)
	s := gi.segment(0)
	n := 0
	if s != nil {
		n = s.Count
	}
	if n != len(g.Layers) {
		l := &g.Layers[n]
		where := fmt.Sprintf("join the KPU segment of layers 0..%d", n-1)
		if n == 0 {
			where = "start a KPU segment"
		}
		return nil, Info{}, fmt.Errorf("layer %d (%v %v → %v) cannot %s: %w", n, l.Kind, l.In, l.Out, where, reason(g, n))
	}
	data, _, err := s.kmodel()
	if err != nil {
		return nil, Info{}, err
	}
	last := &g.Layers[len(g.Layers)-1]
	return data, Info{
		Input: g.Layers[0].In, InputQ: g.Layers[0].InQ,
		Output: last.Out, OutputQ: last.OutQ, KPULayers: len(s.Groups),
	}, nil
}

// reason explains why layer i does not lower: Lower's error in the most
// specific mode that applies, or ErrUnsupported when l lowers alone (then
// its sampling reader, KPU memory or the graph's structure stopped it).
func reason(g *qnet.Graph, i int) error {
	l := &g.Layers[i]
	var f Fuse
	if l.Kind == qnet.Pad && i+1 < len(g.Layers) {
		// A Pad is lowered with the conv after it: that conv's refusal is
		// the reason (e.g. the 256-row input limit).
		f.Pad, l = l, &g.Layers[i+1]
	}
	if l.Kind != qnet.Conv2D && l.Kind != qnet.DepthwiseConv2D {
		return fmt.Errorf("%w: the KPU runs convolutions (with a fused Pad, ReLU/ReLU6 or pool)", ErrUnsupported)
	}
	full := f
	full.Full = true
	_, err := Lower(l, full, 0, 0)
	if errors.Is(err, errNotFull) {
		_, err = Lower(l, f, 0, 0)
	}
	if err == nil {
		return fmt.Errorf("%w: its sampling reader, KPU memory (2 MiB for its input and output) or a second reader of its input", ErrUnsupported)
	}
	return err
}

// Cut returns a graph of g's layers 0..last (sharing their parameters),
// whose output is layer last.
func Cut(g *qnet.Graph, last int) (*qnet.Graph, error) {
	if last < 0 || last >= len(g.Layers) {
		return nil, fmt.Errorf("compile: cut at layer %d of %d", last, len(g.Layers))
	}
	c := &qnet.Graph{Layers: append([]qnet.Layer(nil), g.Layers[:last+1]...)}
	return c, c.Validate()
}

// Resize returns g with the input set to h×w (same channels) and every
// layer's shapes re-derived, sharing the parameters. Convolutions, Pad,
// element-wise layers and pools are size-agnostic; a global pool stays
// global. Layers whose parameters depend on the size (fully connected,
// reshape, concat, resize, constants) are refused.
func Resize(g *qnet.Graph, h, w int) (*qnet.Graph, error) {
	r := &qnet.Graph{Layers: append([]qnet.Layer(nil), g.Layers...), Outputs: g.Outputs}
	for i := range r.Layers {
		l := &r.Layers[i]
		old := g.Layers[i].In
		in := qi8.Shape{H: h, W: w, C: l.In.C}
		if p := producer(r, i); p >= 0 {
			in = r.Layers[p].Out
		} else if p == -2 && len(l.From) > 0 && l.From[0] >= 0 {
			// Add, Mul: In is the first input's shape (only the second may
			// broadcast, as qnet checks); a first input of −1 is the graph
			// input, the resized shape above.
			in = r.Layers[l.From[0]].Out
		}
		l.In = in
		switch l.Kind {
		case qnet.Conv2D, qnet.DepthwiseConv2D:
			l.Out.H, l.Out.W = 0, 0
		case qnet.Pad:
			p := l.Pads
			l.Out = qi8.Shape{H: in.H + p[0] + p[1], W: in.W + p[2] + p[3], C: in.C + p[4] + p[5]}
			continue
		case qnet.MaxPool2D, qnet.AvgPool2D:
			if l.KH == old.H && l.KW == old.W && l.Padding == qi8.PaddingValid { // global
				l.KH, l.KW = in.H, in.W
			}
		case qnet.ReLU, qnet.ReLU6, qnet.LeakyReLU, qnet.Logistic, qnet.Softmax, qnet.Requantize, qnet.Add, qnet.Mul:
			l.Out = in
			continue
		case qnet.Mean:
			l.Out = qi8.Shape{H: 1, W: 1, C: in.C}
			continue
		default:
			return nil, fmt.Errorf("%w: layer %d (%v) depends on the input size", ErrUnsupported, i, l.Kind)
		}
		out, err := l.OutShape()
		if err != nil {
			return nil, fmt.Errorf("compile: layer %d (%v) at %v: %w", i, l.Kind, in, err)
		}
		l.Out = out
	}
	return r, r.Validate()
}
