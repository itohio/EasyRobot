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
		return nil, Info{}, fmt.Errorf("layer %d (%v %v → %v) cannot %s: %w", n, l.Kind, l.In, l.Out, where, Reason(g, n))
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

// Reason explains why layer i of g, a layer Compile left to the CPU, is not
// given to the KPU: Lower's error in the most specific mode that applies, a
// map smaller than MinSide, or ErrUnsupported when the layer lowers alone
// (then its sampling reader, KPU memory or the graph's structure stopped
// it).
func Reason(g *qnet.Graph, i int) error {
	l := &g.Layers[i]
	var f Fuse
	if l.Kind == qnet.Pad {
		// A Pad is lowered with the conv after it, when that conv reads it:
		// then the conv's refusal is the reason (e.g. the 256-row input
		// limit).
		if i+1 >= len(g.Layers) || producer(g, i+1) != i {
			return fmt.Errorf("%w: a Pad lowers only together with the conv right after it that reads it", ErrUnsupported)
		}
		f.Pad, l = l, &g.Layers[i+1]
	}
	if l.Kind != qnet.Conv2D && l.Kind != qnet.DepthwiseConv2D {
		return fmt.Errorf("%w: the KPU runs convolutions (with a fused Pad, ReLU/ReLU6 or pool)", ErrUnsupported)
	}
	if min(l.In.H, l.In.W, l.Out.H, l.Out.W) < MinSide {
		return fmt.Errorf("%w: a map smaller than %d×%d (%v → %v) is not given to the KPU", ErrUnsupported, MinSide, MinSide, l.In, l.Out)
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
// global. A Reshape whose input size changed keeps its channel count and
// whichever of H and W is 1, the other taking the rest (an SSD head's
// [cells·A·k] → [boxes, 1, k] or [1, boxes, k]); a Concat sums its inputs
// along its axis, its Rescale maps carried over. Layers whose parameters
// depend on the size (fully connected, resizes, constants, any other
// reshape) are refused.
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
		case qnet.Reshape:
			out, ok := reshaped(g.Layers[i].Out, old, in)
			if !ok {
				return nil, fmt.Errorf("%w: layer %d (reshape %v → %v) at %v: only a reshape keeping its channels with one of H, W at 1 follows the input size", ErrUnsupported, i, old, l.Out, in)
			}
			l.Out = out
			continue
		case qnet.Concat:
			d := [3]int{in.H, in.W, in.C}
			if l.Axis < 0 || l.Axis > 2 {
				return nil, fmt.Errorf("compile: layer %d (concat): axis %d", i, l.Axis)
			}
			d[l.Axis] = 0
			for _, p := range l.From {
				s := qi8.Shape{H: h, W: w, C: g.Layers[0].In.C} // −1: the graph input
				if p >= i {
					return nil, fmt.Errorf("compile: layer %d (concat) reads layer %d", i, p)
				}
				if p >= 0 {
					s = r.Layers[p].Out
				}
				d[l.Axis] += [3]int{s.H, s.W, s.C}[l.Axis]
			}
			l.Out = qi8.Shape{H: d[0], W: d[1], C: d[2]}
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

// reshaped returns a Reshape's output for the input shape in, given its
// output out for the input old: out itself when the size is unchanged;
// else the channel count and whichever of H, W is 1 are kept and the other
// takes the remaining elements. ok is false when out has neither or both
// of H, W at 1, or the elements do not divide.
func reshaped(out, old, in qi8.Shape) (qi8.Shape, bool) {
	if in == old {
		return out, true
	}
	if out.C <= 0 || (out.H == 1) == (out.W == 1) || in.Len()%out.C != 0 {
		return qi8.Shape{}, false
	}
	n := in.Len() / out.C
	if out.H == 1 {
		return qi8.Shape{H: 1, W: n, C: out.C}, true
	}
	return qi8.Shape{H: n, W: 1, C: out.C}, true
}

// ResizeAligned is Resize with every strided SAME convolution padded before
// its first row and column as in g. TF SAME pads a stride-2 3×3 convolution
// one row before on an odd input and none on an even one, so at a size that
// changes a layer's parity Resize moves that layer's sampling grid by one
// row, and every later layer's with it. ResizeAligned keeps the grid: such
// a convolution becomes a Pad and a VALID convolution (Pad(1,1,1,1) where
// an odd input became even, the form the KPU lowers; two rows after and
// none before where an even one became odd).
//
// The result run on the top-left h×w pixels of an image gives, in each cell
// that reads nothing beyond them, the value g gives on the whole image. A
// detector's outputs therefore keep their positions: its anchors at h×w are
// g's anchors for the cells that remain, in pixels. Inserted Pad layers
// shift the layer indices after them; From and Outputs follow. A strided
// SAME pool whose padding would move is refused: a pool's padding is not a
// Pad layer's.
func ResizeAligned(g *qnet.Graph, h, w int) (*qnet.Graph, error) {
	r, err := Resize(g, h, w)
	if err != nil {
		return nil, err
	}
	ls := make([]qnet.Layer, 0, len(r.Layers))
	at := make([]int, len(r.Layers)) // layer i's index in ls
	for i := range r.Layers {
		l := r.Layers[i]
		if len(l.From) > 0 {
			l.From = append([]int(nil), l.From...)
			for k, f := range l.From {
				if f >= 0 {
					l.From[k] = at[f]
				}
			}
		}
		switch l.Kind {
		case qnet.Conv2D, qnet.DepthwiseConv2D, qnet.MaxPool2D, qnet.AvgPool2D:
			if l.Padding != qi8.PaddingSame || (l.StrideH == 1 && l.StrideW == 1) {
				break
			}
			o := &g.Layers[i]
			top, bottom, movedH := alignedPad(o.In.H, l.In.H, l.Out.H, l.KH, l.StrideH)
			left, right, movedW := alignedPad(o.In.W, l.In.W, l.Out.W, l.KW, l.StrideW)
			if !movedH && !movedW {
				break
			}
			if l.Kind == qnet.MaxPool2D || l.Kind == qnet.AvgPool2D {
				return nil, fmt.Errorf("%w: layer %d (%v, stride %d×%d, SAME) would move its padding at %v", ErrUnsupported, i, l.Kind, l.StrideH, l.StrideW, l.In)
			}
			pad := qnet.Layer{Kind: qnet.Pad, From: l.From, In: l.In, InQ: l.InQ, OutQ: l.InQ,
				Out:  qi8.Shape{H: l.In.H + top + bottom, W: l.In.W + left + right, C: l.In.C},
				Pads: [6]int{top, bottom, left, right, 0, 0}, PadValue: l.InQ.ZP}
			ls = append(ls, pad)
			l.From, l.In, l.Padding = nil, pad.Out, qi8.PaddingValid
		}
		at[i] = len(ls)
		ls = append(ls, l)
	}
	a := &qnet.Graph{Layers: ls}
	if r.Outputs != nil {
		a.Outputs = make([]int, len(r.Outputs))
		for k, v := range r.Outputs {
			a.Outputs[k] = at[v]
		}
	}
	return a, a.Validate()
}

// alignedPad returns the padding of one axis of a SAME convolution (kernel
// k, stride s, n inputs, out outputs) that keeps the rows before the first
// as the same convolution has them on n0 inputs; moved reports that TF SAME
// on n inputs pads differently before. The rows after are those the last
// output needs, or as many as before when that reads no further (the
// symmetric Pad the KPU lowers).
func alignedPad(n0, n, out, k, s int) (before, after int, moved bool) {
	same := func(n int) int { return max(((n+s-1)/s-1)*s+k-n, 0) / 2 }
	before = same(n0)
	after = max((out-1)*s+k-n-before, 0)
	if after < before && (n+2*before-k)/s+1 == out {
		after = before
	}
	return before, after, before != same(n)
}
