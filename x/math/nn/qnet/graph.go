package qnet

import (
	"errors"

	"github.com/itohio/EasyRobot/x/math/primitive/qi8"
)

// Sentinel errors, returned unwrapped (no allocation); compare with ==.
// Kernel failures surface as qi8 sentinels.
var (
	// ErrEmpty reports a graph without layers.
	ErrEmpty = errors.New("qnet: empty graph")
	// ErrKind reports an unknown layer kind.
	ErrKind = errors.New("qnet: unknown layer kind")
	// ErrShape reports an invalid geometry, an Out that differs from the
	// shape implied by In and the geometry, or a layer whose In does not equal
	// the previous layer's Out.
	ErrShape = errors.New("qnet: invalid or mismatched shape")
	// ErrQuant reports a non-positive/non-finite scale, a layer whose InQ does
	// not equal the previous OutQ, or a pool/ReLU whose OutQ differs from InQ.
	ErrQuant = errors.New("qnet: invalid or mismatched quantisation")
	// ErrActivation reports ActMin > ActMax, or a ReLU6 whose quantised 6 is
	// below the zero point.
	ErrActivation = errors.New("qnet: invalid activation range")
	// ErrParams reports missing or wrongly sized weights/bias/requant, an
	// invalid Requant, or parameters on a parameter-free layer.
	ErrParams = errors.New("qnet: invalid layer parameters")
	// ErrCapacity reports a caller-provided slice that is too small.
	ErrCapacity = errors.New("qnet: capacity exceeded")
)

// Graph is a sequential network: Layers[i].Out feeds Layers[i+1].In.
type Graph struct {
	Layers []Layer
}

// Validate checks the whole graph, including bound parameters. See Check.
func (g *Graph) Validate() error {
	_, err := g.Check()
	return err
}

// Check validates the graph and returns the index of the first bad layer
// (-1 when the graph is valid or empty) and the error.
//
// Per layer: known kind, positive In, Out equal to OutShape, valid scales,
// In/InQ equal to the previous Out/OutQ, OutQ == InQ for pools and ReLUs,
// ActMin <= ActMax, len(Weights) and len(Requant) as ParamCounts, Bias nil or
// as ParamCounts, every Requant Valid; parameter-free kinds carry none.
func (g *Graph) Check() (int, error) {
	if len(g.Layers) == 0 {
		return -1, ErrEmpty
	}
	for i := range g.Layers {
		if err := g.checkLayer(i, true); err != nil {
			return i, err
		}
	}
	return -1, nil
}

// checkShapes validates everything except parameter binding.
func (g *Graph) checkShapes() error {
	if len(g.Layers) == 0 {
		return ErrEmpty
	}
	for i := range g.Layers {
		if err := g.checkLayer(i, false); err != nil {
			return err
		}
	}
	return nil
}

func (g *Graph) checkLayer(i int, params bool) error {
	l := &g.Layers[i]
	out, err := l.OutShape()
	if err != nil {
		return err
	}
	if l.Out != out {
		return ErrShape
	}
	if !l.InQ.valid() || !l.OutQ.valid() {
		return ErrQuant
	}
	if i > 0 {
		p := &g.Layers[i-1]
		if l.In != p.Out {
			return ErrShape
		}
		if l.InQ != p.OutQ {
			return ErrQuant
		}
	}
	if !l.Kind.hasParams() {
		if l.InQ != l.OutQ {
			return ErrQuant
		}
		if l.Kind == ReLU6 && qi8.Quantize6(l.InQ.Scale, l.InQ.ZP) < l.InQ.ZP {
			return ErrActivation
		}
		if len(l.Weights) != 0 || len(l.Bias) != 0 || len(l.Requant) != 0 {
			return ErrParams
		}
		return nil
	}
	if l.ActMin > l.ActMax {
		return ErrActivation
	}
	if !params {
		return nil
	}
	nw, nb, nr := l.ParamCounts()
	if len(l.Weights) != nw || len(l.Requant) != nr || (len(l.Bias) != 0 && len(l.Bias) != nb) {
		return ErrParams
	}
	for _, r := range l.Requant {
		if !r.Valid() {
			return ErrParams
		}
	}
	return nil
}

// LayerBudget is the static cost of one layer in bytes and multiply-adds.
type LayerBudget struct {
	In, Out int // activation bytes
	Weights int // weight bytes
	Params  int // bias (4 B each) + requant (8 B each) bytes
	// MACs counts nominal multiply-adds including padded taps (Conv2D:
	// outH·outW·OutC·KH·KW·InC; DepthwiseConv2D: outH·outW·C·KH·KW;
	// FullyConnected: In·Out). Pools and ReLUs count 0.
	MACs int64
}

// requantBytes is the in-memory size of one qi8.Requant.
const requantBytes = 8

// Budget returns the budget of layer i, from shapes only (parameters need
// not be bound). It panics if i is out of range.
func (g *Graph) Budget(i int) LayerBudget {
	l := &g.Layers[i]
	nw, nb, nr := l.ParamCounts()
	b := LayerBudget{In: l.In.Len(), Out: l.Out.Len(), Weights: nw, Params: 4*nb + requantBytes*nr}
	// Every weight is used once per output pixel (FC: Out is 1×1).
	b.MACs = int64(nw) * int64(l.Out.H*l.Out.W)
	return b
}

// WeightBytes returns the total weight bytes of the graph.
func (g *Graph) WeightBytes() int {
	n := 0
	for i := range g.Layers {
		n += g.Budget(i).Weights
	}
	return n
}

// ParamBytes returns the total bias + requant bytes of the graph.
func (g *Graph) ParamBytes() int {
	n := 0
	for i := range g.Layers {
		n += g.Budget(i).Params
	}
	return n
}

// MACs returns the total nominal multiply-adds of one inference.
func (g *Graph) MACs() int64 {
	var n int64
	for i := range g.Layers {
		n += g.Budget(i).MACs
	}
	return n
}

// ArenaSize returns the ping-pong arena bytes the Runner needs when every
// layer is its own segment (the all-Generic plan): the maximum over layers of
// In+Out bytes, except that layer 0 reads the caller's input and needs only
// its Out. The arena holds one tensor at its start and the next at its end,
// alternating. A fused segment needs its first In plus its last Out, which
// can exceed this; Runner.ArenaSize reports the plan's exact need.
func (g *Graph) ArenaSize() int {
	n := 0
	for i := range g.Layers {
		n = max(n, boundaryBytes(g.Layers, i, i+1))
	}
	return n
}

// boundaryBytes is the arena requirement of running layers[first:end] as one
// segment.
func boundaryBytes(layers []Layer, first, end int) int {
	n := layers[end-1].Out.Len()
	if first > 0 {
		n += layers[first].In.Len()
	}
	return n
}

// ParamCounts returns the total weights, bias and requant entries of g, in
// the layout BindParams expects.
func ParamCounts(g *Graph) (weights, bias, requant int) {
	for i := range g.Layers {
		w, b, r := g.Layers[i].ParamCounts()
		weights, bias, requant = weights+w, bias+b, requant+r
	}
	return weights, bias, requant
}

// BindParams slices weights, bias and rq (each exactly as long as
// ParamCounts) into the layers in order, without copying or allocating.
// Parameter-free layers get nil. bias may be nil for an all-nil bias.
// Returns ErrParams on a length mismatch.
func BindParams(g *Graph, weights []int8, bias []int32, rq []qi8.Requant) error {
	nw, nb, nr := ParamCounts(g)
	if len(weights) != nw || len(rq) != nr || (len(bias) != 0 && len(bias) != nb) {
		return ErrParams
	}
	var ow, ob, or int
	for i := range g.Layers {
		l := &g.Layers[i]
		w, b, r := l.ParamCounts()
		l.Weights, l.Bias, l.Requant = nil, nil, nil
		if !l.Kind.hasParams() {
			continue
		}
		l.Weights = weights[ow : ow+w : ow+w]
		if len(bias) != 0 {
			l.Bias = bias[ob : ob+b : ob+b]
		}
		l.Requant = rq[or : or+r : or+r]
		ow, ob, or = ow+w, ob+b, or+r
	}
	return nil
}
