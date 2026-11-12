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
	// shape implied by In and the geometry, a layer whose In does not equal
	// its producer's Out, a Mul second input that does not broadcast, Concat
	// inputs that disagree off the axis, or a bad From (out of range, not
	// earlier than the layer, wrong count for the kind, a Const at layer 0).
	ErrShape = errors.New("qnet: invalid or mismatched shape")
	// ErrQuant reports a non-positive/non-finite scale, a layer whose InQ (or
	// Add's/Mul's In2Q) does not equal its producer's OutQ, a pool/ReLU/
	// Reshape/Const whose OutQ differs from InQ, a Concat input whose
	// quantisation differs from OutQ, or a Softmax/Logistic whose OutQ is
	// not SoftmaxOutQuant/LogisticOutQuant.
	ErrQuant = errors.New("qnet: invalid or mismatched quantisation")
	// ErrActivation reports ActMin > ActMax, or a ReLU6 whose quantised 6 is
	// below the zero point.
	ErrActivation = errors.New("qnet: invalid activation range")
	// ErrParams reports missing or wrongly sized weights/bias/requant, an
	// invalid Requant, rounding mode, Add multipliers, softmax or logistic
	// scaling, Const data of the wrong length, or parameters on a
	// parameter-free layer.
	ErrParams = errors.New("qnet: invalid layer parameters")
	// ErrCapacity reports a caller-provided slice that is too small.
	ErrCapacity = errors.New("qnet: capacity exceeded")
)

// Graph is a network in execution order: layer i reads the layers in its
// From (by default layer i−1; −1 is the graph input). Layer 0 reads the graph
// input, whose shape and quantisation are Layers[0].In/InQ. The graph
// outputs are the layers in Outputs, in order (nil: the last layer).
type Graph struct {
	Layers []Layer
	// Outputs lists the layers whose outputs the graph returns (nil means
	// the last layer). Each entry is a layer index; entries may repeat.
	Outputs []int
}

// outputLayer returns the layer index of graph output k.
func (g *Graph) outputLayer(k int) int {
	if g.Outputs == nil {
		return len(g.Layers) - 1
	}
	return g.Outputs[k]
}

// NumOutputs returns the number of graph outputs.
func (g *Graph) NumOutputs() int {
	if g.Outputs == nil {
		return 1
	}
	return len(g.Outputs)
}

// checkOutputs validates Outputs (ErrShape: empty or out of range).
func (g *Graph) checkOutputs() error {
	if g.Outputs != nil && len(g.Outputs) == 0 {
		return ErrShape
	}
	for _, o := range g.Outputs {
		if o < 0 || o >= len(g.Layers) {
			return ErrShape
		}
	}
	return nil
}

// Validate checks the whole graph, including bound parameters. See Check.
func (g *Graph) Validate() error {
	_, err := g.Check()
	return err
}

// Check validates the graph and returns the index of the first bad layer
// (-1 when the graph is valid or empty) and the error.
//
// Per layer: known kind, From valid (earlier layers or −1; one entry for
// single-input kinds, two for Add and Mul, one or more for Concat, none for
// Const, which must not be layer 0; layer 0 reads only the graph input),
// positive In, Out equal to OutShape, valid scales, In/InQ (and the second
// input's In2Q) equal to the producers' Out/OutQ (Mul's second input may be
// In, 1×1×In.C or 1×1×1; Concat inputs agree with In off the axis, sum to Out
// on it and are quantised as OutQ), OutQ == InQ for pools, ReLUs, Reshape
// and Const (whose In is Out), Softmax/Logistic OutQ ==
// SoftmaxOutQuant/LogisticOutQuant, ActMin <= ActMax, len(Weights) and
// len(Requant) as ParamCounts (Const: Out.Len() data bytes), Bias nil or as
// ParamCounts, every Requant (and RQ: Valid or zero) Valid, valid Round, Add
// multipliers, softmax and logistic scaling; parameter-free kinds carry no
// weights. Outputs, when set, must be non-empty and name existing layers
// (ErrShape, reported at index −1).
func (g *Graph) Check() (int, error) {
	if len(g.Layers) == 0 {
		return -1, ErrEmpty
	}
	for i := range g.Layers {
		if err := g.checkLayer(i, true); err != nil {
			return i, err
		}
	}
	return -1, g.checkOutputs()
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
	return g.checkOutputs()
}

// producer returns layer i's k-th producer (−1 = graph input).
func (g *Graph) producer(i, k int) int {
	if from := g.Layers[i].From; from != nil {
		return from[k]
	}
	return i - 1
}

// inputs returns how many producers layer i reads.
func (g *Graph) inputs(i int) int {
	l := &g.Layers[i]
	if n := l.Kind.inputs(); n >= 0 {
		return n
	}
	if l.From == nil {
		return 1
	}
	return len(l.From)
}

// output returns the shape and quantisation produced by p (−1 = input).
func (g *Graph) output(p int) (qi8.Shape, Quant) {
	if p < 0 {
		return g.Layers[0].In, g.Layers[0].InQ
	}
	return g.Layers[p].Out, g.Layers[p].OutQ
}

func (g *Graph) checkFrom(i int) error {
	l := &g.Layers[i]
	switch n := l.Kind.inputs(); {
	case n == 0:
		if l.From != nil || i == 0 {
			return ErrShape
		}
		return nil
	case l.From == nil:
		if n == 2 {
			return ErrShape
		}
		return nil
	case len(l.From) == 0 || (n > 0 && len(l.From) != n):
		return ErrShape
	}
	for _, p := range l.From {
		if p < -1 || p >= i || (i == 0 && p != -1) {
			return ErrShape
		}
	}
	return nil
}

func dims(s qi8.Shape) [3]int { return [3]int{s.H, s.W, s.C} }

// checkConcat checks Concat layer i against all its producers.
func (g *Graph) checkConcat(i int) error {
	l := &g.Layers[i]
	in := dims(l.In)
	sum := 0
	for k := range g.inputs(i) {
		ps, pq := g.output(g.producer(i, k))
		if pq != l.OutQ {
			return ErrQuant
		}
		d := dims(ps)
		for a := range d {
			if a != l.Axis && d[a] != in[a] {
				return ErrShape
			}
		}
		sum += d[l.Axis]
	}
	in[l.Axis] = sum
	if l.Out != (qi8.Shape{H: in[0], W: in[1], C: in[2]}) {
		return ErrShape
	}
	return nil
}

func (g *Graph) checkLayer(i int, params bool) error {
	l := &g.Layers[i]
	if err := g.checkFrom(i); err != nil {
		return err
	}
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
	if i > 0 && l.Kind != Const {
		ps, pq := g.output(g.producer(i, 0))
		if l.In != ps {
			return ErrShape
		}
		if l.InQ != pq {
			return ErrQuant
		}
	}
	switch l.Kind {
	case Add, Mul:
		ps, pq := g.output(g.producer(i, 1))
		bc := l.Kind == Mul && (ps == qi8.Shape{H: 1, W: 1, C: l.In.C} || ps == qi8.Shape{H: 1, W: 1, C: 1})
		if ps != l.In && !bc {
			return ErrShape
		}
		if !l.In2Q.valid() || l.In2Q != pq {
			return ErrQuant
		}
	case Softmax:
		if l.OutQ != SoftmaxOutQuant {
			return ErrQuant
		}
	case Logistic:
		if l.OutQ != LogisticOutQuant {
			return ErrQuant
		}
	case Concat:
		if err := g.checkConcat(i); err != nil {
			return err
		}
	case Const:
		if l.In != l.Out {
			return ErrShape
		}
		if l.InQ != l.OutQ {
			return ErrQuant
		}
	case Conv2D, DepthwiseConv2D, FullyConnected, Mean, Requantize, LeakyReLU, Pad, ResizeNearest, ResizeBilinear:
		// Pad and the resizes work on raw values (TFLite does, whatever OutQ).
	default:
		if l.InQ != l.OutQ {
			return ErrQuant
		}
	}
	if l.Kind == ReLU6 && qi8.Quantize6(l.InQ.Scale, l.InQ.ZP) < l.InQ.ZP {
		return ErrActivation
	}
	if (l.Kind.hasParams() || l.Kind == Add || l.Kind == Mul) && l.ActMin > l.ActMax {
		return ErrActivation
	}
	if !l.Kind.hasParams() && ((l.Kind != Const && len(l.Weights) != 0) || len(l.Bias) != 0 || len(l.Requant) != 0 || l.WZP != 0) {
		return ErrParams
	}
	if l.TiesUp && l.Kind != AvgPool2D {
		return ErrParams
	}
	if !params {
		return nil
	}
	switch l.Kind {
	case Add:
		if qi8.Add(nil, nil, nil, l.AddParams()) != nil {
			return ErrParams
		}
		return nil
	case Softmax:
		if qi8.Softmax(nil, nil, l.In.C, l.SoftmaxParams) != nil {
			return ErrParams
		}
		return nil
	case Logistic:
		if l.LogisticParams == nil || qi8.Logistic(nil, nil, *l.LogisticParams) != nil {
			return ErrParams
		}
		return nil
	case Mul, Mean, Requantize, LeakyReLU:
		okRQ := func(r qi8.Requant) bool { return r == qi8.Requant{} || r.Valid() }
		if (l.Round != qi8.RoundExact && l.Round != qi8.RoundTFLite) || !okRQ(l.RQ) || (l.Kind == LeakyReLU && !okRQ(l.RQAlpha)) {
			return ErrParams
		}
		return nil
	case Const:
		if len(l.Weights) != l.Out.Len() {
			return ErrParams
		}
		return nil
	}
	if !l.Kind.hasParams() {
		return nil
	}
	if l.Round != qi8.RoundExact && l.Round != qi8.RoundTFLite {
		return ErrParams
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
	// FullyConnected: In·Out). Other kinds count 0.
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
	if l.Kind == Const {
		b.Weights = l.Out.Len()
	}
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
