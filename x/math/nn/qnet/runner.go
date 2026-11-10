package qnet

import (
	"errors"

	"github.com/itohio/EasyRobot/x/math/primitive/qi8"
)

// Runner errors.
var (
	// ErrClaim reports a backend claim past the end of the graph, or over
	// layers that are not a single-input chain (see PlanMemory).
	ErrClaim = errors.New("qnet: invalid backend claim")
	// ErrArena reports an arena smaller than the plan requires.
	ErrArena = errors.New("qnet: arena too small")
	// ErrInput reports an input whose length is not Layers[0].In.Len().
	ErrInput = errors.New("qnet: input length mismatch")
	// ErrNotReady reports a Runner used before a successful Init.
	ErrNotReady = errors.New("qnet: runner not initialised")
	// ErrSegment reports a backend asked to run a segment it cannot run.
	ErrSegment = errors.New("qnet: unsupported segment")
)

// Backend executes runs of layers itself (an accelerator, or fused kernels).
// Layers no backend claims run on the built-in CPU kernels: the qi8 prepared
// layers, which use the build's qi8 backend (generic, or PIE on the
// ESP32-S3). Implementations must not allocate in Run.
type Backend interface {
	// Name identifies the backend in plans.
	Name() string
	// Claim returns how many consecutive layers starting at layers[0] this
	// backend will run as one segment (0 = none). The claim must be a chain
	// of single-input layers (see PlanMemory).
	Claim(layers []Layer) int
	// Run executes the claimed segment from src to dst (NHWC int8, quantised
	// as the first layer's In and the last layer's Out). No allocation.
	Run(layers []Layer, dst, src []int8) error
}

// Segment is one planned step: Backend runs Layers[First:First+Count], or
// (Backend nil) the built-in CPU kernels run the single layer First.
type Segment struct {
	Backend      Backend
	First, Count int
}

// cpuOp is the prepared form of one CPU layer.
type cpuOp struct {
	op  qi8.Op
	add *qi8.AddLayer
}

// Runner executes a validated Graph with a fixed MemPlan over a
// caller-provided arena. The graph's layers must not be changed while the
// Runner is in use.
type Runner struct {
	g     *Graph
	plan  *MemPlan
	arena []int8 // aligned
	ops   []cpuOp
	srcs  [][]int8 // producer tensors of the running layer
}

// NewRunner allocates a Runner and calls Init.
func NewRunner(g *Graph, arena []int8, backends ...Backend) (*Runner, error) {
	r := &Runner{}
	if err := r.Init(g, arena, backends...); err != nil {
		return nil, err
	}
	return r, nil
}

// Init validates g (parameters must be bound), plans it (PlanMemory),
// checks len(arena) >= Plan().Size() (ErrArena) and prepares every CPU layer
// with its scratch region, packing weights for the qi8 backend. It
// allocates the plan and the prepared layers once; Run allocates nothing.
// On error r is left uninitialised.
func (r *Runner) Init(g *Graph, arena []int8, backends ...Backend) error {
	*r = Runner{}
	if err := g.Validate(); err != nil {
		return err
	}
	plan, err := PlanMemory(g, backends...)
	if err != nil {
		return err
	}
	if len(arena) < plan.Size() {
		return ErrArena
	}
	ops := make([]cpuOp, len(g.Layers))
	maxIn := 0
	for _, s := range plan.Segments {
		if s.Backend != nil {
			continue
		}
		maxIn = max(maxIn, g.inputs(s.First))
		if err := prepare(&ops[s.First], &g.Layers[s.First], plan.ScratchSlice(arena, s.First)); err != nil {
			return err
		}
	}
	*r = Runner{g: g, plan: plan, arena: align(arena), ops: ops, srcs: make([][]int8, maxIn)}
	return nil
}

func prepare(o *cpuOp, l *Layer, scratch []int8) error {
	switch l.Kind {
	case Conv2D:
		c := &qi8.Conv2DLayer{P: l.Conv2DParams(), W: l.Weights, Bias: l.Bias, RQ: l.Requant, Scratch: scratch}
		o.op = c
		return c.Prepare()
	case DepthwiseConv2D:
		c := &qi8.DepthwiseConv2DLayer{P: l.DepthwiseParams(), W: l.Weights, Bias: l.Bias, RQ: l.Requant, Scratch: scratch}
		o.op = c
		return c.Prepare()
	case FullyConnected:
		c := &qi8.FCLayer{P: l.FCParams(), W: l.Weights, Bias: l.Bias, RQ: l.Requant, Scratch: scratch}
		o.op = c
		return c.Prepare()
	case MaxPool2D:
		c := &qi8.MaxPool2DLayer{P: l.PoolParams()}
		o.op = c
		return c.Prepare()
	case AvgPool2D:
		c := &qi8.AvgPool2DLayer{P: l.PoolParams()}
		o.op = c
		return c.Prepare()
	case ReLU:
		c := &qi8.ReLULayer{Len: l.In.Len(), ZP: l.InQ.ZP}
		o.op = c
		return c.Prepare()
	case ReLU6:
		c := &qi8.ReLU6Layer{Len: l.In.Len(), ZP: l.InQ.ZP, Q6: qi8.Quantize6(l.InQ.Scale, l.InQ.ZP)}
		o.op = c
		return c.Prepare()
	case Add:
		o.add = &qi8.AddLayer{Len: l.In.Len(), P: l.AddParams()}
		return o.add.Prepare()
	}
	return nil // Reshape and Const (alias) and the stateless kinds need no preparation
}

// Plan returns the execution and memory plan (aliases the Runner).
func (r *Runner) Plan() *MemPlan { return r.plan }

// tensor returns the bytes of layer i's output (input for i == −1).
func (r *Runner) tensor(i int, input []int8) []int8 {
	if i < 0 {
		return input
	}
	o := r.plan.Out[i]
	switch {
	case o.Off == -1:
		return input[:o.Len]
	case o.Off == -2:
		for r.g.Layers[i].Kind == Reshape {
			i = r.g.producer(i, 0)
		}
		return r.g.Layers[i].Weights[:o.Len:o.Len]
	}
	return r.arena[o.Off : o.Off+o.Len : o.Off+o.Len]
}

// Run executes the plan on input (Layers[0].In.Len() bytes, must not overlap
// the arena) and returns graph output 0 (the last layer unless g.Outputs is
// set), which aliases the arena (or input, for a Reshape of it, or a
// Const's Weights) and is valid until the next Run. Output returns the
// others. Run does not allocate.
func (r *Runner) Run(input []int8) ([]int8, error) {
	if r.g == nil {
		return nil, ErrNotReady
	}
	for k := range r.plan.Segments {
		if err := r.RunSegment(input, k); err != nil {
			return nil, err
		}
	}
	return r.tensor(r.g.outputLayer(0), input), nil
}

// Output returns graph output k (0 ≤ k < g.NumOutputs()) of the last Run on
// input; valid until the next Run.
func (r *Runner) Output(input []int8, k int) []int8 {
	return r.tensor(r.g.outputLayer(k), input)
}

// RunSegment executes segment k of the plan on input; running every segment
// in order is Run (for profiling per segment). Segment k's inputs must have
// been produced by earlier segments with the same input.
func (r *Runner) RunSegment(input []int8, k int) error {
	if r.g == nil {
		return ErrNotReady
	}
	layers := r.g.Layers
	if len(input) != layers[0].In.Len() {
		return ErrInput
	}
	s := r.plan.Segments[k]
	last := s.First + s.Count - 1
	dst := r.tensor(last, input)
	src := r.tensor(r.g.producer(s.First, 0), input)
	if s.Backend != nil {
		return s.Backend.Run(layers[s.First:last+1], dst, src)
	}
	l := &layers[s.First]
	o := &r.ops[s.First]
	switch {
	case o.op != nil:
		return o.op.Run(dst, src)
	case o.add != nil:
		return o.add.Run(dst, src, r.tensor(r.g.producer(s.First, 1), input))
	case l.Kind == Reshape || l.Kind == Const:
		return nil // the output aliases the input or the constant
	}
	srcs := r.srcs[:r.g.inputs(s.First)]
	for k := range srcs {
		srcs[k] = r.tensor(r.g.producer(s.First, k), input)
	}
	return runStateless(l, dst, srcs)
}

// SegmentOutput returns segment k's output tensor (valid after it ran).
func (r *Runner) SegmentOutput(input []int8, k int) []int8 {
	s := r.plan.Segments[k]
	return r.tensor(s.First+s.Count-1, input)
}
