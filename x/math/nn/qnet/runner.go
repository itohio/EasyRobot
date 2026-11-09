package qnet

import "errors"

// Runner errors.
var (
	// ErrUnclaimed reports a layer that no backend claims.
	ErrUnclaimed = errors.New("qnet: no backend claims layer")
	// ErrClaim reports a backend claiming more layers than remain.
	ErrClaim = errors.New("qnet: backend claimed past the end of the graph")
	// ErrArena reports an arena smaller than the plan requires.
	ErrArena = errors.New("qnet: arena too small")
	// ErrInput reports an input whose length is not Layers[0].In.Len().
	ErrInput = errors.New("qnet: input length mismatch")
	// ErrNotReady reports a Runner used before a successful Init.
	ErrNotReady = errors.New("qnet: runner not initialised")
	// ErrSegment reports a backend asked to run a segment it cannot run.
	ErrSegment = errors.New("qnet: unsupported segment")
)

// Backend executes runs of layers. Implementations must not allocate in Run.
type Backend interface {
	// Name identifies the backend in plans.
	Name() string
	// Claim returns how many consecutive layers starting at layers[0] this
	// backend will run as one segment (0 = none). A KPU backend can claim
	// whole runs of conv/depthwise layers it can fuse; the generic backend
	// claims 1 layer.
	Claim(layers []Layer) int
	// Run executes the claimed segment from src to dst (NHWC int8, quantised
	// as the first layer's In and the last layer's Out). No allocation.
	Run(layers []Layer, dst, src []int8) error
}

// Generic runs every layer kind with the qi8 reference kernels, one layer
// per segment.
type Generic struct{}

// Name returns "generic".
func (Generic) Name() string { return "generic" }

// Claim returns 1 for any known kind, else 0.
func (Generic) Claim(layers []Layer) int {
	if len(layers) == 0 {
		return 0
	}
	switch layers[0].Kind {
	case Conv2D, DepthwiseConv2D, FullyConnected, MaxPool2D, AvgPool2D, ReLU, ReLU6:
		return 1
	}
	return 0
}

// Run executes a single layer (ErrSegment for any other count).
func (Generic) Run(layers []Layer, dst, src []int8) error {
	if len(layers) != 1 {
		return ErrSegment
	}
	return RunLayer(&layers[0], dst, src)
}

// MaxSegments is the plan capacity of a Runner.
const MaxSegments = 64

// Segment is one planned step: Backend runs Layers[First:First+Count].
type Segment struct {
	Backend Backend
	First   int
	Count   int
}

// Runner executes a validated Graph with a fixed plan over a caller-provided
// ping-pong arena. Segment k writes its output at the end of the arena when k
// is even and at the start when k is odd; segment 0 reads the caller's input.
// The graph's layers must not be changed while the Runner is in use.
type Runner struct {
	g     *Graph
	arena []int8
	plan  [MaxSegments]Segment
	n     int
	need  int
}

// NewRunner allocates a Runner and calls Init. Use Init on a Runner you own
// (e.g. a global) for an allocation-free setup.
func NewRunner(g *Graph, arena []int8, backends ...Backend) (*Runner, error) {
	r := new(Runner)
	if err := r.Init(g, arena, backends...); err != nil {
		return nil, err
	}
	return r, nil
}

// Init validates g (parameters must be bound), plans segments and checks the
// arena. Planning is greedy: at each layer the backends are asked in order and
// the first non-zero Claim forms the next segment. Errors: graph validation
// errors, ErrUnclaimed, ErrClaim, ErrCapacity (more than MaxSegments
// segments), ErrArena (len(arena) < ArenaSize). Init does not allocate. On
// error r is left uninitialised.
func (r *Runner) Init(g *Graph, arena []int8, backends ...Backend) error {
	r.g, r.arena, r.n, r.need = nil, nil, 0, 0
	if err := g.Validate(); err != nil {
		return err
	}
	n, need := 0, 0
	for i := 0; i < len(g.Layers); {
		var seg Segment
		for _, b := range backends {
			c := b.Claim(g.Layers[i:])
			if c == 0 {
				continue
			}
			if c < 0 || c > len(g.Layers)-i {
				return ErrClaim
			}
			seg = Segment{Backend: b, First: i, Count: c}
			break
		}
		if seg.Count == 0 {
			return ErrUnclaimed
		}
		if n == MaxSegments {
			return ErrCapacity
		}
		r.plan[n] = seg
		n++
		need = max(need, boundaryBytes(g.Layers, seg.First, seg.First+seg.Count))
		i += seg.Count
	}
	if len(arena) < need {
		return ErrArena
	}
	r.g, r.arena, r.n, r.need = g, arena, n, need
	return nil
}

// Plan returns the planned segments (aliases the Runner; do not modify).
func (r *Runner) Plan() []Segment { return r.plan[:r.n] }

// ArenaSize returns the arena bytes the plan needs (<= Graph.ArenaSize).
func (r *Runner) ArenaSize() int { return r.need }

// Run executes the plan on input (Layers[0].In.Len() bytes, must not overlap
// the arena) and returns the output, which aliases the arena and is valid
// until the next Run. Run does not allocate.
func (r *Runner) Run(input []int8) ([]int8, error) {
	if r.g == nil {
		return nil, ErrNotReady
	}
	layers := r.g.Layers
	if len(input) != layers[0].In.Len() {
		return nil, ErrInput
	}
	src := input
	for k := range r.n {
		s := &r.plan[k]
		seg := layers[s.First : s.First+s.Count]
		m := seg[len(seg)-1].Out.Len()
		var dst []int8
		if k%2 == 0 {
			dst = r.arena[len(r.arena)-m:]
		} else {
			dst = r.arena[:m]
		}
		if err := s.Backend.Run(seg, dst, src); err != nil {
			return nil, err
		}
		src = dst
	}
	return src, nil
}
