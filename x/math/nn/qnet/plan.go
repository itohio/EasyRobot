package qnet

import (
	"sort"
	"unsafe"

	"github.com/itohio/EasyRobot/x/math/primitive/qi8"
)

// Region is a byte range of the (16-byte-aligned) arena.
type Region struct {
	Off, Len int
}

// MemPlan is the execution and memory plan of a graph: which layers run as
// which segment, and where every tensor and every CPU layer's scratch live
// in one arena.
//
// The arena (after aligning its start to 16 bytes) holds the persistent
// region first — the scratch of CPU layers whose qi8 backend repacks
// weights (the ESP32-S3 PIE build) — then the activation region. Activations
// are segment outputs; a tensor's bytes are reused once every segment that
// reads it has run, placed greedily by size (largest first, lowest offset
// that does not overlap a tensor with an overlapping lifetime). Reshape
// outputs alias their input; the graph input is the caller's buffer; Const
// outputs are their Weights.
type MemPlan struct {
	Segments []Segment
	// Scratch is per layer (Len 0: none).
	Scratch []Region
	// Out is per layer: Off −1 means the caller's input (a Reshape of it),
	// Off −2 a Const's Weights (or a Reshape of a Const); Len 0 means the
	// tensor stays inside a backend's fused segment.
	Out []Region
	// Persistent and Activations are the two regions' sizes in bytes.
	Persistent, Activations int
}

const arenaAlign = 16

func alignUp(n int) int { return (n + arenaAlign - 1) &^ (arenaAlign - 1) }

// Size returns the arena bytes the plan needs, including slack to align an
// arbitrary buffer's start.
func (p *MemPlan) Size() int { return arenaAlign - 1 + p.Persistent + p.Activations }

// align returns arena from its first 16-byte-aligned element.
func align(arena []int8) []int8 {
	if len(arena) == 0 {
		return arena
	}
	off := int((arenaAlign - uintptr(unsafe.Pointer(&arena[0]))%arenaAlign) % arenaAlign)
	if off > len(arena) {
		return arena[:0]
	}
	return arena[off:]
}

// ScratchSlice returns layer i's scratch inside arena (nil when it has
// none). Before NewRunner, a caller may place the layer's weights at its
// start (Weights = ScratchSlice(...)[:n]) so that the backend packs them in
// place and only one copy exists.
func (p *MemPlan) ScratchSlice(arena []int8, i int) []int8 {
	r := p.Scratch[i]
	if r.Len == 0 {
		return nil
	}
	return align(arena)[r.Off : r.Off+r.Len : r.Off+r.Len]
}

// PlanMemory validates g's shapes (parameters need not be bound) and plans
// segments and memory for the given backends. Planning is greedy: at each
// layer the backends are asked in order and the first non-zero Claim forms
// the next segment; layers nobody claims run alone on the built-in CPU
// kernels (qi8 prepared layers). A claim must cover single-input layers
// that read only their predecessor, and no layer outside it may read the
// claim's intermediate outputs (ErrClaim otherwise).
func PlanMemory(g *Graph, backends ...Backend) (*MemPlan, error) {
	if err := g.checkShapes(); err != nil {
		return nil, err
	}
	n := len(g.Layers)
	p := &MemPlan{Scratch: make([]Region, n), Out: make([]Region, n)}

	readers := make([]int, n) // last layer that reads each layer's output
	for i := range readers {
		readers[i] = -1
	}
	for i := range g.Layers {
		for k := range g.inputs(i) {
			if pr := g.producer(i, k); pr >= 0 {
				readers[pr] = max(readers[pr], i)
			}
		}
	}
	isOut := make([]bool, n) // graph outputs: live to the end, never fused away
	for k := range g.NumOutputs() {
		isOut[g.outputLayer(k)] = true
	}
	segOf := make([]int, n)
	for i := 0; i < n; {
		seg := Segment{First: i, Count: 1}
		for _, b := range backends {
			c := b.Claim(g.Layers[i:])
			if c == 0 {
				continue
			}
			if c < 0 || c > n-i || !g.chain(i, c, readers, isOut) {
				return nil, ErrClaim
			}
			seg = Segment{Backend: b, First: i, Count: c}
			break
		}
		for k := range seg.Count {
			segOf[i+k] = len(p.Segments)
		}
		p.Segments = append(p.Segments, seg)
		i += seg.Count
	}

	// Persistent: CPU layers' backend scratch.
	for _, s := range p.Segments {
		if s.Backend != nil {
			continue
		}
		if m := scratchLen(&g.Layers[s.First]); m > 0 {
			p.Scratch[s.First] = Region{Off: p.Persistent, Len: m}
			p.Persistent += alignUp(m)
		}
	}

	// Tensors: owner[i] is the layer whose arena tensor holds i's output
	// (−1: the caller's input; −2: inside a fused segment; −3: constant).
	owner := make([]int, n)
	born := make([]int, n)
	last := make([]int, n)
	for i := range g.Layers {
		s := &p.Segments[segOf[i]]
		switch {
		case i != s.First+s.Count-1:
			owner[i] = -2
		case s.Backend == nil && g.Layers[i].Kind == Const:
			owner[i] = -3
		case s.Backend == nil && g.Layers[i].Kind == Reshape:
			owner[i] = -1
			if pr := g.producer(i, 0); pr >= 0 {
				owner[i] = owner[pr]
			}
		default:
			owner[i] = i
			born[i] = segOf[i]
			last[i] = segOf[i]
		}
	}
	for i := range g.Layers {
		if o := owner[i]; o >= 0 && readers[i] >= 0 {
			last[o] = max(last[o], segOf[readers[i]])
		}
	}
	for i, out := range isOut {
		if o := owner[i]; out && o >= 0 {
			last[o] = len(p.Segments) // graph outputs stay live
		}
	}

	var ts []int
	for i := range owner {
		if owner[i] == i {
			ts = append(ts, i)
		}
	}
	sort.SliceStable(ts, func(a, b int) bool {
		la, lb := g.Layers[ts[a]].Out.Len(), g.Layers[ts[b]].Out.Len()
		if la != lb {
			return la > lb
		}
		return born[ts[a]] < born[ts[b]]
	})
	off := make([]int, n)
	var placed []int
	for _, t := range ts {
		size := alignUp(g.Layers[t].Out.Len())
		var busy []Region
		for _, q := range placed {
			if born[t] <= last[q] && born[q] <= last[t] {
				busy = append(busy, Region{Off: off[q], Len: alignUp(g.Layers[q].Out.Len())})
			}
		}
		sort.Slice(busy, func(a, b int) bool { return busy[a].Off < busy[b].Off })
		o := 0
		for _, r := range busy {
			if o+size <= r.Off {
				break
			}
			o = max(o, r.Off+r.Len)
		}
		off[t] = o
		placed = append(placed, t)
		p.Activations = max(p.Activations, o+size)
	}
	for i := range g.Layers {
		switch o := owner[i]; {
		case o == -1:
			p.Out[i] = Region{Off: -1, Len: g.Layers[i].Out.Len()}
		case o == -3:
			p.Out[i] = Region{Off: -2, Len: g.Layers[i].Out.Len()}
		case o >= 0:
			p.Out[i] = Region{Off: p.Persistent + off[o], Len: g.Layers[i].Out.Len()}
		}
	}
	return p, nil
}

// chain reports whether layers [i, i+c) form a claimable chain: single-input
// layers, each after the first reading its predecessor, and no intermediate
// output read outside the chain or returned by the graph.
func (g *Graph) chain(i, c int, readers []int, isOut []bool) bool {
	for k := i; k < i+c; k++ {
		if g.inputs(k) != 1 {
			return false
		}
		if k > i && g.producer(k, 0) != k-1 {
			return false
		}
		if k < i+c-1 && (readers[k] >= i+c || isOut[k]) {
			return false
		}
	}
	return true
}

// scratchLen is the qi8 backend scratch of a CPU layer (0 for the generic
// build and parameter-free kinds).
func scratchLen(l *Layer) int {
	switch l.Kind {
	case Conv2D:
		return (&qi8.Conv2DLayer{P: l.Conv2DParams()}).ScratchLen()
	case DepthwiseConv2D:
		return (&qi8.DepthwiseConv2DLayer{P: l.DepthwiseParams()}).ScratchLen()
	case FullyConnected:
		return (&qi8.FCLayer{P: l.FCParams()}).ScratchLen()
	}
	return 0
}
