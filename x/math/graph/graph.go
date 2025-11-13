package graph

// Node represents a node in a graph
type Node interface {
	// Equal checks if two nodes are the same
	Equal(other Node) bool
}

// Graph provides neighbors and edge costs
type Graph interface {
	Neighbors(n Node) []Node
	Cost(from, to Node) float32
}

// GenericGraph is an optimized generic graph implementation
// Uses adjacency lists internally for fast neighbor lookup
type GenericGraph struct {
	neighbors map[Node][]Node
	costs     map[Node]map[Node]float32
}

// NewGenericGraph creates a new generic graph
func NewGenericGraph() *GenericGraph {
	return &GenericGraph{
		neighbors: make(map[Node][]Node),
		costs:     make(map[Node]map[Node]float32),
	}
}

// AddEdge adds a directed edge from 'from' to 'to' with given cost
func (g *GenericGraph) AddEdge(from, to Node, cost float32) {
	g.neighbors[from] = append(g.neighbors[from], to)
	if g.costs[from] == nil {
		g.costs[from] = make(map[Node]float32)
	}
	g.costs[from][to] = cost
}

// AddBidirectionalEdge adds edges in both directions with the same cost
func (g *GenericGraph) AddBidirectionalEdge(from, to Node, cost float32) {
	g.AddEdge(from, to, cost)
	g.AddEdge(to, from, cost)
}

func (g *GenericGraph) Neighbors(n Node) []Node {
	return g.neighbors[n]
}

func (g *GenericGraph) Cost(from, to Node) float32 {
	if g.costs[from] == nil {
		return 0
	}
	cost, exists := g.costs[from][to]
	if !exists {
		return 0
	}
	return cost
}
