package graph

import "iter"

// GenericNode is a concrete implementation of Node[N, E]
// It stores a reference to its index in the graph's node array
type GenericNode[N any, E any] struct {
	id           int64
	data         N
	nodeIdx      int   // Index in graph.nodes array
	edgeIdxs     []int // Indices in graph.edges array
	neighborIdxs []int // Indices of neighbor nodes in graph.nodes array
	graph        *GenericGraph[N, E]
}

// ID returns the unique identifier for this node
func (n *GenericNode[N, E]) ID() int64 {
	return n.id
}

// Equal implements Comparable[N, E]
func (n *GenericNode[N, E]) Equal(other Node[N, E]) bool {
	if other == nil {
		return false
	}
	// Compare by ID for efficiency
	return n.id == other.ID()
}

// Compare implements Comparable[N, E]
func (n *GenericNode[N, E]) Compare(other Node[N, E]) int {
	if other == nil {
		return 1
	}
	// Compare by ID
	if n.id < other.ID() {
		return -1
	}
	if n.id > other.ID() {
		return 1
	}
	return 0
}

// Data returns the node's data
func (n *GenericNode[N, E]) Data() N {
	return n.data
}

// setData updates the node's data (internal use)
func (n *GenericNode[N, E]) setData(data N) {
	n.data = data
}

// Neighbors returns an iterator over neighboring nodes
func (n *GenericNode[N, E]) Neighbors() iter.Seq[Node[N, E]] {
	return func(yield func(Node[N, E]) bool) {
		if n.graph == nil {
			return
		}
		for _, idx := range n.neighborIdxs {
			if idx >= 0 && idx < len(n.graph.nodes) {
				if !yield(&n.graph.nodes[idx]) {
					return
				}
			}
		}
	}
}

// Edges returns an iterator over edges from this node
func (n *GenericNode[N, E]) Edges() iter.Seq[Edge[N, E]] {
	return func(yield func(Edge[N, E]) bool) {
		if n.graph == nil {
			return
		}
		for _, idx := range n.edgeIdxs {
			if idx >= 0 && idx < len(n.graph.edges) {
				if !yield(&n.graph.edges[idx]) {
					return
				}
			}
		}
	}
}

// NumNeighbors returns the number of neighboring nodes
func (n *GenericNode[N, E]) NumNeighbors() int {
	if n.graph == nil {
		return 0
	}
	count := 0
	for _, idx := range n.neighborIdxs {
		if idx >= 0 && idx < len(n.graph.nodes) {
			count++
		}
	}
	return count
}

// Cost calculates the cost from this node to another node
func (n *GenericNode[N, E]) Cost(toOther Node[N, E]) float32 {
	if toOther == nil || n.graph == nil {
		return 0
	}
	// Find edge from this node to the other node
	for edge := range n.Edges() {
		if edge.To().ID() == toOther.ID() {
			return edge.Cost()
		}
	}
	// If no edge found, use cost calculator if available
	if n.graph.costCalculator != nil {
		return n.graph.costCalculator(n, toOther)
	}
	return 0
}

// AddNeighbor adds a neighbor to this node
func (n *GenericNode[N, E]) AddNeighbor(neighbor Node[N, E]) error {
	if neighbor == nil || n.graph == nil {
		return nil
	}
	neigh, ok := neighbor.(*GenericNode[N, E])
	if !ok {
		return nil // Only support GenericNode for now
	}
	// Check if already a neighbor
	for _, idx := range n.neighborIdxs {
		if idx == neigh.nodeIdx {
			return nil // Already a neighbor
		}
	}
	n.neighborIdxs = append(n.neighborIdxs, neigh.nodeIdx)
	return nil
}

// DeleteNeighbor removes a neighbor from this node
func (n *GenericNode[N, E]) DeleteNeighbor(neighbor Node[N, E]) error {
	if neighbor == nil {
		return nil
	}
	neigh, ok := neighbor.(*GenericNode[N, E])
	if !ok {
		return nil
	}
	for i, idx := range n.neighborIdxs {
		if idx == neigh.nodeIdx {
			n.neighborIdxs = append(n.neighborIdxs[:i], n.neighborIdxs[i+1:]...)
			break
		}
	}
	return nil
}

// UpdateNeighbor updates a neighbor relationship
func (n *GenericNode[N, E]) UpdateNeighbor(oldNeighbor, newNeighbor Node[N, E]) error {
	if err := n.DeleteNeighbor(oldNeighbor); err != nil {
		return err
	}
	return n.AddNeighbor(newNeighbor)
}

// GenericEdge is a concrete implementation of Edge[N, E]
// It stores indices to nodes in the graph's node array
type GenericEdge[N any, E any] struct {
	id      int64
	fromIdx int // Index in graph.nodes array
	toIdx   int // Index in graph.nodes array
	data    E
	graph   *GenericGraph[N, E]
}

// ID returns the unique identifier for this edge
func (e *GenericEdge[N, E]) ID() int64 {
	return e.id
}

// From returns the source node
func (e *GenericEdge[N, E]) From() Node[N, E] {
	if e.graph == nil || e.fromIdx < 0 || e.fromIdx >= len(e.graph.nodes) {
		return nil
	}
	return &e.graph.nodes[e.fromIdx]
}

// To returns the destination node
func (e *GenericEdge[N, E]) To() Node[N, E] {
	if e.graph == nil || e.toIdx < 0 || e.toIdx >= len(e.graph.nodes) {
		return nil
	}
	return &e.graph.nodes[e.toIdx]
}

// Data returns the edge's data
func (e *GenericEdge[N, E]) Data() E {
	return e.data
}

// Cost returns the cost/weight of this edge
func (e *GenericEdge[N, E]) Cost() float32 {
	if e.graph == nil {
		return 0
	}
	if e.graph.costCalculator != nil {
		from := e.From()
		to := e.To()
		if from != nil && to != nil {
			return e.graph.costCalculator(from, to)
		}
	}
	// Try to convert edge data to float32
	switch v := any(e.data).(type) {
	case float32:
		return v
	case float64:
		return float32(v)
	case int:
		return float32(v)
	case int32:
		return float32(v)
	case int64:
		return float32(v)
	}
	return 0
}

// setData updates the edge's data (internal use)
func (e *GenericEdge[N, E]) setData(data E) {
	e.data = data
}

// GenericGraph is an optimized generic graph implementation
// Stores nodes and edges as values in arrays for cache locality
type GenericGraph[N any, E any] struct {
	nodes          []GenericNode[N, E] // Value array
	edges          []GenericEdge[N, E] // Value array
	nodeMap        map[int64]int       // ID -> index in nodes array
	edgeMap        map[int64]int       // ID -> index in edges array
	nextNodeID     int64
	nextEdgeID     int64
	costCalculator func(from, to Node[N, E]) float32
}

// NewGenericGraph creates a new generic graph
func NewGenericGraph[N any, E any]() *GenericGraph[N, E] {
	return &GenericGraph[N, E]{
		nodes:      make([]GenericNode[N, E], 0),
		edges:      make([]GenericEdge[N, E], 0),
		nodeMap:    make(map[int64]int),
		edgeMap:    make(map[int64]int),
		nextNodeID: 1,
		nextEdgeID: 1,
	}
}

// Nodes returns an iterator over all nodes
func (g *GenericGraph[N, E]) Nodes() iter.Seq[Node[N, E]] {
	return func(yield func(Node[N, E]) bool) {
		for i := range g.nodes {
			if !yield(&g.nodes[i]) {
				return
			}
		}
	}
}

// Edges returns an iterator over all edges
func (g *GenericGraph[N, E]) Edges() iter.Seq[Edge[N, E]] {
	return func(yield func(Edge[N, E]) bool) {
		for i := range g.edges {
			if !yield(&g.edges[i]) {
				return
			}
		}
	}
}

// NumNodes returns the total number of nodes in the graph
func (g *GenericGraph[N, E]) NumNodes() int {
	return len(g.nodes)
}

// NumEdges returns the total number of edges in the graph
func (g *GenericGraph[N, E]) NumEdges() int {
	return len(g.edges)
}

// SetCostCalculator sets the cost calculator function
func (g *GenericGraph[N, E]) SetCostCalculator(calculator func(from, to Node[N, E]) float32) {
	g.costCalculator = calculator
}

// AddNode adds a node to the graph
func (g *GenericGraph[N, E]) AddNode(node Node[N, E]) error {
	gn, ok := node.(*GenericNode[N, E])
	if !ok {
		// Create a new GenericNode from the provided node
		nodeID := node.ID()
		if nodeID == 0 {
			nodeID = g.nextNodeID
			g.nextNodeID++
		}
		if _, exists := g.nodeMap[nodeID]; exists {
			return nil // Node already exists
		}
		idx := len(g.nodes)
		g.nodes = append(g.nodes, GenericNode[N, E]{
			id:           nodeID,
			data:         node.Data(),
			nodeIdx:      idx,
			edgeIdxs:     make([]int, 0),
			neighborIdxs: make([]int, 0),
			graph:        g,
		})
		g.nodeMap[nodeID] = idx
		return nil
	}

	// If it's already a GenericNode, check if it's in this graph
	if gn.graph == g {
		return nil // Already in graph
	}

	// Add to graph
	nodeID := gn.id
	if nodeID == 0 {
		nodeID = g.nextNodeID
		g.nextNodeID++
		gn.id = nodeID
	}
	if _, exists := g.nodeMap[nodeID]; exists {
		return nil // Node already exists
	}

	idx := len(g.nodes)
	gn.nodeIdx = idx
	gn.graph = g
	g.nodes = append(g.nodes, *gn)
	g.nodeMap[nodeID] = idx
	return nil
}

// AddEdge adds an edge to the graph
func (g *GenericGraph[N, E]) AddEdge(edge Edge[N, E]) error {
	if edge == nil {
		return nil
	}

	if ge, ok := edge.(*GenericEdge[N, E]); ok {
		return g.addGenericEdge(ge)
	}

	return g.addEdgeFromInterface(edge)
}

func (g *GenericGraph[N, E]) addEdgeFromInterface(edge Edge[N, E]) error {
	edgeID := edge.ID()
	if edgeID == 0 {
		edgeID = g.nextEdgeID
		g.nextEdgeID++
	} else if _, exists := g.edgeMap[edgeID]; exists {
		return nil
	}

	from := edge.From()
	to := edge.To()
	if from == nil || to == nil {
		return nil
	}

	g.AddNode(from)
	g.AddNode(to)

	fromIdx, ok := g.nodeIndex(from)
	if !ok {
		return nil
	}
	toIdx, ok := g.nodeIndex(to)
	if !ok {
		return nil
	}

	idx := len(g.edges)
	g.edges = append(g.edges, GenericEdge[N, E]{
		id:      edgeID,
		fromIdx: fromIdx,
		toIdx:   toIdx,
		data:    edge.Data(),
		graph:   g,
	})
	g.edgeMap[edgeID] = idx
	g.nodes[fromIdx].edgeIdxs = append(g.nodes[fromIdx].edgeIdxs, idx)
	g.nodes[fromIdx].AddNeighbor(&g.nodes[toIdx])
	return nil
}

func (g *GenericGraph[N, E]) addGenericEdge(edge *GenericEdge[N, E]) error {
	edgeID := edge.id
	if edgeID == 0 {
		edgeID = g.nextEdgeID
		g.nextEdgeID++
	} else if _, exists := g.edgeMap[edgeID]; exists {
		return nil
	}

	fromIdx := edge.fromIdx
	toIdx := edge.toIdx

	if edge.graph != nil && edge.graph != g {
		if from := edge.From(); from != nil {
			g.AddNode(from)
			if idx, ok := g.nodeIndex(from); ok {
				fromIdx = idx
			}
		}
		if to := edge.To(); to != nil {
			g.AddNode(to)
			if idx, ok := g.nodeIndex(to); ok {
				toIdx = idx
			}
		}
	} else {
		if from := edge.From(); from != nil {
			g.AddNode(from)
			if idx, ok := g.nodeIndex(from); ok {
				fromIdx = idx
			}
		}
		if to := edge.To(); to != nil {
			g.AddNode(to)
			if idx, ok := g.nodeIndex(to); ok {
				toIdx = idx
			}
		}
	}

	if fromIdx < 0 || fromIdx >= len(g.nodes) || toIdx < 0 || toIdx >= len(g.nodes) {
		return nil
	}

	idx := len(g.edges)
	g.edges = append(g.edges, GenericEdge[N, E]{
		id:      edgeID,
		fromIdx: fromIdx,
		toIdx:   toIdx,
		data:    edge.data,
		graph:   g,
	})
	g.edgeMap[edgeID] = idx
	g.nodes[fromIdx].edgeIdxs = append(g.nodes[fromIdx].edgeIdxs, idx)
	g.nodes[fromIdx].AddNeighbor(&g.nodes[toIdx])
	return nil
}

func (g *GenericGraph[N, E]) nodeIndex(node Node[N, E]) (int, bool) {
	if node == nil {
		return -1, false
	}
	if gn, ok := node.(*GenericNode[N, E]); ok && gn.graph == g {
		if gn.nodeIdx >= 0 && gn.nodeIdx < len(g.nodes) {
			return gn.nodeIdx, true
		}
	}
	if idx, ok := g.nodeMap[node.ID()]; ok {
		return idx, true
	}
	return -1, false
}

// DeleteNode removes a node from the graph
func (g *GenericGraph[N, E]) DeleteNode(node Node[N, E]) error {
	nodeID := node.ID()
	idx, exists := g.nodeMap[nodeID]
	if !exists {
		return nil // Node not in graph
	}

	// Remove all edges connected to this node
	newEdges := make([]GenericEdge[N, E], 0)
	newEdgeIdxs := make([]int, 0)
	for i, edge := range g.edges {
		edgeFrom := edge.From()
		edgeTo := edge.To()
		if edgeFrom != nil && edgeFrom.ID() == nodeID {
			continue // Skip edge from this node
		}
		if edgeTo != nil && edgeTo.ID() == nodeID {
			continue // Skip edge to this node
		}
		newEdges = append(newEdges, edge)
		newEdgeIdxs = append(newEdgeIdxs, i)
	}
	g.edges = newEdges

	// Rebuild edgeMap
	g.edgeMap = make(map[int64]int)
	for i, edge := range g.edges {
		g.edgeMap[edge.id] = i
	}

	// Remove from nodes list and update indices
	newNodes := make([]GenericNode[N, E], 0)
	for i, n := range g.nodes {
		if i == idx {
			continue // Skip deleted node
		}
		// Update node index
		n.nodeIdx = len(newNodes)
		// Update edge indices in node
		newEdgeIdxs := make([]int, 0)
		for _, edgeIdx := range n.edgeIdxs {
			// Find new index for this edge
			for newIdx, oldIdx := range newEdgeIdxs {
				if oldIdx == edgeIdx {
					newEdgeIdxs = append(newEdgeIdxs, newIdx)
					break
				}
			}
		}
		n.edgeIdxs = newEdgeIdxs
		// Update neighbor indices
		newNeighborIdxs := make([]int, 0)
		for _, neighborIdx := range n.neighborIdxs {
			if neighborIdx < idx {
				newNeighborIdxs = append(newNeighborIdxs, neighborIdx)
			} else if neighborIdx > idx {
				newNeighborIdxs = append(newNeighborIdxs, neighborIdx-1)
			}
		}
		n.neighborIdxs = newNeighborIdxs
		newNodes = append(newNodes, n)
	}
	g.nodes = newNodes

	// Rebuild nodeMap
	g.nodeMap = make(map[int64]int)
	for i, n := range g.nodes {
		g.nodeMap[n.id] = i
	}

	return nil
}

// DeleteEdge removes an edge from the graph
func (g *GenericGraph[N, E]) DeleteEdge(edge Edge[N, E]) error {
	edgeID := edge.ID()
	idx, exists := g.edgeMap[edgeID]
	if !exists {
		return nil // Edge not in graph
	}

	// Remove from graph's edges
	newEdges := make([]GenericEdge[N, E], 0)
	for i, e := range g.edges {
		if i == idx {
			continue
		}
		newEdges = append(newEdges, e)
	}
	g.edges = newEdges

	// Rebuild edgeMap
	g.edgeMap = make(map[int64]int)
	for i, edge := range g.edges {
		g.edgeMap[edge.id] = i
	}

	// Remove from from node's edges and neighbors
	ge := &g.edges[idx]
	fromIdx := ge.fromIdx
	if fromIdx >= 0 && fromIdx < len(g.nodes) {
		// Remove edge index from node
		newEdgeIdxs := make([]int, 0)
		for _, edgeIdx := range g.nodes[fromIdx].edgeIdxs {
			if edgeIdx < idx {
				newEdgeIdxs = append(newEdgeIdxs, edgeIdx)
			} else if edgeIdx > idx {
				newEdgeIdxs = append(newEdgeIdxs, edgeIdx-1)
			}
		}
		g.nodes[fromIdx].edgeIdxs = newEdgeIdxs
		// Remove neighbor
		toIdx := ge.toIdx
		if toIdx >= 0 && toIdx < len(g.nodes) {
			g.nodes[fromIdx].DeleteNeighbor(&g.nodes[toIdx])
		}
	}

	return nil
}

// UpdateNode updates a node's data
func (g *GenericGraph[N, E]) UpdateNode(node Node[N, E]) error {
	nodeID := node.ID()
	idx, exists := g.nodeMap[nodeID]
	if !exists {
		return nil // Node not in graph
	}
	g.nodes[idx].setData(node.Data())
	return nil
}

// UpdateEdge updates an edge's data
func (g *GenericGraph[N, E]) UpdateEdge(edge Edge[N, E]) error {
	edgeID := edge.ID()
	idx, exists := g.edgeMap[edgeID]
	if !exists {
		return nil // Edge not in graph
	}
	g.edges[idx].setData(edge.Data())
	return nil
}
