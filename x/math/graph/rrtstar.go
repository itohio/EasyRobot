package graph

import (
	"math/rand"
)

// RRTStarGraph is the graph interface required by RRT*
// It combines Graph, Adder, and Deleter into a single constraint
// Nodes must implement NodeAdder for connection management
type RRTStarGraph[N any] interface {
	Graph[N, float32]
	Adder[N, float32]
	Deleter[N, float32]
}

// Sampler generates random nodes in the search space
type Sampler[N any] func() Node[N, float32]

// GoalChecker determines if a node has reached the goal
type GoalChecker[N any] func(node, goal Node[N, float32]) bool

// RRTStar is a Rapidly-exploring Random Tree Star search algorithm implementation
// It builds a tree by sampling the search space and connecting nodes
// Node data type N should have a Distance method (signature may vary, e.g., Distance(vecTypes.Vector) or Distance(Self))
// Edge type is always float32 (cost/distance)
// Distance calculation uses node.Cost() method which should use the graph's cost calculator
type RRTStar[N any] struct {
	graph       RRTStarGraph[N]
	sampler     Sampler[N]
	goalChecker GoalChecker[N]

	// Algorithm parameters
	rewireRadius  float32 // Radius for rewiring optimization
	maxIterations int     // Maximum iterations before giving up

	// Reusable buffers
	nodeMap map[int64]Node[N, float32] // ID -> Node for quick lookup
	rng     *rand.Rand                 // Random number generator
}

// RRTStarOptions configures RRT* algorithm parameters
type RRTStarOptions struct {
	RewireRadius  float32 // Radius for rewiring optimization (default: 0.0 = auto)
	MaxIterations int     // Maximum iterations (default: 1000)
	Seed          int64   // Random seed (default: 0 = time-based)
}

// DefaultRRTStarOptions returns default options for RRT*
func DefaultRRTStarOptions() RRTStarOptions {
	return RRTStarOptions{
		RewireRadius:  0.0, // Auto-calculate based on search space
		MaxIterations: 1000,
		Seed:          0,
	}
}

// NewRRTStar creates a new RRT* search instance
// graph must implement RRTStarGraph[N] which combines Graph, Adder, and Deleter
// The graph should have a cost calculator set up for distance calculation
func NewRRTStar[N any](
	graph RRTStarGraph[N],
	sampler Sampler[N],
	goalChecker GoalChecker[N],
	opts ...RRTStarOptions,
) *RRTStar[N] {
	options := DefaultRRTStarOptions()
	if len(opts) > 0 {
		options = opts[0]
	}

	var rng *rand.Rand
	if options.Seed == 0 {
		rng = rand.New(rand.NewSource(rand.Int63()))
	} else {
		rng = rand.New(rand.NewSource(options.Seed))
	}

	return &RRTStar[N]{
		graph:         graph,
		sampler:       sampler,
		goalChecker:   goalChecker,
		rewireRadius:  options.RewireRadius,
		maxIterations: options.MaxIterations,
		nodeMap:       make(map[int64]Node[N, float32]),
		rng:           rng,
	}
}

// Search finds a path from start to goal using RRT* algorithm
// Returns path as list of nodes, or nil if no path exists within max iterations
func (r *RRTStar[N]) Search(start, goal Node[N, float32]) Path[N, float32] {
	if start == nil || goal == nil || r.graph == nil {
		return nil
	}

	// Clear buffers for reuse
	r.clear()

	// Add start node to graph if not already present
	start = r.ensureNodeInGraph(start)
	if start == nil {
		return nil
	}

	// Cache goal for goal checking
	goalID := goal.ID()
	r.nodeMap[goalID] = goal

	// Initialize tree with start node
	startID := start.ID()
	r.nodeMap[startID] = start

	// Calculate rewiring radius if auto
	rewireRadius := r.rewireRadius
	if rewireRadius <= 0 {
		// Auto-calculate: use average distance from start to a few samples
		rewireRadius = r.estimateRewireRadius(start)
	}

	// Main RRT* loop
	for iteration := 0; iteration < r.maxIterations; iteration++ {
		// Sample random node
		sampleNode := r.sampler()
		if sampleNode == nil {
			continue
		}

		// Find nearest node in tree
		nearest := r.findNearest(sampleNode)
		if nearest == nil {
			continue
		}

		// Ensure nearest is in graph (should already be, but ensure anyway)
		nearest = r.ensureNodeInGraph(nearest)
		if nearest == nil {
			continue
		}
		r.nodeMap[nearest.ID()] = nearest

		// Ensure sample node is in graph first
		newNode := r.ensureNodeInGraph(sampleNode)
		if newNode == nil {
			continue
		}
		r.nodeMap[newNode.ID()] = newNode

		// Connect nearest to newNode using NodeAdder
		if err := r.connectNodes(nearest, newNode); err != nil {
			continue // Connection failed, skip this sample
		}

		// Rewire: try to improve paths through new node
		r.rewire(newNode, rewireRadius)

		// Check if goal is reached
		if r.goalChecker(newNode, goal) {
			// Try to connect to goal
			goalNode := r.ensureNodeInGraph(goal)
			if goalNode != nil {
				if r.connectNodes(newNode, goalNode) == nil {
					return r.reconstructPath(goalNode)
				}
			}
		}
	}

	// Check if we can reach goal from any node in tree
	goalNode := r.ensureNodeInGraph(goal)
	if goalNode != nil {
		nearestToGoal := r.findNearest(goalNode)
		if nearestToGoal != nil {
			if r.connectNodes(nearestToGoal, goalNode) == nil {
				return r.reconstructPath(goalNode)
			}
		}
	}

	return nil
}

// findNearest finds the nearest node in the tree to the given node
func (r *RRTStar[N]) findNearest(target Node[N, float32]) Node[N, float32] {
	var nearest Node[N, float32]
	minDist := float32(1e9)

	for _, node := range r.nodeMap {
		// Use node.Cost() for distance calculation
		dist := node.Cost(target)
		if dist < minDist {
			minDist = dist
			nearest = node
		}
	}

	return nearest
}

// rrtStarEdge is a simple edge implementation for RRT* that doesn't depend on GenericEdge
// It implements Edge[N, float32] interface
type rrtStarEdge[N any] struct {
	from Node[N, float32]
	to   Node[N, float32]
	cost float32
}

func (e *rrtStarEdge[N]) ID() int64 {
	return 0 // Graph will assign ID
}

func (e *rrtStarEdge[N]) From() Node[N, float32] {
	return e.from
}

func (e *rrtStarEdge[N]) To() Node[N, float32] {
	return e.to
}

func (e *rrtStarEdge[N]) Data() float32 {
	return e.cost
}

func (e *rrtStarEdge[N]) Cost() float32 {
	return e.cost
}

// connectNodes connects two nodes using NodeAdder
// The from node decides how to add the neighbor, and we also create an edge for the graph
func (r *RRTStar[N]) connectNodes(from, to Node[N, float32]) error {
	if from == nil || to == nil {
		return nil
	}

	// Ensure both nodes are in the graph first
	from = r.ensureNodeInGraph(from)
	to = r.ensureNodeInGraph(to)
	if from == nil || to == nil {
		return nil
	}

	// Calculate cost for the edge
	cost := from.Cost(to)

	// Create edge using simple implementation (doesn't depend on GenericEdge)
	edge := &rrtStarEdge[N]{
		from: from,
		to:   to,
		cost: cost,
	}

	// Add edge to graph first (this will also call AddNeighbor internally in GenericGraph)
	if err := r.graph.AddEdge(edge); err != nil {
		return err
	}

	// Also call AddNeighbor on the node to let it decide direction if it implements NodeAdder
	// Note: AddEdge already calls AddNeighbor internally, but calling it again ensures
	// the node can manage the relationship if needed. Since AddNeighbor is idempotent,
	// this is safe and we don't need to handle errors (AddNeighbor returns nil for duplicates).
	if nodeAdder, ok := from.(NodeAdder[N, float32]); ok {
		_ = nodeAdder.AddNeighbor(to) // Ignore error - AddEdge already succeeded
	}

	return nil
}

// rewire attempts to improve paths by connecting nearby nodes through the new node
func (r *RRTStar[N]) rewire(newNode Node[N, float32], radius float32) {

	for _, node := range r.nodeMap {
		if node.ID() == newNode.ID() {
			continue
		}

		// Use node.Cost() for distance calculation
		dist := newNode.Cost(node)
		if dist > radius {
			continue
		}

		// Calculate cost through new node
		costThroughNew := r.getPathCostToRoot(newNode) + dist
		costCurrent := r.getPathCostToRoot(node)

		// If path through new node is better, rewire
		if costThroughNew < costCurrent {
			// Remove old parent edge
			r.removeParentEdge(node)
			// Connect through new node
			r.connectNodes(newNode, node)
		}
	}
}

// getPathCostToRoot calculates the cost from a node to the root (start)
func (r *RRTStar[N]) getPathCostToRoot(node Node[N, float32]) float32 {
	// Simple BFS to find path cost to root
	visited := make(map[int64]bool)
	queue := []struct {
		node Node[N, float32]
		cost float32
	}{{node: node, cost: 0}}

	for len(queue) > 0 {
		current := queue[0]
		queue = queue[1:]

		if visited[current.node.ID()] {
			continue
		}
		visited[current.node.ID()] = true

		// Check if this is the root (has no incoming edges from tree nodes)
		hasParent := false
		for _, otherNode := range r.nodeMap {
			if otherNode.ID() == current.node.ID() {
				continue
			}
			// Check if there's an edge from otherNode to current.node
			for edge := range otherNode.Edges() {
				if edge.To().ID() == current.node.ID() {
					hasParent = true
					cost := edge.Cost()
					queue = append(queue, struct {
						node Node[N, float32]
						cost float32
					}{node: otherNode, cost: current.cost + cost})
					break
				}
			}
			if hasParent {
				break
			}
		}

		if !hasParent {
			// This is the root
			return current.cost
		}
	}

	return float32(1e9) // No path found
}

// removeParentEdge removes the edge from a node's parent
func (r *RRTStar[N]) removeParentEdge(node Node[N, float32]) {

	// Iterate through all nodes to find edges TO this node
	for _, otherNode := range r.nodeMap {
		if otherNode.ID() == node.ID() {
			continue
		}
		for edge := range otherNode.Edges() {
			if edge.To().ID() == node.ID() {
				r.graph.DeleteEdge(edge)
				return
			}
		}
	}
}

// estimateRewireRadius estimates a good rewiring radius by sampling
func (r *RRTStar[N]) estimateRewireRadius(start Node[N, float32]) float32 {
	const numSamples = 10
	var totalDist float32
	count := 0

	for i := 0; i < numSamples; i++ {
		sampleNode := r.sampler()
		if sampleNode != nil {
			// Use node.Cost() for distance calculation
			dist := start.Cost(sampleNode)
			totalDist += dist
			count++
		}
	}

	if count == 0 {
		return 1.0 // Default
	}

	avgDist := totalDist / float32(count)
	return avgDist * 2.0 // Use 2x average distance as radius
}

// ensureNodeInGraph ensures a node is in the graph and returns the graph's version
func (r *RRTStar[N]) ensureNodeInGraph(node Node[N, float32]) Node[N, float32] {
	if node == nil || r.graph == nil {
		return nil
	}

	// Try to find in graph by ID
	nodeID := node.ID()
	if existing, ok := r.nodeMap[nodeID]; ok {
		// Check if this node is already in the graph
		for n := range r.graph.Nodes() {
			if n.ID() == existing.ID() {
				return n
			}
		}
	}

	// Add node to graph
	if err := r.graph.AddNode(node); err != nil {
		return nil
	}

	// Get the node from graph (it may have been merged/created differently)
	// Use the node's current ID (which may have been assigned by AddNode if it was 0)
	currentID := node.ID()
	for n := range r.graph.Nodes() {
		if n.ID() == currentID {
			return n
		}
	}

	return nil
}

// reconstructPath reconstructs the path from start to goal
func (r *RRTStar[N]) reconstructPath(goal Node[N, float32]) Path[N, float32] {
	// Build path by following parent edges backwards
	path := []Node[N, float32]{goal}
	current := goal
	visited := make(map[int64]bool)

	for {
		visited[current.ID()] = true

		// Find parent (node that has an edge TO current)
		var parent Node[N, float32]
		for _, node := range r.nodeMap {
			if visited[node.ID()] {
				continue
			}
			for edge := range node.Edges() {
				if edge.To().ID() == current.ID() {
					parent = node
					break
				}
			}
			if parent != nil {
				break
			}
		}

		if parent == nil {
			break // Reached root
		}

		path = append(path, parent)
		current = parent
	}

	// Reverse path to get start->goal
	result := make(Path[N, float32], len(path))
	for i := 0; i < len(path); i++ {
		result[i] = path[len(path)-1-i]
	}

	return result
}

// clear resets buffers for reuse
func (r *RRTStar[N]) clear() {
	for k := range r.nodeMap {
		delete(r.nodeMap, k)
	}
}

// RRTStarFunc is a convenience function that creates a temporary RRTStar instance
// For multiple searches, use NewRRTStar() and reuse the instance
func RRTStarFunc[N any](
	graph RRTStarGraph[N],
	sampler Sampler[N],
	goalChecker GoalChecker[N],
	start, goal Node[N, float32],
	opts ...RRTStarOptions,
) Path[N, float32] {
	rrtstar := NewRRTStar(graph, sampler, goalChecker, opts...)
	return rrtstar.Search(start, goal)
}
