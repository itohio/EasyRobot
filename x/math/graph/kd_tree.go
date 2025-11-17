package graph

import (
	"iter"

	vecTypes "github.com/itohio/EasyRobot/x/math/vec/types"
)

// KDNodeValue represents a node in a k-d tree stored as a value
// Uses indices instead of pointers for left/right/parent relationships
type KDNodeValue[V vecTypes.Vector] struct {
	id        int64
	point     V
	dimension int // Which dimension this node splits on (0 to k-1)
	parentIdx int // Index of parent node (-1 for root)
	leftIdx   int // Index of left child (-1 if none)
	rightIdx  int // Index of right child (-1 if none)
}

// GenericKDTree represents a k-dimensional tree for spatial queries
// Node data type: V (Vector interface)
// Edge data type: E (generic, typically float32 for cost/distance)
type GenericKDTree[V vecTypes.Vector, E any] struct {
	nodes   []KDNodeValue[V] // Array of node values
	nodeMap map[int64]int    // ID -> index in nodes array
	nextID  int64
	costMap map[int]map[int]E // parentIdx -> childIdx -> cost
	rootIdx int               // Index of root node (-1 if empty)
	k       int               // Number of dimensions (extracted from first vector)
}

// NewGenericKDTree creates a new generic k-d tree from points
func NewGenericKDTree[V vecTypes.Vector, E any](points []V) *GenericKDTree[V, E] {
	if len(points) == 0 {
		return &GenericKDTree[V, E]{
			nodes:   make([]KDNodeValue[V], 0),
			nodeMap: make(map[int64]int),
			nextID:  1,
			costMap: make(map[int]map[int]E),
			rootIdx: -1,
			k:       0,
		}
	}

	// Extract dimension from first point
	firstPoint := points[0]
	k := firstPoint.Len()

	kt := &GenericKDTree[V, E]{
		nodes:   make([]KDNodeValue[V], 0),
		nodeMap: make(map[int64]int),
		nextID:  1,
		costMap: make(map[int]map[int]E),
		rootIdx: -1,
		k:       k,
	}

	kt.rootIdx = kt.buildTree(points, 0)
	return kt
}

// buildTree builds a k-d tree recursively
func (kt *GenericKDTree[V, E]) buildTree(points []V, depth int) int {
	if len(points) == 0 {
		return -1
	}

	if len(points) == 1 {
		return kt.addNode(points[0], depth%kt.k, -1)
	}

	// Select dimension to split on (alternating)
	dimension := depth % kt.k

	// Find median point for this dimension
	medianIndex := kt.findMedian(points, dimension)
	medianPoint := points[medianIndex]

	// Split points
	leftPoints := make([]V, 0)
	rightPoints := make([]V, 0)

	for i, point := range points {
		if i == medianIndex {
			continue
		}

		// Compare points by dimension value
		if kt.compareByDimension(point, medianPoint, dimension) <= 0 {
			leftPoints = append(leftPoints, point)
		} else {
			rightPoints = append(rightPoints, point)
		}
	}

	// Add median node
	nodeIdx := kt.addNode(medianPoint, dimension, -1)

	// Build left and right subtrees
	leftIdx := kt.buildTree(leftPoints, depth+1)
	rightIdx := kt.buildTree(rightPoints, depth+1)

	// Update node's children
	if nodeIdx >= 0 && nodeIdx < len(kt.nodes) {
		kt.nodes[nodeIdx].leftIdx = leftIdx
		kt.nodes[nodeIdx].rightIdx = rightIdx
	}

	// Update children's parent pointers
	if leftIdx >= 0 && leftIdx < len(kt.nodes) {
		kt.nodes[leftIdx].parentIdx = nodeIdx
	}
	if rightIdx >= 0 && rightIdx < len(kt.nodes) {
		kt.nodes[rightIdx].parentIdx = nodeIdx
	}

	return nodeIdx
}

// addNode adds a node to the tree and returns its index
func (kt *GenericKDTree[V, E]) addNode(point V, dimension, parentIdx int) int {
	id := kt.nextID
	kt.nextID++

	node := KDNodeValue[V]{
		id:        id,
		point:     point,
		dimension: dimension,
		parentIdx: parentIdx,
		leftIdx:   -1,
		rightIdx:  -1,
	}

	idx := len(kt.nodes)
	kt.nodes = append(kt.nodes, node)
	kt.nodeMap[id] = idx

	return idx
}

// compareByDimension compares two vectors by a specific dimension
func (kt *GenericKDTree[V, E]) compareByDimension(v1, v2 V, dim int) float32 {
	// Check bounds using Len()
	if dim >= v1.Len() || dim >= v2.Len() {
		return 0
	}

	// Use Slice to get single component, then Sum() to extract the value
	val1 := v1.Slice(dim, dim+1).Sum()
	val2 := v2.Slice(dim, dim+1).Sum()
	return val1 - val2
}

// getDimensionValue gets the value at a specific dimension
func (kt *GenericKDTree[V, E]) getDimensionValue(v V, dim int) float32 {
	// Check bounds using Len()
	if dim >= v.Len() {
		return 0
	}

	// Use Slice to get single component, then Sum() to extract the value
	return v.Slice(dim, dim+1).Sum()
}

// findMedian finds the median index for points sorted by dimension
func (kt *GenericKDTree[V, E]) findMedian(points []V, dimension int) int {
	if len(points) == 0 {
		return 0
	}

	// Create indices and sort by dimension value
	indices := make([]int, len(points))
	for i := range indices {
		indices[i] = i
	}

	// Simple selection algorithm to find median
	medianPos := len(points) / 2

	for i := 0; i < len(points); i++ {
		for j := i + 1; j < len(points); j++ {
			valI := kt.getDimensionValue(points[indices[i]], dimension)
			valJ := kt.getDimensionValue(points[indices[j]], dimension)
			if valI > valJ {
				indices[i], indices[j] = indices[j], indices[i]
			}
		}
		if i == medianPos {
			break // Early exit when we reach median
		}
	}

	return indices[medianPos]
}

// NearestNeighbor finds the nearest neighbor to the query point
func (kt *GenericKDTree[V, E]) NearestNeighbor(query V) V {
	if kt.rootIdx < 0 {
		var zero V
		return zero
	}

	bestIdx, _ := kt.nearestNeighborRecursive(kt.rootIdx, query, -1, 1e10)
	if bestIdx < 0 {
		var zero V
		return zero
	}
	return kt.nodes[bestIdx].point
}

func (kt *GenericKDTree[V, E]) nearestNeighborRecursive(nodeIdx int, query V, bestIdx int, bestDist float32) (int, float32) {
	if nodeIdx < 0 || nodeIdx >= len(kt.nodes) {
		return bestIdx, bestDist
	}

	node := kt.nodes[nodeIdx]

	// Calculate distance to current node
	currentDist := query.Distance(node.point)
	if currentDist < bestDist {
		bestDist = currentDist
		bestIdx = nodeIdx
	}

	// Determine which side to search first
	dim := node.dimension
	queryVal := kt.getDimensionValue(query, dim)
	nodeVal := kt.getDimensionValue(node.point, dim)
	diff := queryVal - nodeVal

	var nearSubtree, farSubtree int
	if diff <= 0 {
		nearSubtree = node.leftIdx
		farSubtree = node.rightIdx
	} else {
		nearSubtree = node.rightIdx
		farSubtree = node.leftIdx
	}

	// Search near subtree
	bestIdx, bestDist = kt.nearestNeighborRecursive(nearSubtree, query, bestIdx, bestDist)

	// Check if we need to search far subtree
	if diff*diff < bestDist {
		bestIdx, bestDist = kt.nearestNeighborRecursive(farSubtree, query, bestIdx, bestDist)
	}

	return bestIdx, bestDist
}

// SetCost sets the cost for edge from parent to child
func (kt *GenericKDTree[V, E]) SetCost(parentIdx, childIdx int, cost E) {
	if parentIdx < 0 || parentIdx >= len(kt.nodes) ||
		childIdx < 0 || childIdx >= len(kt.nodes) {
		return
	}

	if kt.costMap[parentIdx] == nil {
		kt.costMap[parentIdx] = make(map[int]E)
	}
	kt.costMap[parentIdx][childIdx] = cost
}

// GetCost returns the cost for edge from parent to child
func (kt *GenericKDTree[V, E]) GetCost(parentIdx, childIdx int) (E, bool) {
	if parentIdx < 0 || parentIdx >= len(kt.nodes) ||
		childIdx < 0 || childIdx >= len(kt.nodes) {
		var zero E
		return zero, false
	}

	if costs, exists := kt.costMap[parentIdx]; exists {
		if cost, exists := costs[childIdx]; exists {
			return cost, true
		}
	}
	var zero E
	return zero, false
}

// RootIdx returns the index of the root node
func (kt *GenericKDTree[V, E]) RootIdx() int {
	return kt.rootIdx
}

// Root returns the root node of the tree, or nil if empty
func (kt *GenericKDTree[V, E]) Root() Node[V, E] {
	if kt.rootIdx < 0 || kt.rootIdx >= len(kt.nodes) {
		return nil
	}
	return KDTreeGraphNode[V, E]{tree: kt, idx: kt.rootIdx}
}

// NodeCount returns the number of nodes in the tree
func (kt *GenericKDTree[V, E]) NodeCount() int {
	return len(kt.nodes)
}

// GetNode returns the node at the given index
func (kt *GenericKDTree[V, E]) GetNode(idx int) *KDNodeValue[V] {
	if idx < 0 || idx >= len(kt.nodes) {
		return nil
	}
	return &kt.nodes[idx]
}

// GetHeight returns the height of the tree
func (kt *GenericKDTree[V, E]) GetHeight() int {
	if kt.rootIdx < 0 {
		return 0
	}
	return kt.getHeightRecursive(kt.rootIdx)
}

func (kt *GenericKDTree[V, E]) getHeightRecursive(nodeIdx int) int {
	if nodeIdx < 0 || nodeIdx >= len(kt.nodes) {
		return -1
	}

	node := kt.nodes[nodeIdx]
	leftHeight := kt.getHeightRecursive(node.leftIdx)
	rightHeight := kt.getHeightRecursive(node.rightIdx)

	if leftHeight > rightHeight {
		return leftHeight + 1
	}

	return rightHeight + 1
}

// KDTreeGraphNode implements Node[V, E] for GenericKDTree
type KDTreeGraphNode[V vecTypes.Vector, E any] struct {
	tree *GenericKDTree[V, E]
	idx  int
}

// ID returns a unique identifier for this node
func (n KDTreeGraphNode[V, E]) ID() int64 {
	if n.tree == nil || n.idx < 0 || n.idx >= len(n.tree.nodes) {
		return 0
	}
	return n.tree.nodes[n.idx].id
}

// Equal implements Comparable[V, E]
func (n KDTreeGraphNode[V, E]) Equal(other Node[V, E]) bool {
	if other == nil {
		return false
	}
	o, ok := other.(KDTreeGraphNode[V, E])
	if !ok {
		return false
	}
	return n.tree == o.tree && n.idx == o.idx
}

// Compare implements Comparable[V, E]
func (n KDTreeGraphNode[V, E]) Compare(other Node[V, E]) int {
	if other == nil {
		return 1
	}
	o, ok := other.(KDTreeGraphNode[V, E])
	if !ok {
		return 0
	}
	if n.tree != o.tree {
		return 0 // Can't compare nodes from different trees
	}
	// Compare by index
	if n.idx < o.idx {
		return -1
	}
	if n.idx > o.idx {
		return 1
	}
	return 0
}

// Data returns the node's data (the vector point)
func (n KDTreeGraphNode[V, E]) Data() V {
	if n.tree == nil || n.idx < 0 || n.idx >= len(n.tree.nodes) {
		var zero V
		return zero
	}
	return n.tree.nodes[n.idx].point
}

// Neighbors returns an iterator over neighboring nodes (left and right children)
func (n KDTreeGraphNode[V, E]) Neighbors() iter.Seq[Node[V, E]] {
	return func(yield func(Node[V, E]) bool) {
		if n.tree == nil || n.idx < 0 || n.idx >= len(n.tree.nodes) {
			return
		}

		node := n.tree.nodes[n.idx]

		// Add left child if exists
		if node.leftIdx >= 0 && node.leftIdx < len(n.tree.nodes) {
			neighbor := KDTreeGraphNode[V, E]{
				tree: n.tree,
				idx:  node.leftIdx,
			}
			if !yield(neighbor) {
				return
			}
		}

		// Add right child if exists
		if node.rightIdx >= 0 && node.rightIdx < len(n.tree.nodes) {
			neighbor := KDTreeGraphNode[V, E]{
				tree: n.tree,
				idx:  node.rightIdx,
			}
			if !yield(neighbor) {
				return
			}
		}
	}
}

// NumNeighbors returns the number of neighboring nodes (left and right children)
func (n KDTreeGraphNode[V, E]) NumNeighbors() int {
	if n.tree == nil || n.idx < 0 || n.idx >= len(n.tree.nodes) {
		return 0
	}
	node := n.tree.nodes[n.idx]
	count := 0
	if node.leftIdx >= 0 && node.leftIdx < len(n.tree.nodes) {
		count++
	}
	if node.rightIdx >= 0 && node.rightIdx < len(n.tree.nodes) {
		count++
	}
	return count
}

// Edges returns an iterator over edges from this node
func (n KDTreeGraphNode[V, E]) Edges() iter.Seq[Edge[V, E]] {
	return func(yield func(Edge[V, E]) bool) {
		if n.tree == nil || n.idx < 0 || n.idx >= len(n.tree.nodes) {
			return
		}

		node := n.tree.nodes[n.idx]

		// Add left child edge if exists
		if node.leftIdx >= 0 && node.leftIdx < len(n.tree.nodes) {
			toNode := KDTreeGraphNode[V, E]{
				tree: n.tree,
				idx:  node.leftIdx,
			}
			var cost E
			if c, ok := n.tree.GetCost(n.idx, node.leftIdx); ok {
				cost = c
			} else {
				// Default cost is distance
				var zero E
				if any(zero) == float32(0) {
					dist := node.point.Distance(n.tree.nodes[node.leftIdx].point)
					cost = any(dist).(E)
				}
			}

			edge := KDTreeGraphEdge[V, E]{
				from: n,
				to:   toNode,
				data: cost,
			}
			if !yield(edge) {
				return
			}
		}

		// Add right child edge if exists
		if node.rightIdx >= 0 && node.rightIdx < len(n.tree.nodes) {
			toNode := KDTreeGraphNode[V, E]{
				tree: n.tree,
				idx:  node.rightIdx,
			}
			var cost E
			if c, ok := n.tree.GetCost(n.idx, node.rightIdx); ok {
				cost = c
			} else {
				// Default cost is distance
				var zero E
				if any(zero) == float32(0) {
					dist := node.point.Distance(n.tree.nodes[node.rightIdx].point)
					cost = any(dist).(E)
				}
			}

			edge := KDTreeGraphEdge[V, E]{
				from: n,
				to:   toNode,
				data: cost,
			}
			if !yield(edge) {
				return
			}
		}
	}
}

// Cost calculates the cost from this node to another node
func (n KDTreeGraphNode[V, E]) Cost(toOther Node[V, E]) float32 {
	if toOther == nil || n.tree == nil {
		return 0
	}
	to, ok := toOther.(KDTreeGraphNode[V, E])
	if !ok || n.tree != to.tree {
		return 0
	}
	// Find edge from this node to the other node
	for edge := range n.Edges() {
		if edge.To().ID() == toOther.ID() {
			return edge.Cost()
		}
	}
	// Fallback to distance if no edge found
	return n.Data().Distance(to.Data())
}

// KDTreeGraphEdge implements Edge[V, E] for GenericKDTree
type KDTreeGraphEdge[V vecTypes.Vector, E any] struct {
	from KDTreeGraphNode[V, E]
	to   KDTreeGraphNode[V, E]
	data E
	id   int64
}

// ID returns a unique identifier for this edge
func (e KDTreeGraphEdge[V, E]) ID() int64 {
	return e.from.ID()*1000000 + e.to.ID()
}

// From returns the source node
func (e KDTreeGraphEdge[V, E]) From() Node[V, E] {
	return e.from
}

// To returns the destination node
func (e KDTreeGraphEdge[V, E]) To() Node[V, E] {
	return e.to
}

// Data returns the edge's data
func (e KDTreeGraphEdge[V, E]) Data() E {
	return e.data
}

// Cost returns the cost/weight of this edge
func (e KDTreeGraphEdge[V, E]) Cost() float32 {
	// Try to convert E to float32
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
	default:
		// Fallback to distance between vector points
		from := e.from.Data()
		to := e.to.Data()
		return from.Distance(to)
	}
}

// GenericKDTree implements Graph[V, E] interface
var _ Graph[vecTypes.Vector, float32] = (*GenericKDTree[vecTypes.Vector, float32])(nil)

// Nodes returns an iterator over all nodes in the k-d tree
func (kt *GenericKDTree[V, E]) Nodes() iter.Seq[Node[V, E]] {
	return func(yield func(Node[V, E]) bool) {
		for i := 0; i < len(kt.nodes); i++ {
			// Skip removed nodes (parentIdx == -2)
			if kt.nodes[i].parentIdx == -2 {
				continue
			}
			node := KDTreeGraphNode[V, E]{
				tree: kt,
				idx:  i,
			}
			if !yield(node) {
				return
			}
		}
	}
}

// Edges returns an iterator over all edges in the k-d tree
func (kt *GenericKDTree[V, E]) Edges() iter.Seq[Edge[V, E]] {
	return func(yield func(Edge[V, E]) bool) {
		for node := range kt.Nodes() {
			kn := node.(KDTreeGraphNode[V, E])
			for edge := range kn.Edges() {
				if !yield(edge) {
					return
				}
			}
		}
	}
}

// Neighbors returns an iterator over neighbors of a given node
func (kt *GenericKDTree[V, E]) Neighbors(n Node[V, E]) iter.Seq[Node[V, E]] {
	return n.Neighbors()
}

// NumNodes returns the total number of nodes in the k-d tree
func (kt *GenericKDTree[V, E]) NumNodes() int {
	count := 0
	for i := 0; i < len(kt.nodes); i++ {
		// Skip removed nodes (parentIdx == -2)
		if kt.nodes[i].parentIdx != -2 {
			count++
		}
	}
	return count
}

// NumEdges returns the total number of edges in the k-d tree
func (kt *GenericKDTree[V, E]) NumEdges() int {
	count := 0
	for i := 0; i < len(kt.nodes); i++ {
		// Skip removed nodes
		if kt.nodes[i].parentIdx == -2 {
			continue
		}
		// Count left and right children (each is an edge)
		if kt.nodes[i].leftIdx >= 0 && kt.nodes[i].leftIdx < len(kt.nodes) {
			count++
		}
		if kt.nodes[i].rightIdx >= 0 && kt.nodes[i].rightIdx < len(kt.nodes) {
			count++
		}
	}
	return count
}
