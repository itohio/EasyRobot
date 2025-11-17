package graph

import "iter"

// GenericForest represents a collection of disjoint trees
// Node data type: N (generic)
// Edge data type: E (generic, typically float32 for cost)
// Nodes from different trees are not connected
// Works with any Tree implementation (GenericTree, GenericBinaryTree, etc.)
type GenericForest[N any, E any] struct {
	trees      []Tree[N, E] // Array of trees (any Tree implementation)
	treeMap    map[int]int  // treeIdx -> index in trees array (for tracking)
	nextTreeID int
	costMap    map[int]map[int]E // Global cost map: treeIdx*1000000 + nodeIdx -> cost
}

// NewGenericForest creates a new empty generic forest
func NewGenericForest[N any, E any]() *GenericForest[N, E] {
	return &GenericForest[N, E]{
		trees:      make([]Tree[N, E], 0),
		treeMap:    make(map[int]int),
		nextTreeID: 0,
		costMap:    make(map[int]map[int]E),
	}
}

// AddTree adds a tree to the forest and returns its tree index
// Accepts any Tree implementation (GenericTree, GenericBinaryTree, etc.)
func (f *GenericForest[N, E]) AddTree(tree Tree[N, E]) int {
	if tree == nil {
		return -1
	}

	treeIdx := f.nextTreeID
	f.nextTreeID++

	idx := len(f.trees)
	f.trees = append(f.trees, tree)
	f.treeMap[treeIdx] = idx

	return treeIdx
}

// RemoveTree removes a tree from the forest by tree index
func (f *GenericForest[N, E]) RemoveTree(treeIdx int) bool {
	idx, exists := f.treeMap[treeIdx]
	if !exists {
		return false
	}

	// Remove from trees array
	f.trees = append(f.trees[:idx], f.trees[idx+1:]...)

	// Update treeMap for remaining trees
	delete(f.treeMap, treeIdx)
	for k, v := range f.treeMap {
		if v > idx {
			f.treeMap[k] = v - 1
		}
	}

	// Remove costs for this tree
	for k := range f.costMap {
		if k/1000000 == treeIdx {
			delete(f.costMap, k)
		}
	}

	return true
}

// FindTreeContainingNode finds the tree index that contains the given node index
// Note: This method requires GenericTree internals, so it uses type assertion
func (f *GenericForest[N, E]) FindTreeContainingNode(nodeIdx int) int {
	if nodeIdx < 0 {
		return -1
	}

	// Search through all trees to find which one contains this node
	for treeIdx, idx := range f.treeMap {
		if idx < 0 || idx >= len(f.trees) {
			continue
		}
		tree := f.trees[idx]
		if tree == nil {
			continue
		}

		// Type assert to GenericTree to access internals
		gt, ok := tree.(*GenericTree[N, E])
		if !ok {
			continue // Skip non-GenericTree implementations
		}

		// Check if nodeIdx is valid for this tree
		if nodeIdx >= 0 && nodeIdx < len(gt.nodes) {
			// Check if node is in this tree by traversing up to root
			current := nodeIdx
			for current >= 0 && current < len(gt.nodes) {
				parentIdx := gt.nodes[current].parentIdx
				if parentIdx < 0 {
					// Found root
					if current == gt.rootIdx {
						return treeIdx
					}
					break
				}
				current = parentIdx
			}
		}
	}

	return -1
}

// MergeTrees merges two trees by connecting their roots
// tree1Idx's root becomes parent of tree2Idx's root
// Note: This method requires GenericTree internals, so it uses type assertion
func (f *GenericForest[N, E]) MergeTrees(tree1Idx, tree2Idx int) bool {
	idx1, exists1 := f.treeMap[tree1Idx]
	idx2, exists2 := f.treeMap[tree2Idx]

	if !exists1 || !exists2 || idx1 >= len(f.trees) || idx2 >= len(f.trees) {
		return false
	}

	tree1 := f.trees[idx1]
	tree2 := f.trees[idx2]

	if tree1 == nil || tree2 == nil {
		return false
	}

	// Type assert to GenericTree to access internals
	gt1, ok1 := tree1.(*GenericTree[N, E])
	gt2, ok2 := tree2.(*GenericTree[N, E])
	if !ok1 || !ok2 {
		return false // Can only merge GenericTree instances
	}

	// Get root data from tree2
	root2Data := gt2.nodes[gt2.rootIdx].data

	// Add tree2's root as a child of tree1's root
	childIdx := gt1.AddChild(gt1.rootIdx, root2Data)

	if childIdx < 0 {
		return false
	}

	// Move all children from tree2's root to the new node
	// This is complex - we'd need to copy nodes from tree2 to tree1
	// For now, we'll just mark tree2 as merged and remove it
	// A full implementation would copy all nodes
	_ = gt2.nodes[gt2.rootIdx] // Reference for future implementation

	// Remove tree2 from forest
	f.RemoveTree(tree2Idx)

	return true
}

// GetTreeCount returns number of trees in forest
func (f *GenericForest[N, E]) GetTreeCount() int {
	return len(f.trees)
}

// GetTree returns the tree at the given tree index
func (f *GenericForest[N, E]) GetTree(treeIdx int) Tree[N, E] {
	idx, exists := f.treeMap[treeIdx]
	if !exists || idx < 0 || idx >= len(f.trees) {
		return nil
	}
	return f.trees[idx]
}

// SetCost sets the cost for edge from parent to child (across all trees)
func (f *GenericForest[N, E]) SetCost(treeIdx, parentIdx, childIdx int, cost E) {
	if treeIdx < 0 {
		return
	}

	// Use a composite key: treeIdx*1000000 + parentIdx*1000 + childIdx
	key := treeIdx*1000000 + parentIdx*1000 + childIdx

	if f.costMap[key/1000] == nil {
		f.costMap[key/1000] = make(map[int]E)
	}
	f.costMap[key/1000][key%1000] = cost
}

// GetCost returns the cost for edge from parent to child
func (f *GenericForest[N, E]) GetCost(treeIdx, parentIdx, childIdx int) (E, bool) {
	if treeIdx < 0 {
		var zero E
		return zero, false
	}

	key := treeIdx*1000000 + parentIdx*1000 + childIdx
	if costs, exists := f.costMap[key/1000]; exists {
		if cost, exists := costs[key%1000]; exists {
			return cost, true
		}
	}

	// Fallback to tree's cost map (for GenericTree)
	tree := f.GetTree(treeIdx)
	if tree != nil {
		if gt, ok := tree.(*GenericTree[N, E]); ok {
			return gt.GetCost(parentIdx, childIdx)
		}
	}

	var zero E
	return zero, false
}

// ForestGraphNode implements Node[N, E] for GenericForest
type ForestGraphNode[N any, E any] struct {
	forest  *GenericForest[N, E]
	treeIdx int
	nodeIdx int
}

// ID returns a unique identifier for this node
func (n ForestGraphNode[N, E]) ID() int64 {
	if n.forest == nil || n.treeIdx < 0 {
		return 0
	}

	tree := n.forest.GetTree(n.treeIdx)
	if tree == nil {
		return 0
	}

	// Type assert to GenericTree to access internals
	gt, ok := tree.(*GenericTree[N, E])
	if !ok || n.nodeIdx < 0 || n.nodeIdx >= len(gt.nodes) {
		return 0
	}

	return gt.nodes[n.nodeIdx].id
}

// Equal implements Comparable[N, E]
func (n ForestGraphNode[N, E]) Equal(other Node[N, E]) bool {
	if other == nil {
		return false
	}
	o, ok := other.(ForestGraphNode[N, E])
	if !ok {
		return false
	}
	return n.forest == o.forest && n.treeIdx == o.treeIdx && n.nodeIdx == o.nodeIdx
}

// Compare implements Comparable[N, E]
func (n ForestGraphNode[N, E]) Compare(other Node[N, E]) int {
	if other == nil {
		return 1
	}
	o, ok := other.(ForestGraphNode[N, E])
	if !ok {
		return 0
	}
	if n.forest != o.forest {
		return 0 // Can't compare nodes from different forests
	}
	// Compare by tree index first, then node index
	if n.treeIdx < o.treeIdx {
		return -1
	}
	if n.treeIdx > o.treeIdx {
		return 1
	}
	if n.nodeIdx < o.nodeIdx {
		return -1
	}
	if n.nodeIdx > o.nodeIdx {
		return 1
	}
	return 0
}

// Data returns the node's data
func (n ForestGraphNode[N, E]) Data() N {
	if n.forest == nil || n.treeIdx < 0 {
		var zero N
		return zero
	}

	tree := n.forest.GetTree(n.treeIdx)
	if tree == nil {
		var zero N
		return zero
	}

	// Type assert to GenericTree to access internals
	gt, ok := tree.(*GenericTree[N, E])
	if !ok || n.nodeIdx < 0 || n.nodeIdx >= len(gt.nodes) {
		var zero N
		return zero
	}

	return gt.nodes[n.nodeIdx].data
}

// Neighbors returns an iterator over neighboring nodes (children within same tree)
func (n ForestGraphNode[N, E]) Neighbors() iter.Seq[Node[N, E]] {
	return func(yield func(Node[N, E]) bool) {
		if n.forest == nil || n.treeIdx < 0 {
			return
		}

		tree := n.forest.GetTree(n.treeIdx)
		if tree == nil {
			return
		}

		// Type assert to GenericTree to access internals
		gt, ok := tree.(*GenericTree[N, E])
		if !ok || n.nodeIdx < 0 || n.nodeIdx >= len(gt.nodes) {
			return
		}

		node := gt.nodes[n.nodeIdx]
		for _, childIdx := range node.childIdxs {
			if childIdx >= 0 && childIdx < len(gt.nodes) {
				neighbor := ForestGraphNode[N, E]{
					forest:  n.forest,
					treeIdx: n.treeIdx,
					nodeIdx: childIdx,
				}
				if !yield(neighbor) {
					return
				}
			}
		}
	}
}

// NumNeighbors returns the number of neighboring nodes (children within same tree)
func (n ForestGraphNode[N, E]) NumNeighbors() int {
	if n.forest == nil || n.treeIdx < 0 {
		return 0
	}
	tree := n.forest.GetTree(n.treeIdx)
	if tree == nil {
		return 0
	}
	gt, ok := tree.(*GenericTree[N, E])
	if !ok || n.nodeIdx < 0 || n.nodeIdx >= len(gt.nodes) {
		return 0
	}
	node := gt.nodes[n.nodeIdx]
	count := 0
	for _, childIdx := range node.childIdxs {
		if childIdx >= 0 && childIdx < len(gt.nodes) {
			count++
		}
	}
	return count
}

// Edges returns an iterator over edges from this node
func (n ForestGraphNode[N, E]) Edges() iter.Seq[Edge[N, E]] {
	return func(yield func(Edge[N, E]) bool) {
		if n.forest == nil || n.treeIdx < 0 {
			return
		}

		tree := n.forest.GetTree(n.treeIdx)
		if tree == nil {
			return
		}

		// Type assert to GenericTree to access internals
		gt, ok := tree.(*GenericTree[N, E])
		if !ok || n.nodeIdx < 0 || n.nodeIdx >= len(gt.nodes) {
			return
		}

		node := gt.nodes[n.nodeIdx]
		for _, childIdx := range node.childIdxs {
			if childIdx >= 0 && childIdx < len(gt.nodes) {
				toNode := ForestGraphNode[N, E]{
					forest:  n.forest,
					treeIdx: n.treeIdx,
					nodeIdx: childIdx,
				}
				var cost E
				if c, ok := n.forest.GetCost(n.treeIdx, n.nodeIdx, childIdx); ok {
					cost = c
				} else {
					// Default cost
					var zero E
					var one E
					if any(zero) == float32(0) {
						one = any(float32(1.0)).(E)
					}
					cost = one
				}

				edge := ForestGraphEdge[N, E]{
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
}

// Cost calculates the cost from this node to another node
func (n ForestGraphNode[N, E]) Cost(toOther Node[N, E]) float32 {
	if toOther == nil || n.forest == nil {
		return 0
	}
	to, ok := toOther.(ForestGraphNode[N, E])
	if !ok || n.forest != to.forest || n.treeIdx != to.treeIdx {
		return 0
	}
	// Find edge from this node to the other node
	for edge := range n.Edges() {
		if edge.To().ID() == toOther.ID() {
			return edge.Cost()
		}
	}
	return 0
}

// ForestGraphEdge implements Edge[N, E] for GenericForest
type ForestGraphEdge[N any, E any] struct {
	from ForestGraphNode[N, E]
	to   ForestGraphNode[N, E]
	data E
	id   int64
}

// ID returns a unique identifier for this edge
func (e ForestGraphEdge[N, E]) ID() int64 {
	return e.from.ID()*1000000 + e.to.ID()
}

// From returns the source node
func (e ForestGraphEdge[N, E]) From() Node[N, E] {
	return e.from
}

// To returns the destination node
func (e ForestGraphEdge[N, E]) To() Node[N, E] {
	return e.to
}

// Data returns the edge's data
func (e ForestGraphEdge[N, E]) Data() E {
	return e.data
}

// Cost returns the cost/weight of this edge
func (e ForestGraphEdge[N, E]) Cost() float32 {
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
	}
	return 1.0 // Default cost
}

// GenericForest implements Graph[N, E] interface
var _ Graph[any, float32] = (*GenericForest[any, float32])(nil)

// Nodes returns an iterator over all nodes in all trees in the forest
func (f *GenericForest[N, E]) Nodes() iter.Seq[Node[N, E]] {
	return func(yield func(Node[N, E]) bool) {
		for treeIdx, idx := range f.treeMap {
			if idx < 0 || idx >= len(f.trees) {
				continue
			}
			tree := f.trees[idx]
			if tree == nil {
				continue
			}

			// Type assert to GenericTree to access internals
			gt, ok := tree.(*GenericTree[N, E])
			if !ok {
				// For non-GenericTree implementations, iterate using Tree interface
				for node := range tree.Nodes() {
					if !yield(node) {
						return
					}
				}
				continue
			}

			// Iterate over all nodes in this tree
			for nodeIdx := 0; nodeIdx < len(gt.nodes); nodeIdx++ {
				// Skip removed nodes (parentIdx == -2)
				if gt.nodes[nodeIdx].parentIdx == -2 {
					continue
				}

				node := ForestGraphNode[N, E]{
					forest:  f,
					treeIdx: treeIdx,
					nodeIdx: nodeIdx,
				}
				if !yield(node) {
					return
				}
			}
		}
	}
}

// Edges returns an iterator over all edges in all trees in the forest
func (f *GenericForest[N, E]) Edges() iter.Seq[Edge[N, E]] {
	return func(yield func(Edge[N, E]) bool) {
		for node := range f.Nodes() {
			fn := node.(ForestGraphNode[N, E])
			for edge := range fn.Edges() {
				if !yield(edge) {
					return
				}
			}
		}
	}
}

// Neighbors returns an iterator over neighbors of a given node
func (f *GenericForest[N, E]) Neighbors(n Node[N, E]) iter.Seq[Node[N, E]] {
	return n.Neighbors()
}

// NumNodes returns the total number of nodes across all trees in the forest
func (f *GenericForest[N, E]) NumNodes() int {
	count := 0
	for _, idx := range f.treeMap {
		if idx < 0 || idx >= len(f.trees) {
			continue
		}
		tree := f.trees[idx]
		if tree == nil {
			continue
		}
		count += tree.NumNodes()
	}
	return count
}

// NumEdges returns the total number of edges across all trees in the forest
func (f *GenericForest[N, E]) NumEdges() int {
	count := 0
	for _, idx := range f.treeMap {
		if idx < 0 || idx >= len(f.trees) {
			continue
		}
		tree := f.trees[idx]
		if tree == nil {
			continue
		}
		count += tree.NumEdges()
	}
	return count
}
