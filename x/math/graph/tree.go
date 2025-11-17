package graph

import "iter"

// TreeNodeValue represents a node in a generic tree stored as a value
// Uses indices instead of pointers for parent/child relationships
type TreeNodeValue[N any] struct {
	id        int64
	data      N
	parentIdx int   // Index of parent node (-1 for root)
	childIdxs []int // Indices of child nodes
}

// GenericTree represents a hierarchical tree structure using value arrays
// Node data type: N (generic)
// Edge data type: E (generic, typically float32 for cost)
type GenericTree[N any, E any] struct {
	nodes   []TreeNodeValue[N] // Array of node values
	nodeMap map[int64]int      // ID -> index in nodes array
	nextID  int64
	costMap map[int]map[int]E // parentIdx -> childIdx -> cost
	rootIdx int               // Index of root node (-1 if empty)
}

// NewGenericTree creates a new generic tree with a root node
func NewGenericTree[N any, E any](rootData N) *GenericTree[N, E] {
	t := &GenericTree[N, E]{
		nodes:   make([]TreeNodeValue[N], 0),
		nodeMap: make(map[int64]int),
		nextID:  1,
		costMap: make(map[int]map[int]E),
		rootIdx: -1,
	}

	// Add root node
	rootIdx := t.addNode(rootData, -1)
	t.rootIdx = rootIdx
	return t
}

// addNode adds a node to the tree and returns its index
func (t *GenericTree[N, E]) addNode(data N, parentIdx int) int {
	id := t.nextID
	t.nextID++

	node := TreeNodeValue[N]{
		id:        id,
		data:      data,
		parentIdx: parentIdx,
		childIdxs: make([]int, 0),
	}

	idx := len(t.nodes)
	t.nodes = append(t.nodes, node)
	t.nodeMap[id] = idx

	// Update parent's child list
	if parentIdx >= 0 && parentIdx < len(t.nodes) {
		t.nodes[parentIdx].childIdxs = append(t.nodes[parentIdx].childIdxs, idx)
	}

	return idx
}

// AddChild adds a child node to the given parent and returns the new node's index
func (t *GenericTree[N, E]) AddChild(parentIdx int, childData N) int {
	if parentIdx < 0 || parentIdx >= len(t.nodes) {
		return -1
	}
	return t.addNode(childData, parentIdx)
}

// AddChildAtPosition adds a child node at a specific position in the parent's child list.
// position: 0 = first child, 1 = second child, etc. Use -1 to append at end.
// Returns the new node's index, or -1 on error.
func (t *GenericTree[N, E]) AddChildAtPosition(parentIdx int, childData N, position int) int {
	if parentIdx < 0 || parentIdx >= len(t.nodes) {
		return -1
	}

	childIdx := t.addNode(childData, parentIdx)
	if childIdx < 0 {
		return -1
	}

	parent := &t.nodes[parentIdx]
	if position < 0 || position >= len(parent.childIdxs) {
		// Append at end (already done by addNode)
		return childIdx
	}

	// Insert at specific position
	// Remove from end (where addNode added it)
	parent.childIdxs = parent.childIdxs[:len(parent.childIdxs)-1]
	// Insert at desired position
	parent.childIdxs = append(parent.childIdxs[:position], append([]int{childIdx}, parent.childIdxs[position:]...)...)

	return childIdx
}

// ReorderChildren changes the order of children for a given parent node.
// newOrder: slice of child indices in the desired order.
// Returns true if reordering succeeded, false otherwise.
func (t *GenericTree[N, E]) ReorderChildren(parentIdx int, newOrder []int) bool {
	if parentIdx < 0 || parentIdx >= len(t.nodes) {
		return false
	}

	parent := &t.nodes[parentIdx]

	// Validate that all indices in newOrder are valid children
	childSet := make(map[int]bool)
	for _, childIdx := range parent.childIdxs {
		childSet[childIdx] = true
	}

	for _, idx := range newOrder {
		if !childSet[idx] {
			return false // Invalid child index
		}
		if idx < 0 || idx >= len(t.nodes) {
			return false // Index out of bounds
		}
	}

	// Validate that all children are included
	if len(newOrder) != len(parent.childIdxs) {
		return false // Missing or extra children
	}

	// Update child order
	parent.childIdxs = make([]int, len(newOrder))
	copy(parent.childIdxs, newOrder)

	return true
}

// SwapChildren swaps two children at given positions.
// Returns true if swap succeeded, false otherwise.
func (t *GenericTree[N, E]) SwapChildren(parentIdx int, pos1, pos2 int) bool {
	if parentIdx < 0 || parentIdx >= len(t.nodes) {
		return false
	}

	parent := &t.nodes[parentIdx]
	if pos1 < 0 || pos1 >= len(parent.childIdxs) ||
		pos2 < 0 || pos2 >= len(parent.childIdxs) {
		return false
	}

	parent.childIdxs[pos1], parent.childIdxs[pos2] = parent.childIdxs[pos2], parent.childIdxs[pos1]
	return true
}

// Insert inserts a node using the provided insertion strategy
func (t *GenericTree[N, E]) Insert(data N, strategy InsertionStrategy[N, E]) int {
	if strategy == nil {
		// Default: insert at root
		if t.rootIdx < 0 {
			return -1
		}
		return t.AddChild(t.rootIdx, data)
	}

	parentIdx := strategy(t, data, t.rootIdx)
	if parentIdx < 0 {
		return -1
	}
	return t.AddChild(parentIdx, data)
}

// countDescendants counts the number of descendants of a node (including the node itself)
func (t *GenericTree[N, E]) countDescendants(nodeIdx int) int {
	if nodeIdx < 0 || nodeIdx >= len(t.nodes) {
		return 0
	}

	count := 1 // Count this node
	for _, childIdx := range t.nodes[nodeIdx].childIdxs {
		count += t.countDescendants(childIdx)
	}
	return count
}

// Balance attempts to balance the tree by redistributing nodes.
// This reorders children so that subtrees with fewer nodes come first,
// helping to balance the tree structure.
func (t *GenericTree[N, E]) Balance() {
	if t.rootIdx < 0 {
		return
	}
	t.balanceRecursive(t.rootIdx)
}

func (t *GenericTree[N, E]) balanceRecursive(nodeIdx int) {
	if nodeIdx < 0 || nodeIdx >= len(t.nodes) {
		return
	}

	node := &t.nodes[nodeIdx]
	if len(node.childIdxs) <= 1 {
		// Already balanced or no children
		return
	}

	// Count descendants for each child and sort by count (ascending)
	type childInfo struct {
		idx   int
		count int
	}
	children := make([]childInfo, len(node.childIdxs))
	for i, childIdx := range node.childIdxs {
		count := t.countDescendants(childIdx)
		children[i] = childInfo{idx: childIdx, count: count}
	}

	// Sort by count (ascending) - smaller subtrees first
	for i := 0; i < len(children)-1; i++ {
		for j := i + 1; j < len(children); j++ {
			if children[i].count > children[j].count {
				children[i], children[j] = children[j], children[i]
			}
		}
	}

	// Reorder children based on sorted order
	newOrder := make([]int, len(children))
	for i, info := range children {
		newOrder[i] = info.idx
	}
	t.ReorderChildren(nodeIdx, newOrder)

	// Recursively balance children
	for _, childIdx := range node.childIdxs {
		t.balanceRecursive(childIdx)
	}
}

// RemoveChild removes a child node from the given parent
func (t *GenericTree[N, E]) RemoveChild(parentIdx, childIdx int) bool {
	if parentIdx < 0 || parentIdx >= len(t.nodes) ||
		childIdx < 0 || childIdx >= len(t.nodes) {
		return false
	}

	// Check if child belongs to parent
	parent := &t.nodes[parentIdx]
	found := false
	for i, cIdx := range parent.childIdxs {
		if cIdx == childIdx {
			parent.childIdxs = append(parent.childIdxs[:i], parent.childIdxs[i+1:]...)
			found = true
			break
		}
	}

	if !found {
		return false
	}

	// Mark node as removed (set parentIdx to -2 to indicate removed)
	t.nodes[childIdx].parentIdx = -2

	// Remove from cost map
	if costs, exists := t.costMap[parentIdx]; exists {
		delete(costs, childIdx)
	}

	return true
}

// SetCost sets the cost for edge from parent to child
func (t *GenericTree[N, E]) SetCost(parentIdx, childIdx int, cost E) {
	if parentIdx < 0 || parentIdx >= len(t.nodes) ||
		childIdx < 0 || childIdx >= len(t.nodes) {
		return
	}

	if t.costMap[parentIdx] == nil {
		t.costMap[parentIdx] = make(map[int]E)
	}
	t.costMap[parentIdx][childIdx] = cost
}

// GetCost returns the cost for edge from parent to child
func (t *GenericTree[N, E]) GetCost(parentIdx, childIdx int) (E, bool) {
	if parentIdx < 0 || parentIdx >= len(t.nodes) ||
		childIdx < 0 || childIdx >= len(t.nodes) {
		var zero E
		return zero, false
	}

	if costs, exists := t.costMap[parentIdx]; exists {
		if cost, exists := costs[childIdx]; exists {
			return cost, true
		}
	}
	var zero E
	return zero, false
}

// RootIdx returns the index of the root node
func (t *GenericTree[N, E]) RootIdx() int {
	return t.rootIdx
}

// Root returns the root node of the tree, or nil if empty
func (t *GenericTree[N, E]) Root() Node[N, E] {
	if t.rootIdx < 0 || t.rootIdx >= len(t.nodes) {
		return nil
	}
	return TreeGraphNode[N, E]{tree: t, idx: t.rootIdx}
}

// NodeCount returns the number of nodes in the tree
func (t *GenericTree[N, E]) NodeCount() int {
	return len(t.nodes)
}

// GetNode returns the node at the given index
func (t *GenericTree[N, E]) GetNode(idx int) *TreeNodeValue[N] {
	if idx < 0 || idx >= len(t.nodes) {
		return nil
	}
	return &t.nodes[idx]
}

// FindPathToRoot returns path from node to root as indices
func (t *GenericTree[N, E]) FindPathToRoot(nodeIdx int) []int {
	if nodeIdx < 0 || nodeIdx >= len(t.nodes) {
		return nil
	}

	var path []int
	current := nodeIdx

	for current >= 0 && current < len(t.nodes) {
		path = append(path, current)
		parentIdx := t.nodes[current].parentIdx
		if parentIdx < 0 {
			break
		}
		current = parentIdx
	}

	return path
}

// GetDepth returns depth of node (distance from root)
func (t *GenericTree[N, E]) GetDepth(nodeIdx int) int {
	if nodeIdx < 0 || nodeIdx >= len(t.nodes) {
		return -1
	}

	depth := 0
	current := nodeIdx

	for current >= 0 && current < len(t.nodes) {
		parentIdx := t.nodes[current].parentIdx
		if parentIdx < 0 {
			break
		}
		depth++
		current = parentIdx
	}

	return depth
}

// GetHeight returns height of tree (max depth from root)
func (t *GenericTree[N, E]) GetHeight() int {
	if t.rootIdx < 0 {
		return -1
	}
	return t.getHeightRecursive(t.rootIdx)
}

func (t *GenericTree[N, E]) getHeightRecursive(nodeIdx int) int {
	if nodeIdx < 0 || nodeIdx >= len(t.nodes) {
		return -1
	}

	node := t.nodes[nodeIdx]
	maxChildHeight := -1

	for _, childIdx := range node.childIdxs {
		childHeight := t.getHeightRecursive(childIdx)
		if childHeight > maxChildHeight {
			maxChildHeight = childHeight
		}
	}

	return maxChildHeight + 1
}

// FindNodeByData finds first node with matching data and returns its index
// It uses Node.Equal() and Node.Compare() methods for comparison
func (t *GenericTree[N, E]) FindNodeByData(data N) int {
	return t.findNodeRecursive(t.rootIdx, data)
}

func (t *GenericTree[N, E]) findNodeRecursive(nodeIdx int, data N) int {
	if nodeIdx < 0 || nodeIdx >= len(t.nodes) {
		return -1
	}

	node := TreeGraphNode[N, E]{
		tree: t,
		idx:  nodeIdx,
	}

	// Create a temporary node with the target data for comparison
	dataNode := &dataNodeWrapper[N, E]{data: data}

	if node.Equal(dataNode) {
		return nodeIdx
	}

	for _, childIdx := range t.nodes[nodeIdx].childIdxs {
		if found := t.findNodeRecursive(childIdx, data); found >= 0 {
			return found
		}
	}

	return -1
}

// dataNodeWrapper is a helper to compare node data using Node.Equal/Compare
type dataNodeWrapper[N any, E any] struct {
	data N
}

func (d *dataNodeWrapper[N, E]) ID() int64 {
	return 0
}

func (d *dataNodeWrapper[N, E]) Equal(other Node[N, E]) bool {
	if other == nil {
		return false
	}
	// Compare data values
	var dAny any = d.data
	var oAny any = other.Data()
	return dAny == oAny
}

func (d *dataNodeWrapper[N, E]) Compare(other Node[N, E]) int {
	if other == nil {
		return 1
	}
	// Compare data values
	var dAny any = d.data
	var oAny any = other.Data()
	// Try to use < operator if available
	switch da := dAny.(type) {
	case int:
		if db, ok := oAny.(int); ok {
			if da < db {
				return -1
			}
			if da > db {
				return 1
			}
			return 0
		}
	case float32:
		if db, ok := oAny.(float32); ok {
			if da < db {
				return -1
			}
			if da > db {
				return 1
			}
			return 0
		}
	case float64:
		if db, ok := oAny.(float64); ok {
			if da < db {
				return -1
			}
			if da > db {
				return 1
			}
			return 0
		}
	case string:
		if db, ok := oAny.(string); ok {
			if da < db {
				return -1
			}
			if da > db {
				return 1
			}
			return 0
		}
	}
	// Fallback: use equality
	if dAny == oAny {
		return 0
	}
	return 0
}

func (d *dataNodeWrapper[N, E]) Data() N {
	return d.data
}

func (d *dataNodeWrapper[N, E]) Neighbors() iter.Seq[Node[N, E]] {
	return func(yield func(Node[N, E]) bool) {}
}

func (d *dataNodeWrapper[N, E]) Edges() iter.Seq[Edge[N, E]] {
	return func(yield func(Edge[N, E]) bool) {}
}

func (d *dataNodeWrapper[N, E]) NumNeighbors() int {
	return 0
}

func (d *dataNodeWrapper[N, E]) Cost(toOther Node[N, E]) float32 {
	return 0
}

// TreeGraphNode implements Node[N, E] for GenericTree
type TreeGraphNode[N any, E any] struct {
	tree *GenericTree[N, E]
	idx  int
}

// ID returns a unique identifier for this node
func (n TreeGraphNode[N, E]) ID() int64 {
	if n.tree == nil || n.idx < 0 || n.idx >= len(n.tree.nodes) {
		return 0
	}
	return n.tree.nodes[n.idx].id
}

// Equal implements Comparable[N, E]
func (n TreeGraphNode[N, E]) Equal(other Node[N, E]) bool {
	if other == nil {
		return false
	}
	o, ok := other.(TreeGraphNode[N, E])
	if !ok {
		return false
	}
	return n.tree == o.tree && n.idx == o.idx
}

// Compare implements Comparable[N, E]
func (n TreeGraphNode[N, E]) Compare(other Node[N, E]) int {
	if other == nil {
		return 1
	}
	o, ok := other.(TreeGraphNode[N, E])
	if !ok {
		return 0 // Can't compare different node types
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

// Data returns the node's data
func (n TreeGraphNode[N, E]) Data() N {
	if n.tree == nil || n.idx < 0 || n.idx >= len(n.tree.nodes) {
		var zero N
		return zero
	}
	return n.tree.nodes[n.idx].data
}

// Neighbors returns an iterator over neighboring nodes (children)
func (n TreeGraphNode[N, E]) Neighbors() iter.Seq[Node[N, E]] {
	return func(yield func(Node[N, E]) bool) {
		if n.tree == nil || n.idx < 0 || n.idx >= len(n.tree.nodes) {
			return
		}

		node := n.tree.nodes[n.idx]
		for _, childIdx := range node.childIdxs {
			if childIdx >= 0 && childIdx < len(n.tree.nodes) {
				neighbor := TreeGraphNode[N, E]{
					tree: n.tree,
					idx:  childIdx,
				}
				if !yield(neighbor) {
					return
				}
			}
		}
	}
}

// NumNeighbors returns the number of neighboring nodes (children)
func (n TreeGraphNode[N, E]) NumNeighbors() int {
	if n.tree == nil || n.idx < 0 || n.idx >= len(n.tree.nodes) {
		return 0
	}
	node := n.tree.nodes[n.idx]
	count := 0
	for _, childIdx := range node.childIdxs {
		if childIdx >= 0 && childIdx < len(n.tree.nodes) {
			count++
		}
	}
	return count
}

// Edges returns an iterator over edges from this node
func (n TreeGraphNode[N, E]) Edges() iter.Seq[Edge[N, E]] {
	return func(yield func(Edge[N, E]) bool) {
		if n.tree == nil || n.idx < 0 || n.idx >= len(n.tree.nodes) {
			return
		}

		node := n.tree.nodes[n.idx]
		for _, childIdx := range node.childIdxs {
			if childIdx >= 0 && childIdx < len(n.tree.nodes) {
				toNode := TreeGraphNode[N, E]{
					tree: n.tree,
					idx:  childIdx,
				}
				var cost E
				if c, ok := n.tree.GetCost(n.idx, childIdx); ok {
					cost = c
				} else {
					// Default cost (for float32, this would be 1.0)
					var zero E
					var one E
					if any(zero) == float32(0) {
						one = any(float32(1.0)).(E)
					}
					cost = one
				}

				edge := TreeGraphEdge[N, E]{
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
func (n TreeGraphNode[N, E]) Cost(toOther Node[N, E]) float32 {
	if toOther == nil || n.tree == nil {
		return 0
	}
	to, ok := toOther.(TreeGraphNode[N, E])
	if !ok || n.tree != to.tree {
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

// TreeGraphEdge implements Edge[N, E] for GenericTree
type TreeGraphEdge[N any, E any] struct {
	from TreeGraphNode[N, E]
	to   TreeGraphNode[N, E]
	data E
	id   int64
}

// ID returns a unique identifier for this edge
func (e TreeGraphEdge[N, E]) ID() int64 {
	return e.from.ID()*1000000 + e.to.ID()
}

// From returns the source node
func (e TreeGraphEdge[N, E]) From() Node[N, E] {
	return e.from
}

// To returns the destination node
func (e TreeGraphEdge[N, E]) To() Node[N, E] {
	return e.to
}

// Data returns the edge's data
func (e TreeGraphEdge[N, E]) Data() E {
	return e.data
}

// Cost returns the cost/weight of this edge
func (e TreeGraphEdge[N, E]) Cost() float32 {
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

// GenericTree implements Graph[N, E] interface
var _ Graph[any, float32] = (*GenericTree[any, float32])(nil)

// Nodes returns an iterator over all nodes in the tree
func (t *GenericTree[N, E]) Nodes() iter.Seq[Node[N, E]] {
	return func(yield func(Node[N, E]) bool) {
		for i := 0; i < len(t.nodes); i++ {
			// Skip removed nodes (parentIdx == -2)
			if t.nodes[i].parentIdx == -2 {
				continue
			}
			node := TreeGraphNode[N, E]{
				tree: t,
				idx:  i,
			}
			if !yield(node) {
				return
			}
		}
	}
}

// Edges returns an iterator over all edges in the tree
func (t *GenericTree[N, E]) Edges() iter.Seq[Edge[N, E]] {
	return func(yield func(Edge[N, E]) bool) {
		for node := range t.Nodes() {
			tn := node.(TreeGraphNode[N, E])
			for edge := range tn.Edges() {
				if !yield(edge) {
					return
				}
			}
		}
	}
}

// Neighbors returns an iterator over neighbors of a given node
func (t *GenericTree[N, E]) Neighbors(n Node[N, E]) iter.Seq[Node[N, E]] {
	return n.Neighbors()
}

// NumNodes returns the total number of nodes in the tree
func (t *GenericTree[N, E]) NumNodes() int {
	count := 0
	for i := 0; i < len(t.nodes); i++ {
		// Skip removed nodes (parentIdx == -2)
		if t.nodes[i].parentIdx != -2 {
			count++
		}
	}
	return count
}

// NumEdges returns the total number of edges in the tree
func (t *GenericTree[N, E]) NumEdges() int {
	count := 0
	for i := 0; i < len(t.nodes); i++ {
		// Skip removed nodes
		if t.nodes[i].parentIdx == -2 {
			continue
		}
		// Count children (each child is an edge)
		count += len(t.nodes[i].childIdxs)
	}
	return count
}
