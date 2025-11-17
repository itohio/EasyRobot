package graph

import "iter"

// BinaryTreeNodeValue represents a node in a generic binary tree stored as a value
// Uses indices instead of pointers for left/right/parent relationships
type BinaryTreeNodeValue[N any] struct {
	id        int64
	data      N
	parentIdx int // Index of parent node (-1 for root)
	leftIdx   int // Index of left child (-1 if none)
	rightIdx  int // Index of right child (-1 if none)
}

// GenericBinaryTree represents a binary tree structure using value arrays
// Node data type: N (generic)
// Edge data type: E (generic, typically float32 for cost)
type GenericBinaryTree[N any, E any] struct {
	nodes   []BinaryTreeNodeValue[N] // Array of node values
	nodeMap map[int64]int            // ID -> index in nodes array
	nextID  int64
	costMap map[int]map[int]E // parentIdx -> childIdx -> cost
	rootIdx int               // Index of root node (-1 if empty)
}

// NewGenericBinaryTree creates a new generic binary tree with a root node
func NewGenericBinaryTree[N any, E any](rootData N) *GenericBinaryTree[N, E] {
	t := &GenericBinaryTree[N, E]{
		nodes:   make([]BinaryTreeNodeValue[N], 0),
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
func (t *GenericBinaryTree[N, E]) addNode(data N, parentIdx int) int {
	id := t.nextID
	t.nextID++

	node := BinaryTreeNodeValue[N]{
		id:        id,
		data:      data,
		parentIdx: parentIdx,
		leftIdx:   -1,
		rightIdx:  -1,
	}

	idx := len(t.nodes)
	t.nodes = append(t.nodes, node)
	t.nodeMap[id] = idx

	return idx
}

// SetLeft sets the left child of the given node and returns the new node's index
func (t *GenericBinaryTree[N, E]) SetLeft(parentIdx int, data N) int {
	if parentIdx < 0 || parentIdx >= len(t.nodes) {
		return -1
	}

	// Check if left child already exists
	if t.nodes[parentIdx].leftIdx >= 0 {
		// Update existing node's data
		leftIdx := t.nodes[parentIdx].leftIdx
		t.nodes[leftIdx].data = data
		return leftIdx
	}

	// Create new left child
	leftIdx := t.addNode(data, parentIdx)
	t.nodes[parentIdx].leftIdx = leftIdx
	return leftIdx
}

// SetRight sets the right child of the given node and returns the new node's index
func (t *GenericBinaryTree[N, E]) SetRight(parentIdx int, data N) int {
	if parentIdx < 0 || parentIdx >= len(t.nodes) {
		return -1
	}

	// Check if right child already exists
	if t.nodes[parentIdx].rightIdx >= 0 {
		// Update existing node's data
		rightIdx := t.nodes[parentIdx].rightIdx
		t.nodes[rightIdx].data = data
		return rightIdx
	}

	// Create new right child
	rightIdx := t.addNode(data, parentIdx)
	t.nodes[parentIdx].rightIdx = rightIdx
	return rightIdx
}

// Insert inserts a node using the provided binary insertion strategy
func (t *GenericBinaryTree[N, E]) Insert(data N, strategy BinaryInsertionStrategy[N, E]) int {
	if strategy == nil {
		// Default: insert at root as left child if available, else right
		if t.rootIdx < 0 {
			return -1
		}
		if t.nodes[t.rootIdx].leftIdx < 0 {
			return t.SetLeft(t.rootIdx, data)
		}
		return t.SetRight(t.rootIdx, data)
	}

	parentIdx, shouldGoLeft := strategy(t, data, t.rootIdx)
	if parentIdx < 0 {
		return -1
	}

	if shouldGoLeft {
		return t.SetLeft(parentIdx, data)
	}
	return t.SetRight(parentIdx, data)
}

// nodeData represents a node's data and ID for tree rebuilding
type nodeData[N any] struct {
	data N
	id   int64
}

// Balance attempts to balance the binary tree
// Uses a simple approach: rebuild tree from sorted nodes
func (t *GenericBinaryTree[N, E]) Balance(compare ComparisonStrategy[N]) {
	if t.rootIdx < 0 || len(t.nodes) == 0 {
		return
	}

	// Collect all nodes with their data
	var nodes []nodeData[N]
	t.collectNodes(t.rootIdx, &nodes)

	// Sort nodes by data using comparison strategy
	if compare != nil {
		// Simple bubble sort (could be optimized)
		for i := 0; i < len(nodes)-1; i++ {
			for j := i + 1; j < len(nodes); j++ {
				if compare(nodes[i].data, nodes[j].data) > 0 {
					nodes[i], nodes[j] = nodes[j], nodes[i]
				}
			}
		}
	}

	// Rebuild tree as balanced BST
	t.nodes = make([]BinaryTreeNodeValue[N], 0)
	t.nodeMap = make(map[int64]int)
	t.nextID = 1
	t.rootIdx = -1
	t.costMap = make(map[int]map[int]E)

	// Build balanced tree from sorted nodes
	if len(nodes) > 0 {
		t.rootIdx = t.buildBalancedBST(nodes, 0, len(nodes)-1, compare, -1)
	}
}

func (t *GenericBinaryTree[N, E]) collectNodes(nodeIdx int, nodes *[]nodeData[N]) {
	if nodeIdx < 0 || nodeIdx >= len(t.nodes) {
		return
	}

	node := t.nodes[nodeIdx]
	*nodes = append(*nodes, nodeData[N]{data: node.data, id: node.id})

	if node.leftIdx >= 0 {
		t.collectNodes(node.leftIdx, nodes)
	}
	if node.rightIdx >= 0 {
		t.collectNodes(node.rightIdx, nodes)
	}
}

func (t *GenericBinaryTree[N, E]) buildBalancedBST(nodes []nodeData[N], start, end int, compare ComparisonStrategy[N], parentIdx int) int {
	if start > end {
		return -1
	}

	mid := (start + end) / 2
	nodeData := nodes[mid]

	// Create node
	id := t.nextID
	t.nextID++

	node := BinaryTreeNodeValue[N]{
		id:        id,
		data:      nodeData.data,
		parentIdx: parentIdx,
		leftIdx:   -1,
		rightIdx:  -1,
	}

	idx := len(t.nodes)
	t.nodes = append(t.nodes, node)
	t.nodeMap[id] = idx

	// Recursively build left and right subtrees
	leftIdx := t.buildBalancedBST(nodes, start, mid-1, compare, idx)
	rightIdx := t.buildBalancedBST(nodes, mid+1, end, compare, idx)

	// Update node with children indices
	t.nodes[idx].leftIdx = leftIdx
	t.nodes[idx].rightIdx = rightIdx

	// Update parent pointers for children
	if leftIdx >= 0 && leftIdx < len(t.nodes) {
		t.nodes[leftIdx].parentIdx = idx
	}
	if rightIdx >= 0 && rightIdx < len(t.nodes) {
		t.nodes[rightIdx].parentIdx = idx
	}

	return idx
}

// RemoveLeft removes the left child of the given node
func (t *GenericBinaryTree[N, E]) RemoveLeft(parentIdx int) bool {
	if parentIdx < 0 || parentIdx >= len(t.nodes) {
		return false
	}

	leftIdx := t.nodes[parentIdx].leftIdx
	if leftIdx < 0 {
		return false
	}

	// Mark node as removed (set parentIdx to -2)
	t.nodes[leftIdx].parentIdx = -2
	t.nodes[parentIdx].leftIdx = -1

	// Remove from cost map
	if costs, exists := t.costMap[parentIdx]; exists {
		delete(costs, leftIdx)
	}

	return true
}

// RemoveRight removes the right child of the given node
func (t *GenericBinaryTree[N, E]) RemoveRight(parentIdx int) bool {
	if parentIdx < 0 || parentIdx >= len(t.nodes) {
		return false
	}

	rightIdx := t.nodes[parentIdx].rightIdx
	if rightIdx < 0 {
		return false
	}

	// Mark node as removed (set parentIdx to -2)
	t.nodes[rightIdx].parentIdx = -2
	t.nodes[parentIdx].rightIdx = -1

	// Remove from cost map
	if costs, exists := t.costMap[parentIdx]; exists {
		delete(costs, rightIdx)
	}

	return true
}

// SetCost sets the cost for edge from parent to child
func (t *GenericBinaryTree[N, E]) SetCost(parentIdx, childIdx int, cost E) {
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
func (t *GenericBinaryTree[N, E]) GetCost(parentIdx, childIdx int) (E, bool) {
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
func (t *GenericBinaryTree[N, E]) RootIdx() int {
	return t.rootIdx
}

// Root returns the root node of the tree, or nil if empty
func (t *GenericBinaryTree[N, E]) Root() Node[N, E] {
	if t.rootIdx < 0 || t.rootIdx >= len(t.nodes) {
		return nil
	}
	return BinaryTreeGraphNode[N, E]{tree: t, idx: t.rootIdx}
}

// NodeCount returns the number of nodes in the tree
func (t *GenericBinaryTree[N, E]) NodeCount() int {
	return len(t.nodes)
}

// GetNode returns the node at the given index
func (t *GenericBinaryTree[N, E]) GetNode(idx int) *BinaryTreeNodeValue[N] {
	if idx < 0 || idx >= len(t.nodes) {
		return nil
	}
	return &t.nodes[idx]
}

// FindPathToRoot returns path from node to root as indices
func (t *GenericBinaryTree[N, E]) FindPathToRoot(nodeIdx int) []int {
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
func (t *GenericBinaryTree[N, E]) GetDepth(nodeIdx int) int {
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
func (t *GenericBinaryTree[N, E]) GetHeight() int {
	if t.rootIdx < 0 {
		return -1
	}
	return t.getHeightRecursive(t.rootIdx)
}

func (t *GenericBinaryTree[N, E]) getHeightRecursive(nodeIdx int) int {
	if nodeIdx < 0 || nodeIdx >= len(t.nodes) {
		return -1
	}

	node := t.nodes[nodeIdx]
	leftHeight := t.getHeightRecursive(node.leftIdx)
	rightHeight := t.getHeightRecursive(node.rightIdx)

	maxChildHeight := leftHeight
	if rightHeight > leftHeight {
		maxChildHeight = rightHeight
	}

	return maxChildHeight + 1
}

// FindNodeByData finds first node with matching data and returns its index
// It uses Node.Equal() and Node.Compare() methods for comparison
func (t *GenericBinaryTree[N, E]) FindNodeByData(data N) int {
	return t.findNodeRecursive(t.rootIdx, data)
}

func (t *GenericBinaryTree[N, E]) findNodeRecursive(nodeIdx int, data N) int {
	if nodeIdx < 0 || nodeIdx >= len(t.nodes) {
		return -1
	}

	node := BinaryTreeGraphNode[N, E]{
		tree: t,
		idx:  nodeIdx,
	}

	// Create a temporary node with the target data for comparison
	dataNode := &dataNodeWrapper[N, E]{data: data}

	// Compare data values (call Equal on dataNode to use its comparison logic)
	if dataNode.Equal(node) {
		return nodeIdx
	}

	// Search both subtrees (binary tree, not necessarily a BST)
	// Search left subtree
	if t.nodes[nodeIdx].leftIdx >= 0 {
		if found := t.findNodeRecursive(t.nodes[nodeIdx].leftIdx, data); found >= 0 {
			return found
		}
	}

	// Search right subtree
	if t.nodes[nodeIdx].rightIdx >= 0 {
		if found := t.findNodeRecursive(t.nodes[nodeIdx].rightIdx, data); found >= 0 {
			return found
		}
	}

	return -1
}

// dataNodeWrapper is defined in tree.go - reuse it here

// BinaryTreeGraphNode implements Node[N, E] for GenericBinaryTree
type BinaryTreeGraphNode[N any, E any] struct {
	tree *GenericBinaryTree[N, E]
	idx  int
}

// ID returns a unique identifier for this node
func (n BinaryTreeGraphNode[N, E]) ID() int64 {
	if n.tree == nil || n.idx < 0 || n.idx >= len(n.tree.nodes) {
		return 0
	}
	return n.tree.nodes[n.idx].id
}

// Equal implements Comparable[N, E]
func (n BinaryTreeGraphNode[N, E]) Equal(other Node[N, E]) bool {
	if other == nil {
		return false
	}
	o, ok := other.(BinaryTreeGraphNode[N, E])
	if !ok {
		return false
	}
	return n.tree == o.tree && n.idx == o.idx
}

// Compare implements Comparable[N, E]
func (n BinaryTreeGraphNode[N, E]) Compare(other Node[N, E]) int {
	if other == nil {
		return 1
	}
	o, ok := other.(BinaryTreeGraphNode[N, E])
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
func (n BinaryTreeGraphNode[N, E]) Data() N {
	if n.tree == nil || n.idx < 0 || n.idx >= len(n.tree.nodes) {
		var zero N
		return zero
	}
	return n.tree.nodes[n.idx].data
}

// Neighbors returns an iterator over neighboring nodes (left and right children)
func (n BinaryTreeGraphNode[N, E]) Neighbors() iter.Seq[Node[N, E]] {
	return func(yield func(Node[N, E]) bool) {
		if n.tree == nil || n.idx < 0 || n.idx >= len(n.tree.nodes) {
			return
		}

		node := n.tree.nodes[n.idx]

		// Add left child if exists
		if node.leftIdx >= 0 && node.leftIdx < len(n.tree.nodes) {
			neighbor := BinaryTreeGraphNode[N, E]{
				tree: n.tree,
				idx:  node.leftIdx,
			}
			if !yield(neighbor) {
				return
			}
		}

		// Add right child if exists
		if node.rightIdx >= 0 && node.rightIdx < len(n.tree.nodes) {
			neighbor := BinaryTreeGraphNode[N, E]{
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
func (n BinaryTreeGraphNode[N, E]) NumNeighbors() int {
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
func (n BinaryTreeGraphNode[N, E]) Edges() iter.Seq[Edge[N, E]] {
	return func(yield func(Edge[N, E]) bool) {
		if n.tree == nil || n.idx < 0 || n.idx >= len(n.tree.nodes) {
			return
		}

		node := n.tree.nodes[n.idx]

		// Add left child edge if exists
		if node.leftIdx >= 0 && node.leftIdx < len(n.tree.nodes) {
			toNode := BinaryTreeGraphNode[N, E]{
				tree: n.tree,
				idx:  node.leftIdx,
			}
			var cost E
			if c, ok := n.tree.GetCost(n.idx, node.leftIdx); ok {
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

			edge := BinaryTreeGraphEdge[N, E]{
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
			toNode := BinaryTreeGraphNode[N, E]{
				tree: n.tree,
				idx:  node.rightIdx,
			}
			var cost E
			if c, ok := n.tree.GetCost(n.idx, node.rightIdx); ok {
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

			edge := BinaryTreeGraphEdge[N, E]{
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
func (n BinaryTreeGraphNode[N, E]) Cost(toOther Node[N, E]) float32 {
	if toOther == nil || n.tree == nil {
		return 0
	}
	to, ok := toOther.(BinaryTreeGraphNode[N, E])
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

// BinaryTreeGraphEdge implements Edge[N, E] for GenericBinaryTree
type BinaryTreeGraphEdge[N any, E any] struct {
	from BinaryTreeGraphNode[N, E]
	to   BinaryTreeGraphNode[N, E]
	data E
	id   int64
}

// ID returns a unique identifier for this edge
func (e BinaryTreeGraphEdge[N, E]) ID() int64 {
	return e.from.ID()*1000000 + e.to.ID()
}

// From returns the source node
func (e BinaryTreeGraphEdge[N, E]) From() Node[N, E] {
	return e.from
}

// To returns the destination node
func (e BinaryTreeGraphEdge[N, E]) To() Node[N, E] {
	return e.to
}

// Data returns the edge's data
func (e BinaryTreeGraphEdge[N, E]) Data() E {
	return e.data
}

// Cost returns the cost/weight of this edge
func (e BinaryTreeGraphEdge[N, E]) Cost() float32 {
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

// GenericBinaryTree implements Graph[N, E] interface
var _ Graph[any, float32] = (*GenericBinaryTree[any, float32])(nil)

// Nodes returns an iterator over all nodes in the binary tree
func (t *GenericBinaryTree[N, E]) Nodes() iter.Seq[Node[N, E]] {
	return func(yield func(Node[N, E]) bool) {
		for i := 0; i < len(t.nodes); i++ {
			// Skip removed nodes (parentIdx == -2)
			if t.nodes[i].parentIdx == -2 {
				continue
			}
			node := BinaryTreeGraphNode[N, E]{
				tree: t,
				idx:  i,
			}
			if !yield(node) {
				return
			}
		}
	}
}

// Edges returns an iterator over all edges in the binary tree
func (t *GenericBinaryTree[N, E]) Edges() iter.Seq[Edge[N, E]] {
	return func(yield func(Edge[N, E]) bool) {
		for node := range t.Nodes() {
			btn := node.(BinaryTreeGraphNode[N, E])
			for edge := range btn.Edges() {
				if !yield(edge) {
					return
				}
			}
		}
	}
}

// Neighbors returns an iterator over neighbors of a given node
func (t *GenericBinaryTree[N, E]) Neighbors(n Node[N, E]) iter.Seq[Node[N, E]] {
	return n.Neighbors()
}

// NumNodes returns the total number of nodes in the binary tree
func (t *GenericBinaryTree[N, E]) NumNodes() int {
	count := 0
	for i := 0; i < len(t.nodes); i++ {
		// Skip removed nodes (parentIdx == -2)
		if t.nodes[i].parentIdx != -2 {
			count++
		}
	}
	return count
}

// NumEdges returns the total number of edges in the binary tree
func (t *GenericBinaryTree[N, E]) NumEdges() int {
	count := 0
	for i := 0; i < len(t.nodes); i++ {
		// Skip removed nodes
		if t.nodes[i].parentIdx == -2 {
			continue
		}
		// Count left and right children (each is an edge)
		if t.nodes[i].leftIdx >= 0 && t.nodes[i].leftIdx < len(t.nodes) {
			count++
		}
		if t.nodes[i].rightIdx >= 0 && t.nodes[i].rightIdx < len(t.nodes) {
			count++
		}
	}
	return count
}
