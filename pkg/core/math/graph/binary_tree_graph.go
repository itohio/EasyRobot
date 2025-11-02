package graph

// BinaryTreeGraph adapts BinaryTree to Graph interface
type BinaryTreeGraph struct {
	tree  *BinaryTree
	costs map[*BinaryTreeNode]map[*BinaryTreeNode]float32
}

// NewBinaryTreeGraph creates a new BinaryTreeGraph from a binary tree
func NewBinaryTreeGraph(tree *BinaryTree) *BinaryTreeGraph {
	return &BinaryTreeGraph{
		tree:  tree,
		costs: make(map[*BinaryTreeNode]map[*BinaryTreeNode]float32),
	}
}

// SetCost sets the cost for edge from parent to child
func (btg *BinaryTreeGraph) SetCost(parent, child *BinaryTreeNode, cost float32) {
	if btg.costs[parent] == nil {
		btg.costs[parent] = make(map[*BinaryTreeNode]float32)
	}
	btg.costs[parent][child] = cost
}

// BinaryTreeNodeNode adapts BinaryTreeNode to Node interface
type BinaryTreeNodeNode struct {
	binaryTreeNode *BinaryTreeNode
}

func (n BinaryTreeNodeNode) Equal(other Node) bool {
	o, ok := other.(BinaryTreeNodeNode)
	if !ok {
		return false
	}
	return n.binaryTreeNode == o.binaryTreeNode
}

func (btg *BinaryTreeGraph) Neighbors(n Node) []Node {
	btn, ok := n.(BinaryTreeNodeNode)
	if !ok {
		return nil
	}

	if btn.binaryTreeNode == nil {
		return nil
	}

	neighbors := make([]Node, 0, 2) // Binary tree has max 2 children

	// Add left child if exists
	if btn.binaryTreeNode.Left != nil {
		neighbors = append(neighbors, BinaryTreeNodeNode{binaryTreeNode: btn.binaryTreeNode.Left})
	}

	// Add right child if exists
	if btn.binaryTreeNode.Right != nil {
		neighbors = append(neighbors, BinaryTreeNodeNode{binaryTreeNode: btn.binaryTreeNode.Right})
	}

	return neighbors
}

func (btg *BinaryTreeGraph) Cost(from, to Node) float32 {
	fromNode, ok := from.(BinaryTreeNodeNode)
	if !ok {
		return 0
	}
	toNode, ok := to.(BinaryTreeNodeNode)
	if !ok {
		return 0
	}

	if fromNode.binaryTreeNode == nil || toNode.binaryTreeNode == nil {
		return 0
	}

	// Check if toNode is a child of fromNode
	isLeftChild := fromNode.binaryTreeNode.Left == toNode.binaryTreeNode
	isRightChild := fromNode.binaryTreeNode.Right == toNode.binaryTreeNode

	if !isLeftChild && !isRightChild {
		return 0 // Not a child, no edge
	}

	// Check if custom cost is set
	if costs, exists := btg.costs[fromNode.binaryTreeNode]; exists {
		if cost, exists := costs[toNode.binaryTreeNode]; exists {
			return cost
		}
	}

	// Default unit cost
	return 1.0
}
