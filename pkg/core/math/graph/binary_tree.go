package graph

// BinaryTreeNode represents a node in a binary tree
type BinaryTreeNode struct {
	Data   any
	Left   *BinaryTreeNode
	Right  *BinaryTreeNode
	Parent *BinaryTreeNode
}

// BinaryTree represents a binary tree structure
type BinaryTree struct {
	Root *BinaryTreeNode
}

// NewBinaryTree creates a new binary tree with a root node
func NewBinaryTree(rootData any) *BinaryTree {
	return &BinaryTree{
		Root: &BinaryTreeNode{
			Data:   rootData,
			Left:   nil,
			Right:  nil,
			Parent: nil,
		},
	}
}

// SetLeft sets the left child of the given node
func (bt *BinaryTree) SetLeft(parent *BinaryTreeNode, data any) *BinaryTreeNode {
	if parent == nil {
		return nil
	}

	child := &BinaryTreeNode{
		Data:   data,
		Left:   nil,
		Right:  nil,
		Parent: parent,
	}

	parent.Left = child
	return child
}

// SetRight sets the right child of the given node
func (bt *BinaryTree) SetRight(parent *BinaryTreeNode, data any) *BinaryTreeNode {
	if parent == nil {
		return nil
	}

	child := &BinaryTreeNode{
		Data:   data,
		Left:   nil,
		Right:  nil,
		Parent: parent,
	}

	parent.Right = child
	return child
}

// RemoveLeft removes the left child of the given node
func (bt *BinaryTree) RemoveLeft(node *BinaryTreeNode) bool {
	if node == nil || node.Left == nil {
		return false
	}

	node.Left.Parent = nil
	node.Left = nil
	return true
}

// RemoveRight removes the right child of the given node
func (bt *BinaryTree) RemoveRight(node *BinaryTreeNode) bool {
	if node == nil || node.Right == nil {
		return false
	}

	node.Right.Parent = nil
	node.Right = nil
	return true
}

// FindPathToRoot returns path from node to root
func (bt *BinaryTree) FindPathToRoot(node *BinaryTreeNode) []*BinaryTreeNode {
	if node == nil {
		return nil
	}

	var path []*BinaryTreeNode
	current := node

	for current != nil {
		path = append(path, current)
		current = current.Parent
	}

	return path
}

// GetDepth returns depth of node (distance from root)
func (bt *BinaryTree) GetDepth(node *BinaryTreeNode) int {
	if node == nil {
		return -1
	}

	depth := 0
	current := node

	for current.Parent != nil {
		depth++
		current = current.Parent
	}

	return depth
}

// GetHeight returns height of tree (max depth from root)
func (bt *BinaryTree) GetHeight() int {
	return bt.getHeightRecursive(bt.Root)
}

func (bt *BinaryTree) getHeightRecursive(node *BinaryTreeNode) int {
	if node == nil {
		return -1
	}

	leftHeight := bt.getHeightRecursive(node.Left)
	rightHeight := bt.getHeightRecursive(node.Right)

	maxChildHeight := leftHeight
	if rightHeight > leftHeight {
		maxChildHeight = rightHeight
	}

	return maxChildHeight + 1
}

// FindNodeByData finds first node with matching data (pre-order traversal)
func (bt *BinaryTree) FindNodeByData(data any) *BinaryTreeNode {
	return bt.findNodeRecursive(bt.Root, data)
}

func (bt *BinaryTree) findNodeRecursive(node *BinaryTreeNode, data any) *BinaryTreeNode {
	if node == nil {
		return nil
	}

	if node.Data == data {
		return node
	}

	// Search left subtree
	if found := bt.findNodeRecursive(node.Left, data); found != nil {
		return found
	}

	// Search right subtree
	return bt.findNodeRecursive(node.Right, data)
}

// InOrderTraversal performs in-order traversal and calls fn for each node
func (bt *BinaryTree) InOrderTraversal(fn func(*BinaryTreeNode)) {
	bt.inOrderRecursive(bt.Root, fn)
}

func (bt *BinaryTree) inOrderRecursive(node *BinaryTreeNode, fn func(*BinaryTreeNode)) {
	if node == nil {
		return
	}

	bt.inOrderRecursive(node.Left, fn)
	fn(node)
	bt.inOrderRecursive(node.Right, fn)
}

// PreOrderTraversal performs pre-order traversal and calls fn for each node
func (bt *BinaryTree) PreOrderTraversal(fn func(*BinaryTreeNode)) {
	bt.preOrderRecursive(bt.Root, fn)
}

func (bt *BinaryTree) preOrderRecursive(node *BinaryTreeNode, fn func(*BinaryTreeNode)) {
	if node == nil {
		return
	}

	fn(node)
	bt.preOrderRecursive(node.Left, fn)
	bt.preOrderRecursive(node.Right, fn)
}

// PostOrderTraversal performs post-order traversal and calls fn for each node
func (bt *BinaryTree) PostOrderTraversal(fn func(*BinaryTreeNode)) {
	bt.postOrderRecursive(bt.Root, fn)
}

func (bt *BinaryTree) postOrderRecursive(node *BinaryTreeNode, fn func(*BinaryTreeNode)) {
	if node == nil {
		return
	}

	bt.postOrderRecursive(node.Left, fn)
	bt.postOrderRecursive(node.Right, fn)
	fn(node)
}
