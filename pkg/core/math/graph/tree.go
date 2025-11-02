package graph

// TreeNode represents a node in a tree
type TreeNode struct {
	Data     any
	Parent   *TreeNode
	Children []*TreeNode
}

// Tree represents a hierarchical tree structure
type Tree struct {
	Root *TreeNode
}

// NewTree creates a new tree with a root node
func NewTree(rootData any) *Tree {
	return &Tree{
		Root: &TreeNode{
			Data:     rootData,
			Parent:   nil,
			Children: make([]*TreeNode, 0),
		},
	}
}

// AddChild adds a child node to the given parent
func (t *Tree) AddChild(parent *TreeNode, childData any) *TreeNode {
	if parent == nil {
		return nil
	}

	child := &TreeNode{
		Data:     childData,
		Parent:   parent,
		Children: make([]*TreeNode, 0),
	}

	parent.Children = append(parent.Children, child)
	return child
}

// RemoveChild removes a child node from the given parent
func (t *Tree) RemoveChild(parent, child *TreeNode) bool {
	if parent == nil || child == nil {
		return false
	}

	for i, c := range parent.Children {
		if c == child {
			parent.Children = append(parent.Children[:i], parent.Children[i+1:]...)
			child.Parent = nil
			return true
		}
	}

	return false
}

// FindPathToRoot returns path from node to root
func (t *Tree) FindPathToRoot(node *TreeNode) []*TreeNode {
	if node == nil {
		return nil
	}

	var path []*TreeNode
	current := node

	for current != nil {
		path = append(path, current)
		current = current.Parent
	}

	return path
}

// GetDepth returns depth of node (distance from root)
func (t *Tree) GetDepth(node *TreeNode) int {
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
func (t *Tree) GetHeight() int {
	return t.getHeightRecursive(t.Root)
}

func (t *Tree) getHeightRecursive(node *TreeNode) int {
	if node == nil {
		return -1
	}

	maxChildHeight := -1
	for _, child := range node.Children {
		childHeight := t.getHeightRecursive(child)
		if childHeight > maxChildHeight {
			maxChildHeight = childHeight
		}
	}

	return maxChildHeight + 1
}

// FindNodeByData finds first node with matching data
func (t *Tree) FindNodeByData(data any) *TreeNode {
	return t.findNodeRecursive(t.Root, data)
}

func (t *Tree) findNodeRecursive(node *TreeNode, data any) *TreeNode {
	if node == nil {
		return nil
	}

	if node.Data == data {
		return node
	}

	for _, child := range node.Children {
		if found := t.findNodeRecursive(child, data); found != nil {
			return found
		}
	}

	return nil
}
