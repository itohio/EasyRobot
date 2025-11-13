package graph

// Forest represents a collection of disjoint trees
type Forest struct {
	Trees []*Tree
}

// NewForest creates a new empty forest
func NewForest() *Forest {
	return &Forest{
		Trees: make([]*Tree, 0),
	}
}

// AddTree adds a tree to the forest
func (f *Forest) AddTree(tree *Tree) {
	if tree == nil {
		return
	}

	f.Trees = append(f.Trees, tree)
}

// RemoveTree removes a tree from the forest
func (f *Forest) RemoveTree(tree *Tree) bool {
	if tree == nil {
		return false
	}

	for i, t := range f.Trees {
		if t == tree {
			f.Trees = append(f.Trees[:i], f.Trees[i+1:]...)
			return true
		}
	}

	return false
}

// FindTreeContainingNode finds the tree that contains the given node
func (f *Forest) FindTreeContainingNode(node *TreeNode) *Tree {
	if node == nil {
		return nil
	}

	// Find root by traversing up
	root := node
	for root.Parent != nil {
		root = root.Parent
	}

	// Find tree with this root
	for _, tree := range f.Trees {
		if tree.Root == root {
			return tree
		}
	}

	return nil
}

// MergeTrees merges two trees by connecting their roots
// tree1 becomes parent of tree2's root
func (f *Forest) MergeTrees(tree1, tree2 *Tree) bool {
	if tree1 == nil || tree2 == nil {
		return false
	}

	// Make tree2's root a child of tree1's root
	tree1.AddChild(tree1.Root, tree2.Root.Data)

	// Update tree2's root children to be children of new node
	newNode := tree1.Root.Children[len(tree1.Root.Children)-1]
	newNode.Children = tree2.Root.Children

	// Update parent pointers
	for _, child := range newNode.Children {
		child.Parent = newNode
	}

	// Remove tree2 from forest
	f.RemoveTree(tree2)

	return true
}

// GetTreeCount returns number of trees in forest
func (f *Forest) GetTreeCount() int {
	return len(f.Trees)
}
