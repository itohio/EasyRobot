package graph

// TreeGraph adapts Tree to Graph interface
type TreeGraph struct {
	tree  *Tree
	costs map[*TreeNode]map[*TreeNode]float32
}

// NewTreeGraph creates a new TreeGraph from a tree
func NewTreeGraph(tree *Tree) *TreeGraph {
	return &TreeGraph{
		tree:  tree,
		costs: make(map[*TreeNode]map[*TreeNode]float32),
	}
}

// SetCost sets the cost for edge from parent to child
func (tg *TreeGraph) SetCost(parent, child *TreeNode, cost float32) {
	if tg.costs[parent] == nil {
		tg.costs[parent] = make(map[*TreeNode]float32)
	}
	tg.costs[parent][child] = cost
}

// TreeNodeNode adapts TreeNode to Node interface
type TreeNodeNode struct {
	treeNode *TreeNode
}

func (n TreeNodeNode) Equal(other Node) bool {
	o, ok := other.(TreeNodeNode)
	if !ok {
		return false
	}
	return n.treeNode == o.treeNode
}

func (tg *TreeGraph) Neighbors(n Node) []Node {
	tn, ok := n.(TreeNodeNode)
	if !ok {
		return nil
	}

	if tn.treeNode == nil {
		return nil
	}

	neighbors := make([]Node, 0, len(tn.treeNode.Children))
	for _, child := range tn.treeNode.Children {
		neighbors = append(neighbors, TreeNodeNode{treeNode: child})
	}

	return neighbors
}

func (tg *TreeGraph) Cost(from, to Node) float32 {
	fromNode, ok := from.(TreeNodeNode)
	if !ok {
		return 0
	}
	toNode, ok := to.(TreeNodeNode)
	if !ok {
		return 0
	}

	if fromNode.treeNode == nil || toNode.treeNode == nil {
		return 0
	}

	// Check if toNode is a child of fromNode
	for _, child := range fromNode.treeNode.Children {
		if child == toNode.treeNode {
			// Check if custom cost is set
			if costs, exists := tg.costs[fromNode.treeNode]; exists {
				if cost, exists := costs[toNode.treeNode]; exists {
					return cost
				}
			}
			// Default unit cost
			return 1.0
		}
	}

	return 0
}
