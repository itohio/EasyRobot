package graph

// ForestGraph adapts Forest to Graph interface
// Nodes from different trees are not connected
type ForestGraph struct {
	forest *Forest
	costs  map[*TreeNode]map[*TreeNode]float32
}

// NewForestGraph creates a new ForestGraph from a forest
func NewForestGraph(forest *Forest) *ForestGraph {
	return &ForestGraph{
		forest: forest,
		costs:  make(map[*TreeNode]map[*TreeNode]float32),
	}
}

// SetCost sets the cost for edge from parent to child
func (fg *ForestGraph) SetCost(parent, child *TreeNode, cost float32) {
	if fg.costs[parent] == nil {
		fg.costs[parent] = make(map[*TreeNode]float32)
	}
	fg.costs[parent][child] = cost
}

func (fg *ForestGraph) Neighbors(n Node) []Node {
	tn, ok := n.(TreeNodeNode)
	if !ok {
		return nil
	}

	if tn.treeNode == nil {
		return nil
	}

	// Only return neighbors within the same tree
	neighbors := make([]Node, 0, len(tn.treeNode.Children))
	for _, child := range tn.treeNode.Children {
		neighbors = append(neighbors, TreeNodeNode{treeNode: child})
	}

	return neighbors
}

func (fg *ForestGraph) Cost(from, to Node) float32 {
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

	// Check if nodes are in the same tree
	fromTree := fg.forest.FindTreeContainingNode(fromNode.treeNode)
	toTree := fg.forest.FindTreeContainingNode(toNode.treeNode)

	if fromTree != toTree || fromTree == nil {
		return 0 // Nodes in different trees are not connected
	}

	// Check if toNode is a child of fromNode
	for _, child := range fromNode.treeNode.Children {
		if child == toNode.treeNode {
			// Check if custom cost is set
			if costs, exists := fg.costs[fromNode.treeNode]; exists {
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
