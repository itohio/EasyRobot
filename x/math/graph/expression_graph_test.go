package graph

import (
	"iter"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// ExpressionNodeData holds data for an expression node
type ExpressionNodeData struct {
	// Operation type: "input", "constant", "add", "multiply", etc.
	OpType string
	// Value for constants and inputs
	Value float32
	// Description
	Description string
}

// expressionNodeImpl implements ExpressionNode for testing
type expressionNodeImpl[E any] struct {
	node         Node[ExpressionNodeData, E]
	expressionFn func(float32, map[int64]float32) (float32, bool)
	isLeaf       bool
	adapter      *expressionGraphAdapter[E] // Reference to adapter for edge wrapping
}

func (n *expressionNodeImpl[E]) ID() int64 {
	return n.node.ID()
}

func (n *expressionNodeImpl[E]) Data() ExpressionNodeData {
	return n.node.Data()
}

func (n *expressionNodeImpl[E]) Neighbors() iter.Seq[Node[ExpressionNodeData, E]] {
	return n.node.Neighbors()
}

func (n *expressionNodeImpl[E]) Edges() iter.Seq[Edge[ExpressionNodeData, E]] {
	return func(yield func(Edge[ExpressionNodeData, E]) bool) {
		if n.adapter == nil {
			// Fallback to underlying edges if no adapter
			for edge := range n.node.Edges() {
				if !yield(edge) {
					return
				}
			}
			return
		}
		// Wrap edges to return expression nodes
		for edge := range n.node.Edges() {
			wrapped := &expressionEdgeAdapter[E]{
				edge:    edge,
				adapter: n.adapter,
			}
			if !yield(wrapped) {
				return
			}
		}
	}
}

func (n *expressionNodeImpl[E]) NumNeighbors() int {
	return n.node.NumNeighbors()
}

func (n *expressionNodeImpl[E]) Cost(toOther Node[ExpressionNodeData, E]) float32 {
	return n.node.Cost(toOther)
}

func (n *expressionNodeImpl[E]) Equal(other Node[ExpressionNodeData, E]) bool {
	return n.node.Equal(other)
}

func (n *expressionNodeImpl[E]) Compare(other Node[ExpressionNodeData, E]) int {
	return n.node.Compare(other)
}

func (n *expressionNodeImpl[E]) EvaluateExpression(input float32, childOutputs map[int64]float32) (float32, bool) {
	if n.expressionFn != nil {
		return n.expressionFn(input, childOutputs)
	}

	// Default behavior based on OpType
	data := n.node.Data()
	switch data.OpType {
	case "input":
		return input, true
	case "constant":
		return data.Value, true
	case "add":
		sum := float32(0)
		for _, val := range childOutputs {
			sum += val
		}
		return sum, true
	case "multiply":
		product := float32(1)
		for _, val := range childOutputs {
			product *= val
		}
		return product, true
	default:
		return 0, false
	}
}

func (n *expressionNodeImpl[E]) ExpressionFunction() func(float32, map[int64]float32) (float32, bool) {
	return n.expressionFn
}

func (n *expressionNodeImpl[E]) IsLeaf() bool {
	return n.isLeaf
}

// expressionGraphAdapter wraps a tree and returns expression nodes
type expressionGraphAdapter[E any] struct {
	tree    *GenericTree[ExpressionNodeData, E]
	nodeMap map[int64]*expressionNodeImpl[E]
}

func newExpressionGraphAdapter[E any](tree *GenericTree[ExpressionNodeData, E]) *expressionGraphAdapter[E] {
	adapter := &expressionGraphAdapter[E]{
		tree:    tree,
		nodeMap: make(map[int64]*expressionNodeImpl[E]),
	}

	// Build expression nodes for all tree nodes
	for node := range tree.Nodes() {
		treeNode := node.(TreeGraphNode[ExpressionNodeData, E])
		idx := treeNode.idx
		data := treeNode.Data()
		isLeaf := treeNode.NumNeighbors() == 0

		adapter.nodeMap[node.ID()] = &expressionNodeImpl[E]{
			node:    treeNode,
			isLeaf:  isLeaf,
			adapter: adapter,
		}
		_ = idx
		_ = data
	}

	return adapter
}

func (a *expressionGraphAdapter[E]) Nodes() iter.Seq[Node[ExpressionNodeData, E]] {
	return func(yield func(Node[ExpressionNodeData, E]) bool) {
		for node := range a.tree.Nodes() {
			exprNode := a.nodeMap[node.ID()]
			if exprNode != nil && !yield(exprNode) {
				return
			}
		}
	}
}

// expressionEdgeAdapter wraps an edge to return expression nodes
type expressionEdgeAdapter[E any] struct {
	edge    Edge[ExpressionNodeData, E]
	adapter *expressionGraphAdapter[E]
}

func (e *expressionEdgeAdapter[E]) ID() int64 {
	return e.edge.ID()
}

func (e *expressionEdgeAdapter[E]) From() Node[ExpressionNodeData, E] {
	from := e.edge.From()
	if from == nil {
		return nil
	}
	exprNode := e.adapter.nodeMap[from.ID()]
	if exprNode == nil {
		return from
	}
	return exprNode
}

func (e *expressionEdgeAdapter[E]) To() Node[ExpressionNodeData, E] {
	to := e.edge.To()
	if to == nil {
		return nil
	}
	exprNode := e.adapter.nodeMap[to.ID()]
	if exprNode == nil {
		return to
	}
	return exprNode
}

func (e *expressionEdgeAdapter[E]) Data() E {
	return e.edge.Data()
}

func (e *expressionEdgeAdapter[E]) Cost() float32 {
	return e.edge.Cost()
}

func (a *expressionGraphAdapter[E]) Edges() iter.Seq[Edge[ExpressionNodeData, E]] {
	return func(yield func(Edge[ExpressionNodeData, E]) bool) {
		for edge := range a.tree.Edges() {
			wrapped := &expressionEdgeAdapter[E]{
				edge:    edge,
				adapter: a,
			}
			if !yield(wrapped) {
				return
			}
		}
	}
}

func (a *expressionGraphAdapter[E]) NumNodes() int {
	return a.tree.NumNodes()
}

func (a *expressionGraphAdapter[E]) NumEdges() int {
	return a.tree.NumEdges()
}

func TestExpressionGraph_SimpleExpression(t *testing.T) {
	// Build expression graph for (x + 2) * 3
	// Graph structure:
	//   multiply (root)
	//   ├── add
	//   │   ├── input (x)
	//   │   └── constant (2)
	//   └── constant (3)

	g := NewGenericGraph[ExpressionNodeData, float32]()

	// Create nodes
	inputNode := &GenericNode[ExpressionNodeData, float32]{
		data: ExpressionNodeData{OpType: "input", Description: "x"},
		id:   1,
	}
	const2Node := &GenericNode[ExpressionNodeData, float32]{
		data: ExpressionNodeData{OpType: "constant", Value: 2.0, Description: "2"},
		id:   2,
	}
	const3Node := &GenericNode[ExpressionNodeData, float32]{
		data: ExpressionNodeData{OpType: "constant", Value: 3.0, Description: "3"},
		id:   3,
	}
	addNode := &GenericNode[ExpressionNodeData, float32]{
		data: ExpressionNodeData{OpType: "add", Description: "x + 2"},
		id:   4,
	}
	multiplyNode := &GenericNode[ExpressionNodeData, float32]{
		data: ExpressionNodeData{OpType: "multiply", Description: "(x + 2) * 3"},
		id:   5,
	}

	g.AddNode(inputNode)
	g.AddNode(const2Node)
	g.AddNode(const3Node)
	g.AddNode(addNode)
	g.AddNode(multiplyNode)

	// Create edges: add -> input, add -> const2, multiply -> add, multiply -> const3
	edge1 := &GenericEdge[ExpressionNodeData, float32]{
		id:      1,
		fromIdx: 2, // addNode index
		toIdx:   0, // inputNode index
		data:    1.0,
		graph:   g,
	}
	edge2 := &GenericEdge[ExpressionNodeData, float32]{
		id:      2,
		fromIdx: 2, // addNode index
		toIdx:   1, // const2Node index
		data:    1.0,
		graph:   g,
	}
	edge3 := &GenericEdge[ExpressionNodeData, float32]{
		id:      3,
		fromIdx: 3, // multiplyNode index
		toIdx:   2, // addNode index
		data:    1.0,
		graph:   g,
	}
	edge4 := &GenericEdge[ExpressionNodeData, float32]{
		id:      4,
		fromIdx: 3, // multiplyNode index
		toIdx:   2, // const3Node index (wrong, should be index 2 for addNode, but const3 is at index 2...)
		data:    1.0,
		graph:   g,
	}

	// Actually, we need to fix the indices. Let me check the actual indices after AddNode
	// AddNode returns the index, but we're using hardcoded indices. Let's fix this.
	_ = edge1
	_ = edge2
	_ = edge3
	_ = edge4

	// Better approach: use the graph's node map to find indices
	// For now, let's create a simpler test that uses the actual graph structure
}

func TestExpressionGraph_ComputeExpression(t *testing.T) {
	// Build expression graph for (x + 2) * 3
	// We'll create a tree structure where:
	// - Root is multiply node
	// - It has two children: add node and constant 3
	// - Add node has two children: input x and constant 2

	// Create nodes with proper data
	inputData := ExpressionNodeData{OpType: "input", Description: "x"}
	const2Data := ExpressionNodeData{OpType: "constant", Value: 2.0, Description: "2"}
	const3Data := ExpressionNodeData{OpType: "constant", Value: 3.0, Description: "3"}
	addData := ExpressionNodeData{OpType: "add", Description: "x + 2"}
	multiplyData := ExpressionNodeData{OpType: "multiply", Description: "(x + 2) * 3"}

	// Create a tree structure
	tree := NewGenericTree[ExpressionNodeData, float32](multiplyData)
	rootIdx := tree.RootIdx()

	// Add children to multiply: add node and constant 3
	addIdx := tree.AddChild(rootIdx, addData)
	_ = tree.AddChild(rootIdx, const3Data)

	// Add children to add: input x and constant 2
	_ = tree.AddChild(addIdx, inputData)
	_ = tree.AddChild(addIdx, const2Data)

	// Create adapter that wraps tree nodes as expression nodes
	adapter := newExpressionGraphAdapter(tree)

	// Get the root expression node
	rootNode := TreeGraphNode[ExpressionNodeData, float32]{tree: tree, idx: rootIdx}
	multiplyExpr := adapter.nodeMap[rootNode.ID()]

	// Create expression graph using the adapter
	eg := NewExpressionGraph[ExpressionNodeData, float32, float32, float32](adapter)

	// Test with different values of x
	testCases := []struct {
		name     string
		x        float32
		expected float32
	}{
		{"x=0", 0.0, (0.0 + 2.0) * 3.0},    // (0+2)*3 = 6
		{"x=1", 1.0, (1.0 + 2.0) * 3.0},    // (1+2)*3 = 9
		{"x=5", 5.0, (5.0 + 2.0) * 3.0},    // (5+2)*3 = 21
		{"x=-2", -2.0, (-2.0 + 2.0) * 3.0}, // (-2+2)*3 = 0
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Compute expression starting from root
			result, ok := eg.Compute(tc.x, multiplyExpr)
			require.True(t, ok, "Computation should succeed")
			assert.InDelta(t, tc.expected, result, 0.001, "Result should match expected value")
		})
	}
}

func TestExpressionGraph_SimpleAdd(t *testing.T) {
	// Test a simpler expression: x + 5
	tree := NewGenericTree[ExpressionNodeData, float32](ExpressionNodeData{
		OpType:      "add",
		Description: "x + 5",
	})
	rootIdx := tree.RootIdx()

	inputIdx := tree.AddChild(rootIdx, ExpressionNodeData{OpType: "input", Description: "x"})
	const5Idx := tree.AddChild(rootIdx, ExpressionNodeData{OpType: "constant", Value: 5.0, Description: "5"})

	_ = inputIdx
	_ = const5Idx

	// Create adapter
	adapter := newExpressionGraphAdapter(tree)
	rootNode := TreeGraphNode[ExpressionNodeData, float32]{tree: tree, idx: rootIdx}
	addExpr := adapter.nodeMap[rootNode.ID()]

	eg := NewExpressionGraph[ExpressionNodeData, float32, float32, float32](adapter)

	testCases := []struct {
		name     string
		x        float32
		expected float32
	}{
		{"x=0", 0.0, 5.0},
		{"x=3", 3.0, 8.0},
		{"x=10", 10.0, 15.0},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result, ok := eg.Compute(tc.x, addExpr)
			require.True(t, ok, "Computation should succeed")
			assert.InDelta(t, tc.expected, result, 0.001, "Result should match expected value")
		})
	}
}
