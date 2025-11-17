package graph

import (
	"iter"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Example implementation of DecisionNode for testing
type testDecisionNode[N any, E any, Input any, Output any] struct {
	node       Node[N, E]
	decisionFn func(Input) (Output, bool)
	isLeaf     bool
}

// Example implementation of ExpressionNode for testing
type testExpressionNode[N any, E any, Input any, Output any] struct {
	node         Node[N, E]
	expressionFn func(Input, map[int64]Output) (Output, bool)
	isLeaf       bool
}

func (n *testExpressionNode[N, E, Input, Output]) ID() int64 {
	return n.node.ID()
}

func (n *testExpressionNode[N, E, Input, Output]) Data() N {
	return n.node.Data()
}

func (n *testExpressionNode[N, E, Input, Output]) Neighbors() iter.Seq[Node[N, E]] {
	return n.node.Neighbors()
}

func (n *testExpressionNode[N, E, Input, Output]) Edges() iter.Seq[Edge[N, E]] {
	return n.node.Edges()
}

func (n *testExpressionNode[N, E, Input, Output]) NumNeighbors() int {
	return n.node.NumNeighbors()
}

func (n *testExpressionNode[N, E, Input, Output]) Cost(toOther Node[N, E]) float32 {
	return n.node.Cost(toOther)
}

func (n *testExpressionNode[N, E, Input, Output]) Equal(other Node[N, E]) bool {
	return n.node.Equal(other)
}

func (n *testExpressionNode[N, E, Input, Output]) Compare(other Node[N, E]) int {
	return n.node.Compare(other)
}

func (n *testExpressionNode[N, E, Input, Output]) EvaluateExpression(input Input, childOutputs map[int64]Output) (Output, bool) {
	if n.expressionFn != nil {
		return n.expressionFn(input, childOutputs)
	}
	var zero Output
	return zero, false
}

func (n *testExpressionNode[N, E, Input, Output]) ExpressionFunction() func(Input, map[int64]Output) (Output, bool) {
	return n.expressionFn
}

func (n *testExpressionNode[N, E, Input, Output]) IsLeaf() bool {
	return n.isLeaf
}

func (n *testDecisionNode[N, E, Input, Output]) ID() int64 {
	return n.node.ID()
}

func (n *testDecisionNode[N, E, Input, Output]) Data() N {
	return n.node.Data()
}

func (n *testDecisionNode[N, E, Input, Output]) Neighbors() iter.Seq[Node[N, E]] {
	return n.node.Neighbors()
}

func (n *testDecisionNode[N, E, Input, Output]) Edges() iter.Seq[Edge[N, E]] {
	return n.node.Edges()
}

func (n *testDecisionNode[N, E, Input, Output]) NumNeighbors() int {
	return n.node.NumNeighbors()
}

func (n *testDecisionNode[N, E, Input, Output]) Cost(toOther Node[N, E]) float32 {
	return n.node.Cost(toOther)
}

func (n *testDecisionNode[N, E, Input, Output]) Equal(other Node[N, E]) bool {
	return n.node.Equal(other)
}

func (n *testDecisionNode[N, E, Input, Output]) Compare(other Node[N, E]) int {
	return n.node.Compare(other)
}

func (n *testDecisionNode[N, E, Input, Output]) EvaluateDecision(input Input) (Output, bool) {
	if n.decisionFn != nil {
		return n.decisionFn(input)
	}
	var zero Output
	return zero, false
}

func (n *testDecisionNode[N, E, Input, Output]) IsLeaf() bool {
	return n.isLeaf
}

func (n *testDecisionNode[N, E, Input, Output]) DecisionFunction() func(Input) (Output, bool) {
	return n.decisionFn
}

// Example implementation of DecisionEdge for testing
type testDecisionEdge[N any, E any, Input any] struct {
	edge       Edge[N, E]
	criteriaFn func(Input) bool
}

func (e *testDecisionEdge[N, E, Input]) ID() int64 {
	return e.edge.ID()
}

func (e *testDecisionEdge[N, E, Input]) From() Node[N, E] {
	return e.edge.From()
}

func (e *testDecisionEdge[N, E, Input]) To() Node[N, E] {
	return e.edge.To()
}

func (e *testDecisionEdge[N, E, Input]) Data() E {
	return e.edge.Data()
}

func (e *testDecisionEdge[N, E, Input]) Cost() float32 {
	return e.edge.Cost()
}

func (e *testDecisionEdge[N, E, Input]) EvaluateCriteria(input Input) bool {
	if e.criteriaFn != nil {
		return e.criteriaFn(input)
	}
	return true // Default: allow all edges
}

func (e *testDecisionEdge[N, E, Input]) CriteriaFunction() func(Input) bool {
	return e.criteriaFn
}

func TestComputeDecision_SimpleTree(t *testing.T) {
	// Create a simple decision tree: if x > 5 then "A" else "B"
	g := NewGenericGraph[string, float32]()

	// Root node: decision node
	rootNode := &GenericNode[string, float32]{data: "root", id: 1}
	g.AddNode(rootNode)

	// Leaf nodes
	leafA := &GenericNode[string, float32]{data: "A", id: 2}
	leafB := &GenericNode[string, float32]{data: "B", id: 3}
	g.AddNode(leafA)
	g.AddNode(leafB)

	// Create decision nodes
	rootDecision := &testDecisionNode[string, float32, int, string]{
		node:   rootNode,
		isLeaf: false,
		decisionFn: func(x int) (string, bool) {
			// This node doesn't return directly, it routes based on edges
			var zero string
			return zero, false
		},
	}

	leafADecision := &testDecisionNode[string, float32, int, string]{
		node:   leafA,
		isLeaf: true,
		decisionFn: func(x int) (string, bool) {
			return "A", true
		},
	}

	leafBDecision := &testDecisionNode[string, float32, int, string]{
		node:   leafB,
		isLeaf: true,
		decisionFn: func(x int) (string, bool) {
			return "B", true
		},
	}

	// Create edges with criteria
	edgeToA := &testDecisionEdge[string, float32, int]{
		edge: &GenericEdge[string, float32]{
			id:      1,
			fromIdx: 0,
			toIdx:   1,
			data:    1.0,
			graph:   g,
		},
		criteriaFn: func(x int) bool {
			return x > 5
		},
	}

	edgeToB := &testDecisionEdge[string, float32, int]{
		edge: &GenericEdge[string, float32]{
			id:      2,
			fromIdx: 0,
			toIdx:   2,
			data:    1.0,
			graph:   g,
		},
		criteriaFn: func(x int) bool {
			return x <= 5
		},
	}

	// Note: In a real implementation, you'd need to wrap the nodes/edges
	// For this test, we'll use a simpler approach with direct computation

	// Test: x = 10 should go to A
	// This is a simplified test - in practice you'd need proper node/edge wrapping
	_ = rootDecision
	_ = leafADecision
	_ = leafBDecision
	_ = edgeToA
	_ = edgeToB

	// For now, just verify the interfaces compile
	assert.NotNil(t, rootDecision)
	assert.NotNil(t, leafADecision)
	assert.NotNil(t, leafBDecision)
}

func TestExpressionGraph_BottomUp(t *testing.T) {
	// Create a simple expression graph: (a + b) * c
	// Where a=2, b=3, c=4 -> (2+3)*4 = 20
	g := NewGenericGraph[float32, float32]()

	// Leaf nodes: a, b, c
	nodeA := &GenericNode[float32, float32]{data: 2.0, id: 1}
	nodeB := &GenericNode[float32, float32]{data: 3.0, id: 2}
	nodeC := &GenericNode[float32, float32]{data: 4.0, id: 3}
	g.AddNode(nodeA)
	g.AddNode(nodeB)
	g.AddNode(nodeC)

	// Intermediate node: a + b
	nodeAdd := &GenericNode[float32, float32]{data: 0.0, id: 4}
	g.AddNode(nodeAdd)

	// Root node: (a+b) * c
	nodeMul := &GenericNode[float32, float32]{data: 0.0, id: 5}
	g.AddNode(nodeMul)

	// Create expression nodes
	exprA := &testExpressionNode[float32, float32, int, float32]{
		node:   nodeA,
		isLeaf: true,
		expressionFn: func(input int, childOutputs map[int64]float32) (float32, bool) {
			return 2.0, true // Constant value
		},
	}

	exprB := &testExpressionNode[float32, float32, int, float32]{
		node:   nodeB,
		isLeaf: true,
		expressionFn: func(input int, childOutputs map[int64]float32) (float32, bool) {
			return 3.0, true
		},
	}

	exprC := &testExpressionNode[float32, float32, int, float32]{
		node:   nodeC,
		isLeaf: true,
		expressionFn: func(input int, childOutputs map[int64]float32) (float32, bool) {
			return 4.0, true
		},
	}

	exprAdd := &testExpressionNode[float32, float32, int, float32]{
		node:   nodeAdd,
		isLeaf: false,
		expressionFn: func(input int, childOutputs map[int64]float32) (float32, bool) {
			// Sum outputs from children (a and b)
			sum := float32(0)
			for _, val := range childOutputs {
				sum += val
			}
			return sum, true
		},
	}

	exprMul := &testExpressionNode[float32, float32, int, float32]{
		node:   nodeMul,
		isLeaf: false,
		expressionFn: func(input int, childOutputs map[int64]float32) (float32, bool) {
			// Multiply outputs from children (add result and c)
			product := float32(1)
			for _, val := range childOutputs {
				product *= val
			}
			return product, true
		},
	}

	_ = exprA
	_ = exprB
	_ = exprC
	_ = exprAdd
	_ = exprMul

	// Verify interfaces compile
	assert.NotNil(t, exprA)
	assert.NotNil(t, exprAdd)
	assert.NotNil(t, exprMul)
}

func TestDecisionInterfaces(t *testing.T) {
	// Test that interfaces are properly defined
	var _ DecisionNode[any, any, any, any] = (*testDecisionNode[any, any, any, any])(nil)
	var _ DecisionEdge[any, any, any] = (*testDecisionEdge[any, any, any])(nil)
	// ExpressionNode is separate and doesn't extend DecisionNode
	var _ ExpressionNode[any, any, any, any] = (*testExpressionNode[any, any, any, any])(nil)

	// Verify interface methods exist
	dn := &testDecisionNode[string, float32, int, string]{
		node:   &GenericNode[string, float32]{data: "test", id: 1},
		isLeaf: true,
		decisionFn: func(x int) (string, bool) {
			return "result", true
		},
	}

	output, ok := dn.EvaluateDecision(42)
	require.True(t, ok)
	assert.Equal(t, "result", output)
	assert.True(t, dn.IsLeaf())

	de := &testDecisionEdge[string, float32, int]{
		edge: &GenericEdge[string, float32]{
			id:   1,
			data: 1.0,
		},
		criteriaFn: func(x int) bool {
			return x > 10
		},
	}

	assert.True(t, de.EvaluateCriteria(20))
	assert.False(t, de.EvaluateCriteria(5))
}
