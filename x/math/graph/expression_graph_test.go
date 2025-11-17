package graph

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type ExpressionNodeData struct {
	Description string
}

func TestExpressionGraph_Compute(t *testing.T) {
	graph := NewGenericExpressionGraph[ExpressionNodeData, float32, float32, float32]()

	mul, err := graph.AddNode(ExpressionNodeData{Description: "mul"}, nil)
	require.NoError(t, err)
	add, err := graph.AddNode(ExpressionNodeData{Description: "add"}, nil)
	require.NoError(t, err)
	input, err := graph.AddNode(ExpressionNodeData{Description: "x"}, func(x float32, _ map[int64]float32) (float32, bool) { return x, true })
	require.NoError(t, err)
	c2, err := graph.AddNode(ExpressionNodeData{Description: "const2"}, func(_ float32, _ map[int64]float32) (float32, bool) { return 2, true })
	require.NoError(t, err)
	c3, err := graph.AddNode(ExpressionNodeData{Description: "const3"}, func(_ float32, _ map[int64]float32) (float32, bool) { return 3, true })
	require.NoError(t, err)

	require.NoError(t, graph.AddEdge(add, input, 0))
	require.NoError(t, graph.AddEdge(add, c2, 0))
	require.NoError(t, graph.AddEdge(mul, add, 0))
	require.NoError(t, graph.AddEdge(mul, c3, 0))

	graph.SetNodeOpByID(add.ID(), func(_ float32, children map[int64]float32) (float32, bool) {
		var sum float32
		for _, v := range children {
			sum += v
		}
		return sum, true
	})
	graph.SetNodeOpByID(mul.ID(), func(_ float32, children map[int64]float32) (float32, bool) {
		product := float32(1)
		for _, v := range children {
			product *= v
		}
		return product, true
	})

	results, err := graph.Compute(nil, 0, 1, 5, -2)
	require.NoError(t, err)
	require.Len(t, results, 4)
	assert.InDelta(t, 6, results[0], 1e-3)
	assert.InDelta(t, 9, results[1], 1e-3)
	assert.InDelta(t, 21, results[2], 1e-3)
	assert.InDelta(t, 0, results[3], 1e-3)
}

func TestExpressionGraph_SimpleAdd(t *testing.T) {
	graph := NewGenericExpressionGraph[ExpressionNodeData, float32, float32, float32]()
	add, err := graph.AddNode(ExpressionNodeData{Description: "add"}, nil)
	require.NoError(t, err)
	input, err := graph.AddNode(ExpressionNodeData{Description: "x"}, func(x float32, _ map[int64]float32) (float32, bool) { return x, true })
	require.NoError(t, err)
	constant, err := graph.AddNode(ExpressionNodeData{Description: "const"}, func(_ float32, _ map[int64]float32) (float32, bool) { return 5, true })
	require.NoError(t, err)

	require.NoError(t, graph.AddEdge(add, input, 0))
	require.NoError(t, graph.AddEdge(add, constant, 0))

	graph.SetNodeOpByID(add.ID(), func(_ float32, children map[int64]float32) (float32, bool) {
		var sum float32
		for _, v := range children {
			sum += v
		}
		return sum, true
	})

	results, err := graph.Compute(nil, 0, 3, 10)
	require.NoError(t, err)
	require.Len(t, results, 3)
	assert.InDelta(t, 5, results[0], 1e-3)
	assert.InDelta(t, 8, results[1], 1e-3)
	assert.InDelta(t, 15, results[2], 1e-3)
}

func TestGenericExpressionGraph_Interfaces(t *testing.T) {
	graph := NewGenericExpressionGraph[ExpressionNodeData, float32, float32, float32]()
	node, err := graph.AddNode(ExpressionNodeData{Description: "leaf"}, func(x float32, _ map[int64]float32) (float32, bool) { return x, true })
	require.NoError(t, err)

	if _, ok := node.(ExpressionNode[ExpressionNodeData, float32, float32, float32]); !ok {
		t.Fatalf("node should implement ExpressionNode")
	}
	if _, ok := interface{}(graph).(ExpressionGraph[ExpressionNodeData, float32, float32, float32]); !ok {
		t.Fatalf("graph should implement ExpressionGraph interface")
	}
}
