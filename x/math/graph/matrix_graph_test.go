package graph

import (
	"testing"

	"github.com/itohio/EasyRobot/x/math/mat"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// testEdge is a simple edge implementation for testing
type testEdge[N any, E any] struct {
	from Node[N, E]
	to   Node[N, E]
	data E
	id   int64
}

var testEdgeIDCounter int64 = 1

func (e *testEdge[N, E]) ID() int64 {
	if e.id == 0 {
		e.id = testEdgeIDCounter
		testEdgeIDCounter++
	}
	return e.id
}

func (e *testEdge[N, E]) From() Node[N, E] {
	return e.from
}

func (e *testEdge[N, E]) To() Node[N, E] {
	return e.to
}

func (e *testEdge[N, E]) Data() E {
	return e.data
}

func (e *testEdge[N, E]) Cost() float32 {
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
	default:
		return 0
	}
}

func TestToMatrix_GenericGraph(t *testing.T) {
	// Create a generic graph
	g := NewGenericGraph[string, float32]()

	// Add nodes
	nodeA := &GenericNode[string, float32]{data: "A", id: 1}
	nodeB := &GenericNode[string, float32]{data: "B", id: 2}
	nodeC := &GenericNode[string, float32]{data: "C", id: 3}

	g.AddNode(nodeA)
	g.AddNode(nodeB)
	g.AddNode(nodeC)

	// Add edges with costs - create edges with proper node references
	// We need to create a helper edge type that implements Edge interface
	// For now, let's use a simpler approach: create edges after nodes are added
	edgeAB := &testEdge[string, float32]{
		from: nodeA,
		to:   nodeB,
		data: 1.5,
	}
	g.AddEdge(edgeAB)

	edgeBC := &testEdge[string, float32]{
		from: nodeB,
		to:   nodeC,
		data: 2.0,
	}
	g.AddEdge(edgeBC)

	edgeCA := &testEdge[string, float32]{
		from: nodeC,
		to:   nodeA,
		data: 0.5,
	}
	g.AddEdge(edgeCA)

	// Create matrix with correct size
	matrix := mat.New(3, 3)

	// Convert graph to matrix
	err := ToMatrix(g, matrix)
	require.NoError(t, err, "ToMatrix should succeed")

	// Verify matrix contents
	// A -> B: cost 1.5
	assert.Equal(t, float32(1.5), matrix[0][1], "A->B should have cost 1.5")
	// B -> C: cost 2.0
	assert.Equal(t, float32(2.0), matrix[1][2], "B->C should have cost 2.0")
	// C -> A: cost 0.5
	assert.Equal(t, float32(0.5), matrix[2][0], "C->A should have cost 0.5")

	// Verify no other edges
	assert.Equal(t, float32(0), matrix[0][0], "A->A should be 0")
	assert.Equal(t, float32(0), matrix[0][2], "A->C should be 0")
	assert.Equal(t, float32(0), matrix[1][0], "B->A should be 0")
	assert.Equal(t, float32(0), matrix[1][1], "B->B should be 0")
	assert.Equal(t, float32(0), matrix[2][1], "C->B should be 0")
	assert.Equal(t, float32(0), matrix[2][2], "C->C should be 0")
}

func TestToMatrix_GridGraph(t *testing.T) {
	// Create a simple grid graph
	matrixData := mat.New(3, 3)
	matrixData[0][0] = 1.0
	matrixData[0][1] = 1.0
	matrixData[1][1] = 1.0
	matrixData[1][2] = 1.0
	matrixData[2][2] = 1.0

	g := &GridGraph{
		Matrix:    matrixData,
		AllowDiag: false,
		Obstacle:  0,
	}

	// Count nodes
	nodeCount := 0
	for range g.Nodes() {
		nodeCount++
	}

	// Create output matrix
	outputMatrix := mat.New(nodeCount, nodeCount)

	// Convert to matrix
	err := ToMatrix(g, outputMatrix)
	require.NoError(t, err, "ToMatrix should succeed")

	// Verify some edges exist (grid graph has many edges)
	hasEdges := false
	for i := range outputMatrix {
		for j := range outputMatrix[i] {
			if outputMatrix[i][j] > 0 {
				hasEdges = true
				break
			}
		}
		if hasEdges {
			break
		}
	}
	assert.True(t, hasEdges, "Grid graph should have edges")
}

func TestToMatrix_MatrixSizeError(t *testing.T) {
	g := NewGenericGraph[int, float32]()

	// Add 3 nodes
	g.AddNode(&GenericNode[int, float32]{data: 1, id: 1})
	g.AddNode(&GenericNode[int, float32]{data: 2, id: 2})
	g.AddNode(&GenericNode[int, float32]{data: 3, id: 3})

	// Create matrix with wrong size
	matrix := mat.New(2, 2) // Should be 3x3

	err := ToMatrix(g, matrix)
	require.Error(t, err, "Should return error for wrong matrix size")

	matrixSizeErr, ok := err.(*MatrixSizeError)
	require.True(t, ok, "Should be MatrixSizeError")
	assert.Equal(t, 3, matrixSizeErr.Expected, "Expected 3 nodes")
	assert.Equal(t, 2, matrixSizeErr.Actual, "Got 2 rows")
}

func TestToMatrix_MatrixColumnSizeError(t *testing.T) {
	g := NewGenericGraph[int, float32]()

	// Add 3 nodes
	g.AddNode(&GenericNode[int, float32]{data: 1, id: 1})
	g.AddNode(&GenericNode[int, float32]{data: 2, id: 2})
	g.AddNode(&GenericNode[int, float32]{data: 3, id: 3})

	// Create matrix with wrong column size
	matrix := mat.New(3, 2) // Should be 3x3

	err := ToMatrix(g, matrix)
	require.Error(t, err, "Should return error for wrong column size")

	matrixSizeErr, ok := err.(*MatrixSizeError)
	require.True(t, ok, "Should be MatrixSizeError")
	assert.Equal(t, 3, matrixSizeErr.Expected, "Expected 3 columns")
}

func TestToMatrix_EmptyGraph(t *testing.T) {
	g := NewGenericGraph[int, float32]()

	matrix := mat.New(0, 0)

	err := ToMatrix(g, matrix)
	require.NoError(t, err, "Should handle empty graph")
}

func TestToMatrix_ArbitraryNodeIDs(t *testing.T) {
	// Test that ToMatrix correctly maps arbitrary node IDs to matrix indices
	g := NewGenericGraph[string, float32]()

	// Add nodes with non-sequential IDs
	nodeA := &GenericNode[string, float32]{data: "A", id: 100}
	nodeB := &GenericNode[string, float32]{data: "B", id: 200}
	nodeC := &GenericNode[string, float32]{data: "C", id: 300}

	g.AddNode(nodeA)
	g.AddNode(nodeB)
	g.AddNode(nodeC)

	// Add edges
	edgeAB := &testEdge[string, float32]{
		from: nodeA,
		to:   nodeB,
		data: 1.0,
	}
	g.AddEdge(edgeAB)

	edgeBC := &testEdge[string, float32]{
		from: nodeB,
		to:   nodeC,
		data: 2.0,
	}
	g.AddEdge(edgeBC)

	matrix := mat.New(3, 3)

	err := ToMatrix(g, matrix)
	require.NoError(t, err, "Should handle arbitrary node IDs")

	// Verify edges are correctly mapped
	assert.Equal(t, float32(1.0), matrix[0][1], "A->B should be mapped correctly")
	assert.Equal(t, float32(2.0), matrix[1][2], "B->C should be mapped correctly")
}

func TestToMatrix_ZeroMatrixFirst(t *testing.T) {
	// Test that matrix is zeroed out first
	g := NewGenericGraph[int, float32]()

	g.AddNode(&GenericNode[int, float32]{data: 1, id: 1})
	g.AddNode(&GenericNode[int, float32]{data: 2, id: 2})

	matrix := mat.New(2, 2)

	// Fill matrix with non-zero values
	matrix[0][0] = 999.0
	matrix[0][1] = 999.0
	matrix[1][0] = 999.0
	matrix[1][1] = 999.0

	// Add one edge
	node1 := &GenericNode[int, float32]{data: 1, id: 1}
	node2 := &GenericNode[int, float32]{data: 2, id: 2}
	g.AddNode(node1)
	g.AddNode(node2)

	edge := &testEdge[int, float32]{
		from: node1,
		to:   node2,
		data: 5.0,
	}
	g.AddEdge(edge)

	err := ToMatrix(g, matrix)
	require.NoError(t, err)

	// Verify matrix was zeroed first
	assert.Equal(t, float32(0), matrix[0][0], "Should be zeroed")
	assert.Equal(t, float32(0), matrix[1][0], "Should be zeroed")
	assert.Equal(t, float32(0), matrix[1][1], "Should be zeroed")

	// Only the edge should have a value
	assert.Equal(t, float32(5.0), matrix[0][1], "Edge should have correct cost")
}

func TestToMatrix_EdgeCosts(t *testing.T) {
	// Test that edge costs are correctly transferred
	g := NewGenericGraph[int, float32]()

	g.AddNode(&GenericNode[int, float32]{data: 1, id: 1})
	g.AddNode(&GenericNode[int, float32]{data: 2, id: 2})
	g.AddNode(&GenericNode[int, float32]{data: 3, id: 3})

	// Add edges with different costs
	edge1 := &GenericEdge[int, float32]{
		fromIdx: 0,
		toIdx:   1,
		data:    1.5,
	}
	g.AddEdge(edge1)

	edge2 := &GenericEdge[int, float32]{
		fromIdx: 1,
		toIdx:   2,
		data:    2.5,
	}
	g.AddEdge(edge2)

	edge3 := &GenericEdge[int, float32]{
		fromIdx: 0,
		toIdx:   2,
		data:    10.0,
	}
	g.AddEdge(edge3)

	matrix := mat.New(3, 3)

	err := ToMatrix(g, matrix)
	require.NoError(t, err)

	// Verify all edge costs
	assert.Equal(t, float32(1.5), matrix[0][1], "Edge 0->1 cost")
	assert.Equal(t, float32(2.5), matrix[1][2], "Edge 1->2 cost")
	assert.Equal(t, float32(10.0), matrix[0][2], "Edge 0->2 cost")
}
