package graph

import (
	"testing"

	"github.com/itohio/EasyRobot/x/math/mat"
	"github.com/stretchr/testify/assert"
)

func TestLoopDetection_NoCycle(t *testing.T) {
	matrix := mat.New(1, 3)
	for i := 0; i < 3; i++ {
		matrix[0][i] = 1.0
	}

	g := &GridGraph{
		Matrix:    matrix,
		AllowDiag: false,
		Obstacle:  0,
	}

	nodeA := GridNode{Row: 0, Col: 0, graph: g}
	// Linear path: A->B->C, no cycle

	hasCycle := LoopDetection(g, nodeA)
	assert.False(t, hasCycle, "Graph should not have cycle")
}

func TestLoopDetection_WithCycle(t *testing.T) {
	// Create a matrix graph with a cycle
	matrix := mat.New(3, 3)
	matrix[0][1] = 1.0 // A->B
	matrix[1][2] = 1.0 // B->C
	matrix[2][0] = 1.0 // C->A (creates cycle)

	g := &MatrixGraph{
		Matrix:   matrix,
		Obstacle: 0,
	}

	nodeA := MatrixNode{Index: 0, Graph: g}

	hasCycle := LoopDetection(g, nodeA)
	assert.True(t, hasCycle, "Graph should have cycle")
}

func TestLoopDetection_SelfLoop(t *testing.T) {
	// Create a matrix graph with self-loop
	matrix := mat.New(1, 1)
	matrix[0][0] = 1.0 // A->A (self-loop)

	g := &MatrixGraph{
		Matrix:   matrix,
		Obstacle: 0,
	}

	nodeA := MatrixNode{Index: 0, Graph: g}

	hasCycle := LoopDetection(g, nodeA)
	assert.True(t, hasCycle, "Self-loop should be detected as cycle")
}

func TestConnectedComponents_SingleComponent(t *testing.T) {
	matrix := mat.New(1, 3)
	for i := 0; i < 3; i++ {
		matrix[0][i] = 1.0
	}

	g := &GridGraph{
		Matrix:    matrix,
		AllowDiag: false,
		Obstacle:  0,
	}

	nodeA := GridNode{Row: 0, Col: 0, graph: g}
	nodeB := GridNode{Row: 0, Col: 1, graph: g}
	nodeC := GridNode{Row: 0, Col: 2, graph: g}

	nodes := []Node[GridNode, float32]{nodeA, nodeB, nodeC}
	components := ConnectedComponents(g, nodes)

	assert.Equal(t, 1, len(components), "Should have one component")
	assert.Equal(t, 3, len(components[0]), "Component should have 3 nodes")
}

func TestConnectedComponents_MultipleComponents(t *testing.T) {
	// Create two separate components
	matrix := mat.New(1, 4)
	matrix[0][0] = 1.0
	matrix[0][1] = 1.0
	matrix[0][2] = 0.0 // Obstacle
	matrix[0][3] = 1.0
	// Component 1: A-B, Component 2: D (isolated)

	g := &GridGraph{
		Matrix:    matrix,
		AllowDiag: false,
		Obstacle:  0,
	}

	nodeA := GridNode{Row: 0, Col: 0, graph: g}
	nodeB := GridNode{Row: 0, Col: 1, graph: g}
	nodeD := GridNode{Row: 0, Col: 3, graph: g}

	nodes := []Node[GridNode, float32]{nodeA, nodeB, nodeD}
	components := ConnectedComponents(g, nodes)

	assert.GreaterOrEqual(t, len(components), 1, "Should have at least one component")
}

func TestConnectedComponents_Empty(t *testing.T) {
	matrix := mat.New(1, 1)
	matrix[0][0] = 1.0

	g := &GridGraph{
		Matrix:    matrix,
		AllowDiag: false,
		Obstacle:  0,
	}

	components := ConnectedComponents(g, nil)
	assert.Nil(t, components, "Empty nodes should return nil")

	components = ConnectedComponents(g, []Node[GridNode, float32]{})
	assert.Nil(t, components, "Empty nodes should return nil")
}
