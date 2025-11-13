package graph

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestLoopDetection_NoCycle(t *testing.T) {
	g := NewGenericGraph()

	nodeA := GridNode{Row: 0, Col: 0}
	nodeB := GridNode{Row: 0, Col: 1}
	nodeC := GridNode{Row: 0, Col: 2}

	g.AddEdge(nodeA, nodeB, 1.0)
	g.AddEdge(nodeB, nodeC, 1.0)
	// No cycle

	hasCycle := LoopDetection(g, nodeA)
	assert.False(t, hasCycle, "Graph should not have cycle")
}

func TestLoopDetection_WithCycle(t *testing.T) {
	g := NewGenericGraph()

	nodeA := GridNode{Row: 0, Col: 0}
	nodeB := GridNode{Row: 0, Col: 1}
	nodeC := GridNode{Row: 0, Col: 2}

	g.AddEdge(nodeA, nodeB, 1.0)
	g.AddEdge(nodeB, nodeC, 1.0)
	g.AddEdge(nodeC, nodeA, 1.0) // Creates cycle

	hasCycle := LoopDetection(g, nodeA)
	assert.True(t, hasCycle, "Graph should have cycle")
}

func TestLoopDetection_SelfLoop(t *testing.T) {
	g := NewGenericGraph()

	nodeA := GridNode{Row: 0, Col: 0}

	g.AddEdge(nodeA, nodeA, 1.0) // Self-loop

	hasCycle := LoopDetection(g, nodeA)
	assert.True(t, hasCycle, "Self-loop should be detected as cycle")
}

func TestConnectedComponents_SingleComponent(t *testing.T) {
	g := NewGenericGraph()

	nodeA := GridNode{Row: 0, Col: 0}
	nodeB := GridNode{Row: 0, Col: 1}
	nodeC := GridNode{Row: 0, Col: 2}

	g.AddEdge(nodeA, nodeB, 1.0)
	g.AddEdge(nodeB, nodeC, 1.0)

	nodes := []Node{nodeA, nodeB, nodeC}
	components := ConnectedComponents(g, nodes)
	
	assert.Equal(t, 1, len(components), "Should have one component")
	assert.Equal(t, 3, len(components[0]), "Component should have 3 nodes")
}

func TestConnectedComponents_MultipleComponents(t *testing.T) {
	g := NewGenericGraph()

	nodeA := GridNode{Row: 0, Col: 0}
	nodeB := GridNode{Row: 0, Col: 1}
	nodeC := GridNode{Row: 0, Col: 2}
	nodeD := GridNode{Row: 0, Col: 3}

	g.AddEdge(nodeA, nodeB, 1.0)
	g.AddEdge(nodeC, nodeD, 1.0)
	// Two separate components

	nodes := []Node{nodeA, nodeB, nodeC, nodeD}
	components := ConnectedComponents(g, nodes)
	
	assert.Equal(t, 2, len(components), "Should have two components")
	
	// Check component sizes
	if len(components[0]) == 2 {
		assert.Equal(t, 2, len(components[1]), "Both components should have 2 nodes")
	} else {
		assert.Equal(t, 2, len(components[0]), "Both components should have 2 nodes")
	}
}

func TestConnectedComponents_Empty(t *testing.T) {
	g := NewGenericGraph()
	components := ConnectedComponents(g, nil)
	assert.Nil(t, components, "Empty nodes should return nil")
	
	components = ConnectedComponents(g, []Node{})
	assert.Nil(t, components, "Empty nodes should return nil")
}

