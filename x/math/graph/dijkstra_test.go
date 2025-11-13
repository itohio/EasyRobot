package graph

import (
	"testing"

	"github.com/itohio/EasyRobot/x/math/mat"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDijkstra_SimplePath(t *testing.T) {
	g := NewGenericGraph()

	// Create a simple graph: A -> B -> C -> D
	nodeA := GridNode{Row: 0, Col: 0}
	nodeB := GridNode{Row: 0, Col: 1}
	nodeC := GridNode{Row: 0, Col: 2}
	nodeD := GridNode{Row: 0, Col: 3}

	g.AddEdge(nodeA, nodeB, 1.0)
	g.AddEdge(nodeB, nodeC, 1.0)
	g.AddEdge(nodeC, nodeD, 1.0)

	d := NewDijkstra(g)
	path := d.Search(nodeA, nodeD)
	require.NotNil(t, path, "Path should exist")
	assert.Equal(t, 4, len(path), "Path should have 4 nodes")
	assert.True(t, path[0].Equal(nodeA), "Path should start at A")
	assert.True(t, path[3].Equal(nodeD), "Path should end at D")
}

func TestDijkstra_ShortestPath(t *testing.T) {
	g := NewGenericGraph()

	// Create graph with two paths: short (A->D=3) and long (A->B->C->D=5)
	nodeA := GridNode{Row: 0, Col: 0}
	nodeB := GridNode{Row: 0, Col: 1}
	nodeC := GridNode{Row: 0, Col: 2}
	nodeD := GridNode{Row: 0, Col: 3}

	g.AddEdge(nodeA, nodeB, 2.0)
	g.AddEdge(nodeB, nodeC, 2.0)
	g.AddEdge(nodeC, nodeD, 1.0)
	g.AddEdge(nodeA, nodeD, 3.0) // Direct path

	d := NewDijkstra(g)
	path := d.Search(nodeA, nodeD)
	require.NotNil(t, path, "Path should exist")
	assert.Equal(t, 2, len(path), "Path should use direct edge")
	assert.True(t, path[0].Equal(nodeA), "Path should start at A")
	assert.True(t, path[1].Equal(nodeD), "Path should end at D")
}

func TestDijkstra_NoPath(t *testing.T) {
	g := NewGenericGraph()

	nodeA := GridNode{Row: 0, Col: 0}
	nodeB := GridNode{Row: 0, Col: 1}
	nodeC := GridNode{Row: 0, Col: 2}

	g.AddEdge(nodeA, nodeB, 1.0)
	// No path from A to C

	d := NewDijkstra(g)
	path := d.Search(nodeA, nodeC)
	assert.Nil(t, path, "Path should not exist")
}

func TestDijkstra_ReuseInstance(t *testing.T) {
	g := NewGenericGraph()

	nodeA := GridNode{Row: 0, Col: 0}
	nodeB := GridNode{Row: 0, Col: 1}
	nodeC := GridNode{Row: 0, Col: 2}

	g.AddEdge(nodeA, nodeB, 1.0)
	g.AddEdge(nodeB, nodeC, 1.0)

	d := NewDijkstra(g)

	// First search
	path1 := d.Search(nodeA, nodeC)
	require.NotNil(t, path1, "First path should exist")

	// Second search (should reuse buffers)
	path2 := d.Search(nodeA, nodeB)
	require.NotNil(t, path2, "Second path should exist")
}

func TestDijkstra_ImplementsSearcher(t *testing.T) {
	g := NewGenericGraph()
	d := NewDijkstra(g)

	var searcher Searcher = d
	assert.NotNil(t, searcher, "Dijkstra should implement Searcher interface")
}

func TestDijkstra_GridGraph(t *testing.T) {
	matrix := mat.New(5, 5)
	for i := 0; i < 5; i++ {
		for j := 0; j < 5; j++ {
			matrix[i][j] = 1.0
		}
	}

	g := &GridGraph{
		Matrix:    matrix,
		AllowDiag: false,
		Obstacle:  0,
	}

	start := GridNode{Row: 0, Col: 0}
	goal := GridNode{Row: 4, Col: 4}

	d := NewDijkstra(g)
	path := d.Search(start, goal)
	require.NotNil(t, path, "Path should exist")
	assert.Greater(t, len(path), 0, "Path should have points")
}

func TestDijkstra_GridGraphWithObstacles(t *testing.T) {
	matrix := mat.New(5, 5)
	for i := 0; i < 5; i++ {
		for j := 0; j < 5; j++ {
			matrix[i][j] = 1.0
		}
	}
	// Add obstacle wall (but leave a gap)
	for i := 0; i < 4; i++ {
		matrix[i][2] = 0.0
	}
	matrix[4][2] = 1.0 // Gap

	g := &GridGraph{
		Matrix:    matrix,
		AllowDiag: false,
		Obstacle:  0,
	}

	start := GridNode{Row: 0, Col: 0}
	goal := GridNode{Row: 4, Col: 4}

	d := NewDijkstra(g)
	path := d.Search(start, goal)
	require.NotNil(t, path, "Path should exist despite obstacles")
}
