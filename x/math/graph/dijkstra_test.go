package graph

import (
	"testing"

	"github.com/itohio/EasyRobot/x/math/mat"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDijkstra_SimplePath(t *testing.T) {
	matrix := mat.New(1, 4)
	for i := 0; i < 4; i++ {
		matrix[0][i] = 1.0
	}

	g := &GridGraph{
		Matrix:    matrix,
		AllowDiag: false,
		Obstacle:  0,
	}

	nodeA := GridNode{Row: 0, Col: 0, graph: g}
	nodeD := GridNode{Row: 0, Col: 3, graph: g}

	d := NewDijkstra(g)
	path := d.Search(nodeA, nodeD)
	require.NotNil(t, path, "Path should exist")
	assert.GreaterOrEqual(t, len(path), 1, "Path should have at least 1 node")
}

func TestDijkstra_ShortestPath(t *testing.T) {
	// Create a matrix graph with two paths
	matrix := mat.New(4, 4)
	// Initialize all to obstacle
	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			matrix[i][j] = 0.0
		}
	}
	// Path 1: A->B->C->D (cost 2+2+1=5)
	matrix[0][1] = 2.0
	matrix[1][2] = 2.0
	matrix[2][3] = 1.0
	// Path 2: A->D (direct, cost 3)
	matrix[0][3] = 3.0

	g := &MatrixGraph{
		Matrix:   matrix,
		Obstacle: 0,
	}

	nodeA := MatrixNode{Index: 0, Graph: g}
	nodeD := MatrixNode{Index: 3, Graph: g}

	d := NewDijkstra(g)
	path := d.Search(nodeA, nodeD)
	require.NotNil(t, path, "Path should exist")
	assert.Equal(t, 2, len(path), "Path should use direct edge")
}

func TestDijkstra_NoPath(t *testing.T) {
	matrix := mat.New(1, 3)
	matrix[0][0] = 1.0
	matrix[0][1] = 1.0
	matrix[0][2] = 0.0 // Obstacle

	g := &GridGraph{
		Matrix:    matrix,
		AllowDiag: false,
		Obstacle:  0,
	}

	nodeA := GridNode{Row: 0, Col: 0, graph: g}
	nodeC := GridNode{Row: 0, Col: 2, graph: g}

	d := NewDijkstra(g)
	path := d.Search(nodeA, nodeC)
	assert.Nil(t, path, "Path should not exist")
}

func TestDijkstra_ReuseInstance(t *testing.T) {
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

	d := NewDijkstra(g)

	// First search
	path1 := d.Search(nodeA, nodeC)
	require.NotNil(t, path1, "First path should exist")

	// Second search (should reuse buffers)
	path2 := d.Search(nodeA, nodeB)
	require.NotNil(t, path2, "Second path should exist")
}

func TestDijkstra_ImplementsSearcher(t *testing.T) {
	matrix := mat.New(1, 2)
	matrix[0][0] = 1.0
	matrix[0][1] = 1.0

	g := &GridGraph{
		Matrix:    matrix,
		AllowDiag: false,
		Obstacle:  0,
	}

	d := NewDijkstra(g)

	var searcher Searcher[GridNode, float32] = d
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

	start := GridNode{Row: 0, Col: 0, graph: g}
	goal := GridNode{Row: 4, Col: 4, graph: g}

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

	start := GridNode{Row: 0, Col: 0, graph: g}
	goal := GridNode{Row: 4, Col: 4, graph: g}

	d := NewDijkstra(g)
	path := d.Search(start, goal)
	require.NotNil(t, path, "Path should exist despite obstacles")
}
