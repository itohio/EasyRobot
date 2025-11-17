package graph

import (
	"testing"

	"github.com/itohio/EasyRobot/x/math/mat"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestBFS_SimplePath(t *testing.T) {
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

	bfs := NewBFS(g)
	path := bfs.Search(nodeA, nodeD)
	require.NotNil(t, path, "Path should exist")
	assert.GreaterOrEqual(t, len(path), 1, "Path should have at least 1 node")
}

func TestBFS_ReuseInstance(t *testing.T) {
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
	nodeC := GridNode{Row: 0, Col: 2, graph: g}
	nodeD := GridNode{Row: 0, Col: 3, graph: g}

	bfs := NewBFS(g)

	// First search
	path1 := bfs.Search(nodeA, nodeD)
	require.NotNil(t, path1, "First path should exist")

	// Second search (should reuse buffers)
	path2 := bfs.Search(nodeA, nodeC)
	require.NotNil(t, path2, "Second path should exist")
}

func TestBFS_ImplementsSearcher(t *testing.T) {
	matrix := mat.New(1, 2)
	matrix[0][0] = 1.0
	matrix[0][1] = 1.0

	g := &GridGraph{
		Matrix:    matrix,
		AllowDiag: false,
		Obstacle:  0,
	}

	bfs := NewBFS(g)

	var searcher Searcher[GridNode, float32] = bfs
	assert.NotNil(t, searcher, "BFS should implement Searcher interface")
}

func TestBFS_GridGraph(t *testing.T) {
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

	bfs := NewBFS(g)
	start := GridNode{Row: 0, Col: 0, graph: g}
	goal := GridNode{Row: 4, Col: 4, graph: g}

	path := bfs.Search(start, goal)
	require.NotNil(t, path, "Path should exist")
	assert.Greater(t, len(path), 0, "Path should have points")
}

func TestBFS_NoPath(t *testing.T) {
	matrix := mat.New(1, 3)
	matrix[0][0] = 1.0
	matrix[0][1] = 1.0
	matrix[0][2] = 0.0 // Obstacle

	g := &GridGraph{
		Matrix:    matrix,
		AllowDiag: false,
		Obstacle:  0,
	}

	bfs := NewBFS(g)
	start := GridNode{Row: 0, Col: 0, graph: g}
	goal := GridNode{Row: 0, Col: 2, graph: g}

	path := bfs.Search(start, goal)
	assert.Nil(t, path, "Path should not exist")
}
