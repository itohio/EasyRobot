package graph

import (
	"testing"

	"github.com/itohio/EasyRobot/x/math/mat"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestBFS_SimplePath(t *testing.T) {
	g := NewGenericGraph()

	nodeA := GridNode{Row: 0, Col: 0}
	nodeB := GridNode{Row: 0, Col: 1}
	nodeC := GridNode{Row: 0, Col: 2}
	nodeD := GridNode{Row: 0, Col: 3}

	g.AddEdge(nodeA, nodeB, 1.0)
	g.AddEdge(nodeB, nodeC, 1.0)
	g.AddEdge(nodeC, nodeD, 1.0)

	bfs := NewBFS(g)
	path := bfs.Search(nodeA, nodeD)
	require.NotNil(t, path, "Path should exist")
	assert.Equal(t, 4, len(path), "Path should have 4 nodes")
	assert.True(t, path[0].Equal(nodeA), "Path should start at A")
	assert.True(t, path[3].Equal(nodeD), "Path should end at D")
}

func TestBFS_ReuseInstance(t *testing.T) {
	g := NewGenericGraph()

	nodeA := GridNode{Row: 0, Col: 0}
	nodeB := GridNode{Row: 0, Col: 1}
	nodeC := GridNode{Row: 0, Col: 2}
	nodeD := GridNode{Row: 0, Col: 3}

	g.AddEdge(nodeA, nodeB, 1.0)
	g.AddEdge(nodeB, nodeC, 1.0)
	g.AddEdge(nodeC, nodeD, 1.0)

	bfs := NewBFS(g)

	// First search
	path1 := bfs.Search(nodeA, nodeD)
	require.NotNil(t, path1, "First path should exist")

	// Second search (should reuse buffers)
	path2 := bfs.Search(nodeA, nodeC)
	require.NotNil(t, path2, "Second path should exist")
	assert.True(t, path2[0].Equal(nodeA), "Second path should start at A")
	assert.True(t, path2[2].Equal(nodeC), "Second path should end at C")
}

func TestBFS_ImplementsSearcher(t *testing.T) {
	g := NewGenericGraph()
	bfs := NewBFS(g)

	var searcher Searcher = bfs
	assert.NotNil(t, searcher, "BFS should implement Searcher interface")
}

func TestBFS_MatrixGraph(t *testing.T) {
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
	start := GridNode{Row: 0, Col: 0}
	goal := GridNode{Row: 4, Col: 4}

	path := bfs.Search(start, goal)
	require.NotNil(t, path, "Path should exist")
	assert.Greater(t, len(path), 0, "Path should have points")
}

func TestBFS_NoPath(t *testing.T) {
	g := NewGenericGraph()

	nodeA := GridNode{Row: 0, Col: 0}
	nodeB := GridNode{Row: 0, Col: 1}
	nodeC := GridNode{Row: 0, Col: 2}

	g.AddEdge(nodeA, nodeB, 1.0)
	// No path from A to C

	bfs := NewBFS(g)
	path := bfs.Search(nodeA, nodeC)
	assert.Nil(t, path, "Path should not exist")
}
