package graph

import (
	"testing"

	"github.com/itohio/EasyRobot/x/math/mat"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDFS_SimplePath(t *testing.T) {
	g := NewGenericGraph()

	nodeA := GridNode{Row: 0, Col: 0}
	nodeB := GridNode{Row: 0, Col: 1}
	nodeC := GridNode{Row: 0, Col: 2}
	nodeD := GridNode{Row: 0, Col: 3}

	g.AddEdge(nodeA, nodeB, 1.0)
	g.AddEdge(nodeB, nodeC, 1.0)
	g.AddEdge(nodeC, nodeD, 1.0)

	dfs := NewDFS(g)
	path := dfs.Search(nodeA, nodeD)
	require.NotNil(t, path, "Path should exist")
	assert.GreaterOrEqual(t, len(path), 1, "Path should have at least 1 node")
	assert.True(t, path[0].Equal(nodeA), "Path should start at A")
	assert.True(t, path[len(path)-1].Equal(nodeD), "Path should end at D")
}

func TestDFS_ReuseInstance(t *testing.T) {
	g := NewGenericGraph()

	nodeA := GridNode{Row: 0, Col: 0}
	nodeB := GridNode{Row: 0, Col: 1}
	nodeC := GridNode{Row: 0, Col: 2}

	g.AddEdge(nodeA, nodeB, 1.0)
	g.AddEdge(nodeB, nodeC, 1.0)

	dfs := NewDFS(g)

	// First search
	path1 := dfs.Search(nodeA, nodeC)
	require.NotNil(t, path1, "First path should exist")

	// Second search (should reuse buffers)
	path2 := dfs.Search(nodeA, nodeB)
	require.NotNil(t, path2, "Second path should exist")
}

func TestDFS_ImplementsSearcher(t *testing.T) {
	g := NewGenericGraph()
	dfs := NewDFS(g)

	var searcher Searcher = dfs
	assert.NotNil(t, searcher, "DFS should implement Searcher interface")
}

func TestDFS_MatrixGraph(t *testing.T) {
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

	dfs := NewDFS(g)
	start := GridNode{Row: 0, Col: 0}
	goal := GridNode{Row: 4, Col: 4}

	path := dfs.Search(start, goal)
	require.NotNil(t, path, "Path should exist")
	assert.Greater(t, len(path), 0, "Path should have points")
}

func TestDFS_NoPath(t *testing.T) {
	g := NewGenericGraph()

	nodeA := GridNode{Row: 0, Col: 0}
	nodeB := GridNode{Row: 0, Col: 1}
	nodeC := GridNode{Row: 0, Col: 2}

	g.AddEdge(nodeA, nodeB, 1.0)
	// No path from A to C

	dfs := NewDFS(g)
	path := dfs.Search(nodeA, nodeC)
	assert.Nil(t, path, "Path should not exist")
}
