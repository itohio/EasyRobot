package graph

import (
	"testing"

	"github.com/itohio/EasyRobot/x/math/mat"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestAStar_SimplePath(t *testing.T) {
	g := NewGenericGraph()

	nodeA := GridNode{Row: 0, Col: 0}
	nodeB := GridNode{Row: 0, Col: 1}
	nodeC := GridNode{Row: 0, Col: 2}
	nodeD := GridNode{Row: 0, Col: 3}

	g.AddEdge(nodeA, nodeB, 1.0)
	g.AddEdge(nodeB, nodeC, 1.0)
	g.AddEdge(nodeC, nodeD, 1.0)

	heuristic := func(from, to Node) float32 {
		fromNode := from.(GridNode)
		toNode := to.(GridNode)
		dx := float32(toNode.Col - fromNode.Col)
		dy := float32(toNode.Row - fromNode.Row)
		return dx + dy // Manhattan
	}

	astar := NewAStar(g, heuristic)
	path := astar.Search(nodeA, nodeD)
	require.NotNil(t, path, "Path should exist")
	assert.Equal(t, 4, len(path), "Path should have 4 nodes")
	assert.True(t, path[0].Equal(nodeA), "Path should start at A")
	assert.True(t, path[3].Equal(nodeD), "Path should end at D")
}

func TestAStar_ReuseInstance(t *testing.T) {
	g := NewGenericGraph()

	nodeA := GridNode{Row: 0, Col: 0}
	nodeB := GridNode{Row: 0, Col: 1}
	nodeC := GridNode{Row: 0, Col: 2}

	g.AddEdge(nodeA, nodeB, 1.0)
	g.AddEdge(nodeB, nodeC, 1.0)

	heuristic := func(from, to Node) float32 { return 0 }
	astar := NewAStar(g, heuristic)

	// First search
	path1 := astar.Search(nodeA, nodeC)
	require.NotNil(t, path1, "First path should exist")

	// Second search (should reuse buffers)
	path2 := astar.Search(nodeA, nodeB)
	require.NotNil(t, path2, "Second path should exist")
}

func TestAStar_ImplementsSearcher(t *testing.T) {
	g := NewGenericGraph()
	heuristic := func(from, to Node) float32 { return 0 }
	astar := NewAStar(g, heuristic)

	var searcher Searcher = astar
	assert.NotNil(t, searcher, "AStar should implement Searcher interface")
}

func TestAStar_GridGraph(t *testing.T) {
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

	heuristic := func(from, to Node) float32 { return 0 }
	astar := NewAStar(g, heuristic)
	start := GridNode{Row: 0, Col: 0}
	goal := GridNode{Row: 4, Col: 4}

	path := astar.Search(start, goal)
	require.NotNil(t, path, "Path should exist")
	assert.Greater(t, len(path), 0, "Path should have points")
}

func TestAStar_NoPath(t *testing.T) {
	g := NewGenericGraph()

	nodeA := GridNode{Row: 0, Col: 0}
	nodeB := GridNode{Row: 0, Col: 1}
	nodeC := GridNode{Row: 0, Col: 2}

	g.AddEdge(nodeA, nodeB, 1.0)
	// No path from A to C

	heuristic := func(from, to Node) float32 { return 0 }
	astar := NewAStar(g, heuristic)
	path := astar.Search(nodeA, nodeC)
	assert.Nil(t, path, "Path should not exist")
}
