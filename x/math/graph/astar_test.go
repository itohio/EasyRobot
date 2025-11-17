package graph

import (
	"testing"

	"github.com/itohio/EasyRobot/x/math/mat"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestAStar_SimplePath(t *testing.T) {
	g := NewGenericGraph[GridNode, float32]()

	nodeA := GridNode{Row: 0, Col: 0, graph: nil}
	nodeB := GridNode{Row: 0, Col: 1, graph: nil}
	nodeC := GridNode{Row: 0, Col: 2, graph: nil}
	nodeD := GridNode{Row: 0, Col: 3, graph: nil}

	// Create edges using GenericGraph
	edgeAB := &GenericEdge[GridNode, float32]{
		fromIdx: 0,
		toIdx:   1,
		data:    1.0,
	}
	edgeBC := &GenericEdge[GridNode, float32]{
		fromIdx: 1,
		toIdx:   2,
		data:    1.0,
	}
	edgeCD := &GenericEdge[GridNode, float32]{
		fromIdx: 2,
		toIdx:   3,
		data:    1.0,
	}

	g.AddNode(&GenericNode[GridNode, float32]{data: nodeA, id: 1})
	g.AddNode(&GenericNode[GridNode, float32]{data: nodeB, id: 2})
	g.AddNode(&GenericNode[GridNode, float32]{data: nodeC, id: 3})
	g.AddNode(&GenericNode[GridNode, float32]{data: nodeD, id: 4})

	g.AddEdge(edgeAB)
	g.AddEdge(edgeBC)
	g.AddEdge(edgeCD)

	heuristic := func(from, to Node[GridNode, float32]) float32 {
		fromNode := from.(*GenericNode[GridNode, float32]).Data()
		toNode := to.(*GenericNode[GridNode, float32]).Data()
		dx := float32(toNode.Col - fromNode.Col)
		dy := float32(toNode.Row - fromNode.Row)
		return dx + dy // Manhattan
	}

	astar := NewAStar(g, heuristic)

	// Collect nodes from iterator
	var nodes []Node[GridNode, float32]
	for node := range g.Nodes() {
		nodes = append(nodes, node)
	}
	require.Len(t, nodes, 4, "Should have 4 nodes")

	startNode := nodes[0]
	endNode := nodes[3]
	path := astar.Search(startNode, endNode)
	require.NotNil(t, path, "Path should exist")
	assert.Equal(t, 4, len(path), "Path should have 4 nodes")
}

func TestAStar_ReuseInstance(t *testing.T) {
	g := NewGenericGraph[GridNode, float32]()

	nodeA := GridNode{Row: 0, Col: 0, graph: nil}
	nodeB := GridNode{Row: 0, Col: 1, graph: nil}
	nodeC := GridNode{Row: 0, Col: 2, graph: nil}

	g.AddNode(&GenericNode[GridNode, float32]{data: nodeA, id: 1})
	g.AddNode(&GenericNode[GridNode, float32]{data: nodeB, id: 2})
	g.AddNode(&GenericNode[GridNode, float32]{data: nodeC, id: 3})

	heuristic := func(from, to Node[GridNode, float32]) float32 { return 0 }
	astar := NewAStar(g, heuristic)

	// Collect nodes from iterator
	var nodes []Node[GridNode, float32]
	for node := range g.Nodes() {
		nodes = append(nodes, node)
	}
	require.Len(t, nodes, 3, "Should have 3 nodes")

	// First search
	startNode := nodes[0]
	endNode := nodes[2]
	path1 := astar.Search(startNode, endNode)
	require.NotNil(t, path1, "First path should exist")

	// Second search (should reuse buffers)
	endNode2 := nodes[1]
	path2 := astar.Search(startNode, endNode2)
	require.NotNil(t, path2, "Second path should exist")
}

func TestAStar_ImplementsSearcher(t *testing.T) {
	g := NewGenericGraph[GridNode, float32]()
	heuristic := func(from, to Node[GridNode, float32]) float32 { return 0 }
	astar := NewAStar(g, heuristic)

	var searcher Searcher[GridNode, float32] = astar
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

	heuristic := func(from, to Node[GridNode, float32]) float32 { return 0 }
	astar := NewAStar(g, heuristic)
	start := GridNode{Row: 0, Col: 0}
	goal := GridNode{Row: 4, Col: 4}

	path := astar.Search(start, goal)
	require.NotNil(t, path, "Path should exist")
	assert.Greater(t, len(path), 0, "Path should have points")
}

func TestAStar_NoPath(t *testing.T) {
	g := NewGenericGraph[GridNode, float32]()

	nodeA := GridNode{Row: 0, Col: 0, graph: nil}
	nodeB := GridNode{Row: 0, Col: 1, graph: nil}
	nodeC := GridNode{Row: 0, Col: 2, graph: nil}

	g.AddNode(&GenericNode[GridNode, float32]{data: nodeA, id: 1})
	g.AddNode(&GenericNode[GridNode, float32]{data: nodeB, id: 2})
	g.AddNode(&GenericNode[GridNode, float32]{data: nodeC, id: 3})

	// Only add edge from A to B, no path to C
	edgeAB := &GenericEdge[GridNode, float32]{
		fromIdx: 0,
		toIdx:   1,
		data:    1.0,
	}
	g.AddEdge(edgeAB)

	heuristic := func(from, to Node[GridNode, float32]) float32 { return 0 }
	astar := NewAStar(g, heuristic)

	// Collect nodes from iterator
	var nodes []Node[GridNode, float32]
	for node := range g.Nodes() {
		nodes = append(nodes, node)
	}
	require.Len(t, nodes, 3, "Should have 3 nodes")

	startNode := nodes[0]
	endNode := nodes[2]
	path := astar.Search(startNode, endNode)
	assert.Nil(t, path, "Path should not exist")
}
