package graph

import (
	"github.com/itohio/EasyRobot/x/math/mat"
)

// GridGraph adapts Matrix to Graph interface for grid-based pathfinding
// Uses 2D grid coordinates (row, col) as nodes
// Each node represents a cell, neighbors are adjacent cells
type GridGraph struct {
	Matrix    mat.Matrix
	AllowDiag bool
	Obstacle  float32
}

// GridNode represents a node in a grid graph
type GridNode struct {
	Row, Col int
}

func (n GridNode) Equal(other Node) bool {
	o, ok := other.(GridNode)
	if !ok {
		return false
	}
	return n.Row == o.Row && n.Col == o.Col
}

func (g *GridGraph) Neighbors(n Node) []Node {
	gn, ok := n.(GridNode)
	if !ok {
		return nil
	}

	var neighbors []Node
	dirs := getGridDirections(g.AllowDiag)

	for _, dir := range dirs {
		newRow := gn.Row + dir[0]
		newCol := gn.Col + dir[1]

		if newRow < 0 || newRow >= len(g.Matrix) ||
			newCol < 0 || newCol >= len(g.Matrix[0]) {
			continue
		}

		if g.Matrix[newRow][newCol] <= g.Obstacle {
			continue
		}

		neighbors = append(neighbors, GridNode{Row: newRow, Col: newCol})
	}

	return neighbors
}

func (g *GridGraph) Cost(from, to Node) float32 {
	fromNode, ok := from.(GridNode)
	if !ok {
		return 0
	}
	toNode, ok := to.(GridNode)
	if !ok {
		return 0
	}

	cost := g.Matrix[toNode.Row][toNode.Col]

	if g.AllowDiag && fromNode.Row != toNode.Row && fromNode.Col != toNode.Col {
		cost *= 1.41421356237 // sqrt(2)
	}

	return cost
}

func getGridDirections(allowDiag bool) [][]int {
	if allowDiag {
		return [][]int{
			{-1, 0}, {-1, 1}, {0, 1}, {1, 1},
			{1, 0}, {1, -1}, {0, -1}, {-1, -1},
		}
	}
	return [][]int{
		{-1, 0}, {0, 1}, {1, 0}, {0, -1},
	}
}
