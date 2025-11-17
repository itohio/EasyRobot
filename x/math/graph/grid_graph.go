package graph

import (
	"iter"

	"github.com/itohio/EasyRobot/x/math/mat"
)

// GridGraph adapts Matrix to Graph interface for grid-based pathfinding
// Uses 2D grid coordinates (row, col) as nodes
// Each node represents a cell, neighbors are adjacent cells
// Node data type: GridNode (the coordinates)
// Edge data type: float32 (the cost/weight)
type GridGraph struct {
	Matrix    mat.Matrix
	AllowDiag bool
	Obstacle  float32
}

// GridNode represents a node in a grid graph
// It implements Node[GridNode, float32]
type GridNode struct {
	Row, Col int
	id       int64
	graph    *GridGraph
}

// ID returns a unique identifier for this node
func (n GridNode) ID() int64 {
	// Generate ID from row and col: row * maxCols + col
	// For simplicity, use a hash-like approach
	if n.graph == nil || len(n.graph.Matrix) == 0 {
		return int64(n.Row)*10000 + int64(n.Col)
	}
	maxCols := len(n.graph.Matrix[0])
	return int64(n.Row)*int64(maxCols) + int64(n.Col)
}

// Equal implements Comparable[GridNode, float32]
func (n GridNode) Equal(other Node[GridNode, float32]) bool {
	if other == nil {
		return false
	}
	o, ok := other.(GridNode)
	if !ok {
		return false
	}
	return n.Row == o.Row && n.Col == o.Col
}

// Compare implements Comparable[GridNode, float32]
func (n GridNode) Compare(other Node[GridNode, float32]) int {
	if other == nil {
		return 1
	}
	o, ok := other.(GridNode)
	if !ok {
		return 0
	}
	// Compare by row first, then column
	if n.Row < o.Row {
		return -1
	}
	if n.Row > o.Row {
		return 1
	}
	if n.Col < o.Col {
		return -1
	}
	if n.Col > o.Col {
		return 1
	}
	return 0
}

// Data returns the node's data (the coordinates)
func (n GridNode) Data() GridNode {
	return n
}

// Neighbors returns an iterator over neighboring nodes
func (n GridNode) Neighbors() iter.Seq[Node[GridNode, float32]] {
	return func(yield func(Node[GridNode, float32]) bool) {
		if n.graph == nil {
			return
		}
		dirs := getGridDirections(n.graph.AllowDiag)
		for _, dir := range dirs {
			newRow := n.Row + dir[0]
			newCol := n.Col + dir[1]

			if newRow < 0 || newRow >= len(n.graph.Matrix) ||
				newCol < 0 || newCol >= len(n.graph.Matrix[0]) {
				continue
			}

			if n.graph.Matrix[newRow][newCol] <= n.graph.Obstacle {
				continue
			}

			neighbor := GridNode{Row: newRow, Col: newCol, graph: n.graph}
			if !yield(neighbor) {
				return
			}
		}
	}
}

// Edges returns an iterator over edges from this node
func (n GridNode) Edges() iter.Seq[Edge[GridNode, float32]] {
	return func(yield func(Edge[GridNode, float32]) bool) {
		if n.graph == nil {
			return
		}
		dirs := getGridDirections(n.graph.AllowDiag)
		for _, dir := range dirs {
			newRow := n.Row + dir[0]
			newCol := n.Col + dir[1]

			if newRow < 0 || newRow >= len(n.graph.Matrix) ||
				newCol < 0 || newCol >= len(n.graph.Matrix[0]) {
				continue
			}

			if n.graph.Matrix[newRow][newCol] <= n.graph.Obstacle {
				continue
			}

			to := GridNode{Row: newRow, Col: newCol, graph: n.graph}
			cost := n.Cost(to)
			edge := GridEdge{from: n, to: to, data: cost}
			if !yield(edge) {
				return
			}
		}
	}
}

// NumNeighbors returns the number of neighboring nodes
func (n GridNode) NumNeighbors() int {
	if n.graph == nil {
		return 0
	}
	count := 0
	dirs := getGridDirections(n.graph.AllowDiag)
	for _, dir := range dirs {
		newRow := n.Row + dir[0]
		newCol := n.Col + dir[1]

		if newRow < 0 || newRow >= len(n.graph.Matrix) ||
			newCol < 0 || newCol >= len(n.graph.Matrix[0]) {
			continue
		}

		if n.graph.Matrix[newRow][newCol] <= n.graph.Obstacle {
			continue
		}

		count++
	}
	return count
}

// Cost calculates the cost from this node to another node
func (n GridNode) Cost(toOther Node[GridNode, float32]) float32 {
	if toOther == nil || n.graph == nil {
		return 0
	}
	to, ok := toOther.(GridNode)
	if !ok {
		return 0
	}
	if to.Row < 0 || to.Row >= len(n.graph.Matrix) ||
		to.Col < 0 || to.Col >= len(n.graph.Matrix[0]) {
		return 0
	}
	cost := n.graph.Matrix[to.Row][to.Col]
	if n.graph.AllowDiag && n.Row != to.Row && n.Col != to.Col {
		cost *= 1.41421356237 // sqrt(2)
	}
	return cost
}

// GridEdge represents an edge in a grid graph
type GridEdge struct {
	from GridNode
	to   GridNode
	data float32
	id   int64
}

// ID returns a unique identifier for this edge
func (e GridEdge) ID() int64 {
	// Generate ID from from and to node IDs
	return e.from.ID()*1000000 + e.to.ID()
}

// From returns the source node
func (e GridEdge) From() Node[GridNode, float32] {
	return e.from
}

// To returns the destination node
func (e GridEdge) To() Node[GridNode, float32] {
	return e.to
}

// Data returns the edge's data (the cost)
func (e GridEdge) Data() float32 {
	return e.data
}

// Cost returns the cost/weight of this edge
func (e GridEdge) Cost() float32 {
	return e.data
}

// Nodes returns an iterator over all nodes in the grid
func (g *GridGraph) Nodes() iter.Seq[Node[GridNode, float32]] {
	return func(yield func(Node[GridNode, float32]) bool) {
		if len(g.Matrix) == 0 {
			return
		}
		for row := 0; row < len(g.Matrix); row++ {
			for col := 0; col < len(g.Matrix[row]); col++ {
				if g.Matrix[row][col] <= g.Obstacle {
					continue
				}
				node := GridNode{Row: row, Col: col, graph: g}
				if !yield(node) {
					return
				}
			}
		}
	}
}

// Edges returns an iterator over all edges in the grid
func (g *GridGraph) Edges() iter.Seq[Edge[GridNode, float32]] {
	return func(yield func(Edge[GridNode, float32]) bool) {
		for node := range g.Nodes() {
			gn := node.(GridNode)
			for edge := range gn.Edges() {
				if !yield(edge) {
					return
				}
			}
		}
	}
}

// NumNodes returns the total number of nodes in the graph
func (g *GridGraph) NumNodes() int {
	if len(g.Matrix) == 0 {
		return 0
	}
	count := 0
	for row := 0; row < len(g.Matrix); row++ {
		for col := 0; col < len(g.Matrix[row]); col++ {
			if g.Matrix[row][col] > g.Obstacle {
				count++
			}
		}
	}
	return count
}

// NumEdges returns the total number of edges in the graph
func (g *GridGraph) NumEdges() int {
	if len(g.Matrix) == 0 {
		return 0
	}
	count := 0
	dirs := getGridDirections(g.AllowDiag)
	for row := 0; row < len(g.Matrix); row++ {
		for col := 0; col < len(g.Matrix[row]); col++ {
			if g.Matrix[row][col] <= g.Obstacle {
				continue
			}
			// Count valid neighbors
			for _, dir := range dirs {
				newRow := row + dir[0]
				newCol := col + dir[1]
				if newRow < 0 || newRow >= len(g.Matrix) ||
					newCol < 0 || newCol >= len(g.Matrix[0]) {
					continue
				}
				if g.Matrix[newRow][newCol] > g.Obstacle {
					count++
				}
			}
		}
	}
	return count
}

// NewGridNode creates a GridNode with the given coordinates and graph reference
func NewGridNode(g *GridGraph, row, col int) GridNode {
	return GridNode{Row: row, Col: col, graph: g}
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
