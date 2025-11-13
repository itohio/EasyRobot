package graph

import (
	"github.com/itohio/EasyRobot/x/math/mat"
)

// MatrixGraph represents a graph where each column is a node
// Matrix represents adjacency matrix: matrix[i][j] = weight from node i to node j
// Each row i contains weights to all neighbors (columns j)
type MatrixGraph struct {
	Matrix   mat.Matrix // matrix[i][j] = weight from node (column) i to node (column) j
	Obstacle float32    // Values <= Obstacle are considered non-existent edges
}

// MatrixNode represents a node in a matrix graph (column index)
type MatrixNode int

func (n MatrixNode) Equal(other Node) bool {
	o, ok := other.(MatrixNode)
	if !ok {
		return false
	}
	return int(n) == int(o)
}

func (g *MatrixGraph) Neighbors(n Node) []Node {
	mn, ok := n.(MatrixNode)
	if !ok {
		return nil
	}

	col := int(mn)
	if col < 0 || col >= len(g.Matrix) {
		return nil
	}

	var neighbors []Node

	// Standard adjacency matrix: matrix[i][j] = weight from node i to node j
	// For node col (column col), check row col for all neighbors
	if col < len(g.Matrix) {
		for j := 0; j < len(g.Matrix[col]); j++ {
			if j != col && g.Matrix[col][j] > g.Obstacle {
				neighbors = append(neighbors, MatrixNode(j))
			}
		}
	}

	return neighbors
}

func (g *MatrixGraph) Cost(from, to Node) float32 {
	fromNode, ok := from.(MatrixNode)
	if !ok {
		return 0
	}
	toNode, ok := to.(MatrixNode)
	if !ok {
		return 0
	}

	fromCol := int(fromNode)
	toCol := int(toNode)

	if fromCol < 0 || fromCol >= len(g.Matrix) ||
		toCol < 0 || toCol >= len(g.Matrix[fromCol]) {
		return 0
	}

	// matrix[fromCol][toCol] = weight from node fromCol to node toCol
	return g.Matrix[fromCol][toCol]
}
