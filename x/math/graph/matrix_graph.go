package graph

import (
	"fmt"
	"iter"

	"github.com/itohio/EasyRobot/x/math/mat"
)

// MatrixGraph represents a graph where each column is a node
// Matrix represents adjacency matrix: matrix[i][j] = weight from node i to node j
// Each row i contains weights to all neighbors (columns j)
// Node data type: MatrixNode (the column index)
// Edge data type: float32 (the weight)
type MatrixGraph struct {
	Matrix   mat.Matrix // matrix[i][j] = weight from node (column) i to node (column) j
	Obstacle float32    // Values <= Obstacle are considered non-existent edges
}

// MatrixNode represents a node in a matrix graph (column index), with reference to the graph.
// It implements Node[MatrixNode, float32]
type MatrixNode struct {
	Index int
	Graph *MatrixGraph
}

// ID returns a unique identifier for this node
func (n MatrixNode) ID() int64 {
	return int64(n.Index)
}

// Equal implements Comparable[MatrixNode, float32]
func (n MatrixNode) Equal(other Node[MatrixNode, float32]) bool {
	if other == nil {
		return false
	}
	o, ok := other.(MatrixNode)
	if !ok {
		return false
	}
	return n.Index == o.Index
}

// Compare implements Comparable[MatrixNode, float32]
func (n MatrixNode) Compare(other Node[MatrixNode, float32]) int {
	if other == nil {
		return 1
	}
	o, ok := other.(MatrixNode)
	if !ok {
		return 0
	}
	if n.Index < o.Index {
		return -1
	}
	if n.Index > o.Index {
		return 1
	}
	return 0
}

// Data returns the node's data (the column index)
func (n MatrixNode) Data() MatrixNode {
	return n
}

// Neighbors returns an iterator over neighboring nodes (directly from the underlying matrix)
func (n MatrixNode) Neighbors() iter.Seq[Node[MatrixNode, float32]] {
	return func(yield func(Node[MatrixNode, float32]) bool) {
		if n.Graph == nil || n.Index < 0 || n.Index >= len(n.Graph.Matrix) {
			return
		}
		row := n.Graph.Matrix[n.Index]
		for j, val := range row {
			if val > n.Graph.Obstacle {
				neighbor := MatrixNode{Index: j, Graph: n.Graph}
				if !yield(neighbor) {
					return
				}
			}
		}
	}
}

// Edges returns an iterator over edges from this node (directly from the underlying matrix)
func (n MatrixNode) Edges() iter.Seq[Edge[MatrixNode, float32]] {
	return func(yield func(Edge[MatrixNode, float32]) bool) {
		if n.Graph == nil || n.Index < 0 || n.Index >= len(n.Graph.Matrix) {
			return
		}
		row := n.Graph.Matrix[n.Index]
		for j, val := range row {
			if val > n.Graph.Obstacle {
				toNode := MatrixNode{Index: j, Graph: n.Graph}
				edge := MatrixEdge{
					from: n,
					to:   toNode,
					data: val,
				}
				if !yield(edge) {
					return
				}
			}
		}
	}
}

// NumNeighbors returns the number of neighboring nodes
func (n MatrixNode) NumNeighbors() int {
	if n.Graph == nil || n.Index < 0 || n.Index >= len(n.Graph.Matrix) {
		return 0
	}
	count := 0
	row := n.Graph.Matrix[n.Index]
	for _, val := range row {
		if val > n.Graph.Obstacle {
			count++
		}
	}
	return count
}

// Cost calculates the cost from this node to another node
func (n MatrixNode) Cost(toOther Node[MatrixNode, float32]) float32 {
	if toOther == nil || n.Graph == nil {
		return 0
	}
	to, ok := toOther.(MatrixNode)
	if !ok {
		return 0
	}
	if n.Index < 0 || n.Index >= len(n.Graph.Matrix) ||
		to.Index < 0 || to.Index >= len(n.Graph.Matrix[n.Index]) {
		return 0
	}
	return n.Graph.Matrix[n.Index][to.Index]
}

// MatrixEdge represents an edge in a matrix graph
type MatrixEdge struct {
	from MatrixNode
	to   MatrixNode
	data float32
	id   int64
}

// ID returns a unique identifier for this edge
func (e MatrixEdge) ID() int64 {
	// Generate ID from from and to node IDs
	return int64(e.from.Index)*1000000 + int64(e.to.Index)
}

// From returns the source node
func (e MatrixEdge) From() Node[MatrixNode, float32] {
	return e.from
}

// To returns the destination node
func (e MatrixEdge) To() Node[MatrixNode, float32] {
	return e.to
}

// Data returns the edge's data (the weight)
func (e MatrixEdge) Data() float32 {
	return e.data
}

// Cost returns the cost/weight of this edge
func (e MatrixEdge) Cost() float32 {
	return e.data
}

// Nodes returns an iterator over all nodes in the matrix graph
func (g *MatrixGraph) Nodes() iter.Seq[Node[MatrixNode, float32]] {
	return func(yield func(Node[MatrixNode, float32]) bool) {
		for i := 0; i < len(g.Matrix); i++ {
			node := MatrixNode{Index: i, Graph: g}
			if !yield(node) {
				return
			}
		}
	}
}

// Edges returns an iterator over all edges in the matrix graph
func (g *MatrixGraph) Edges() iter.Seq[Edge[MatrixNode, float32]] {
	return func(yield func(Edge[MatrixNode, float32]) bool) {
		for i := 0; i < len(g.Matrix); i++ {
			fromNode := MatrixNode{Index: i, Graph: g}
			if i >= len(g.Matrix) {
				continue
			}
			for j := 0; j < len(g.Matrix[i]); j++ {
				if g.Matrix[i][j] > g.Obstacle {
					toNode := MatrixNode{Index: j, Graph: g}
					edge := MatrixEdge{
						from: fromNode,
						to:   toNode,
						data: g.Matrix[i][j],
					}
					if !yield(edge) {
						return
					}
				}
			}
		}
	}
}

// NumNodes returns the total number of nodes in the graph
func (g *MatrixGraph) NumNodes() int {
	return len(g.Matrix)
}

// NumEdges returns the total number of edges in the graph
func (g *MatrixGraph) NumEdges() int {
	count := 0
	for i := 0; i < len(g.Matrix); i++ {
		for j := 0; j < len(g.Matrix[i]); j++ {
			if g.Matrix[i][j] > g.Obstacle {
				count++
			}
		}
	}
	return count
}

// ToMatrix converts any graph to an adjacency matrix
// The matrix must have rows and cols equal to the number of nodes in the graph
// First maps every node ID to a matrix index, then fills in edge costs
func ToMatrix[N any, E any](g Graph[N, E], matrix mat.Matrix) error {
	// Count nodes and validate matrix size
	nodeCount := 0
	nodeIDToIndex := make(map[int64]int)

	// First pass: collect all nodes and map IDs to indices
	for node := range g.Nodes() {
		if node == nil {
			continue
		}
		nodeID := node.ID()
		if _, exists := nodeIDToIndex[nodeID]; !exists {
			nodeIDToIndex[nodeID] = nodeCount
			nodeCount++
		}
	}

	// Validate matrix dimensions
	if len(matrix) != nodeCount {
		return &MatrixSizeError{Expected: nodeCount, Actual: len(matrix)}
	}
	for i := range matrix {
		if len(matrix[i]) != nodeCount {
			return &MatrixSizeError{Expected: nodeCount, Actual: len(matrix[i])}
		}
	}

	// Zero out the matrix first
	for i := range matrix {
		for j := range matrix[i] {
			matrix[i][j] = 0
		}
	}

	// Second pass: fill in edge costs
	for edge := range g.Edges() {
		if edge == nil {
			continue
		}
		from := edge.From()
		to := edge.To()
		if from == nil || to == nil {
			continue
		}

		fromIdx, fromExists := nodeIDToIndex[from.ID()]
		toIdx, toExists := nodeIDToIndex[to.ID()]
		if !fromExists || !toExists {
			continue
		}

		cost := edge.Cost()
		matrix[fromIdx][toIdx] = cost
	}

	return nil
}

// MatrixSizeError represents an error when matrix dimensions don't match node count
type MatrixSizeError struct {
	Expected int
	Actual   int
}

func (e *MatrixSizeError) Error() string {
	return fmt.Sprintf("matrix size mismatch: expected %d, got %d", e.Expected, e.Actual)
}
