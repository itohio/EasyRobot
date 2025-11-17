package graph

import (
	"github.com/itohio/EasyRobot/x/math/vec"
)

// Path represents a path through the graph
type Path[N any, E any] []Node[N, E]

// GridPathToVector2D converts a Path containing GridNode to []vec.Vector2D
func GridPathToVector2D[E any](path Path[GridNode, E]) []vec.Vector2D {
	if path == nil {
		return nil
	}

	result := make([]vec.Vector2D, len(path))
	for i, n := range path {
		gn := n.Data()
		result[i] = vec.Vector2D{float32(gn.Col), float32(gn.Row)}
	}

	return result
}

// MatrixPathToVector converts a Path containing MatrixNode to []vec.Vector (indices)
func MatrixPathToVector[E any](path Path[MatrixNode, E]) []vec.Vector {
	if path == nil {
		return nil
	}

	result := make([]vec.Vector, len(path))
	for i, n := range path {
		mn := n.Data()
		result[i] = vec.Vector{float32(mn.Index)}
	}

	return result
}

// TreePathToNodes returns Path as []Node (already is)
func TreePathToNodes[N any, E any](path Path[N, E]) []Node[N, E] {
	return []Node[N, E](path)
}
