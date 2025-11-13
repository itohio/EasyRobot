package graph

import (
	"github.com/itohio/EasyRobot/pkg/core/math/vec"
)

// GridPathToVector2D converts a Path containing GridNode to []vec.Vector2D
func GridPathToVector2D(path Path) []vec.Vector2D {
	if path == nil {
		return nil
	}

	result := make([]vec.Vector2D, len(path))
	for i, n := range path {
		gn, ok := n.(GridNode)
		if !ok {
			return nil
		}
		result[i] = vec.Vector2D{float32(gn.Col), float32(gn.Row)}
	}

	return result
}

// MatrixPathToVector converts a Path containing MatrixNode to []vec.Vector (indices)
func MatrixPathToVector(path Path) []vec.Vector {
	if path == nil {
		return nil
	}

	result := make([]vec.Vector, len(path))
	for i, n := range path {
		mn, ok := n.(MatrixNode)
		if !ok {
			return nil
		}
		// Convert MatrixNode (int) to Vector ([]float32)
		result[i] = vec.Vector{float32(mn)}
	}

	return result
}

// TreePathToNodes returns Path as []Node (already is)
func TreePathToNodes(path Path) []Node {
	return []Node(path)
}
