package grid

import (
	"github.com/itohio/EasyRobot/x/math/graph"
	"github.com/itohio/EasyRobot/x/math/mat"
	"github.com/itohio/EasyRobot/x/math/vec"
)

// DijkstraOptions configures Dijkstra algorithm behavior
type DijkstraOptions struct {
	AllowDiagonal bool
	ObstacleValue float32
}

// Dijkstra finds shortest path in matrix using Dijkstra's algorithm
// Uses optimized heap implementation from graph package
// matrix: 2D grid where values represent costs (values <= obstacleValue = obstacle)
// startRow, startCol: Starting position (row, col)
// goalRow, goalCol: Goal position (row, col)
// opts: Options (nil uses defaults: 4-directional, obstacleValue=0)
// Returns path as []vec.Vector2D, or nil if no path exists
func Dijkstra(
	matrix mat.Matrix,
	startRow, startCol int,
	goalRow, goalCol int,
	opts *DijkstraOptions,
) []vec.Vector2D {
	if !validateInput(matrix, startRow, startCol, goalRow, goalCol) {
		return nil
	}

	if opts == nil {
		opts = &DijkstraOptions{
			AllowDiagonal: false,
			ObstacleValue: 0,
		}
	}

	// Check if start/goal are obstacles
	if matrix[startRow][startCol] <= opts.ObstacleValue ||
		matrix[goalRow][goalCol] <= opts.ObstacleValue {
		return nil
	}

	g := &graph.GridGraph{
		Matrix:    matrix,
		AllowDiag: opts.AllowDiagonal,
		Obstacle:  opts.ObstacleValue,
	}

	start := graph.NewGridNode(g, startRow, startCol)
	goal := graph.NewGridNode(g, goalRow, goalCol)

	d := graph.NewDijkstra(g)
	path := d.Search(start, goal)
	if path == nil {
		return nil
	}

	return graph.GridPathToVector2D(path)
}
