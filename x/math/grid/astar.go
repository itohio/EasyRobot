package grid

import (
	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/x/math/graph"
	"github.com/itohio/EasyRobot/x/math/mat"
	"github.com/itohio/EasyRobot/x/math/vec"
)

// AStarOptions configures A* algorithm behavior
type AStarOptions struct {
	AllowDiagonal bool
	Heuristic     graph.Heuristic[graph.GridNode, float32]
	ObstacleValue float32
}

// AStar finds path in matrix using A* algorithm
// matrix: 2D grid where values represent costs (values <= obstacleValue = obstacle)
// startRow, startCol: Starting position (row, col)
// goalRow, goalCol: Goal position (row, col)
// opts: Options (nil uses defaults: 4-directional, Euclidean heuristic, obstacleValue=0)
// Returns path as []vec.Vector2D, or nil if no path exists
func AStar(
	matrix mat.Matrix,
	startRow, startCol int,
	goalRow, goalCol int,
	opts *AStarOptions,
) []vec.Vector2D {
	if !validateInput(matrix, startRow, startCol, goalRow, goalCol) {
		return nil
	}

	if opts == nil {
		opts = &AStarOptions{
			AllowDiagonal: false,
			Heuristic:     EuclideanHeuristic,
			ObstacleValue: 0,
		}
	}

	if opts.Heuristic == nil {
		opts.Heuristic = EuclideanHeuristic
	}

	g := &graph.GridGraph{
		Matrix:    matrix,
		AllowDiag: opts.AllowDiagonal,
		Obstacle:  opts.ObstacleValue,
	}

	start := graph.NewGridNode(g, startRow, startCol)
	goal := graph.NewGridNode(g, goalRow, goalCol)

	astar := graph.NewAStar(g, opts.Heuristic)
	path := astar.Search(start, goal)
	if path == nil {
		return nil
	}

	return graph.GridPathToVector2D(path)
}

func validateInput(matrix mat.Matrix, startRow, startCol, goalRow, goalCol int) bool {
	if len(matrix) == 0 || len(matrix[0]) == 0 {
		return false
	}
	rows, cols := len(matrix), len(matrix[0])
	if startRow < 0 || startRow >= rows || startCol < 0 || startCol >= cols {
		return false
	}
	if goalRow < 0 || goalRow >= rows || goalCol < 0 || goalCol >= cols {
		return false
	}
	return true
}

// EuclideanHeuristic uses Euclidean distance
func EuclideanHeuristic(from, to graph.Node[graph.GridNode, float32]) float32 {
	fromNode := from.Data()
	toNode := to.Data()

	dx := float32(toNode.Col - fromNode.Col)
	dy := float32(toNode.Row - fromNode.Row)
	return float32(math32.Sqrt(dx*dx + dy*dy))
}

// ManhattanHeuristic uses Manhattan distance
func ManhattanHeuristic(from, to graph.Node[graph.GridNode, float32]) float32 {
	fromNode := from.Data()
	toNode := to.Data()

	dx := float32(toNode.Col - fromNode.Col)
	dy := float32(toNode.Row - fromNode.Row)
	return math32.Abs(dx) + math32.Abs(dy)
}

// ZeroHeuristic returns 0 (equivalent to Dijkstra)
func ZeroHeuristic(from, to graph.Node[graph.GridNode, float32]) float32 {
	return 0
}
