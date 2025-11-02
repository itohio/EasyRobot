package grid

import (
	"testing"

	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/pkg/core/math/graph"
	"github.com/itohio/EasyRobot/pkg/core/math/mat"
	"github.com/itohio/EasyRobot/pkg/core/math/vec"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestAStar_SimplePath(t *testing.T) {
	matrix := mat.New(5, 5)
	for i := 0; i < 5; i++ {
		for j := 0; j < 5; j++ {
			matrix[i][j] = 1.0
		}
	}

	path := AStar(matrix, 0, 0, 4, 4, nil)
	require.NotNil(t, path, "Path should exist")
	assert.Greater(t, len(path), 0, "Path should have points")
	assert.Equal(t, vec.Vector2D{0, 0}, path[0], "Path should start at (0,0)")
}

func TestAStar_WithObstacles(t *testing.T) {
	matrix := mat.New(5, 5)
	for i := 0; i < 5; i++ {
		for j := 0; j < 5; j++ {
			matrix[i][j] = 1.0
		}
	}
	// Add obstacle wall (but leave a gap)
	for i := 0; i < 4; i++ {
		matrix[i][2] = 0.0
	}
	// Leave one cell open for path to go through
	matrix[4][2] = 1.0

	path := AStar(matrix, 0, 0, 4, 4, nil)
	require.NotNil(t, path, "Path should exist despite obstacles")
	assert.Greater(t, len(path), 4, "Path should go around obstacle")
}

func TestAStar_NoPath(t *testing.T) {
	matrix := mat.New(5, 5)
	for i := 0; i < 5; i++ {
		for j := 0; j < 5; j++ {
			matrix[i][j] = 1.0
		}
	}
	// Block entire middle column
	for i := 0; i < 5; i++ {
		matrix[i][2] = 0.0
	}
	// Block path completely
	matrix[0][1] = 0.0
	matrix[0][3] = 0.0

	path := AStar(matrix, 0, 0, 0, 4, nil)
	assert.Nil(t, path, "Path should not exist when blocked")
}

func TestAStar_DiagonalMovement(t *testing.T) {
	matrix := mat.New(5, 5)
	for i := 0; i < 5; i++ {
		for j := 0; j < 5; j++ {
			matrix[i][j] = 1.0
		}
	}

	opts := &AStarOptions{
		AllowDiagonal: true,
		Heuristic:     EuclideanHeuristic,
		ObstacleValue: 0,
	}

	path := AStar(matrix, 0, 0, 4, 4, opts)
	require.NotNil(t, path, "Path should exist")
	assert.LessOrEqual(t, len(path), 5, "Diagonal path should be shorter")
}

func TestAStar_ManhattanHeuristic(t *testing.T) {
	matrix := mat.New(5, 5)
	for i := 0; i < 5; i++ {
		for j := 0; j < 5; j++ {
			matrix[i][j] = 1.0
		}
	}

	opts := &AStarOptions{
		AllowDiagonal: false,
		Heuristic:     ManhattanHeuristic,
		ObstacleValue: 0,
	}

	path := AStar(matrix, 0, 0, 4, 4, opts)
	require.NotNil(t, path, "Path should exist with Manhattan heuristic")
}

func TestAStar_SameStartGoal(t *testing.T) {
	matrix := mat.New(5, 5)
	for i := 0; i < 5; i++ {
		for j := 0; j < 5; j++ {
			matrix[i][j] = 1.0
		}
	}

	path := AStar(matrix, 2, 2, 2, 2, nil)
	require.NotNil(t, path, "Path should exist when start == goal")
	assert.Equal(t, 1, len(path), "Path should have one point")
	assert.Equal(t, vec.Vector2D{2, 2}, path[0], "Path point should be (2,2)")
}

func TestAStar_InvalidInput(t *testing.T) {
	matrix := mat.New(5, 5)

	// Invalid start coordinates
	path := AStar(matrix, -1, 0, 4, 4, nil)
	assert.Nil(t, path, "Should return nil for invalid start")

	// Invalid goal coordinates
	path = AStar(matrix, 0, 0, 10, 10, nil)
	assert.Nil(t, path, "Should return nil for invalid goal")

	// Empty matrix
	empty := mat.New(0, 0)
	path = AStar(empty, 0, 0, 0, 0, nil)
	assert.Nil(t, path, "Should return nil for empty matrix")
}

func TestAStar_CustomObstacleValue(t *testing.T) {
	matrix := mat.New(5, 5)
	for i := 0; i < 5; i++ {
		for j := 0; j < 5; j++ {
			matrix[i][j] = 2.0 // All cells have cost 2.0
		}
	}
	// Obstacles with value 1.0
	matrix[2][2] = 1.0
	matrix[2][3] = 1.0

	opts := &AStarOptions{
		AllowDiagonal: false,
		Heuristic:     EuclideanHeuristic,
		ObstacleValue: 1.5, // Values <= 1.5 are obstacles
	}

	path := AStar(matrix, 0, 0, 4, 4, opts)
	require.NotNil(t, path, "Path should exist")
}

func TestDijkstra_SimplePath(t *testing.T) {
	matrix := mat.New(5, 5)
	for i := 0; i < 5; i++ {
		for j := 0; j < 5; j++ {
			matrix[i][j] = 1.0
		}
	}

	path := Dijkstra(matrix, 0, 0, 4, 4, nil)
	require.NotNil(t, path, "Path should exist")
	assert.Greater(t, len(path), 0, "Path should have points")
}

func TestDijkstra_WithObstacles(t *testing.T) {
	matrix := mat.New(5, 5)
	for i := 0; i < 5; i++ {
		for j := 0; j < 5; j++ {
			matrix[i][j] = 1.0
		}
	}
	// Add obstacle wall (but leave a gap)
	for i := 0; i < 4; i++ {
		matrix[i][2] = 0.0
	}
	// Leave one cell open for path to go through
	matrix[4][2] = 1.0

	path := Dijkstra(matrix, 0, 0, 4, 4, nil)
	require.NotNil(t, path, "Path should exist despite obstacles")
}

func TestDijkstra_DiagonalMovement(t *testing.T) {
	matrix := mat.New(5, 5)
	for i := 0; i < 5; i++ {
		for j := 0; j < 5; j++ {
			matrix[i][j] = 1.0
		}
	}

	opts := &DijkstraOptions{
		AllowDiagonal: true,
		ObstacleValue: 0,
	}

	path := Dijkstra(matrix, 0, 0, 4, 4, opts)
	require.NotNil(t, path, "Path should exist with diagonal movement")
}

func TestHeuristic_Euclidean(t *testing.T) {
	from := graph.GridNode{Row: 0, Col: 0}
	to := graph.GridNode{Row: 3, Col: 4}
	dist := EuclideanHeuristic(from, to)
	expected := math32.Sqrt(3*3 + 4*4) // 5
	assert.InDelta(t, expected, dist, 1e-5, "Euclidean distance")
}

func TestHeuristic_Manhattan(t *testing.T) {
	from := graph.GridNode{Row: 0, Col: 0}
	to := graph.GridNode{Row: 3, Col: 4}
	dist := ManhattanHeuristic(from, to)
	expected := float32(7) // 3 + 4
	assert.InDelta(t, expected, dist, 1e-5, "Manhattan distance")
}

func TestHeuristic_Zero(t *testing.T) {
	from := graph.GridNode{Row: 0, Col: 0}
	to := graph.GridNode{Row: 100, Col: 100}
	dist := ZeroHeuristic(from, to)
	assert.Equal(t, float32(0), dist, "Zero heuristic should always return 0")
}
