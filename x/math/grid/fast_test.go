package grid

import (
	"testing"

	"github.com/itohio/EasyRobot/x/math/mat"
	"github.com/itohio/EasyRobot/x/math/vec"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestFastAStar_SimplePath(t *testing.T) {
	matrix := mat.New(5, 5)
	for i := 0; i < 5; i++ {
		for j := 0; j < 5; j++ {
			matrix[i][j] = 1.0
		}
	}

	fastAStar := NewFastAStar(matrix, false, 0, EuclideanHeuristic)
	path := fastAStar.Search(0, 0, 4, 4)
	require.NotNil(t, path, "Path should exist")
	assert.Greater(t, len(path), 0, "Path should have points")
	assert.Equal(t, vec.Vector2D{0, 0}, path[0], "Path should start at (0,0)")
}

func TestFastAStar_ReuseInstance(t *testing.T) {
	matrix := mat.New(5, 5)
	for i := 0; i < 5; i++ {
		for j := 0; j < 5; j++ {
			matrix[i][j] = 1.0
		}
	}

	fastAStar := NewFastAStar(matrix, false, 0, EuclideanHeuristic)

	// First search
	path1 := fastAStar.Search(0, 0, 4, 4)
	require.NotNil(t, path1, "First path should exist")

	// Second search (should reuse buffers)
	path2 := fastAStar.Search(0, 0, 2, 2)
	require.NotNil(t, path2, "Second path should exist")
	assert.Equal(t, vec.Vector2D{0, 0}, path2[0], "Second path should start at (0,0)")
}

func TestFastAStar_WithObstacles(t *testing.T) {
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
	matrix[4][2] = 1.0 // Gap

	fastAStar := NewFastAStar(matrix, false, 0, EuclideanHeuristic)
	path := fastAStar.Search(0, 0, 4, 4)
	require.NotNil(t, path, "Path should exist despite obstacles")
	assert.Greater(t, len(path), 4, "Path should go around obstacle")
}

func TestFastAStar_DiagonalMovement(t *testing.T) {
	matrix := mat.New(5, 5)
	for i := 0; i < 5; i++ {
		for j := 0; j < 5; j++ {
			matrix[i][j] = 1.0
		}
	}

	fastAStar := NewFastAStar(matrix, true, 0, EuclideanHeuristic)
	path := fastAStar.Search(0, 0, 4, 4)
	require.NotNil(t, path, "Path should exist")
	assert.LessOrEqual(t, len(path), 5, "Diagonal path should be shorter")
}

func TestFastDijkstra_SimplePath(t *testing.T) {
	matrix := mat.New(5, 5)
	for i := 0; i < 5; i++ {
		for j := 0; j < 5; j++ {
			matrix[i][j] = 1.0
		}
	}

	fastDijkstra := NewFastDijkstra(matrix, false, 0)
	path := fastDijkstra.Search(0, 0, 4, 4)
	require.NotNil(t, path, "Path should exist")
	assert.Greater(t, len(path), 0, "Path should have points")
}

func TestFastDijkstra_ReuseInstance(t *testing.T) {
	matrix := mat.New(5, 5)
	for i := 0; i < 5; i++ {
		for j := 0; j < 5; j++ {
			matrix[i][j] = 1.0
		}
	}

	fastDijkstra := NewFastDijkstra(matrix, false, 0)

	// First search
	path1 := fastDijkstra.Search(0, 0, 4, 4)
	require.NotNil(t, path1, "First path should exist")

	// Second search (should reuse buffers)
	path2 := fastDijkstra.Search(0, 0, 2, 2)
	require.NotNil(t, path2, "Second path should exist")
	assert.Equal(t, vec.Vector2D{0, 0}, path2[0], "Second path should start at (0,0)")
}

func TestFastDijkstra_WithObstacles(t *testing.T) {
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
	matrix[4][2] = 1.0 // Gap

	fastDijkstra := NewFastDijkstra(matrix, false, 0)
	path := fastDijkstra.Search(0, 0, 4, 4)
	require.NotNil(t, path, "Path should exist despite obstacles")
}

func TestFastDijkstra_DiagonalMovement(t *testing.T) {
	matrix := mat.New(5, 5)
	for i := 0; i < 5; i++ {
		for j := 0; j < 5; j++ {
			matrix[i][j] = 1.0
		}
	}

	fastDijkstra := NewFastDijkstra(matrix, true, 0)
	path := fastDijkstra.Search(0, 0, 4, 4)
	require.NotNil(t, path, "Path should exist with diagonal movement")
}

func TestFastDijkstra_NoPath(t *testing.T) {
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

	fastDijkstra := NewFastDijkstra(matrix, false, 0)
	path := fastDijkstra.Search(0, 0, 0, 4)
	assert.Nil(t, path, "Path should not exist when blocked")
}
