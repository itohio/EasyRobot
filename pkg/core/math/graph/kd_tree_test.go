package graph

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestKDTree_BuildTree(t *testing.T) {
	points := [][]float32{
		{2.0, 3.0},
		{5.0, 4.0},
		{9.0, 6.0},
		{4.0, 7.0},
		{8.0, 1.0},
		{7.0, 2.0},
	}

	kt := NewKDTree(points)
	require.NotNil(t, kt, "KDTree should be created")
	require.NotNil(t, kt.Root, "Root should be created")
	assert.Equal(t, 2, kt.K, "K should be 2 for 2D points")
}

func TestKDTree_NearestNeighbor(t *testing.T) {
	points := [][]float32{
		{2.0, 3.0},
		{5.0, 4.0},
		{9.0, 6.0},
		{4.0, 7.0},
		{8.0, 1.0},
		{7.0, 2.0},
	}

	kt := NewKDTree(points)

	// Query point close to {5.0, 4.0}
	query := []float32{5.1, 4.1}
	nearest := kt.NearestNeighbor(query)
	require.NotNil(t, nearest, "Nearest neighbor should be found")
	assert.Equal(t, []float32{5.0, 4.0}, nearest, "Should find closest point")
}

func TestKDTree_KNearestNeighbors(t *testing.T) {
	points := [][]float32{
		{1.0, 1.0},
		{2.0, 2.0},
		{3.0, 3.0},
		{4.0, 4.0},
		{5.0, 5.0},
	}

	kt := NewKDTree(points)

	query := []float32{2.5, 2.5}
	k := 3
	neighbors := kt.KNearestNeighbors(query, k)
	require.NotNil(t, neighbors, "K nearest neighbors should be found")
	assert.Equal(t, k, len(neighbors), "Should return k neighbors")
}

func TestKDTree_RangeQuery(t *testing.T) {
	points := [][]float32{
		{1.0, 1.0},
		{2.0, 2.0},
		{3.0, 3.0},
		{4.0, 4.0},
		{5.0, 5.0},
	}

	kt := NewKDTree(points)

	min := []float32{1.5, 1.5}
	max := []float32{3.5, 3.5}
	results := kt.RangeQuery(min, max)
	require.NotNil(t, results, "Range query should return results")
	assert.Greater(t, len(results), 0, "Should find points in range")
}

func TestKDTree_Insert(t *testing.T) {
	points := [][]float32{
		{1.0, 1.0},
		{2.0, 2.0},
	}

	kt := NewKDTree(points)
	require.NotNil(t, kt.Root, "Root should exist")

	kt.Insert([]float32{3.0, 3.0})
	assert.NotNil(t, kt.Root, "Root should still exist after insert")
}

func TestKDTree_GetHeight(t *testing.T) {
	points := [][]float32{
		{1.0, 1.0},
		{2.0, 2.0},
		{3.0, 3.0},
	}

	kt := NewKDTree(points)
	height := kt.GetHeight()
	assert.GreaterOrEqual(t, height, 0, "Height should be non-negative")
}

func TestKDTree_EmptyTree(t *testing.T) {
	kt := NewKDTree([][]float32{})
	assert.Nil(t, kt.Root, "Root should be nil for empty tree")
	assert.Equal(t, 0, kt.K, "K should be 0 for empty tree")
}

func TestKDTree_SinglePoint(t *testing.T) {
	points := [][]float32{
		{1.0, 2.0},
	}

	kt := NewKDTree(points)
	require.NotNil(t, kt.Root, "Root should exist")
	assert.Equal(t, []float32{1.0, 2.0}, kt.Root.Point, "Root should contain the point")

	nearest := kt.NearestNeighbor([]float32{1.1, 2.1})
	assert.Equal(t, []float32{1.0, 2.0}, nearest, "Should find the single point")
}

func TestKDTree_3D(t *testing.T) {
	points := [][]float32{
		{1.0, 2.0, 3.0},
		{4.0, 5.0, 6.0},
		{7.0, 8.0, 9.0},
	}

	kt := NewKDTree(points)
	require.NotNil(t, kt, "KDTree should be created")
	assert.Equal(t, 3, kt.K, "K should be 3 for 3D points")

	query := []float32{4.1, 5.1, 6.1}
	nearest := kt.NearestNeighbor(query)
	require.NotNil(t, nearest, "Nearest neighbor should be found")
	assert.Equal(t, []float32{4.0, 5.0, 6.0}, nearest, "Should find closest 3D point")
}
