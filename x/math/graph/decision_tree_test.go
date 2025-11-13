package graph

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDecisionTree_BuildTree(t *testing.T) {
	// Simple 2D classification problem
	data := [][]float32{
		{1.0, 1.0},
		{1.0, 2.0},
		{2.0, 1.0},
		{2.0, 2.0},
		{5.0, 5.0},
		{5.0, 6.0},
		{6.0, 5.0},
		{6.0, 6.0},
	}

	labels := []any{"A", "A", "A", "A", "B", "B", "B", "B"}

	dt := NewDecisionTree(10, 1)
	err := dt.BuildTree(data, labels)
	require.NoError(t, err, "BuildTree should succeed")
	require.NotNil(t, dt.Root, "Root should be created")
}

func TestDecisionTree_Predict(t *testing.T) {
	data := [][]float32{
		{1.0, 1.0},
		{1.0, 2.0},
		{2.0, 1.0},
		{2.0, 2.0},
		{5.0, 5.0},
		{5.0, 6.0},
		{6.0, 5.0},
		{6.0, 6.0},
	}

	labels := []any{"A", "A", "A", "A", "B", "B", "B", "B"}

	dt := NewDecisionTree(10, 1)
	err := dt.BuildTree(data, labels)
	require.NoError(t, err)

	// Test predictions
	pred1 := dt.Predict([]float32{1.5, 1.5})
	assert.Equal(t, "A", pred1, "Should predict class A for point near A cluster")

	pred2 := dt.Predict([]float32{5.5, 5.5})
	assert.Equal(t, "B", pred2, "Should predict class B for point near B cluster")
}

func TestDecisionTree_PredictProbabilities(t *testing.T) {
	data := [][]float32{
		{1.0, 1.0},
		{1.0, 2.0},
		{2.0, 1.0},
		{2.0, 2.0},
	}

	labels := []any{"A", "A", "A", "A"}

	dt := NewDecisionTree(10, 1)
	err := dt.BuildTree(data, labels)
	require.NoError(t, err)

	probs := dt.PredictProbabilities([]float32{1.5, 1.5})
	require.NotNil(t, probs, "Probabilities should be returned")
	assert.Greater(t, probs["A"], float32(0), "Probability of A should be positive")
}

func TestDecisionTree_GetDepth(t *testing.T) {
	data := [][]float32{
		{1.0, 1.0},
		{2.0, 2.0},
		{5.0, 5.0},
		{6.0, 6.0},
	}

	labels := []any{"A", "A", "B", "B"}

	dt := NewDecisionTree(10, 1)
	err := dt.BuildTree(data, labels)
	require.NoError(t, err)

	depth := dt.GetDepth()
	assert.Greater(t, depth, 0, "Tree should have depth > 0")
}

func TestDecisionTree_CountNodes(t *testing.T) {
	data := [][]float32{
		{1.0, 1.0},
		{2.0, 2.0},
	}

	labels := []any{"A", "B"}

	dt := NewDecisionTree(10, 1)
	err := dt.BuildTree(data, labels)
	require.NoError(t, err)

	count := dt.CountNodes()
	assert.Greater(t, count, 0, "Tree should have nodes")
}

func TestDecisionTree_EmptyData(t *testing.T) {
	dt := NewDecisionTree(10, 1)
	err := dt.BuildTree([][]float32{}, []any{})
	require.NoError(t, err)
	assert.Nil(t, dt.Root, "Root should be nil for empty data")
}

func TestDecisionTree_MaxDepth(t *testing.T) {
	data := [][]float32{
		{1.0}, {2.0}, {3.0}, {4.0}, {5.0},
		{6.0}, {7.0}, {8.0}, {9.0}, {10.0},
	}

	labels := []any{"A", "A", "A", "A", "A", "B", "B", "B", "B", "B"}

	dt := NewDecisionTree(2, 1) // Max depth 2
	err := dt.BuildTree(data, labels)
	require.NoError(t, err)

	depth := dt.GetDepth()
	assert.LessOrEqual(t, depth, 2, "Depth should not exceed max depth")
}
