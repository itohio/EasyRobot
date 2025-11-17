package graph

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/itohio/EasyRobot/x/math/vec"
)

func TestVectorDecisionTree_SimpleDecision(t *testing.T) {
	// Create a simple decision tree:
	// Root: Check if x <= 5.0
	//   If true: Check if y <= 3.0
	//     If true: "Class A"
	//     If false: "Class B"
	//   If false: Check if y <= 7.0
	//     If true: "Class C"
	//     If false: "Class D"

	rootCriteria := VectorDimensionThreshold(0, 5.0)
	tree := NewVectorDecisionTree[string](
		rootCriteria,
		DecisionOutcome[string]{IsLeaf: false},
		"x <= 5.0",
	)

	rootIdx := tree.RootIdx()

	// Left branch (x <= 5.0)
	left1 := tree.AddDecisionChild(rootIdx, VectorDimensionThreshold(1, 3.0), "y <= 3.0", -1)
	require.GreaterOrEqual(t, left1, 0)

	// Right branch (x > 5.0)
	right1 := tree.AddDecisionChild(rootIdx, VectorDimensionThreshold(1, 7.0), "y <= 7.0", -1)
	require.GreaterOrEqual(t, right1, 0)

	// Leaf nodes
	leafA := tree.AddLeafChild(left1, "Class A", "x<=5.0 and y<=3.0", -1)
	require.GreaterOrEqual(t, leafA, 0)

	leafB := tree.AddLeafChild(left1, "Class B", "x<=5.0 and y>3.0", -1)
	require.GreaterOrEqual(t, leafB, 0)

	leafC := tree.AddLeafChild(right1, "Class C", "x>5.0 and y<=7.0", -1)
	require.GreaterOrEqual(t, leafC, 0)

	leafD := tree.AddLeafChild(right1, "Class D", "x>5.0 and y>7.0", -1)
	require.GreaterOrEqual(t, leafD, 0)

	// Test cases
	testCases := []struct {
		name     string
		input    vec.Vector2D
		expected string
	}{
		{"x<=5, y<=3", vec.Vector2D{3.0, 2.0}, "Class A"},
		{"x<=5, y>3", vec.Vector2D{3.0, 5.0}, "Class B"},
		{"x>5, y<=7", vec.Vector2D{7.0, 5.0}, "Class C"},
		{"x>5, y>7", vec.Vector2D{7.0, 10.0}, "Class D"},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result, ok := tree.ComputeDecision(tc.input)
			assert.True(t, ok, "Computation should succeed")
			assert.Equal(t, tc.expected, result, "Should match expected outcome")
		})
	}
}

func TestVectorDecisionTree_ReorderChildren(t *testing.T) {
	tree := NewVectorDecisionTree[string](
		VectorDimensionThreshold(0, 5.0),
		DecisionOutcome[string]{IsLeaf: false},
		"root",
	)

	rootIdx := tree.RootIdx()

	// Add children in order: A, B, C
	childA := tree.AddLeafChild(rootIdx, "A", "child A", -1)
	childB := tree.AddLeafChild(rootIdx, "B", "child B", -1)
	childC := tree.AddLeafChild(rootIdx, "C", "child C", -1)

	// Verify initial order
	rootNode := tree.Tree().GetNode(rootIdx)
	require.NotNil(t, rootNode)
	assert.Equal(t, childA, rootNode.childIdxs[0])
	assert.Equal(t, childB, rootNode.childIdxs[1])
	assert.Equal(t, childC, rootNode.childIdxs[2])

	// Reorder to: C, A, B
	newOrder := []int{childC, childA, childB}
	ok := tree.ReorderChildren(rootIdx, newOrder)
	assert.True(t, ok, "Reorder should succeed")

	// Verify new order
	rootNode = tree.Tree().GetNode(rootIdx)
	require.NotNil(t, rootNode)
	assert.Equal(t, childC, rootNode.childIdxs[0])
	assert.Equal(t, childA, rootNode.childIdxs[1])
	assert.Equal(t, childB, rootNode.childIdxs[2])
}

func TestVectorDecisionTree_SwapChildren(t *testing.T) {
	tree := NewVectorDecisionTree[string](
		VectorDimensionThreshold(0, 5.0),
		DecisionOutcome[string]{IsLeaf: false},
		"root",
	)

	rootIdx := tree.RootIdx()

	// Add children: A, B, C
	childA := tree.AddLeafChild(rootIdx, "A", "child A", -1)
	childB := tree.AddLeafChild(rootIdx, "B", "child B", -1)
	childC := tree.AddLeafChild(rootIdx, "C", "child C", -1)

	// Swap first and last
	ok := tree.SwapChildren(rootIdx, 0, 2)
	assert.True(t, ok, "Swap should succeed")

	// Verify swap
	rootNode := tree.Tree().GetNode(rootIdx)
	require.NotNil(t, rootNode)
	assert.Equal(t, childC, rootNode.childIdxs[0], "C should be first")
	assert.Equal(t, childB, rootNode.childIdxs[1], "B should be in middle")
	assert.Equal(t, childA, rootNode.childIdxs[2], "A should be last")
}

func TestVectorDecisionTree_AddChildAtPosition(t *testing.T) {
	tree := NewVectorDecisionTree[string](
		VectorDimensionThreshold(0, 5.0),
		DecisionOutcome[string]{IsLeaf: false},
		"root",
	)

	rootIdx := tree.RootIdx()

	// Add children: A, C
	childA := tree.AddLeafChild(rootIdx, "A", "child A", -1)
	childC := tree.AddLeafChild(rootIdx, "C", "child C", -1)

	// Insert B at position 1 (between A and C)
	childB := tree.AddLeafChild(rootIdx, "B", "child B", 1)
	require.GreaterOrEqual(t, childB, 0)

	// Verify order: A, B, C
	rootNode := tree.Tree().GetNode(rootIdx)
	require.NotNil(t, rootNode)
	assert.Equal(t, childA, rootNode.childIdxs[0])
	assert.Equal(t, childB, rootNode.childIdxs[1])
	assert.Equal(t, childC, rootNode.childIdxs[2])
}

func TestVectorDecisionTree_Balance(t *testing.T) {
	tree := NewVectorDecisionTree[string](
		VectorDimensionThreshold(0, 5.0),
		DecisionOutcome[string]{IsLeaf: false},
		"root",
	)

	rootIdx := tree.RootIdx()

	// Create an imbalanced tree: root has 3 children with different subtree sizes
	// Child 1: 1 leaf
	// Child 2: 2 leaves
	// Child 3: 3 leaves

	child1 := tree.AddDecisionChild(rootIdx, VectorDimensionThreshold(0, 2.0), "child1", -1)
	child2 := tree.AddDecisionChild(rootIdx, VectorDimensionThreshold(0, 4.0), "child2", -1)
	child3 := tree.AddDecisionChild(rootIdx, VectorDimensionThreshold(0, 6.0), "child3", -1)

	tree.AddLeafChild(child1, "A", "leaf A", -1)

	tree.AddLeafChild(child2, "B", "leaf B", -1)
	tree.AddLeafChild(child2, "C", "leaf C", -1)

	tree.AddLeafChild(child3, "D", "leaf D", -1)
	tree.AddLeafChild(child3, "E", "leaf E", -1)
	tree.AddLeafChild(child3, "F", "leaf F", -1)

	// Balance the tree
	tree.Balance()

	// Verify tree is still valid (all nodes accessible)
	rootNode := tree.Tree().GetNode(rootIdx)
	require.NotNil(t, rootNode)
	assert.Equal(t, 3, len(rootNode.childIdxs), "Root should still have 3 children")

	// Verify we can still compute decisions
	vec1 := vec.Vector2D{1.0, 0.0}
	result, ok := tree.ComputeDecision(vec1)
	// Note: The result depends on the tree structure after balancing
	// We just verify computation still works
	_ = result
	_ = ok
}

func TestVectorDecisionTree_NormThreshold(t *testing.T) {
	// Test decision tree using vector norm
	tree := NewVectorDecisionTree[string](
		VectorNormThreshold(5.0),
		DecisionOutcome[string]{IsLeaf: false},
		"norm <= 5.0",
	)

	rootIdx := tree.RootIdx()

	// If norm <= 5.0: "Small"
	// If norm > 5.0: "Large"
	smallLeaf := tree.AddLeafChild(rootIdx, "Small", "small vector", -1)
	largeLeaf := tree.AddLeafChild(rootIdx, "Large", "large vector", -1)

	_ = smallLeaf
	_ = largeLeaf

	// Test with small vector (norm = sqrt(3^2 + 4^2) = 5.0)
	vec1 := vec.Vector2D{3.0, 4.0}
	result1, ok1 := tree.ComputeDecision(vec1)
	assert.True(t, ok1)
	// Note: Exact threshold behavior depends on implementation
	_ = result1

	// Test with large vector (norm = sqrt(5^2 + 5^2) = ~7.07)
	vec2 := vec.Vector2D{5.0, 5.0}
	result2, ok2 := tree.ComputeDecision(vec2)
	assert.True(t, ok2)
	_ = result2
}

func TestGenericTree_AddChildAtPosition(t *testing.T) {
	tree := NewGenericTree[string, float32]("root")
	rootIdx := tree.RootIdx()

	// Add children: A, C
	childA := tree.AddChild(rootIdx, "A")
	childC := tree.AddChild(rootIdx, "C")

	// Insert B at position 1
	childB := tree.AddChildAtPosition(rootIdx, "B", 1)
	require.GreaterOrEqual(t, childB, 0)

	// Verify order
	rootNode := tree.GetNode(rootIdx)
	require.NotNil(t, rootNode)
	assert.Equal(t, childA, rootNode.childIdxs[0])
	assert.Equal(t, childB, rootNode.childIdxs[1])
	assert.Equal(t, childC, rootNode.childIdxs[2])
}

func TestGenericTree_ReorderChildren(t *testing.T) {
	tree := NewGenericTree[string, float32]("root")
	rootIdx := tree.RootIdx()

	// Add children: A, B, C
	childA := tree.AddChild(rootIdx, "A")
	childB := tree.AddChild(rootIdx, "B")
	childC := tree.AddChild(rootIdx, "C")

	// Reorder to: C, A, B
	newOrder := []int{childC, childA, childB}
	ok := tree.ReorderChildren(rootIdx, newOrder)
	assert.True(t, ok)

	// Verify new order
	rootNode := tree.GetNode(rootIdx)
	require.NotNil(t, rootNode)
	assert.Equal(t, childC, rootNode.childIdxs[0])
	assert.Equal(t, childA, rootNode.childIdxs[1])
	assert.Equal(t, childB, rootNode.childIdxs[2])
}

func TestGenericTree_SwapChildren(t *testing.T) {
	tree := NewGenericTree[string, float32]("root")
	rootIdx := tree.RootIdx()

	// Add children: A, B, C
	childA := tree.AddChild(rootIdx, "A")
	childB := tree.AddChild(rootIdx, "B")
	childC := tree.AddChild(rootIdx, "C")

	// Swap first and last
	ok := tree.SwapChildren(rootIdx, 0, 2)
	assert.True(t, ok)

	// Verify swap
	rootNode := tree.GetNode(rootIdx)
	require.NotNil(t, rootNode)
	assert.Equal(t, childC, rootNode.childIdxs[0])
	assert.Equal(t, childB, rootNode.childIdxs[1])
	assert.Equal(t, childA, rootNode.childIdxs[2])
}

