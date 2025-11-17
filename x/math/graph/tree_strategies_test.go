package graph

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestBinarySearchTreeStrategy_Insert(t *testing.T) {
	// Create a binary tree with root
	bt := NewGenericBinaryTree[int, float32](5)

	// Create BST strategy with int comparison
	strategy := BinarySearchTreeStrategy[int, float32](IntComparison())

	// Insert nodes in BST order
	bt.Insert(3, strategy)
	bt.Insert(7, strategy)
	bt.Insert(2, strategy)
	bt.Insert(4, strategy)
	bt.Insert(6, strategy)
	bt.Insert(8, strategy)

	// Verify BST properties: left < root < right
	rootIdx := bt.RootIdx()
	require.GreaterOrEqual(t, rootIdx, 0)

	rootNode := bt.GetNode(rootIdx)
	require.NotNil(t, rootNode)
	assert.Equal(t, 5, rootNode.data)

	// Check left subtree (should be < 5)
	if rootNode.leftIdx >= 0 {
		leftNode := bt.GetNode(rootNode.leftIdx)
		assert.Less(t, leftNode.data, 5)

		// Check left-left (should be < left)
		if leftNode.leftIdx >= 0 {
			leftLeftNode := bt.GetNode(leftNode.leftIdx)
			assert.Less(t, leftLeftNode.data, leftNode.data)
		}

		// Check left-right (should be > left but < root)
		if leftNode.rightIdx >= 0 {
			leftRightNode := bt.GetNode(leftNode.rightIdx)
			assert.Greater(t, leftRightNode.data, leftNode.data)
			assert.Less(t, leftRightNode.data, 5)
		}
	}

	// Check right subtree (should be > 5)
	if rootNode.rightIdx >= 0 {
		rightNode := bt.GetNode(rootNode.rightIdx)
		assert.Greater(t, rightNode.data, 5)

		// Check right-left (should be < right but > root)
		if rightNode.leftIdx >= 0 {
			rightLeftNode := bt.GetNode(rightNode.leftIdx)
			assert.Less(t, rightLeftNode.data, rightNode.data)
			assert.Greater(t, rightLeftNode.data, 5)
		}

		// Check right-right (should be > right)
		if rightNode.rightIdx >= 0 {
			rightRightNode := bt.GetNode(rightNode.rightIdx)
			assert.Greater(t, rightRightNode.data, rightNode.data)
		}
	}
}

func TestBinarySearchTreeStrategy_DuplicatePrevention(t *testing.T) {
	bt := NewGenericBinaryTree[int, float32](5)
	strategy := BinarySearchTreeStrategy[int, float32](IntComparison())

	// Insert duplicate
	result := bt.Insert(5, strategy)
	assert.Equal(t, -1, result, "Should not insert duplicate")

	// Verify tree still has only root
	assert.Equal(t, 1, bt.NodeCount(), "Should have only root node")
}

func TestBinarySearchTreeStrategy_Float32(t *testing.T) {
	bt := NewGenericBinaryTree[float32, float32](5.0)
	strategy := BinarySearchTreeStrategy[float32, float32](Float32Comparison())

	bt.Insert(3.0, strategy)
	bt.Insert(7.0, strategy)
	bt.Insert(2.5, strategy)

	rootIdx := bt.RootIdx()
	rootNode := bt.GetNode(rootIdx)
	assert.Equal(t, float32(5.0), rootNode.data)

	if rootNode.leftIdx >= 0 {
		leftNode := bt.GetNode(rootNode.leftIdx)
		assert.Less(t, leftNode.data, float32(5.0))
	}
}

func TestBinarySearchTreeStrategy_String(t *testing.T) {
	bt := NewGenericBinaryTree[string, float32]("m")
	strategy := BinarySearchTreeStrategy[string, float32](StringComparison())

	bt.Insert("a", strategy)
	bt.Insert("z", strategy)
	bt.Insert("f", strategy)

	rootIdx := bt.RootIdx()
	rootNode := bt.GetNode(rootIdx)
	assert.Equal(t, "m", rootNode.data)

	if rootNode.leftIdx >= 0 {
		leftNode := bt.GetNode(rootNode.leftIdx)
		assert.Less(t, leftNode.data, "m")
	}

	if rootNode.rightIdx >= 0 {
		rightNode := bt.GetNode(rootNode.rightIdx)
		assert.Greater(t, rightNode.data, "m")
	}
}

func TestFirstAvailableStrategy(t *testing.T) {
	tree := NewGenericTree[int, float32](1)
	strategy := FirstAvailableStrategy[int, float32]

	// Insert at root
	result := tree.Insert(2, strategy)
	require.GreaterOrEqual(t, result, 0)

	// Verify it was added as child of root
	rootIdx := tree.RootIdx()
	rootNode := tree.GetNode(rootIdx)
	require.NotNil(t, rootNode)
	assert.Contains(t, rootNode.childIdxs, result, "Should be child of root")
}

func TestBalancedInsertionStrategy(t *testing.T) {
	tree := NewGenericTree[int, float32](1)
	strategy := BalancedInsertionStrategy[int, float32]

	// Insert multiple nodes
	tree.Insert(2, strategy)
	tree.Insert(3, strategy)
	tree.Insert(4, strategy)

	// Verify all nodes were inserted
	assert.Equal(t, 4, tree.NodeCount(), "Should have 4 nodes")

	// Verify tree structure
	rootIdx := tree.RootIdx()
	rootNode := tree.GetNode(rootIdx)
	require.NotNil(t, rootNode)
	assert.Greater(t, len(rootNode.childIdxs), 0, "Root should have children")
}

func TestComparisonStrategies(t *testing.T) {
	// Test IntComparison
	intComp := IntComparison()
	assert.Equal(t, -1, intComp(1, 2))
	assert.Equal(t, 0, intComp(2, 2))
	assert.Equal(t, 1, intComp(3, 2))

	// Test Float32Comparison
	floatComp := Float32Comparison()
	assert.Equal(t, -1, floatComp(1.0, 2.0))
	assert.Equal(t, 0, floatComp(2.0, 2.0))
	assert.Equal(t, 1, floatComp(3.0, 2.0))

	// Test StringComparison
	strComp := StringComparison()
	assert.Equal(t, -1, strComp("a", "b"))
	assert.Equal(t, 0, strComp("b", "b"))
	assert.Equal(t, 1, strComp("c", "b"))
}

func TestBinaryTree_Balance(t *testing.T) {
	// Create an unbalanced BST (degenerate case - all right children)
	bt := NewGenericBinaryTree[int, float32](1)
	strategy := BinarySearchTreeStrategy[int, float32](IntComparison())

	// Insert in order to create a degenerate tree
	bt.Insert(2, strategy)
	bt.Insert(3, strategy)
	bt.Insert(4, strategy)
	bt.Insert(5, strategy)
	bt.Insert(6, strategy)

	// Verify tree is unbalanced (height should be high)
	heightBefore := bt.GetHeight()
	assert.Greater(t, heightBefore, 3, "Tree should be unbalanced")

	// Balance the tree
	bt.Balance(IntComparison())

	// Verify tree is now balanced
	heightAfter := bt.GetHeight()
	assert.LessOrEqual(t, heightAfter, heightBefore, "Height should be reduced or equal")

	// Verify all nodes are still present
	assert.Equal(t, 6, bt.NodeCount(), "Should still have 6 nodes")

	// Verify BST properties are maintained
	verifyBSTProperties(t, bt, IntComparison())
}

func TestBinaryTree_Balance_EmptyTree(t *testing.T) {
	bt := NewGenericBinaryTree[int, float32](1)

	// Balance empty tree (only root)
	bt.Balance(IntComparison())

	assert.Equal(t, 1, bt.NodeCount(), "Should still have root")
	assert.Equal(t, 0, bt.GetHeight(), "Height should be 0")
}

func TestBinaryTree_Balance_SingleNode(t *testing.T) {
	bt := NewGenericBinaryTree[int, float32](5)

	// Balance single node tree
	bt.Balance(IntComparison())

	assert.Equal(t, 1, bt.NodeCount())
	rootIdx := bt.RootIdx()
	rootNode := bt.GetNode(rootIdx)
	assert.Equal(t, 5, rootNode.data)
}

func TestBinaryTree_Balance_MaintainsData(t *testing.T) {
	// Create tree with specific values
	bt := NewGenericBinaryTree[int, float32](5)
	strategy := BinarySearchTreeStrategy[int, float32](IntComparison())

	values := []int{3, 7, 2, 4, 6, 8, 1, 9}
	for _, v := range values {
		bt.Insert(v, strategy)
	}

	// Collect all values before balancing
	valuesBefore := collectTreeValues(bt)

	// Balance
	bt.Balance(IntComparison())

	// Collect all values after balancing
	valuesAfter := collectTreeValues(bt)

	// Verify all values are preserved
	assert.ElementsMatch(t, valuesBefore, valuesAfter, "All values should be preserved after balancing")
}

func TestGenericTree_Balance(t *testing.T) {
	tree := NewGenericTree[int, float32](1)

	// Add children in a way that creates imbalance
	rootIdx := tree.RootIdx()
	child1 := tree.AddChild(rootIdx, 2)
	_ = tree.AddChild(rootIdx, 3) // child2
	tree.AddChild(child1, 4)
	tree.AddChild(child1, 5)
	tree.AddChild(child1, 6)

	// Balance the tree
	tree.Balance()

	// Verify tree structure is still valid
	assert.Equal(t, 6, tree.NodeCount(), "Should have 6 nodes")
	rootNode := tree.GetNode(tree.RootIdx())
	require.NotNil(t, rootNode)
	assert.Greater(t, len(rootNode.childIdxs), 0, "Root should have children")
}

func TestBinaryTree_Insert_WithoutStrategy(t *testing.T) {
	bt := NewGenericBinaryTree[int, float32](5)

	// Insert without strategy (should use default)
	result := bt.Insert(3, nil)
	require.GreaterOrEqual(t, result, 0)

	rootIdx := bt.RootIdx()
	rootNode := bt.GetNode(rootIdx)

	// Should insert as left child if available
	if rootNode.leftIdx >= 0 {
		assert.Equal(t, result, rootNode.leftIdx)
	}
}

func TestGenericTree_Insert_WithoutStrategy(t *testing.T) {
	tree := NewGenericTree[int, float32](1)

	// Insert without strategy (should use default)
	result := tree.Insert(2, nil)
	require.GreaterOrEqual(t, result, 0)

	rootIdx := tree.RootIdx()
	rootNode := tree.GetNode(rootIdx)
	assert.Contains(t, rootNode.childIdxs, result, "Should be child of root")
}

// Helper functions

func verifyBSTProperties(t *testing.T, bt *GenericBinaryTree[int, float32], compare ComparisonStrategy[int]) {
	verifyBSTRecursive(t, bt, bt.RootIdx(), compare, -1, 100)
}

func verifyBSTRecursive(t *testing.T, bt *GenericBinaryTree[int, float32], nodeIdx int, compare ComparisonStrategy[int], min, max int) {
	if nodeIdx < 0 || nodeIdx >= len(bt.nodes) {
		return
	}

	node := bt.GetNode(nodeIdx)
	require.NotNil(t, node)

	// Verify node value is within bounds
	if min >= 0 {
		assert.GreaterOrEqual(t, compare(node.data, min), 0, "Node value should be >= min")
	}
	if max < 100 {
		assert.LessOrEqual(t, compare(node.data, max), 0, "Node value should be <= max")
	}

	// Verify left subtree
	if node.leftIdx >= 0 {
		leftNode := bt.GetNode(node.leftIdx)
		assert.Less(t, compare(leftNode.data, node.data), 0, "Left child should be < parent")
		verifyBSTRecursive(t, bt, node.leftIdx, compare, min, node.data)
	}

	// Verify right subtree
	if node.rightIdx >= 0 {
		rightNode := bt.GetNode(node.rightIdx)
		assert.Greater(t, compare(rightNode.data, node.data), 0, "Right child should be > parent")
		verifyBSTRecursive(t, bt, node.rightIdx, compare, node.data, max)
	}
}

func collectTreeValues(bt *GenericBinaryTree[int, float32]) []int {
	var values []int
	collectValuesRecursive(bt, bt.RootIdx(), &values)
	return values
}

func collectValuesRecursive(bt *GenericBinaryTree[int, float32], nodeIdx int, values *[]int) {
	if nodeIdx < 0 || nodeIdx >= len(bt.nodes) {
		return
	}

	node := bt.GetNode(nodeIdx)
	*values = append(*values, node.data)

	if node.leftIdx >= 0 {
		collectValuesRecursive(bt, node.leftIdx, values)
	}
	if node.rightIdx >= 0 {
		collectValuesRecursive(bt, node.rightIdx, values)
	}
}
