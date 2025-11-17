package graph

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestBinaryTree_NewTree(t *testing.T) {
	bt := NewGenericBinaryTree[int, float32](42)
	require.NotNil(t, bt, "BinaryTree should be created")
	require.NotNil(t, bt.Root(), "Root should be created")

	rootData := bt.Root().Data()
	assert.Equal(t, 42, rootData, "Root data should be set")

	// Check that root has no children
	hasNeighbors := false
	for range bt.Root().Neighbors() {
		hasNeighbors = true
		break
	}
	assert.False(t, hasNeighbors, "Root should have no children initially")
}

func TestBinaryTree_SetLeft(t *testing.T) {
	bt := NewGenericBinaryTree[int, float32](1)
	rootIdx := bt.RootIdx()

	leftIdx := bt.SetLeft(rootIdx, 2)
	require.GreaterOrEqual(t, leftIdx, 0, "Left child should be created")

	// Get left child node
	leftNode := BinaryTreeGraphNode[int, float32]{tree: bt, idx: leftIdx}
	assert.Equal(t, 2, leftNode.Data(), "Left child data should be set")

	// Check parent relationship
	rootNode := bt.Root()
	hasLeft := false
	for neighbor := range rootNode.Neighbors() {
		if neighbor.ID() == leftNode.ID() {
			hasLeft = true
			break
		}
	}
	assert.True(t, hasLeft, "Root should have left child as neighbor")
}

func TestBinaryTree_SetRight(t *testing.T) {
	bt := NewGenericBinaryTree[int, float32](1)
	rootIdx := bt.RootIdx()

	rightIdx := bt.SetRight(rootIdx, 3)
	require.GreaterOrEqual(t, rightIdx, 0, "Right child should be created")

	// Get right child node
	rightNode := BinaryTreeGraphNode[int, float32]{tree: bt, idx: rightIdx}
	assert.Equal(t, 3, rightNode.Data(), "Right child data should be set")

	// Check parent relationship
	rootNode := bt.Root()
	hasRight := false
	for neighbor := range rootNode.Neighbors() {
		if neighbor.ID() == rightNode.ID() {
			hasRight = true
			break
		}
	}
	assert.True(t, hasRight, "Root should have right child as neighbor")
}

func TestBinaryTree_GetDepth(t *testing.T) {
	bt := NewGenericBinaryTree[int, float32](1)
	rootIdx := bt.RootIdx()
	leftIdx := bt.SetLeft(rootIdx, 2)
	leftLeftIdx := bt.SetLeft(leftIdx, 4)

	assert.Equal(t, 0, bt.GetDepth(rootIdx), "Root depth should be 0")
	assert.Equal(t, 1, bt.GetDepth(leftIdx), "Left child depth should be 1")
	assert.Equal(t, 2, bt.GetDepth(leftLeftIdx), "Left-left child depth should be 2")
}

func TestBinaryTree_GetHeight(t *testing.T) {
	bt := NewGenericBinaryTree[int, float32](1)
	assert.Equal(t, 0, bt.GetHeight(), "Height of single node tree should be 0")

	rootIdx := bt.RootIdx()
	bt.SetLeft(rootIdx, 2)
	assert.Equal(t, 1, bt.GetHeight(), "Height should be 1 after adding one level")

	bt.SetRight(rootIdx, 3)
	assert.Equal(t, 1, bt.GetHeight(), "Height should still be 1")

	leftIdx := bt.GetNode(rootIdx).leftIdx
	bt.SetLeft(leftIdx, 4)
	assert.Equal(t, 2, bt.GetHeight(), "Height should be 2 after adding another level")
}

func TestBinaryTree_FindNodeByData(t *testing.T) {
	bt := NewGenericBinaryTree[int, float32](1)
	rootIdx := bt.RootIdx()
	leftIdx := bt.SetLeft(rootIdx, 2)
	_ = bt.SetRight(rootIdx, 3)
	bt.SetLeft(leftIdx, 4)

	foundIdx := bt.FindNodeByData(4)
	if foundIdx < 0 {
		t.Fatalf("Should find node with data 4, got index %d", foundIdx)
	}

	foundNode := BinaryTreeGraphNode[int, float32]{tree: bt, idx: foundIdx}
	assert.Equal(t, 4, foundNode.Data(), "Found node should have data 4")

	notFoundIdx := bt.FindNodeByData(99)
	assert.Equal(t, -1, notFoundIdx, "Should not find node with data 99")
}

func TestBinaryTree_RemoveLeft(t *testing.T) {
	bt := NewGenericBinaryTree[int, float32](1)
	rootIdx := bt.RootIdx()
	leftIdx := bt.SetLeft(rootIdx, 2)

	result := bt.RemoveLeft(rootIdx)
	assert.True(t, result, "RemoveLeft should succeed")

	// Check that left child is removed
	node := bt.GetNode(rootIdx)
	assert.Equal(t, -1, node.leftIdx, "Root left should be -1")

	// Check that removed node is marked
	removedNode := bt.GetNode(leftIdx)
	assert.Equal(t, -2, removedNode.parentIdx, "Removed node parent should be -2")

	result = bt.RemoveLeft(rootIdx)
	assert.False(t, result, "RemoveLeft should fail if no left child")
}

func TestBinaryTree_RemoveRight(t *testing.T) {
	bt := NewGenericBinaryTree[int, float32](1)
	rootIdx := bt.RootIdx()
	rightIdx := bt.SetRight(rootIdx, 3)

	result := bt.RemoveRight(rootIdx)
	assert.True(t, result, "RemoveRight should succeed")

	// Check that right child is removed
	node := bt.GetNode(rootIdx)
	assert.Equal(t, -1, node.rightIdx, "Root right should be -1")

	// Check that removed node is marked
	removedNode := bt.GetNode(rightIdx)
	assert.Equal(t, -2, removedNode.parentIdx, "Removed node parent should be -2")

	result = bt.RemoveRight(rootIdx)
	assert.False(t, result, "RemoveRight should fail if no right child")
}

func TestBinaryTree_FindPathToRoot(t *testing.T) {
	bt := NewGenericBinaryTree[int, float32](1)
	rootIdx := bt.RootIdx()
	leftIdx := bt.SetLeft(rootIdx, 2)
	leftLeftIdx := bt.SetLeft(leftIdx, 4)

	path := bt.FindPathToRoot(leftLeftIdx)
	require.NotNil(t, path, "Path should exist")
	assert.Equal(t, 3, len(path), "Path should have 3 indices")
	assert.Equal(t, leftLeftIdx, path[0], "Path should start with leftLeftIdx")
	assert.Equal(t, leftIdx, path[1], "Path should contain leftIdx")
	assert.Equal(t, rootIdx, path[2], "Path should end with rootIdx")
}

func TestBinaryTree_GraphInterface(t *testing.T) {
	bt := NewGenericBinaryTree[int, float32](1)
	rootIdx := bt.RootIdx()
	bt.SetLeft(rootIdx, 2)
	bt.SetRight(rootIdx, 3)

	// Test that it implements Graph interface
	var g Graph[int, float32] = bt
	assert.NotNil(t, g, "GenericBinaryTree should implement Graph interface")

	// Test Nodes iterator
	nodeCount := 0
	for range g.Nodes() {
		nodeCount++
	}
	assert.Equal(t, 3, nodeCount, "Should have 3 nodes")

	// Test Root method (Tree interface)
	var tree Tree[int, float32] = bt
	root := tree.Root()
	require.NotNil(t, root, "Root should exist")
	assert.Equal(t, 1, root.Data(), "Root data should be 1")
}

func TestBinaryTree_Neighbors(t *testing.T) {
	bt := NewGenericBinaryTree[int, float32](1)
	rootIdx := bt.RootIdx()
	leftIdx := bt.SetLeft(rootIdx, 2)
	rightIdx := bt.SetRight(rootIdx, 3)

	rootNode := bt.Root()

	// Collect neighbors
	var neighbors []Node[int, float32]
	for neighbor := range rootNode.Neighbors() {
		neighbors = append(neighbors, neighbor)
	}

	require.Equal(t, 2, len(neighbors), "Should have 2 neighbors")

	// Check that neighbors are left and right children
	neighborIds := make(map[int64]bool)
	for _, n := range neighbors {
		neighborIds[n.ID()] = true
	}

	leftNode := BinaryTreeGraphNode[int, float32]{tree: bt, idx: leftIdx}
	rightNode := BinaryTreeGraphNode[int, float32]{tree: bt, idx: rightIdx}

	assert.True(t, neighborIds[leftNode.ID()], "Should have left child as neighbor")
	assert.True(t, neighborIds[rightNode.ID()], "Should have right child as neighbor")
}

func TestBinaryTree_Cost(t *testing.T) {
	bt := NewGenericBinaryTree[int, float32](1)
	rootIdx := bt.RootIdx()
	leftIdx := bt.SetLeft(rootIdx, 2)

	rootNode := bt.Root()
	leftNode := BinaryTreeGraphNode[int, float32]{tree: bt, idx: leftIdx}

	cost := rootNode.Cost(leftNode)
	assert.Equal(t, float32(1.0), cost, "Default cost should be 1.0")

	bt.SetCost(rootIdx, leftIdx, 5.0)
	cost = rootNode.Cost(leftNode)
	assert.Equal(t, float32(5.0), cost, "Custom cost should be 5.0")
}

func TestBinaryTree_ImplementsTree(t *testing.T) {
	bt := NewGenericBinaryTree[int, float32](1)

	var tree Tree[int, float32] = bt
	assert.NotNil(t, tree, "GenericBinaryTree should implement Tree interface")

	root := tree.Root()
	require.NotNil(t, root, "Root should exist")

	height := tree.GetHeight()
	assert.GreaterOrEqual(t, height, 0, "Height should be non-negative")

	count := tree.NodeCount()
	assert.Equal(t, 1, count, "Should have 1 node")
}
