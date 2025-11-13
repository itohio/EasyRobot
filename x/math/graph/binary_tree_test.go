package graph

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestBinaryTree_NewTree(t *testing.T) {
	bt := NewBinaryTree(42)
	require.NotNil(t, bt, "BinaryTree should be created")
	require.NotNil(t, bt.Root, "Root should be created")
	assert.Equal(t, 42, bt.Root.Data, "Root data should be set")
	assert.Nil(t, bt.Root.Left, "Left child should be nil")
	assert.Nil(t, bt.Root.Right, "Right child should be nil")
	assert.Nil(t, bt.Root.Parent, "Root parent should be nil")
}

func TestBinaryTree_SetLeft(t *testing.T) {
	bt := NewBinaryTree(1)
	left := bt.SetLeft(bt.Root, 2)

	require.NotNil(t, left, "Left child should be created")
	assert.Equal(t, 2, left.Data, "Left child data should be set")
	assert.Equal(t, bt.Root, left.Parent, "Left child parent should be root")
	assert.Equal(t, left, bt.Root.Left, "Root should point to left child")
}

func TestBinaryTree_SetRight(t *testing.T) {
	bt := NewBinaryTree(1)
	right := bt.SetRight(bt.Root, 3)

	require.NotNil(t, right, "Right child should be created")
	assert.Equal(t, 3, right.Data, "Right child data should be set")
	assert.Equal(t, bt.Root, right.Parent, "Right child parent should be root")
	assert.Equal(t, right, bt.Root.Right, "Root should point to right child")
}

func TestBinaryTree_GetDepth(t *testing.T) {
	bt := NewBinaryTree(1)
	left := bt.SetLeft(bt.Root, 2)
	leftLeft := bt.SetLeft(left, 4)

	assert.Equal(t, 0, bt.GetDepth(bt.Root), "Root depth should be 0")
	assert.Equal(t, 1, bt.GetDepth(left), "Left child depth should be 1")
	assert.Equal(t, 2, bt.GetDepth(leftLeft), "Left-left child depth should be 2")
}

func TestBinaryTree_GetHeight(t *testing.T) {
	bt := NewBinaryTree(1)
	assert.Equal(t, 0, bt.GetHeight(), "Height of single node tree should be 0")

	bt.SetLeft(bt.Root, 2)
	assert.Equal(t, 1, bt.GetHeight(), "Height should be 1 after adding one level")

	bt.SetRight(bt.Root, 3)
	assert.Equal(t, 1, bt.GetHeight(), "Height should still be 1")

	left := bt.Root.Left
	bt.SetLeft(left, 4)
	assert.Equal(t, 2, bt.GetHeight(), "Height should be 2 after adding another level")
}

func TestBinaryTree_FindNodeByData(t *testing.T) {
	bt := NewBinaryTree(1)
	left := bt.SetLeft(bt.Root, 2)
	_ = bt.SetRight(bt.Root, 3)
	bt.SetLeft(left, 4)

	found := bt.FindNodeByData(4)
	require.NotNil(t, found, "Should find node with data 4")
	assert.Equal(t, 4, found.Data, "Found node should have data 4")

	notFound := bt.FindNodeByData(99)
	assert.Nil(t, notFound, "Should not find node with data 99")
}

func TestBinaryTree_RemoveLeft(t *testing.T) {
	bt := NewBinaryTree(1)
	left := bt.SetLeft(bt.Root, 2)

	result := bt.RemoveLeft(bt.Root)
	assert.True(t, result, "RemoveLeft should succeed")
	assert.Nil(t, bt.Root.Left, "Root left should be nil")
	assert.Nil(t, left.Parent, "Removed node parent should be nil")

	result = bt.RemoveLeft(bt.Root)
	assert.False(t, result, "RemoveLeft should fail if no left child")
}

func TestBinaryTree_RemoveRight(t *testing.T) {
	bt := NewBinaryTree(1)
	right := bt.SetRight(bt.Root, 3)

	result := bt.RemoveRight(bt.Root)
	assert.True(t, result, "RemoveRight should succeed")
	assert.Nil(t, bt.Root.Right, "Root right should be nil")
	assert.Nil(t, right.Parent, "Removed node parent should be nil")

	result = bt.RemoveRight(bt.Root)
	assert.False(t, result, "RemoveRight should fail if no right child")
}

func TestBinaryTree_FindPathToRoot(t *testing.T) {
	bt := NewBinaryTree(1)
	left := bt.SetLeft(bt.Root, 2)
	leftLeft := bt.SetLeft(left, 4)

	path := bt.FindPathToRoot(leftLeft)
	require.NotNil(t, path, "Path should exist")
	assert.Equal(t, 3, len(path), "Path should have 3 nodes")
	assert.Equal(t, leftLeft, path[0], "Path should start with leftLeft")
	assert.Equal(t, left, path[1], "Path should contain left")
	assert.Equal(t, bt.Root, path[2], "Path should end with root")
}

func TestBinaryTree_Traversal(t *testing.T) {
	bt := NewBinaryTree(1)
	left := bt.SetLeft(bt.Root, 2)
	_ = bt.SetRight(bt.Root, 3)
	bt.SetLeft(left, 4)
	bt.SetRight(left, 5)

	var inOrder []int
	bt.InOrderTraversal(func(node *BinaryTreeNode) {
		inOrder = append(inOrder, node.Data.(int))
	})
	assert.Equal(t, []int{4, 2, 5, 1, 3}, inOrder, "In-order traversal")

	var preOrder []int
	bt.PreOrderTraversal(func(node *BinaryTreeNode) {
		preOrder = append(preOrder, node.Data.(int))
	})
	assert.Equal(t, []int{1, 2, 4, 5, 3}, preOrder, "Pre-order traversal")

	var postOrder []int
	bt.PostOrderTraversal(func(node *BinaryTreeNode) {
		postOrder = append(postOrder, node.Data.(int))
	})
	assert.Equal(t, []int{4, 5, 2, 3, 1}, postOrder, "Post-order traversal")
}

func TestBinaryTreeGraph_Neighbors(t *testing.T) {
	bt := NewBinaryTree(1)
	left := bt.SetLeft(bt.Root, 2)
	right := bt.SetRight(bt.Root, 3)

	btg := NewBinaryTreeGraph(bt)
	rootNode := BinaryTreeNodeNode{binaryTreeNode: bt.Root}

	neighbors := btg.Neighbors(rootNode)
	require.NotNil(t, neighbors, "Neighbors should exist")
	assert.Equal(t, 2, len(neighbors), "Should have 2 neighbors")

	leftNode, ok := neighbors[0].(BinaryTreeNodeNode)
	require.True(t, ok, "First neighbor should be BinaryTreeNodeNode")
	assert.Equal(t, left, leftNode.binaryTreeNode, "First neighbor should be left child")

	rightNode, ok := neighbors[1].(BinaryTreeNodeNode)
	require.True(t, ok, "Second neighbor should be BinaryTreeNodeNode")
	assert.Equal(t, right, rightNode.binaryTreeNode, "Second neighbor should be right child")
}

func TestBinaryTreeGraph_Cost(t *testing.T) {
	bt := NewBinaryTree(1)
	left := bt.SetLeft(bt.Root, 2)

	btg := NewBinaryTreeGraph(bt)
	rootNode := BinaryTreeNodeNode{binaryTreeNode: bt.Root}
	leftNode := BinaryTreeNodeNode{binaryTreeNode: left}

	cost := btg.Cost(rootNode, leftNode)
	assert.Equal(t, float32(1.0), cost, "Default cost should be 1.0")

	btg.SetCost(bt.Root, left, 5.0)
	cost = btg.Cost(rootNode, leftNode)
	assert.Equal(t, float32(5.0), cost, "Custom cost should be 5.0")
}

func TestBinaryTreeGraph_ImplementsGraph(t *testing.T) {
	bt := NewBinaryTree(1)
	btg := NewBinaryTreeGraph(bt)

	var graph Graph = btg
	assert.NotNil(t, graph, "BinaryTreeGraph should implement Graph interface")
}
