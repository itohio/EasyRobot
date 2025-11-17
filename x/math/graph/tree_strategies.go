package graph

// InsertionStrategy defines how to insert a new node into a tree
// Returns the index of the parent node where the new node should be inserted
// Returns -1 if insertion should not proceed
type InsertionStrategy[N any, E any] func(tree Tree[N, E], data N, currentNodeIdx int) int

// BinaryInsertionStrategy defines how to insert a new node into a binary tree
// Returns (parentIdx, shouldGoLeft)
// Returns (-1, false) if insertion should not proceed
type BinaryInsertionStrategy[N any, E any] func(tree Tree[N, E], data N, currentNodeIdx int) (parentIdx int, shouldGoLeft bool)

// ComparisonStrategy defines how to compare two node data values
// Returns: -1 if data1 < data2, 0 if data1 == data2, 1 if data1 > data2
type ComparisonStrategy[N any] func(data1, data2 N) int

// BinarySearchTreeStrategy creates a binary search tree insertion strategy
// Uses the provided comparison strategy to determine insertion position
func BinarySearchTreeStrategy[N any, E any](compare ComparisonStrategy[N]) BinaryInsertionStrategy[N, E] {
	return func(tree Tree[N, E], data N, currentNodeIdx int) (parentIdx int, shouldGoLeft bool) {
		if currentNodeIdx < 0 {
			return -1, false
		}

		// Get current node
		bt, ok := tree.(*GenericBinaryTree[N, E])
		if !ok {
			return -1, false
		}

		if currentNodeIdx >= len(bt.nodes) {
			return -1, false
		}

		currentNode := BinaryTreeGraphNode[N, E]{tree: bt, idx: currentNodeIdx}
		currentData := currentNode.Data()

		// Compare new data with current node data
		cmp := compare(data, currentData)

		if cmp < 0 {
			// Go left
			if bt.nodes[currentNodeIdx].leftIdx >= 0 {
				// Recursively search left subtree
				strategy := BinarySearchTreeStrategy[N, E](compare)
				return strategy(tree, data, bt.nodes[currentNodeIdx].leftIdx)
			}
			return currentNodeIdx, true
		} else if cmp > 0 {
			// Go right
			if bt.nodes[currentNodeIdx].rightIdx >= 0 {
				// Recursively search right subtree
				strategy := BinarySearchTreeStrategy[N, E](compare)
				return strategy(tree, data, bt.nodes[currentNodeIdx].rightIdx)
			}
			return currentNodeIdx, false
		} else {
			// Equal - don't insert duplicates (or update existing)
			return -1, false
		}
	}
}

// SimpleComparison creates a simple comparison strategy for basic types
func SimpleComparison[N comparable]() ComparisonStrategy[N] {
	return func(data1, data2 N) int {
		// For comparable types, we can't use < or > directly
		// This is a placeholder - users should provide their own comparison
		if data1 == data2 {
			return 0
		}
		// For non-comparable types, this won't work
		// Users need to provide custom comparison
		return 0
	}
}

// IntComparison creates a comparison strategy for int
func IntComparison() ComparisonStrategy[int] {
	return func(data1, data2 int) int {
		if data1 < data2 {
			return -1
		}
		if data1 > data2 {
			return 1
		}
		return 0
	}
}

// Float32Comparison creates a comparison strategy for float32
func Float32Comparison() ComparisonStrategy[float32] {
	return func(data1, data2 float32) int {
		if data1 < data2 {
			return -1
		}
		if data1 > data2 {
			return 1
		}
		return 0
	}
}

// StringComparison creates a comparison strategy for string
func StringComparison() ComparisonStrategy[string] {
	return func(data1, data2 string) int {
		if data1 < data2 {
			return -1
		}
		if data1 > data2 {
			return 1
		}
		return 0
	}
}

// FirstAvailableStrategy inserts at the first available child position
func FirstAvailableStrategy[N any, E any](tree Tree[N, E], data N, parentIdx int) int {
	gt, ok := tree.(*GenericTree[N, E])
	if !ok {
		return -1
	}

	if parentIdx < 0 || parentIdx >= len(gt.nodes) {
		return -1
	}

	// Return the parent index - insertion will add as first child
	return parentIdx
}

// BalancedInsertionStrategy tries to balance the tree by choosing the subtree with fewer nodes
func BalancedInsertionStrategy[N any, E any](tree Tree[N, E], data N, parentIdx int) int {
	gt, ok := tree.(*GenericTree[N, E])
	if !ok {
		return -1
	}

	if parentIdx < 0 || parentIdx >= len(gt.nodes) {
		return -1
	}

	// Find child with fewest descendants
	parent := &gt.nodes[parentIdx]
	if len(parent.childIdxs) == 0 {
		return parentIdx
	}

	minChildIdx := parent.childIdxs[0]
	minCount := gt.countDescendants(minChildIdx)

	for _, childIdx := range parent.childIdxs[1:] {
		count := gt.countDescendants(childIdx)
		if count < minCount {
			minCount = count
			minChildIdx = childIdx
		}
	}

	// Recursively insert into the subtree with fewest nodes
	return BalancedInsertionStrategy(tree, data, minChildIdx)
}
