package graph

import (
	vecTypes "github.com/itohio/EasyRobot/x/math/vec/types"
)

// DecisionCriteria defines a function that evaluates whether a decision condition is met.
// Returns true if the condition is satisfied (edge should be taken), false otherwise.
type DecisionCriteria[Input any] func(input Input) bool

// DecisionOutcome represents the result of a decision evaluation.
type DecisionOutcome[Output any] struct {
	Output Output
	IsLeaf bool // True if this is a terminal decision (leaf node)
}

// VectorDecisionNodeData holds decision information for a node in a vector decision tree.
type VectorDecisionNodeData[Output any] struct {
	// Criteria is the decision function (nil for leaf nodes)
	Criteria DecisionCriteria[vecTypes.Vector]
	// Outcome is the result for leaf nodes (nil for decision nodes)
	Outcome DecisionOutcome[Output]
	// Description is a human-readable description
	Description string
}

// VectorDecisionTree is a decision tree implementation that works with vectypes.Vector.
// It builds upon GenericTree and implements decision computation interfaces.
type VectorDecisionTree[Output any] struct {
	tree *GenericTree[VectorDecisionNodeData[Output], float32]
}

// NewVectorDecisionTree creates a new vector decision tree with a root node.
// rootCriteria: Decision criteria for the root node (nil for leaf root)
// rootOutcome: Outcome if root is a leaf (ignored if rootCriteria is not nil)
// description: Description of the root node
func NewVectorDecisionTree[Output any](
	rootCriteria DecisionCriteria[vecTypes.Vector],
	rootOutcome DecisionOutcome[Output],
	description string,
) *VectorDecisionTree[Output] {
	rootData := VectorDecisionNodeData[Output]{
		Criteria:    rootCriteria,
		Outcome:     rootOutcome,
		Description: description,
	}
	tree := NewGenericTree[VectorDecisionNodeData[Output], float32](rootData)
	return &VectorDecisionTree[Output]{tree: tree}
}

// AddDecisionChild adds a decision node as a child.
// parentIdx: Index of parent node
// criteria: Decision criteria function
// description: Description of the decision
// position: Position in child list (-1 to append at end)
// Returns the index of the new node.
func (vdt *VectorDecisionTree[Output]) AddDecisionChild(
	parentIdx int,
	criteria DecisionCriteria[vecTypes.Vector],
	description string,
	position int,
) int {
	data := VectorDecisionNodeData[Output]{
		Criteria:    criteria,
		Outcome:     DecisionOutcome[Output]{IsLeaf: false},
		Description: description,
	}
	return vdt.tree.AddChildAtPosition(parentIdx, data, position)
}

// AddLeafChild adds a leaf node with an outcome.
// parentIdx: Index of parent node
// outcome: The decision outcome
// description: Description of the leaf
// position: Position in child list (-1 to append at end)
// Returns the index of the new node.
func (vdt *VectorDecisionTree[Output]) AddLeafChild(
	parentIdx int,
	outcome Output,
	description string,
	position int,
) int {
	data := VectorDecisionNodeData[Output]{
		Criteria:    nil, // Leaf nodes have no criteria
		Outcome:     DecisionOutcome[Output]{Output: outcome, IsLeaf: true},
		Description: description,
	}
	return vdt.tree.AddChildAtPosition(parentIdx, data, position)
}

// ComputeDecision evaluates a decision for the given input vector.
// Traverses the tree from root, evaluating criteria at each node,
// until a leaf node is reached.
// Returns the outcome and true if computation succeeded, false otherwise.
func (vdt *VectorDecisionTree[Output]) ComputeDecision(input vecTypes.Vector) (Output, bool) {
	if vdt.tree == nil || vdt.tree.RootIdx() < 0 {
		var zero Output
		return zero, false
	}

	return vdt.computeDecisionRecursive(vdt.tree.RootIdx(), input)
}

func (vdt *VectorDecisionTree[Output]) computeDecisionRecursive(
	nodeIdx int,
	input vecTypes.Vector,
) (Output, bool) {
	if nodeIdx < 0 || nodeIdx >= len(vdt.tree.nodes) {
		var zero Output
		return zero, false
	}

	node := vdt.tree.nodes[nodeIdx]
	data := node.data

	// If this is a leaf node, return its outcome
	if data.Criteria == nil || data.Outcome.IsLeaf {
		return data.Outcome.Output, true
	}

	// Evaluate criteria to determine which child to traverse
	if data.Criteria == nil {
		var zero Output
		return zero, false
	}

	criteriaMet := data.Criteria(input)

	// For decision trees, we need to route based on criteria
	// Typically, decision nodes have exactly 2 children: one for true, one for false
	// Or they have multiple children with different criteria

	// If criteria is met, take first child (or child that matches)
	// If criteria is not met, take second child (or skip first)
	if len(node.childIdxs) >= 2 {
		if criteriaMet {
			// Take first child (true branch)
			return vdt.computeDecisionRecursive(node.childIdxs[0], input)
		} else {
			// Take second child (false branch)
			return vdt.computeDecisionRecursive(node.childIdxs[1], input)
		}
	} else if len(node.childIdxs) == 1 {
		// Single child - take it if criteria is met
		if criteriaMet {
			return vdt.computeDecisionRecursive(node.childIdxs[0], input)
		}
	}

	// No valid path found
	var zero Output
	return zero, false
}

// ReorderChildren reorders the children of a node.
func (vdt *VectorDecisionTree[Output]) ReorderChildren(parentIdx int, newOrder []int) bool {
	return vdt.tree.ReorderChildren(parentIdx, newOrder)
}

// SwapChildren swaps two children at given positions.
func (vdt *VectorDecisionTree[Output]) SwapChildren(parentIdx int, pos1, pos2 int) bool {
	return vdt.tree.SwapChildren(parentIdx, pos1, pos2)
}

// Balance rebalances the tree structure.
func (vdt *VectorDecisionTree[Output]) Balance() {
	vdt.tree.Balance()
}

// RootIdx returns the root node index.
func (vdt *VectorDecisionTree[Output]) RootIdx() int {
	return vdt.tree.RootIdx()
}

// Tree returns the underlying GenericTree.
func (vdt *VectorDecisionTree[Output]) Tree() *GenericTree[VectorDecisionNodeData[Output], float32] {
	return vdt.tree
}

// GetNodeData returns the data for a node at the given index.
func (vdt *VectorDecisionTree[Output]) GetNodeData(idx int) (VectorDecisionNodeData[Output], bool) {
	if idx < 0 || idx >= len(vdt.tree.nodes) {
		var zero VectorDecisionNodeData[Output]
		return zero, false
	}
	return vdt.tree.nodes[idx].data, true
}

//
// Helper functions for creating common decision criteria
//

// VectorDimensionThreshold creates a decision criteria that checks if a vector dimension is <= threshold.
func VectorDimensionThreshold(dimension int, threshold float32) DecisionCriteria[vecTypes.Vector] {
	return func(v vecTypes.Vector) bool {
		if dimension < 0 || dimension >= v.Len() {
			return false
		}
		slice := v.Slice(dimension, dimension+1)
		value := slice.Sum()
		return value <= threshold
	}
}

// VectorNormThreshold creates a decision criteria based on vector norm.
func VectorNormThreshold(threshold float32) DecisionCriteria[vecTypes.Vector] {
	return func(v vecTypes.Vector) bool {
		norm := v.Magnitude()
		return norm <= threshold
	}
}

// VectorDotProductThreshold creates a decision criteria based on dot product with a reference vector.
func VectorDotProductThreshold(reference vecTypes.Vector, threshold float32) DecisionCriteria[vecTypes.Vector] {
	return func(v vecTypes.Vector) bool {
		dot := v.Dot(reference)
		return dot <= threshold
	}
}
