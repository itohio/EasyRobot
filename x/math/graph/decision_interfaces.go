package graph

// DecisionNode extends Node with decision-making capabilities.
// Any node implementing this interface can participate in decision computation.
// Input: Type of input data for decision evaluation
// Output: Type of output/outcome from decision
type DecisionNode[N any, E any, Input any, Output any] interface {
	Node[N, E]
	// EvaluateDecision evaluates the decision at this node given input data.
	// Returns the output/outcome and true if evaluation succeeded, false otherwise.
	// For leaf nodes, this typically returns a fixed outcome.
	// For decision nodes, this may evaluate criteria and delegate to child nodes.
	EvaluateDecision(input Input) (Output, bool)
	// IsLeaf returns true if this is a leaf node (terminal decision).
	// Leaf nodes typically return fixed outcomes without further traversal.
	IsLeaf() bool
	// DecisionFunction returns the decision function for this node, if available.
	// Returns nil if the node doesn't have an explicit decision function.
	// This allows external code to inspect or modify decision logic.
	DecisionFunction() func(Input) (Output, bool)
}

// DecisionEdge extends Edge with decision criteria evaluation.
// Edges can filter which paths are taken during decision computation.
// Input: Type of input data for criteria evaluation
type DecisionEdge[N any, E any, Input any] interface {
	Edge[N, E]
	// EvaluateCriteria evaluates whether this edge should be taken given input data.
	// Returns true if the edge criteria is met (edge should be traversed), false otherwise.
	// This allows conditional traversal based on input data.
	EvaluateCriteria(input Input) bool
	// CriteriaFunction returns the criteria function for this edge, if available.
	// Returns nil if the edge doesn't have explicit criteria.
	CriteriaFunction() func(Input) bool
}

// DecisionGraph extends Graph with decision computation capabilities.
// Any graph implementing this interface can compute decisions.
// Input: Type of input data for decision evaluation
// Output: Type of output/outcome from decision
type DecisionGraph[N any, E any, Input any, Output any] interface {
	Graph[N, E]
	// ComputeDecision evaluates a decision starting from the given node.
	// Traverses the graph based on decision criteria and returns the outcome.
	// Returns the output and true if computation succeeded, false otherwise.
	ComputeDecision(start Node[N, E], input Input) (Output, bool)
}

// DecisionTree extends Tree with decision computation capabilities.
// Trees are ideal for decision computation as they have a clear root and hierarchical structure.
// Input: Type of input data for decision evaluation
// Output: Type of output/outcome from decision
type DecisionTree[N any, E any, Input any, Output any] interface {
	Tree[N, E]
	// ComputeDecision evaluates a decision starting from the root.
	// Traverses the tree based on decision criteria and returns the outcome.
	// Returns the output and true if computation succeeded, false otherwise.
	ComputeDecision(input Input) (Output, bool)
}

