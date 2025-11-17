package graph

// ExpressionNode extends Node with expression evaluation capabilities.
// Expression nodes compute values from their children (bottom-up evaluation).
// This is used for computation graphs, neural networks, and data flow graphs.
// Input: Type of input data for expression evaluation
// Output: Type of output from expression computation
type ExpressionNode[N any, E any, Input any, Output any] interface {
	Node[N, E]
	// EvaluateExpression computes the expression value from child node outputs.
	// This is called during bottom-up evaluation after all children have been evaluated.
	// input: Input data for the expression
	// childOutputs: Map of child node ID to their computed output values
	// Returns the computed output and true if evaluation succeeded, false otherwise.
	EvaluateExpression(input Input, childOutputs map[int64]Output) (Output, bool)
	// ExpressionFunction returns the expression function for this node, if available.
	// The function receives input and child outputs, returns computed output.
	// Returns nil if the node doesn't have an explicit expression function.
	ExpressionFunction() func(Input, map[int64]Output) (Output, bool)
	// IsLeaf returns true if this is a leaf node (has no children or is an input node).
	// Leaf nodes typically return their own value without depending on children.
	IsLeaf() bool
}

// ExpressionOp represents the computation attached to an expression node.
type ExpressionOp[Input any, Output any] func(Input, map[int64]Output) (Output, bool)

// ExpressionGraph extends the Graph interface with expression-evaluation behaviour.
type ExpressionGraph[N any, E any, Input any, Output any] interface {
	Graph[N, E]
	// Compute evaluates the graph starting at the given node for all provided inputs.
	Compute(start Node[N, E], inputs ...Input) ([]Output, error)
}
