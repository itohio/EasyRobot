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

// ExpressionGraph manages expression evaluation on a graph.
// Expressions are evaluated bottom-up: leaf nodes compute first, then their parents,
// continuing until the root is reached.
//
// This is useful for:
// - Neural network computation graphs
// - Mathematical expression trees
// - Data flow graphs
// - Pipeline processing
type ExpressionGraph[N any, E any, Input any, Output any] struct {
	graph Graph[N, E]
	// Cache for computed node outputs (node ID -> output value)
	computed map[int64]Output
	// Track evaluation order (for debugging)
	evaluationOrder []Node[N, E]
}

// NewExpressionGraph creates a new expression graph evaluator.
func NewExpressionGraph[N any, E any, Input any, Output any](
	g Graph[N, E],
) *ExpressionGraph[N, E, Input, Output] {
	return &ExpressionGraph[N, E, Input, Output]{
		graph:           g,
		computed:        make(map[int64]Output),
		evaluationOrder: make([]Node[N, E], 0),
	}
}

// Compute evaluates the expression graph bottom-up from leaves to root.
// The algorithm:
// 1. Identify leaf nodes (nodes with no outgoing edges or marked as leaves)
// 2. Evaluate leaf nodes first (they have no dependencies)
// 3. Evaluate parent nodes after all their children are evaluated
// 4. Continue until all nodes are evaluated
// 5. Return the output from the root node (or specified start node)
//
// Returns the computed output and true if evaluation succeeded, false otherwise.
func (eg *ExpressionGraph[N, E, Input, Output]) Compute(
	input Input,
	start Node[N, E],
) (Output, bool) {
	if eg.graph == nil {
		var zero Output
		return zero, false
	}

	// Reset state
	eg.computed = make(map[int64]Output)
	eg.evaluationOrder = make([]Node[N, E], 0)

	// Build dependency graph and identify evaluation order
	evalOrder, err := eg.buildEvaluationOrder(start)
	if err != nil {
		var zero Output
		return zero, false
	}

	// Evaluate nodes in order (bottom-up)
	for _, node := range evalOrder {
		output, ok := eg.evaluateNode(node, input)
		if !ok {
			var zero Output
			return zero, false
		}
		eg.computed[node.ID()] = output
		eg.evaluationOrder = append(eg.evaluationOrder, node)
	}

	// Return output from start node
	if output, ok := eg.computed[start.ID()]; ok {
		return output, true
	}

	var zero Output
	return zero, false
}

// buildEvaluationOrder determines the order in which nodes should be evaluated.
// Uses topological sort to ensure children are evaluated before parents.
// In a tree/graph, edges go from parent to child, so we need to:
// 1. Find nodes with no outgoing edges (leaves) - these are evaluated first
// 2. Then evaluate their parents, and so on
func (eg *ExpressionGraph[N, E, Input, Output]) buildEvaluationOrder(
	start Node[N, E],
) ([]Node[N, E], error) {
	// Collect all nodes reachable from start
	nodeSet := make(map[int64]Node[N, E])
	eg.collectReachableNodes(start, nodeSet)

	// Build reverse dependency map: for each node, track its parents
	// In a tree, edges go from parent to child, so we need to find which nodes are parents
	parents := make(map[int64][]Node[N, E])

	// Count outgoing edges for each node (how many children it has)
	// Nodes with 0 outgoing edges are leaves and should be evaluated first
	outDegree := make(map[int64]int)
	for _, node := range nodeSet {
		count := 0
		for edge := range node.Edges() {
			to := edge.To()
			if to != nil {
				count++
				toID := to.ID()
				// Build reverse map: to is a child of node, so node is a parent of to
				parents[toID] = append(parents[toID], node)
			}
		}
		outDegree[node.ID()] = count
	}

	// Topological sort: start with nodes that have no outgoing edges (leaves)
	queue := make([]Node[N, E], 0)
	for _, node := range nodeSet {
		if outDegree[node.ID()] == 0 {
			queue = append(queue, node)
		}
	}

	result := make([]Node[N, E], 0)
	for len(queue) > 0 {
		node := queue[0]
		queue = queue[1:]
		result = append(result, node)

		// Find all parents of this node and decrease their out-degree
		nodeID := node.ID()
		for _, parent := range parents[nodeID] {
			parentID := parent.ID()
			outDegree[parentID]--
			if outDegree[parentID] == 0 {
				// All children of parent have been evaluated
				queue = append(queue, parent)
			}
		}
	}

	return result, nil
}

// collectReachableNodes collects all nodes reachable from the start node.
func (eg *ExpressionGraph[N, E, Input, Output]) collectReachableNodes(
	start Node[N, E],
	nodeSet map[int64]Node[N, E],
) {
	if start == nil {
		return
	}

	startID := start.ID()
	if _, visited := nodeSet[startID]; visited {
		return
	}

	nodeSet[startID] = start

	for edge := range start.Edges() {
		to := edge.To()
		if to != nil {
			eg.collectReachableNodes(to, nodeSet)
		}
	}
}

// evaluateNode evaluates a single node in the expression graph.
func (eg *ExpressionGraph[N, E, Input, Output]) evaluateNode(
	node Node[N, E],
	input Input,
) (Output, bool) {
	// Check if node implements ExpressionNode
	if en, ok := node.(ExpressionNode[N, E, Input, Output]); ok {
		// If it's a leaf, it should compute its own value
		if en.IsLeaf() {
			// For leaf nodes, childOutputs will be empty
			childOutputs := make(map[int64]Output)
			return en.EvaluateExpression(input, childOutputs)
		}

		// Collect outputs from child nodes
		childOutputs := make(map[int64]Output)
		for edge := range node.Edges() {
			to := edge.To()
			if to != nil {
				if output, computed := eg.computed[to.ID()]; computed {
					childOutputs[to.ID()] = output
				}
			}
		}

		// Use expression function if available
		if fn := en.ExpressionFunction(); fn != nil {
			return fn(input, childOutputs)
		}

		// Otherwise use EvaluateExpression
		return en.EvaluateExpression(input, childOutputs)
	}

	// No expression capability
	var zero Output
	return zero, false
}

// GetComputedOutput returns the computed output for a node, if available.
func (eg *ExpressionGraph[N, E, Input, Output]) GetComputedOutput(
	nodeID int64,
) (Output, bool) {
	output, ok := eg.computed[nodeID]
	return output, ok
}

// GetEvaluationOrder returns the order in which nodes were evaluated.
func (eg *ExpressionGraph[N, E, Input, Output]) GetEvaluationOrder() []Node[N, E] {
	return eg.evaluationOrder
}

// ClearCache clears the computed outputs cache.
func (eg *ExpressionGraph[N, E, Input, Output]) ClearCache() {
	eg.computed = make(map[int64]Output)
	eg.evaluationOrder = make([]Node[N, E], 0)
}
