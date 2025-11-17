package graph

// ComputeDecision evaluates a decision on any graph starting from a given node.
// This function works with any graph structure by traversing nodes and edges
// based on their decision criteria.
//
// The algorithm:
// 1. If the start node is a DecisionNode and is a leaf, return its outcome
// 2. If the start node is a DecisionNode, evaluate its decision function
// 3. Otherwise, traverse edges that pass their criteria evaluation
// 4. Recursively compute decisions on child nodes
//
// Returns the output and true if computation succeeded, false otherwise.
func ComputeDecision[N any, E any, Input any, Output any](
	g Graph[N, E],
	start Node[N, E],
	input Input,
) (Output, bool) {
	if start == nil {
		var zero Output
		return zero, false
	}

	// Check if node implements DecisionNode interface
	if dn, ok := start.(DecisionNode[N, E, Input, Output]); ok {
		// If it's a leaf, return its outcome directly
		if dn.IsLeaf() {
			return dn.EvaluateDecision(input)
		}

		// If it has a decision function, use it
		if fn := dn.DecisionFunction(); fn != nil {
			return fn(input)
		}

		// Otherwise, evaluate decision and traverse
		output, ok := dn.EvaluateDecision(input)
		if ok {
			return output, true
		}
	}

	// If not a DecisionNode or evaluation didn't return a result,
	// traverse edges based on their criteria
	for edge := range start.Edges() {
		// Check if edge implements DecisionEdge interface
		if de, ok := edge.(DecisionEdge[N, E, Input]); ok {
			// Evaluate criteria
			if !de.EvaluateCriteria(input) {
				continue // Skip this edge if criteria not met
			}
		}

		// Traverse to destination node
		to := edge.To()
		if to == nil {
			continue
		}

		// Recursively compute decision on child node
		// Note: We need to use type assertion to help with type inference
		var output Output
		var ok bool
		if dn, isDecisionNode := to.(DecisionNode[N, E, Input, Output]); isDecisionNode {
			output, ok = dn.EvaluateDecision(input)
		} else {
			output, ok = ComputeDecision[N, E, Input, Output](g, to, input)
		}
		if ok {
			return output, true
		}
	}

	// No valid path found
	var zero Output
	return zero, false
}

// ComputeDecisionTree evaluates a decision on a tree starting from the root.
// This is optimized for tree structures which have a clear root and hierarchical paths.
//
// The algorithm:
// 1. Start at the root node
// 2. If root is a leaf DecisionNode, return its outcome
// 3. Otherwise, evaluate edges based on criteria and traverse to child nodes
// 4. Continue until a leaf node is reached
//
// Returns the output and true if computation succeeded, false otherwise.
func ComputeDecisionTree[N any, E any, Input any, Output any](
	t Tree[N, E],
	input Input,
) (Output, bool) {
	root := t.Root()
	if root == nil {
		var zero Output
		return zero, false
	}

	return ComputeDecision[N, E, Input, Output](t, root, input)
}

// ComputeDecisionForest evaluates decisions across multiple trees in a forest.
// Returns the first successful outcome, or false if no tree produces a result.
//
// This is useful when you have multiple decision trees and want to try them
// in sequence until one succeeds.
func ComputeDecisionForest[N any, E any, Input any, Output any](
	forest Graph[N, E],
	input Input,
) (Output, bool) {
	// Try each node as a potential root of a decision tree
	for node := range forest.Nodes() {
		// Check if this node could be a root (has no incoming edges or is marked as root)
		output, ok := ComputeDecision[N, E, Input, Output](forest, node, input)
		if ok {
			return output, true
		}
	}

	var zero Output
	return zero, false
}

// FindDecisionPath finds the path taken during decision computation.
// Returns the sequence of nodes visited during decision evaluation.
// This is useful for debugging and understanding decision logic.
func FindDecisionPath[N any, E any, Input any, Output any](
	g Graph[N, E],
	start Node[N, E],
	input Input,
) (Path[N, E], Output, bool) {
	var path Path[N, E]
	var visited map[Node[N, E]]bool = make(map[Node[N, E]]bool)

	var computeWithPath func(Node[N, E]) (Output, bool)
	computeWithPath = func(node Node[N, E]) (Output, bool) {
		if node == nil {
			var zero Output
			return zero, false
		}

		// Avoid cycles
		if visited[node] {
			var zero Output
			return zero, false
		}
		visited[node] = true
		path = append(path, node)

		// Check if node implements DecisionNode interface
		if dn, ok := node.(DecisionNode[N, E, Input, Output]); ok {
			if dn.IsLeaf() {
				output, ok := dn.EvaluateDecision(input)
				return output, ok
			}

			if fn := dn.DecisionFunction(); fn != nil {
				return fn(input)
			}

			output, ok := dn.EvaluateDecision(input)
			if ok {
				return output, true
			}
		}

		// Traverse edges
		for edge := range node.Edges() {
			if de, ok := edge.(DecisionEdge[N, E, Input]); ok {
				if !de.EvaluateCriteria(input) {
					continue
				}
			}

			to := edge.To()
			if to == nil {
				continue
			}

			output, ok := computeWithPath(to)
			if ok {
				return output, true
			}
		}

		// Backtrack
		path = path[:len(path)-1]
		visited[node] = false

		var zero Output
		return zero, false
	}

	output, ok := computeWithPath(start)
	return path, output, ok
}
