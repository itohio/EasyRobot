package graph

import (
	"fmt"
	"iter"
)

// GenericExpressionGraph is a concrete implementation of ExpressionGraph.
// It stores nodes and edges using GenericGraph for cache-friendly storage
// and augments each node with an ExpressionOp for evaluation.
type GenericExpressionGraph[N any, E any, Input any, Output any] struct {
	base    *GenericGraph[N, E]
	nodeOps map[int64]ExpressionOp[Input, Output]
	rootID  int64
}

// NewGenericExpressionGraph creates an empty expression graph.
func NewGenericExpressionGraph[N any, E any, Input any, Output any]() *GenericExpressionGraph[N, E, Input, Output] {
	return &GenericExpressionGraph[N, E, Input, Output]{
		base:    NewGenericGraph[N, E](),
		nodeOps: make(map[int64]ExpressionOp[Input, Output]),
	}
}

// AddNode creates a new node with the provided data and optional operation.
func (g *GenericExpressionGraph[N, E, Input, Output]) AddNode(data N, op ExpressionOp[Input, Output]) (Node[N, E], error) {
	node := &GenericNode[N, E]{data: data}
	if err := g.base.AddNode(node); err != nil {
		return nil, err
	}
	if op != nil {
		g.nodeOps[node.ID()] = op
	}
	if g.rootID == 0 {
		g.rootID = node.ID()
	}
	return g.wrapNode(node), nil
}

// SetRoot updates the default root node used when Compute is called with a nil start.
func (g *GenericExpressionGraph[N, E, Input, Output]) SetRoot(node Node[N, E]) bool {
	baseNode := g.baseNode(node)
	if baseNode == nil {
		return false
	}
	g.rootID = baseNode.ID()
	return true
}

// AddEdge creates a directed edge from -> to with the given data.
func (g *GenericExpressionGraph[N, E, Input, Output]) AddEdge(from, to Node[N, E], data E) error {
	baseFrom := g.baseNode(from)
	baseTo := g.baseNode(to)
	if baseFrom == nil || baseTo == nil {
		return nil
	}
	return g.base.AddEdge(&expressionGraphInputEdge[N, E]{from: baseFrom, to: baseTo, data: data})
}

// SetNodeOpByID assigns or removes an ExpressionOp for the provided node ID.
func (g *GenericExpressionGraph[N, E, Input, Output]) SetNodeOpByID(nodeID int64, op ExpressionOp[Input, Output]) {
	if op == nil {
		delete(g.nodeOps, nodeID)
		return
	}
	g.nodeOps[nodeID] = op
}

// Nodes implements Graph.
func (g *GenericExpressionGraph[N, E, Input, Output]) Nodes() iter.Seq[Node[N, E]] {
	return func(yield func(Node[N, E]) bool) {
		for node := range g.base.Nodes() {
			if !yield(g.wrapNode(node)) {
				return
			}
		}
	}
}

// Edges implements Graph.
func (g *GenericExpressionGraph[N, E, Input, Output]) Edges() iter.Seq[Edge[N, E]] {
	return func(yield func(Edge[N, E]) bool) {
		for edge := range g.base.Edges() {
			if !yield(g.wrapEdge(edge)) {
				return
			}
		}
	}
}

// NumNodes implements Graph.
func (g *GenericExpressionGraph[N, E, Input, Output]) NumNodes() int {
	return g.base.NumNodes()
}

// NumEdges implements Graph.
func (g *GenericExpressionGraph[N, E, Input, Output]) NumEdges() int {
	return g.base.NumEdges()
}

// Compute evaluates the graph starting from the provided node for each input.
func (g *GenericExpressionGraph[N, E, Input, Output]) Compute(
	start Node[N, E],
	inputs ...Input,
) ([]Output, error) {
	if len(inputs) == 0 {
		return nil, fmt.Errorf("no inputs provided")
	}
	if start == nil {
		start = g.nodeByID(g.rootID)
	}
	if start == nil {
		return nil, fmt.Errorf("no start node available")
	}

	evalOrder, err := g.buildEvaluationOrder(start)
	if err != nil {
		return nil, err
	}

	startID := start.ID()
	results := make([]Output, 0, len(inputs))
	for _, input := range inputs {
		output, err := g.computeOnce(evalOrder, startID, input)
		if err != nil {
			return nil, err
		}
		results = append(results, output)
	}
	return results, nil
}

// Evaluation helpers ---------------------------------------------------------

func (g *GenericExpressionGraph[N, E, Input, Output]) buildEvaluationOrder(
	start Node[N, E],
) ([]Node[N, E], error) {
	nodeSet := make(map[int64]Node[N, E])
	g.collectReachableNodes(start, nodeSet)

	parents := make(map[int64][]Node[N, E])
	outDegree := make(map[int64]int)
	for _, node := range nodeSet {
		count := 0
		for edge := range node.Edges() {
			to := edge.To()
			if to != nil {
				count++
				parents[to.ID()] = append(parents[to.ID()], node)
			}
		}
		outDegree[node.ID()] = count
	}

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

		for _, parent := range parents[node.ID()] {
			parentID := parent.ID()
			outDegree[parentID]--
			if outDegree[parentID] == 0 {
				queue = append(queue, parent)
			}
		}
	}

	return result, nil
}

func (g *GenericExpressionGraph[N, E, Input, Output]) collectReachableNodes(
	start Node[N, E],
	nodeSet map[int64]Node[N, E],
) {
	if start == nil {
		return
	}
	if _, visited := nodeSet[start.ID()]; visited {
		return
	}

	nodeSet[start.ID()] = start
	for edge := range start.Edges() {
		if to := edge.To(); to != nil {
			g.collectReachableNodes(to, nodeSet)
		}
	}
}

func (g *GenericExpressionGraph[N, E, Input, Output]) evaluateNode(
	node Node[N, E],
	input Input,
	outputs map[int64]Output,
) (Output, error) {
	var zero Output
	if en, ok := node.(ExpressionNode[N, E, Input, Output]); ok {
		childOutputs := make(map[int64]Output)
		for edge := range node.Edges() {
			to := edge.To()
			if to == nil {
				continue
			}
			val, ok := outputs[to.ID()]
			if !ok {
				return zero, fmt.Errorf("missing value for node %d", to.ID())
			}
			childOutputs[to.ID()] = val
		}

		if fn := en.ExpressionFunction(); fn != nil {
			if out, ok := fn(input, childOutputs); ok {
				return out, nil
			}
			return zero, fmt.Errorf("expression function failed")
		}
		if out, ok := en.EvaluateExpression(input, childOutputs); ok {
			return out, nil
		}
		return zero, fmt.Errorf("expression evaluation failed")
	}
	return zero, fmt.Errorf("node does not implement ExpressionNode")
}

func (g *GenericExpressionGraph[N, E, Input, Output]) nodeByID(id int64) Node[N, E] {
	if id == 0 {
		return nil
	}
	if idx, ok := g.base.nodeMap[id]; ok && idx >= 0 && idx < len(g.base.nodes) {
		return g.wrapNode(&g.base.nodes[idx])
	}
	return nil
}

// Wrapping helpers ----------------------------------------------------------

func (g *GenericExpressionGraph[N, E, Input, Output]) computeOnce(
	order []Node[N, E],
	startID int64,
	input Input,
) (Output, error) {
	values := make(map[int64]Output, len(order))
	var zero Output
	for _, node := range order {
		output, err := g.evaluateNode(node, input, values)
		if err != nil {
			return zero, err
		}
		values[node.ID()] = output
	}
	if output, ok := values[startID]; ok {
		return output, nil
	}
	return zero, fmt.Errorf("start node %d not evaluated", startID)
}
func (g *GenericExpressionGraph[N, E, Input, Output]) wrapNode(node Node[N, E]) Node[N, E] {
	if node == nil {
		return nil
	}
	if en, ok := node.(*expressionGraphNode[N, E, Input, Output]); ok && en.graph == g {
		return en
	}
	return &expressionGraphNode[N, E, Input, Output]{
		inner: g.baseNode(node),
		graph: g,
	}
}

func (g *GenericExpressionGraph[N, E, Input, Output]) wrapEdge(edge Edge[N, E]) Edge[N, E] {
	if edge == nil {
		return nil
	}
	if ee, ok := edge.(*expressionGraphEdge[N, E, Input, Output]); ok && ee.graph == g {
		return ee
	}
	return &expressionGraphEdge[N, E, Input, Output]{
		inner: g.baseEdge(edge),
		graph: g,
	}
}

func (g *GenericExpressionGraph[N, E, Input, Output]) baseNode(node Node[N, E]) Node[N, E] {
	switch n := node.(type) {
	case *expressionGraphNode[N, E, Input, Output]:
		return n.inner
	default:
		return node
	}
}

func (g *GenericExpressionGraph[N, E, Input, Output]) baseEdge(edge Edge[N, E]) Edge[N, E] {
	switch e := edge.(type) {
	case *expressionGraphEdge[N, E, Input, Output]:
		return e.inner
	default:
		return edge
	}
}

// expressionGraphNode implements ExpressionNode by delegating to the underlying
// graph node and the operation registered in GenericExpressionGraph.
type expressionGraphNode[N any, E any, Input any, Output any] struct {
	inner Node[N, E]
	graph *GenericExpressionGraph[N, E, Input, Output]
}

func (n *expressionGraphNode[N, E, Input, Output]) ID() int64         { return n.inner.ID() }
func (n *expressionGraphNode[N, E, Input, Output]) Data() N           { return n.inner.Data() }
func (n *expressionGraphNode[N, E, Input, Output]) NumNeighbors() int { return n.inner.NumNeighbors() }
func (n *expressionGraphNode[N, E, Input, Output]) Cost(to Node[N, E]) float32 {
	return n.inner.Cost(to)
}
func (n *expressionGraphNode[N, E, Input, Output]) Equal(o Node[N, E]) bool { return n.inner.Equal(o) }
func (n *expressionGraphNode[N, E, Input, Output]) Compare(o Node[N, E]) int {
	return n.inner.Compare(o)
}

func (n *expressionGraphNode[N, E, Input, Output]) Neighbors() iter.Seq[Node[N, E]] {
	base := n.inner.Neighbors()
	return func(yield func(Node[N, E]) bool) {
		for neighbor := range base {
			if !yield(n.graph.wrapNode(neighbor)) {
				return
			}
		}
	}
}

func (n *expressionGraphNode[N, E, Input, Output]) Edges() iter.Seq[Edge[N, E]] {
	base := n.inner.Edges()
	return func(yield func(Edge[N, E]) bool) {
		for edge := range base {
			if !yield(n.graph.wrapEdge(edge)) {
				return
			}
		}
	}
}

func (n *expressionGraphNode[N, E, Input, Output]) EvaluateExpression(input Input, childOutputs map[int64]Output) (Output, bool) {
	if op, ok := n.graph.nodeOps[n.ID()]; ok && op != nil {
		return op(input, childOutputs)
	}
	var zero Output
	return zero, false
}

func (n *expressionGraphNode[N, E, Input, Output]) ExpressionFunction() func(Input, map[int64]Output) (Output, bool) {
	if op, ok := n.graph.nodeOps[n.ID()]; ok {
		return op
	}
	return nil
}

func (n *expressionGraphNode[N, E, Input, Output]) IsLeaf() bool {
	return n.NumNeighbors() == 0
}

// expressionGraphEdge wraps GenericEdge so From/To return expression-aware nodes.
type expressionGraphEdge[N any, E any, Input any, Output any] struct {
	inner Edge[N, E]
	graph *GenericExpressionGraph[N, E, Input, Output]
}

func (e *expressionGraphEdge[N, E, Input, Output]) ID() int64     { return e.inner.ID() }
func (e *expressionGraphEdge[N, E, Input, Output]) Data() E       { return e.inner.Data() }
func (e *expressionGraphEdge[N, E, Input, Output]) Cost() float32 { return e.inner.Cost() }

func (e *expressionGraphEdge[N, E, Input, Output]) From() Node[N, E] {
	return e.graph.wrapNode(e.inner.From())
}

func (e *expressionGraphEdge[N, E, Input, Output]) To() Node[N, E] {
	return e.graph.wrapNode(e.inner.To())
}

// expressionGraphInputEdge is a lightweight Edge used when inserting edges.
type expressionGraphInputEdge[N any, E any] struct {
	from, to Node[N, E]
	data     E
	id       int64
}

func (e *expressionGraphInputEdge[N, E]) ID() int64        { return e.id }
func (e *expressionGraphInputEdge[N, E]) Data() E          { return e.data }
func (e *expressionGraphInputEdge[N, E]) Cost() float32    { return 0 }
func (e *expressionGraphInputEdge[N, E]) From() Node[N, E] { return e.from }
func (e *expressionGraphInputEdge[N, E]) To() Node[N, E]   { return e.to }
