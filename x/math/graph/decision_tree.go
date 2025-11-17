package graph

import (
	"fmt"
	"iter"
)

// DecisionOp represents the behaviour of a decision node.
type DecisionOp[Input any, Output any] func(Input) (Output, bool)

// DecisionEdgeOp represents the criteria for traversing an outgoing edge.
type DecisionEdgeOp[Input any] func(Input) bool

type decisionEdgeKey struct {
	parentID int64
	childID  int64
}

// GenericDecisionTree is a DecisionTree built on top of GenericTree.
type GenericDecisionTree[N any, E any, Input any, Output any] struct {
	tree    *GenericTree[N, E]
	nodeOps map[int64]DecisionOp[Input, Output]
	edgeOps map[decisionEdgeKey]DecisionEdgeOp[Input]
}

// NewGenericDecisionTree creates a new decision tree with the provided root data.
func NewGenericDecisionTree[N any, E any, Input any, Output any](rootData N) *GenericDecisionTree[N, E, Input, Output] {
	return &GenericDecisionTree[N, E, Input, Output]{
		tree:    NewGenericTree[N, E](rootData),
		nodeOps: make(map[int64]DecisionOp[Input, Output]),
		edgeOps: make(map[decisionEdgeKey]DecisionEdgeOp[Input]),
	}
}

// BaseTree exposes the underlying GenericTree.
func (dt *GenericDecisionTree[N, E, Input, Output]) BaseTree() *GenericTree[N, E] {
	return dt.tree
}

// RootIdx proxies to the underlying tree.
func (dt *GenericDecisionTree[N, E, Input, Output]) RootIdx() int {
	return dt.tree.RootIdx()
}

// Root returns the wrapped root node.
func (dt *GenericDecisionTree[N, E, Input, Output]) Root() Node[N, E] {
	root := dt.tree.Root()
	if root == nil {
		return nil
	}
	return dt.wrapNode(root)
}

// GetHeight proxies to the underlying tree.
func (dt *GenericDecisionTree[N, E, Input, Output]) GetHeight() int {
	return dt.tree.GetHeight()
}

// NodeCount proxies to the underlying tree.
func (dt *GenericDecisionTree[N, E, Input, Output]) NodeCount() int {
	return dt.tree.NodeCount()
}

// NumNodes implements Graph.
func (dt *GenericDecisionTree[N, E, Input, Output]) NumNodes() int {
	return dt.tree.NumNodes()
}

// NumEdges implements Graph.
func (dt *GenericDecisionTree[N, E, Input, Output]) NumEdges() int {
	return dt.tree.NumEdges()
}

// Nodes returns decision-aware nodes.
func (dt *GenericDecisionTree[N, E, Input, Output]) Nodes() iter.Seq[Node[N, E]] {
	return func(yield func(Node[N, E]) bool) {
		for node := range dt.tree.Nodes() {
			if !yield(dt.wrapNode(node)) {
				return
			}
		}
	}
}

// Edges returns decision-aware edges.
func (dt *GenericDecisionTree[N, E, Input, Output]) Edges() iter.Seq[Edge[N, E]] {
	return func(yield func(Edge[N, E]) bool) {
		for edge := range dt.tree.Edges() {
			if !yield(dt.wrapEdge(edge)) {
				return
			}
		}
	}
}

// AddChild proxies to the underlying tree.
func (dt *GenericDecisionTree[N, E, Input, Output]) AddChild(parentIdx int, childData N) int {
	return dt.tree.AddChild(parentIdx, childData)
}

// AddChildAtPosition proxies to the underlying tree.
func (dt *GenericDecisionTree[N, E, Input, Output]) AddChildAtPosition(parentIdx int, childData N, position int) int {
	return dt.tree.AddChildAtPosition(parentIdx, childData, position)
}

// SetCost proxies to the underlying tree.
func (dt *GenericDecisionTree[N, E, Input, Output]) SetCost(parentIdx, childIdx int, cost E) {
	dt.tree.SetCost(parentIdx, childIdx, cost)
}

// Decide evaluates the decision starting from the provided node (nil uses the root) for each input.
func (dt *GenericDecisionTree[N, E, Input, Output]) Decide(start Node[N, E], inputs ...Input) ([]Output, error) {
	if len(inputs) == 0 {
		return nil, fmt.Errorf("no inputs provided")
	}
	if start == nil {
		start = dt.Root()
	}
	if start == nil {
		return nil, fmt.Errorf("no start node available")
	}

	results := make([]Output, 0, len(inputs))
	for _, input := range inputs {
		output, err := dt.decideFrom(start, input)
		if err != nil {
			return nil, err
		}
		results = append(results, output)
	}
	return results, nil
}

// SetNodeOpByIndex assigns a decision operation to the node at the provided index.
func (dt *GenericDecisionTree[N, E, Input, Output]) SetNodeOpByIndex(nodeIdx int, op DecisionOp[Input, Output]) bool {
	if nodeID, ok := dt.nodeIDByIndex(nodeIdx); ok {
		dt.setNodeOp(nodeID, op)
		return true
	}
	return false
}

// SetNodeOpByID assigns a decision operation using a node ID.
func (dt *GenericDecisionTree[N, E, Input, Output]) SetNodeOpByID(nodeID int64, op DecisionOp[Input, Output]) bool {
	if nodeID == 0 {
		return false
	}
	dt.setNodeOp(nodeID, op)
	return true
}

// NodeOps returns a copy of the registered node operations.
func (dt *GenericDecisionTree[N, E, Input, Output]) NodeOps() map[int64]DecisionOp[Input, Output] {
	copyMap := make(map[int64]DecisionOp[Input, Output], len(dt.nodeOps))
	for id, op := range dt.nodeOps {
		copyMap[id] = op
	}
	return copyMap
}

// SetEdgeOpByIndex assigns criteria for the edge parentIdx->childIdx.
func (dt *GenericDecisionTree[N, E, Input, Output]) SetEdgeOpByIndex(parentIdx, childIdx int, op DecisionEdgeOp[Input]) bool {
	parentID, okParent := dt.nodeIDByIndex(parentIdx)
	childID, okChild := dt.nodeIDByIndex(childIdx)
	if !okParent || !okChild {
		return false
	}
	dt.setEdgeOp(parentID, childID, op)
	return true
}

// SetEdgeOpByID assigns criteria using node IDs.
func (dt *GenericDecisionTree[N, E, Input, Output]) SetEdgeOpByID(parentID, childID int64, op DecisionEdgeOp[Input]) bool {
	if parentID == 0 || childID == 0 {
		return false
	}
	dt.setEdgeOp(parentID, childID, op)
	return true
}

// EdgeOps returns a copy of the registered edge operations.
func (dt *GenericDecisionTree[N, E, Input, Output]) EdgeOps() map[[2]int64]DecisionEdgeOp[Input] {
	copyMap := make(map[[2]int64]DecisionEdgeOp[Input], len(dt.edgeOps))
	for key, op := range dt.edgeOps {
		copyMap[[2]int64{key.parentID, key.childID}] = op
	}
	return copyMap
}

// Internal helpers ----------------------------------------------------------

func (dt *GenericDecisionTree[N, E, Input, Output]) setNodeOp(nodeID int64, op DecisionOp[Input, Output]) {
	if op == nil {
		delete(dt.nodeOps, nodeID)
		return
	}
	if dt.nodeOps == nil {
		dt.nodeOps = make(map[int64]DecisionOp[Input, Output])
	}
	dt.nodeOps[nodeID] = op
}

func (dt *GenericDecisionTree[N, E, Input, Output]) setEdgeOp(parentID, childID int64, op DecisionEdgeOp[Input]) {
	key := decisionEdgeKey{parentID: parentID, childID: childID}
	if op == nil {
		delete(dt.edgeOps, key)
		return
	}
	if dt.edgeOps == nil {
		dt.edgeOps = make(map[decisionEdgeKey]DecisionEdgeOp[Input])
	}
	dt.edgeOps[key] = op
}

func (dt *GenericDecisionTree[N, E, Input, Output]) nodeIDByIndex(idx int) (int64, bool) {
	node := dt.tree.GetNode(idx)
	if node == nil || node.parentIdx == -2 {
		return 0, false
	}
	return node.id, true
}

func (dt *GenericDecisionTree[N, E, Input, Output]) nodeOp(nodeID int64) (DecisionOp[Input, Output], bool) {
	op, ok := dt.nodeOps[nodeID]
	return op, ok
}

func (dt *GenericDecisionTree[N, E, Input, Output]) edgeOp(parentID, childID int64) (DecisionEdgeOp[Input], bool) {
	op, ok := dt.edgeOps[decisionEdgeKey{parentID: parentID, childID: childID}]
	return op, ok
}

func (dt *GenericDecisionTree[N, E, Input, Output]) decideFrom(start Node[N, E], input Input) (Output, error) {
	var zero Output
	if start == nil {
		return zero, fmt.Errorf("nil start node")
	}

	if dn, ok := start.(DecisionNode[N, E, Input, Output]); ok {
		if dn.IsLeaf() {
			if out, ok := dn.EvaluateDecision(input); ok {
				return out, nil
			}
			return zero, fmt.Errorf("failed to evaluate leaf node")
		}
		if fn := dn.DecisionFunction(); fn != nil {
			if out, ok := fn(input); ok {
				return out, nil
			}
		}
		if out, ok := dn.EvaluateDecision(input); ok {
			return out, nil
		}
	}

	for edge := range start.Edges() {
		if de, ok := edge.(DecisionEdge[N, E, Input]); ok && !de.EvaluateCriteria(input) {
			continue
		}
		if to := edge.To(); to != nil {
			if out, err := dt.decideFrom(to, input); err == nil {
				return out, nil
			}
		}
	}

	return zero, fmt.Errorf("no decision path found")
}

func (dt *GenericDecisionTree[N, E, Input, Output]) wrapNode(node Node[N, E]) Node[N, E] {
	if node == nil {
		return nil
	}
	if dn, ok := node.(*decisionTreeNode[N, E, Input, Output]); ok && dn.tree == dt {
		return dn
	}
	if tn, ok := node.(TreeGraphNode[N, E]); ok {
		return &decisionTreeNode[N, E, Input, Output]{TreeGraphNode: tn, tree: dt}
	}
	return node
}

func (dt *GenericDecisionTree[N, E, Input, Output]) wrapEdge(edge Edge[N, E]) Edge[N, E] {
	if edge == nil {
		return nil
	}
	if de, ok := edge.(*decisionTreeEdge[N, E, Input, Output]); ok && de.tree == dt {
		return de
	}
	if te, ok := edge.(TreeGraphEdge[N, E]); ok {
		return &decisionTreeEdge[N, E, Input, Output]{TreeGraphEdge: te, tree: dt}
	}
	return edge
}

// decisionTreeNode implements DecisionNode by wrapping TreeGraphNode.
type decisionTreeNode[N any, E any, Input any, Output any] struct {
	TreeGraphNode[N, E]
	tree *GenericDecisionTree[N, E, Input, Output]
}

func (n *decisionTreeNode[N, E, Input, Output]) Neighbors() iter.Seq[Node[N, E]] {
	base := n.TreeGraphNode.Neighbors()
	return func(yield func(Node[N, E]) bool) {
		for neighbor := range base {
			if !yield(n.tree.wrapNode(neighbor)) {
				return
			}
		}
	}
}

func (n *decisionTreeNode[N, E, Input, Output]) Edges() iter.Seq[Edge[N, E]] {
	base := n.TreeGraphNode.Edges()
	return func(yield func(Edge[N, E]) bool) {
		for edge := range base {
			if !yield(n.tree.wrapEdge(edge)) {
				return
			}
		}
	}
}

func (n *decisionTreeNode[N, E, Input, Output]) EvaluateDecision(input Input) (Output, bool) {
	if op, ok := n.tree.nodeOp(n.ID()); ok && op != nil {
		return op(input)
	}
	var zero Output
	return zero, false
}

func (n *decisionTreeNode[N, E, Input, Output]) IsLeaf() bool {
	return n.NumNeighbors() == 0
}

func (n *decisionTreeNode[N, E, Input, Output]) DecisionFunction() func(Input) (Output, bool) {
	if op, ok := n.tree.nodeOp(n.ID()); ok {
		return op
	}
	return nil
}

// decisionTreeEdge implements DecisionEdge by wrapping TreeGraphEdge.
type decisionTreeEdge[N any, E any, Input any, Output any] struct {
	TreeGraphEdge[N, E]
	tree *GenericDecisionTree[N, E, Input, Output]
}

func (e *decisionTreeEdge[N, E, Input, Output]) From() Node[N, E] {
	return e.tree.wrapNode(e.TreeGraphEdge.From())
}

func (e *decisionTreeEdge[N, E, Input, Output]) To() Node[N, E] {
	return e.tree.wrapNode(e.TreeGraphEdge.To())
}

func (e *decisionTreeEdge[N, E, Input, Output]) EvaluateCriteria(input Input) bool {
	from := e.TreeGraphEdge.From()
	to := e.TreeGraphEdge.To()
	if from == nil || to == nil {
		return false
	}
	if op, ok := e.tree.edgeOp(from.ID(), to.ID()); ok && op != nil {
		return op(input)
	}
	return true
}

func (e *decisionTreeEdge[N, E, Input, Output]) CriteriaFunction() func(Input) bool {
	from := e.TreeGraphEdge.From()
	to := e.TreeGraphEdge.To()
	if from == nil || to == nil {
		return nil
	}
	if op, ok := e.tree.edgeOp(from.ID(), to.ID()); ok {
		return op
	}
	return nil
}
