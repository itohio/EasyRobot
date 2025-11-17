package graph

import (
	"fmt"

	"github.com/itohio/EasyRobot/x/math/graph"
)

var _ graph.ExpressionGraph[any, any, any, any] = (*StoredExpressionGraph)(nil)

// StoredExpressionGraph evaluates expression graphs directly from storage.
type StoredExpressionGraph struct {
	*StoredGraph
	rootID  int64
	nodeOps map[int64]expressionOpInfo
}

func newStoredExpressionGraph(base *StoredGraph, meta *graphMetadata, cfg config) (*StoredExpressionGraph, error) {
	if meta == nil || meta.Expression == nil {
		return nil, fmt.Errorf("expression metadata missing")
	}
	nodeOps := make(map[int64]expressionOpInfo, len(meta.Expression.NodeOps))
	for id, name := range meta.Expression.NodeOps {
		info, ok := cfg.expressionOps[name]
		if !ok {
			return nil, fmt.Errorf("expression op %q not registered", name)
		}
		nodeOps[id] = info
	}
	rootID := meta.Expression.RootID
	if rootID == 0 {
		rootID = meta.RootID
	}
	if rootID == 0 {
		return nil, fmt.Errorf("expression root not specified")
	}
	return &StoredExpressionGraph{
		StoredGraph: base,
		rootID:      rootID,
		nodeOps:     nodeOps,
	}, nil
}

// Compute evaluates expression graph for all inputs (nil start uses default root).
func (g *StoredExpressionGraph) Compute(start graph.Node[any, any], inputs ...any) ([]any, error) {
	if len(inputs) == 0 {
		return nil, fmt.Errorf("no inputs provided")
	}
	if g == nil || g.StoredGraph == nil {
		return nil, fmt.Errorf("expression graph is not initialized")
	}

	startNode, err := g.resolveStart(start)
	if err != nil {
		return nil, err
	}
	order, err := g.buildEvaluationOrder(startNode.ID())
	if err != nil {
		return nil, err
	}

	results := make([]any, 0, len(inputs))
	for _, input := range inputs {
		values := make(map[int64]any, len(order))
		for _, nodeID := range order {
			val, err := g.evaluateNode(nodeID, input, values)
			if err != nil {
				return nil, err
			}
			values[nodeID] = val
		}
		if out, ok := values[startNode.ID()]; ok {
			results = append(results, out)
		} else {
			return nil, fmt.Errorf("start node %d not evaluated", startNode.ID())
		}
	}
	return results, nil
}

func (g *StoredExpressionGraph) resolveStart(start graph.Node[any, any]) (*storedNode, error) {
	if start == nil {
		node, ok := g.nodeByID(g.rootID)
		if !ok {
			return nil, fmt.Errorf("expression root %d not found", g.rootID)
		}
		return node, nil
	}
	node, ok := g.nodeFromGraphNode(start)
	if !ok {
		return nil, fmt.Errorf("node %d not found", start.ID())
	}
	return node, nil
}

func (g *StoredExpressionGraph) evaluateNode(nodeID int64, input any, values map[int64]any) (any, error) {
	info, ok := g.nodeOps[nodeID]
	if !ok {
		return nil, fmt.Errorf("expression op missing for node %d", nodeID)
	}
	childOutputs := make(map[int64]any)
	for _, edgeIdx := range g.neighborsOf(nodeID) {
		edgeRec, ok := g.edgeRecord(edgeIdx)
		if !ok {
			continue
		}
		childID := edgeRec.ToID
		if val, ok := values[childID]; ok {
			childOutputs[childID] = val
		}
	}
	out, okRes, err := callExpressionOp(info, input, childOutputs)
	if err != nil {
		return nil, err
	}
	if !okRes {
		return nil, fmt.Errorf("expression node %d evaluation failed", nodeID)
	}
	return out, nil
}

func (g *StoredExpressionGraph) buildEvaluationOrder(startID int64) ([]int64, error) {
	reachable := make(map[int64]bool)
	g.collectReachable(startID, reachable)
	parents := make(map[int64][]int64)
	outDegree := make(map[int64]int)

	for id := range reachable {
		for _, edgeIdx := range g.edgesByFrom[id] {
			childID := g.edgeRecords[edgeIdx].ToID
			if !reachable[childID] {
				continue
			}
			outDegree[id]++
			parents[childID] = append(parents[childID], id)
		}
	}

	queue := make([]int64, 0)
	for id := range reachable {
		if outDegree[id] == 0 {
			queue = append(queue, id)
		}
	}

	order := make([]int64, 0, len(reachable))
	for len(queue) > 0 {
		nodeID := queue[0]
		queue = queue[1:]
		order = append(order, nodeID)
		for _, parentID := range parents[nodeID] {
			outDegree[parentID]--
			if outDegree[parentID] == 0 {
				queue = append(queue, parentID)
			}
		}
	}

	if len(order) != len(reachable) {
		return nil, fmt.Errorf("cycle detected in expression graph")
	}
	return order, nil
}

func (g *StoredExpressionGraph) collectReachable(startID int64, seen map[int64]bool) {
	if seen[startID] {
		return
	}
	seen[startID] = true
	for _, edgeIdx := range g.neighborsOf(startID) {
		edgeRec, ok := g.edgeRecord(edgeIdx)
		if !ok {
			continue
		}
		g.collectReachable(edgeRec.ToID, seen)
	}
}
