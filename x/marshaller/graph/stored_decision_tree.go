package graph

import (
	"fmt"

	"github.com/itohio/EasyRobot/x/math/graph"
)

var _ graph.DecisionTree[any, any, any, any] = (*StoredDecisionTree)(nil)

// StoredDecisionTree executes decision operations directly from storage.
type StoredDecisionTree struct {
	*StoredTree
	nodeOps map[int64]decisionNodeOpInfo
	edgeOps map[string]decisionEdgeOpInfo
}

func newStoredDecisionTree(base *StoredGraph, meta *graphMetadata, cfg config) (*StoredDecisionTree, error) {
	if meta == nil || meta.Decision == nil {
		return nil, fmt.Errorf("decision tree metadata missing")
	}
	tree, err := newStoredTree(base, meta)
	if err != nil {
		return nil, err
	}

	nodeOps := make(map[int64]decisionNodeOpInfo, len(meta.Decision.NodeOps))
	for id, name := range meta.Decision.NodeOps {
		info, ok := cfg.decisionNodeOps[name]
		if !ok {
			return nil, fmt.Errorf("decision op %q not registered", name)
		}
		nodeOps[id] = info
	}

	edgeOps := make(map[string]decisionEdgeOpInfo, len(meta.Decision.EdgeOps))
	for key, name := range meta.Decision.EdgeOps {
		info, ok := cfg.decisionEdgeOps[name]
		if !ok {
			return nil, fmt.Errorf("decision edge op %q not registered", name)
		}
		edgeOps[key] = info
	}

	return &StoredDecisionTree{
		StoredTree: tree,
		nodeOps:    nodeOps,
		edgeOps:    edgeOps,
	}, nil
}

// Decide evaluates decision tree starting at provided node (nil uses root).
func (t *StoredDecisionTree) Decide(start graph.Node[any, any], inputs ...any) ([]any, error) {
	if len(inputs) == 0 {
		return nil, fmt.Errorf("no inputs provided")
	}
	if t == nil || t.StoredGraph == nil {
		return nil, fmt.Errorf("decision tree is not initialized")
	}

	startNode, err := t.resolveStartNode(start)
	if err != nil {
		return nil, err
	}

	results := make([]any, 0, len(inputs))
	for _, input := range inputs {
		out, err := t.decideFrom(startNode, input)
		if err != nil {
			return nil, err
		}
		results = append(results, out)
	}
	return results, nil
}

func (t *StoredDecisionTree) decideFrom(node *storedNode, input any) (any, error) {
	if node == nil {
		return nil, fmt.Errorf("nil start node")
	}
	if info, ok := t.nodeOps[node.ID()]; ok {
		if out, okRes, err := callDecisionNodeOp(info, input); err != nil {
			return nil, err
		} else if okRes {
			return out, nil
		}
	}

	children := t.neighborsOf(node.ID())
	if len(children) == 0 {
		return nil, fmt.Errorf("no decision path found from node %d", node.ID())
	}

	for _, edgeIdx := range children {
		edgeRec, ok := t.edgeRecord(edgeIdx)
		if !ok {
			continue
		}
		key := edgeKey(edgeRec.FromID, edgeRec.ToID)
		if op, ok := t.edgeOps[key]; ok {
			pass, err := callDecisionEdgeOp(op, input)
			if err != nil {
				return nil, err
			}
			if !pass {
				continue
			}
		}
		child, ok := t.nodeByID(edgeRec.ToID)
		if !ok {
			continue
		}
		if out, err := t.decideFrom(child, input); err == nil {
			return out, nil
		}
	}

	return nil, fmt.Errorf("no decision path found from node %d", node.ID())
}
