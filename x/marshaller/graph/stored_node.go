package graph

import (
	"iter"

	"github.com/itohio/EasyRobot/x/math/graph"
)

type storedNode struct {
	graph *StoredGraph
	index int
}

func (n *storedNode) ID() int64 {
	if n == nil || n.graph == nil {
		return 0
	}
	record, ok := n.graph.nodeRecord(n.index)
	if !ok {
		return 0
	}
	return record.ID
}

func (n *storedNode) Data() any {
	if n == nil || n.graph == nil {
		return nil
	}
	val, err := n.graph.readNodeData(n.index)
	if err != nil {
		return nil
	}
	return val
}

func (n *storedNode) Neighbors() iter.Seq[graph.Node[any, any]] {
	return func(yield func(graph.Node[any, any]) bool) {
		if n == nil || n.graph == nil {
			return
		}
		fromID := n.ID()
		for _, edgeIdx := range n.graph.neighborsOf(fromID) {
			edgeRecord, ok := n.graph.edgeRecord(edgeIdx)
			if !ok {
				continue
			}
			neighbor, ok := n.graph.nodeByID(edgeRecord.ToID)
			if !ok {
				continue
			}
			if !yield(neighbor) {
				return
			}
		}
	}
}

func (n *storedNode) Edges() iter.Seq[graph.Edge[any, any]] {
	return func(yield func(graph.Edge[any, any]) bool) {
		if n == nil || n.graph == nil {
			return
		}
		fromID := n.ID()
		for _, edgeIdx := range n.graph.neighborsOf(fromID) {
			edge := &storedEdge{graph: n.graph, index: edgeIdx}
			if !yield(edge) {
				return
			}
		}
	}
}

func (n *storedNode) NumNeighbors() int {
	if n == nil || n.graph == nil {
		return 0
	}
	return len(n.graph.neighborsOf(n.ID()))
}

func (n *storedNode) Cost(toOther graph.Node[any, any]) float32 {
	if n == nil || n.graph == nil || toOther == nil {
		return 0
	}
	if n.graph.onCost != nil {
		return n.graph.onCost(n, toOther)
	}
	targetID := toOther.ID()
	for _, edgeIdx := range n.graph.neighborsOf(n.ID()) {
		edgeRecord, ok := n.graph.edgeRecord(edgeIdx)
		if !ok {
			continue
		}
		if edgeRecord.ToID == targetID {
			return n.graph.edgeCost(edgeIdx)
		}
	}
	return 0
}

func (n *storedNode) Equal(other graph.Node[any, any]) bool {
	if n == nil || other == nil {
		return false
	}
	if n.graph != nil && n.graph.onEqual != nil {
		return n.graph.onEqual(n, other)
	}
	return n.ID() == other.ID()
}

func (n *storedNode) Compare(other graph.Node[any, any]) int {
	if n == nil {
		return -1
	}
	if other == nil {
		return 1
	}
	if n.graph != nil && n.graph.onCompare != nil {
		return n.graph.onCompare(n, other)
	}
	otherID := other.ID()
	switch {
	case n.ID() < otherID:
		return -1
	case n.ID() > otherID:
		return 1
	default:
		return 0
	}
}
