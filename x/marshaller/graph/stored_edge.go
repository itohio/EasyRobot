package graph

import "github.com/itohio/EasyRobot/x/math/graph"

type storedEdge struct {
	graph *StoredGraph
	index int
}

func (e *storedEdge) ID() int64 {
	return int64(e.index + 1)
}

func (e *storedEdge) From() graph.Node[any, any] {
	if e == nil || e.graph == nil {
		return nil
	}
	record, ok := e.graph.edgeRecord(e.index)
	if !ok {
		return nil
	}
	node, _ := e.graph.nodeByID(record.FromID)
	return node
}

func (e *storedEdge) To() graph.Node[any, any] {
	if e == nil || e.graph == nil {
		return nil
	}
	record, ok := e.graph.edgeRecord(e.index)
	if !ok {
		return nil
	}
	node, _ := e.graph.nodeByID(record.ToID)
	return node
}

func (e *storedEdge) Data() any {
	if e == nil || e.graph == nil {
		return nil
	}
	val, err := e.graph.readEdgeData(e.index)
	if err != nil {
		return nil
	}
	return val
}

func (e *storedEdge) Cost() float32 {
	if e == nil || e.graph == nil {
		return 0
	}
	if e.graph.onCost != nil {
		from := e.From()
		to := e.To()
		if from != nil && to != nil {
			return e.graph.onCost(from, to)
		}
	}
	return e.graph.edgeCost(e.index)
}
