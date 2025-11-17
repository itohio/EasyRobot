package graph

import (
	"fmt"

	"iter"

	"github.com/itohio/EasyRobot/x/marshaller/types"
	"github.com/itohio/EasyRobot/x/math/graph"
	"google.golang.org/protobuf/proto"
)

var _ graph.Graph[any, any] = (*StoredGraph)(nil)

func newStoredGraph(nodeStorage, edgeStorage, dataStorage types.MappedStorage, registry map[string]proto.Message, kind graphKind) (*StoredGraph, error) {
	g := &StoredGraph{
		nodeStorage: nodeStorage,
		edgeStorage: edgeStorage,
		dataStorage: dataStorage,
		nodeIndex:   make(map[int64]int),
		edgesByFrom: make(map[int64][]int),
		kind:        kind,
	}
	g.SetTypeRegistry(registry)

	nodeHeader, err := readNodeFileHeader(nodeStorage)
	if err != nil {
		return nil, fmt.Errorf("failed to read node header: %w", err)
	}
	for i := 0; i < int(nodeHeader.NodeCount); i++ {
		record, err := readNodeRecord(nodeStorage, i)
		if err != nil {
			return nil, fmt.Errorf("failed to read node record %d: %w", i, err)
		}
		g.nodeRecords = append(g.nodeRecords, *record)
		g.nodeIndex[record.ID] = i
	}

	edgeHeader, err := readEdgeFileHeader(edgeStorage)
	if err != nil {
		return nil, fmt.Errorf("failed to read edge header: %w", err)
	}
	for i := 0; i < int(edgeHeader.EdgeCount); i++ {
		record, err := readEdgeRecord(edgeStorage, i)
		if err != nil {
			return nil, fmt.Errorf("failed to read edge record %d: %w", i, err)
		}
		g.edgeRecords = append(g.edgeRecords, *record)
		g.edgesByFrom[record.FromID] = append(g.edgesByFrom[record.FromID], i)
	}

	return g, nil
}

// Close releases underlying storages.
func (g *StoredGraph) Close() error {
	var firstErr error
	closeOne := func(storage types.MappedStorage) {
		if storage != nil {
			if err := storage.Close(); err != nil && firstErr == nil {
				firstErr = err
			}
		}
	}
	closeOne(g.nodeStorage)
	closeOne(g.edgeStorage)
	closeOne(g.dataStorage)
	g.nodeStorage = nil
	g.edgeStorage = nil
	g.dataStorage = nil
	return firstErr
}

// Nodes returns an iterator over all nodes in the stored graph.
func (g *StoredGraph) Nodes() iter.Seq[graph.Node[any, any]] {
	return func(yield func(graph.Node[any, any]) bool) {
		for idx := range g.nodeRecords {
			node := &storedNode{graph: g, index: idx}
			if !yield(node) {
				return
			}
		}
	}
}

// Edges returns an iterator over all edges in the stored graph.
func (g *StoredGraph) Edges() iter.Seq[graph.Edge[any, any]] {
	return func(yield func(graph.Edge[any, any]) bool) {
		for idx := range g.edgeRecords {
			edge := &storedEdge{graph: g, index: idx}
			if !yield(edge) {
				return
			}
		}
	}
}

// NumNodes returns the total number of nodes.
func (g *StoredGraph) NumNodes() int {
	return len(g.nodeRecords)
}

// NumEdges returns the total number of edges.
func (g *StoredGraph) NumEdges() int {
	return len(g.edgeRecords)
}

func (g *StoredGraph) nodeByID(id int64) (*storedNode, bool) {
	idx, ok := g.nodeIndex[id]
	if !ok {
		return nil, false
	}
	return &storedNode{graph: g, index: idx}, true
}

func (g *StoredGraph) nodeFromGraphNode(node graph.Node[any, any]) (*storedNode, bool) {
	if node == nil {
		return nil, false
	}
	if sn, ok := node.(*storedNode); ok && sn.graph == g {
		return sn, true
	}
	return g.nodeByID(node.ID())
}

func (g *StoredGraph) nodeRecord(index int) (*NodeRecord, bool) {
	if index < 0 || index >= len(g.nodeRecords) {
		return nil, false
	}
	return &g.nodeRecords[index], true
}

func (g *StoredGraph) edgeRecord(index int) (*EdgeRecord, bool) {
	if index < 0 || index >= len(g.edgeRecords) {
		return nil, false
	}
	return &g.edgeRecords[index], true
}

func (g *StoredGraph) neighborsOf(fromID int64) []int {
	if g == nil {
		return nil
	}
	return g.edgesByFrom[fromID]
}

func (g *StoredGraph) readNodeData(index int) (any, error) {
	record, ok := g.nodeRecord(index)
	if !ok {
		return nil, fmt.Errorf("node index %d out of bounds", index)
	}
	if record.DataOffset == 0 {
		return nil, nil
	}
	return readDataEntryValue(g.dataStorage, record.DataOffset, g.typeRegistry)
}

func (g *StoredGraph) readEdgeData(index int) (any, error) {
	record, ok := g.edgeRecord(index)
	if !ok {
		return nil, fmt.Errorf("edge index %d out of bounds", index)
	}
	offset := uint64(record.DataOffset)
	if offset == 0 {
		return nil, nil
	}
	return readDataEntryValue(g.dataStorage, offset, g.typeRegistry)
}

func (g *StoredGraph) edgeCost(index int) float32 {
	val, err := g.readEdgeData(index)
	if err != nil {
		return 0
	}
	return valueToFloat32(val)
}

func valueToFloat32(value any) float32 {
	switch v := value.(type) {
	case nil:
		return 0
	case float32:
		return v
	case float64:
		return float32(v)
	case int:
		return float32(v)
	case int32:
		return float32(v)
	case int64:
		return float32(v)
	case uint:
		return float32(v)
	case uint32:
		return float32(v)
	case uint64:
		return float32(v)
	default:
		return 0
	}
}
