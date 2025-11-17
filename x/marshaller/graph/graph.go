package graph

import (
	"github.com/itohio/EasyRobot/x/marshaller/types"
	"github.com/itohio/EasyRobot/x/math/graph"
	"google.golang.org/protobuf/proto"
)

// StoredGraph implements graph.Graph[any, any] interface with memory-mapped storage.
// It does not require full reconstruction - nodes and edges are accessed directly from mmap.
type StoredGraph struct {
	nodeStorage  types.MappedStorage
	edgeStorage  types.MappedStorage
	dataStorage  types.MappedStorage
	onEqual      func(a, b graph.Node[any, any]) bool
	onCompare    func(a, b graph.Node[any, any]) int
	onCost       func(from, to graph.Node[any, any]) float32
	typeRegistry map[string]proto.Message
	kind         graphKind

	nodeRecords []NodeRecord
	edgeRecords []EdgeRecord
	nodeIndex   map[int64]int
	edgesByFrom map[int64][]int
}

// SetOnEqual sets the callback function for node equality comparison.
// The callback receives two nodes and returns true if they are equal.
// This is used by StoredNode.Equal() method.
func (g *StoredGraph) SetOnEqual(fn func(a, b graph.Node[any, any]) bool) {
	g.onEqual = fn
}

// SetOnCompare sets the callback function for node comparison.
// The callback receives two nodes and returns:
//   - -1 if a < b
//   - 0 if a == b
//   - +1 if a > b
//
// This is used by StoredNode.Compare() method.
func (g *StoredGraph) SetOnCompare(fn func(a, b graph.Node[any, any]) int) {
	g.onCompare = fn
}

// SetOnCost sets the callback function for edge cost calculation.
// Cost is algorithm-dependent and calculated dynamically, not stored.
// The calculator function receives (from, to) node pairs and returns the cost.
// This allows different algorithms to use different cost calculations.
// This is used by StoredNode.Cost() and StoredEdge.Cost() methods.
func (g *StoredGraph) SetOnCost(fn func(from, to graph.Node[any, any]) float32) {
	g.onCost = fn
}

// SetCostCalculator is a deprecated alias for SetOnCost.
// Use SetOnCost instead.
func (g *StoredGraph) SetCostCalculator(calculator func(from, to graph.Node[any, any]) float32) {
	g.SetOnCost(calculator)
}

// SetTypeRegistry registers the proto.Message types used when deserializing node or edge data.
func (g *StoredGraph) SetTypeRegistry(types map[string]proto.Message) {
	if types == nil {
		g.typeRegistry = nil
		return
	}
	g.typeRegistry = make(map[string]proto.Message, len(types))
	for name, prototype := range types {
		if prototype == nil {
			continue
		}
		g.typeRegistry[name] = proto.Clone(prototype)
	}
}

func (g *StoredGraph) clonePrototype(typeName string) (proto.Message, bool) {
	if g.typeRegistry == nil {
		return nil, false
	}
	prototype, ok := g.typeRegistry[typeName]
	if !ok || prototype == nil {
		return nil, false
	}
	return proto.Clone(prototype), true
}

// GraphStorage provides access to a graph stored in mmap-backed storage.
type GraphStorage struct {
	nodeStorage types.MappedStorage
	edgeStorage types.MappedStorage
	dataStorage types.MappedStorage
	graph       *StoredGraph
}

// BeginTransaction starts a new transaction and returns a transaction wrapper.
// Similar to GORM pattern - operations on the wrapper are transactional.
func (s *GraphStorage) BeginTransaction() (*GraphTransaction, error) {
	return &GraphTransaction{
		storage: s,
		changes: make([]change, 0),
	}, nil
}

// GraphTransaction wraps graph operations in a transaction.
type GraphTransaction struct {
	storage *GraphStorage
	changes []change
}

// change represents a single change in a transaction
type change struct {
	op     string // "addNode", "addEdge", "deleteNode", "deleteEdge", "updateNode", "updateEdge"
	nodeID int64
	fromID int64
	toID   int64
	data   any // Node or edge data (cost is calculated, not stored)
}

// Commit commits all changes atomically to storage.
// Uses atomic write patterns to prevent broken state on power loss.
func (tx *GraphTransaction) Commit() error {
	// TODO: Implement atomic commit
	// 1. Write changes to temporary locations
	// 2. Update checksums
	// 3. Atomically update headers (single write operation)
	// 4. Sync to disk
	return nil
}

// Rollback discards all changes in the transaction.
func (tx *GraphTransaction) Rollback() error {
	tx.changes = nil
	return nil
}

// AddNode adds a node to the graph storage.
// Changes are recorded but not committed until Commit().
func (tx *GraphTransaction) AddNode(data any) (int64, error) {
	// TODO: Implement node addition
	// Record change in transaction
	return 0, nil
}

// AddEdge adds one or more edges to the graph storage.
// Cost (if needed) should be included in the edge data.
func (tx *GraphTransaction) AddEdge(fromID, toID int64, data any, edges ...EdgeSpec) error {
	// TODO: Implement edge addition
	// Record changes in transaction
	return nil
}

// DeleteNode marks a node as deleted (defragmentation handles cleanup).
func (tx *GraphTransaction) DeleteNode(nodeID int64) error {
	// TODO: Implement node deletion
	// Record change in transaction
	return nil
}

// DeleteEdge marks one or more edges as deleted.
func (tx *GraphTransaction) DeleteEdge(fromID, toID int64, edges ...EdgeSpec) error {
	// TODO: Implement edge deletion
	// Record changes in transaction
	return nil
}

// UpdateNode updates a node's data in storage.
// Update strategy:
//   - If new data size differs from old: append new record to data.graph, update offset in node record
//   - If new data size is same or smaller: update in place at existing offset
//   - Old data location becomes a hole (defragmentation handles cleanup)
func (tx *GraphTransaction) UpdateNode(nodeID int64, data any) error {
	// TODO: Implement node update with size check
	// Record change in transaction
	return nil
}

// UpdateEdge updates an edge's data in storage.
// Cost (if needed) should be included in the edge data.
// Update strategy:
//   - If new data size differs from old: append new record to data.graph, update offset in edge record
//   - If new data size is same or smaller: update in place at existing offset
//   - Old data location becomes a hole (defragmentation handles cleanup)
func (tx *GraphTransaction) UpdateEdge(fromID, toID int64, data any) error {
	// TODO: Implement edge update with size check
	// Record change in transaction
	return nil
}

// EdgeSpec specifies an edge for batch operations.
type EdgeSpec struct {
	FromID int64
	ToID   int64
	Data   any // Edge data (cost included if needed)
}
