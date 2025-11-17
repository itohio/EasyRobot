package graph

import "iter"

// Comparable defines comparison operations for node types.
// Node implementations MAY implement this interface if they want to be used in searches and trees.
// This is used by tree structures for ordering and searching.
type Comparable[N any, E any] interface {
	// Equal returns true if this node equals other
	Equal(other Node[N, E]) bool
	// Compare returns:
	//   -1 if this < other
	//    0 if this == other
	//   +1 if this > other
	Compare(other Node[N, E]) int
}

// Searcher is the interface for graph search algorithms
type Searcher[N any, E any] interface {
	// Search finds a path from start to goal
	// Returns path as list of nodes, or nil if no path exists
	Search(start, goal Node[N, E]) Path[N, E]
}

// Accessor is the interface for accessing nodes and edges by their IDs.
type Accessor[N any, E any] interface {
	// NodeByID retrieves a node by its ID. Returns the Node and true if found, else nil and false.
	NodeByID(id int64) (Node[N, E], bool)
	// EdgeByID retrieves an edge by its ID. Returns the Edge and true if found, else nil and false.
	EdgeByID(id int64) (Edge[N, E], bool)
}

// Node represents a node in a graph.
// It is generic - users can specify any types for node and edge data.
// Node implementations MAY implement Comparable[N, E] for tree operations.
type Node[N any, E any] interface {
	Comparable[N, E]
	// ID returns a unique identifier for this node
	ID() int64
	// Data returns the node's data
	Data() N
	// Neighbors returns an iterator over neighboring nodes (for range loop)
	// Usage: for neighbor := range node.Neighbors() { ... }
	Neighbors() iter.Seq[Node[N, E]]
	// Edges returns an iterator over edges from this node (for range loop)
	// Usage: for edge := range node.Edges() { ... }
	Edges() iter.Seq[Edge[N, E]]
	// NumNeighbors returns the number of neighboring nodes
	NumNeighbors() int
	// Cost calculates the cost from this node to another node
	// Cost is algorithm-dependent and calculated, not stored
	Cost(toOther Node[N, E]) float32
}

// Edge represents an edge in a graph.
// It is generic - users can specify any types for node and edge data.
type Edge[N any, E any] interface {
	// ID returns a unique identifier for this edge
	ID() int64
	// From returns the source node
	From() Node[N, E]
	// To returns the destination node
	To() Node[N, E]
	// Data returns the edge's data
	Data() E
	// Cost returns the cost/weight of this edge
	// Cost is algorithm-dependent and calculated, not stored
	Cost() float32
}

// Graph represents a graph structure.
// It is generic - users can specify any types for node and edge data.
type Graph[N any, E any] interface {
	// Nodes returns an iterator over all nodes (for range loop)
	// Usage: for node := range graph.Nodes() { ... }
	Nodes() iter.Seq[Node[N, E]]
	// Edges returns an iterator over all edges (for range loop)
	// Usage: for edge := range graph.Edges() { ... }
	Edges() iter.Seq[Edge[N, E]]
	// NumNodes returns the total number of nodes in the graph
	NumNodes() int
	// NumEdges returns the total number of edges in the graph
	NumEdges() int
}

// Adder interface for adding nodes and edges
type Adder[N any, E any] interface {
	// AddNode adds a node to the graph
	AddNode(node Node[N, E]) error
	// AddEdge adds an edge to the graph
	AddEdge(edge Edge[N, E]) error
}

// Deleter interface for deleting nodes and edges
type Deleter[N any, E any] interface {
	// DeleteNode removes a node from the graph
	DeleteNode(node Node[N, E]) error
	// DeleteEdge removes an edge from the graph
	DeleteEdge(edge Edge[N, E]) error
}

// Updater interface for updating nodes and edges
type Updater[N any, E any] interface {
	// UpdateNode updates a node's data
	UpdateNode(node Node[N, E]) error
	// UpdateEdge updates an edge's data
	UpdateEdge(edge Edge[N, E]) error
}

// NodeAdder interface for node-level neighbor management
type NodeAdder[N any, E any] interface {
	// AddNeighbor adds a neighbor to this node
	AddNeighbor(neighbor Node[N, E]) error
}

// NodeDeleter interface for node-level neighbor management
type NodeDeleter[N any, E any] interface {
	// DeleteNeighbor removes a neighbor from this node
	DeleteNeighbor(neighbor Node[N, E]) error
}

// NodeUpdater interface for node-level neighbor management
type NodeUpdater[N any, E any] interface {
	// UpdateNeighbor updates a neighbor relationship
	UpdateNeighbor(oldNeighbor, newNeighbor Node[N, E]) error
}

// NodeTransactioner interface for transactional node operations
type NodeTransactioner[N any, E any] interface {
	// BeginNodeTransaction starts a transaction for node modifications
	BeginNodeTransaction() (NodeTransaction[N, E], error)
}

// GraphTransactioner interface for transactional graph operations
type GraphTransactioner[N any, E any] interface {
	// BeginGraphTransaction starts a transaction for graph modifications
	BeginGraphTransaction() (GraphTransaction[N, E], error)
}

// NodeTransaction represents a transaction for node operations
type NodeTransaction[N any, E any] interface {
	// Commit commits all changes in the transaction
	Commit() error
	// Rollback discards all changes in the transaction
	Rollback() error
}

// GraphTransaction represents a transaction for graph operations
type GraphTransaction[N any, E any] interface {
	// Commit commits all changes in the transaction
	Commit() error
	// Rollback discards all changes in the transaction
	Rollback() error
	// AddNode adds a node within the transaction
	AddNode(node Node[N, E]) error
	// AddEdge adds an edge within the transaction
	AddEdge(edge Edge[N, E]) error
	// DeleteNode deletes a node within the transaction
	DeleteNode(node Node[N, E]) error
	// DeleteEdge deletes an edge within the transaction
	DeleteEdge(edge Edge[N, E]) error
	// UpdateNode updates a node within the transaction
	UpdateNode(node Node[N, E]) error
	// UpdateEdge updates an edge within the transaction
	UpdateEdge(edge Edge[N, E]) error
}

// Tree extends Graph with tree-specific methods.
// Trees are directed acyclic graphs with a single root and hierarchical structure.
type Tree[N any, E any] interface {
	Graph[N, E]
	// Root returns the root node of the tree, or nil if empty
	Root() Node[N, E]
	// GetHeight returns the height of the tree (longest path from root to leaf)
	GetHeight() int
	// NodeCount returns the total number of nodes in the tree
	NodeCount() int
}
