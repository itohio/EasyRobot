# Graph Package Specification

## Overview

The `graph` package provides a generic, type-safe graph data structure library with support for various graph algorithms, tree structures, and pathfinding. It is designed for embedded systems and robotics applications, emphasizing efficiency, cache locality, and minimal allocations.

## Purpose

This package provides:
- Generic graph data structures with type-safe node and edge data
- Efficient graph algorithms (A*, Dijkstra, BFS, DFS)
- Tree structures (generic trees, binary trees, k-d trees, forests)
- Grid and matrix graph adapters
- Iterator-based traversal using Go's `iter.Seq` for zero-allocation iteration
- Support for dynamic cost calculation (algorithm-dependent, not stored)

## Design Principles

1. **Generics First**: All interfaces and implementations are generic over node data type `N` and edge data type `E`
2. **Iterator Pattern**: Use `iter.Seq[T]` for all traversal operations, enabling `for range` loops
3. **Value Arrays**: Store nodes and edges as values in arrays for cache locality
4. **Zero-Copy**: Minimize allocations through iterator-based traversal
5. **Cost Calculation**: Cost is calculated dynamically, not stored in edge records
6. **Interface Segregation**: Small, focused interfaces (Graph, Tree, Adder, Deleter, etc.)
7. **Composition over Inheritance**: Use interfaces and composition, not inheritance

## Core Interfaces

### Node[N, E]

Represents a node in a graph. Node implementations MAY implement `Comparable[N, E]` when ordering is required (trees, KD-trees, etc.).

```go
type Node[N any, E any] interface {
    Comparable[N, E]
    ID() int64
    Data() N
    Neighbors() iter.Seq[Node[N, E]]
    Edges() iter.Seq[Edge[N, E]]
    NumNeighbors() int
    Cost(toOther Node[N, E]) float32
}
```

**Key Methods:**
- `ID()`: Unique identifier for the node
- `Data()`: Returns the node's data (type `N`)
- `Neighbors()`: Iterator over neighboring nodes (call directly on the node)
- `Edges()`: Iterator over edges from this node
- `NumNeighbors()`: Quick access to neighbor count
- `Cost(toOther)`: Calculates cost to another node (algorithm-dependent)

### Edge[N, E]

Represents an edge in a graph.

```go
type Edge[N any, E any] interface {
    ID() int64
    From() Node[N, E]
    To() Node[N, E]
    Data() E
    Cost() float32
}
```

**Key Methods:**
- `Cost()`: Returns the cost/weight of this edge (calculated, not stored)

### Graph[N, E]

Core graph interface providing traversal and metadata.

```go
type Graph[N any, E any] interface {
    Nodes() iter.Seq[Node[N, E]]
    Edges() iter.Seq[Edge[N, E]]
    NumNodes() int
    NumEdges() int
}
```

**Usage:**
```go
for node := range graph.Nodes() {
    // Process node
}
for edge := range graph.Edges() {
    // Process edge
}
for neighbor := range node.Neighbors() {
    // Process neighbor
}
```

### Tree[N, E]

Extends `Graph[N, E]` with tree-specific methods.

```go
type Tree[N any, E any] interface {
    Graph[N, E]
    Root() Node[N, E]
    GetHeight() int
    NodeCount() int
}
```

### Comparable[N, E]

Optional interface for nodes used in tree operations and searches.

```go
type Comparable[N any, E any] interface {
    Equal(other Node[N, E]) bool
    Compare(other Node[N, E]) int  // -1: <, 0: ==, 1: >
}
```

**Note**: Node implementations MAY implement this interface if they want to be used in searches and trees. Generic graph implementations accept non-comparable nodes, but tree structures and insertion strategies rely on these methods when present.

### Accessor[N, E]

Interface for accessing nodes and edges by ID.

```go
type Accessor[N any, E any] interface {
    NodeByID(id int64) (Node[N, E], bool)
    EdgeByID(id int64) (Edge[N, E], bool)
}
```

### Modification Interfaces

- **Adder[N, E]**: `AddNode`, `AddEdge`
- **Deleter[N, E]**: `DeleteNode`, `DeleteEdge`
- **Updater[N, E]**: `UpdateNode`, `UpdateEdge`
- **NodeAdder[N, E]**: `AddNeighbor`
- **NodeDeleter[N, E]**: `DeleteNeighbor`
- **NodeUpdater[N, E]**: `UpdateNeighbor`

### Transaction Interfaces

- **NodeTransactioner[N, E]**: `BeginNodeTransaction()`
- **GraphTransactioner[N, E]**: `BeginGraphTransaction()`

## Implementations

### GenericGraph[N, E]

General-purpose graph implementation using value arrays.

**Features:**
- Stores nodes and edges as values in arrays for cache locality
- Supports dynamic cost calculation via `SetCostCalculator`
- Implements `Graph`, `Adder`, `Deleter`, `Updater`, `Accessor`

**Usage:**
```go
g := NewGenericGraph[string, float32]()
node := &GenericNode[string, float32]{data: "A", id: 1, graph: g}
g.AddNode(node)
```

### GridGraph

Adapter that treats a 2D matrix as a graph for grid-based pathfinding.

**Features:**
- Each cell (row, col) is a node
- Neighbors are adjacent cells (4 or 8-directional)
- Edge cost is the matrix value at the destination cell
- Supports diagonal movement

**Usage:**
```go
g := &GridGraph{
    Matrix:    matrix,
    AllowDiag: false,
    Obstacle:  0,
}
start := NewGridNode(g, 0, 0)
goal := NewGridNode(g, 10, 10)
```

### MatrixGraph

Adapter that treats a matrix as an adjacency matrix.

**Features:**
- Each column is a node
- `matrix[i][j]` = weight from node i to node j
- Values <= Obstacle are considered non-existent edges
- `ToMatrix` helper converts any `Graph[N, E]` into the provided matrix (zeros it first, maps arbitrary node IDs to indices)

### GenericTree[N, E]

Hierarchical tree structure with arbitrary branching.

**Features:**
- Value-based storage (arrays, not pointers)
- Supports multiple children per node
- Implements `Tree[N, E]` interface
- Provides ordered insertion (`AddChildAtPosition`, `ReorderChildren`, `SwapChildren`)
- Balancing helpers (`Balance`, `BalancedInsertionStrategy`)

**Usage:**
```go
tree := NewGenericTree[int, float32](1)  // Root with data 1
rootIdx := tree.RootIdx()
childIdx := tree.AddChild(rootIdx, 2)
tree.Balance()
```

### GenericBinaryTree[N, E]

Binary tree with left/right children.

**Features:**
- Value-based storage
- Supports left/right child operations
- Implements `Tree[N, E]` interface
- Accepts insertion/comparison strategies (e.g. `BinarySearchTreeStrategy`, `IntComparison`)

**Usage:**
```go
tree := NewGenericBinaryTree[int, float32](1)
rootIdx := tree.RootIdx()
leftIdx := tree.SetLeft(rootIdx, 2)
rightIdx := tree.SetRight(rootIdx, 3)
tree.ApplyInsertionStrategy(BinarySearchTreeStrategy[int, float32](IntComparison()), 5)
```

### GenericKDTree[V, E]

K-dimensional tree for spatial queries.

**Features:**
- Works with `vecTypes.Vector` interface
- Efficient nearest neighbor search
- Supports arbitrary dimensions via `Vector.Len()`

**Usage:**
```go
points := []vec.Vector{
    vec.NewFrom(1.0, 2.0),
    vec.NewFrom(3.0, 4.0),
}
tree := NewGenericKDTree[vec.Vector, float32](points)
query := vec.NewFrom(1.1, 2.1)
nearest := tree.NearestNeighbor(query)
```

### GenericForest[N, E]

Collection of disjoint trees.

**Features:**
- Works with any `Tree[N, E]` implementation
- Supports adding/removing trees
- Can merge trees (requires GenericTree internals)

**Usage:**
```go
forest := NewGenericForest[int, float32]()
tree1 := NewGenericTree[int, float32](1)
tree2 := NewGenericBinaryTree[int, float32](10)
treeIdx1 := forest.AddTree(tree1)
treeIdx2 := forest.AddTree(tree2)
```

### Decision & Expression Stack

Decision/Expression helpers layer behaviour on top of core graph/tree storage so
layout stays serializable while operations remain in user code.

#### Decision Interfaces & Operations

- `DecisionNode`, `DecisionEdge`, and `DecisionTree` extend the base contracts.
- `DecisionOp[Input, Output]` (`func(Input) (Output, bool)`) register per-node,
  while `DecisionEdgeOp[Input]` gate traversal (`func(Input) bool`).
- `GenericDecisionTree` embeds a `*GenericTree` and wraps `TreeGraphNode/Edge`
  so traversals yield decision-aware implementations without copying storage.
- Node/edge operations live in `map[int64]DecisionOp` and
  `map[parentID,childID]DecisionEdgeOp`; helpers bind ops by index or ID
  (`SetNodeOpByIndex`, `SetNodeOpByID`, `SetEdgeOpByID`, etc.).
- `Decide(start, inputs...)` walks the tree recursively per input, honoring edge
  criteria, and stops at the first successful path (errors bubble for nil roots,
  missing ops, or blocked paths).
- `decisionTreeNode` / `decisionTreeEdge` wrappers delegate traversal to the
  underlying tree, keeping iterators allocation-free.

#### VectorDecisionTree

`vector_decision_tree.go` is a specialized decision tree for `vec/types.Vector`
inputs:

- Stores `VectorDecisionNodeData` inside `GenericTree`, pairing optional
  `DecisionCriteria` predicates with `DecisionOutcome` leaf metadata.
- Provides `AddDecisionChild`, `AddLeafChild`, `ReorderChildren`,
  `SwapChildren`, and `Balance`, mirroring `GenericTree` ergonomics.
- `ComputeDecision(vec)` evaluates criteria (commonly binary true/false
  branches) until reaching a leaf outcome.
- Helper builders (`VectorDimensionThreshold`, `VectorNormThreshold`,
  `VectorDotProductThreshold`) cover typical robotics predicates such as
  dimensional slices, magnitude limits, or dot-product comparisons.

#### Expression Graph

- `ExpressionNode` adds `EvaluateExpression`/`ExpressionFunction`, consuming
  child outputs.
- `ExpressionOp[Input, Output]` functions bind via `SetNodeOpByID`.
- `GenericExpressionGraph` wraps a `*GenericGraph`, tracks a default root ID,
  and surfaces iterators that yield expression-aware wrappers while delegating
  storage to the base graph.
- `Compute(start, inputs...)` builds a reverse topological order of all nodes
  reachable from `start`, then for each input:
  1. Evaluates nodes bottom-up, caching results by node ID.
  2. Invokes the registered `ExpressionOp` with the current input plus child
     outputs.
  3. Returns the cached value for the requested start ID.
- Explicit errors cover missing inputs, absent start nodes, or nodes without an
  attached operation, keeping failure modes predictable.

## Algorithms

### A* Search

```go
type Heuristic[N any, E any] func(from, to Node[N, E]) float32

astar := NewAStar[GridNode, float32](graph, heuristic)
path := astar.Search(start, goal)
```

**Features:**
- Reusable buffers to avoid allocations
- Generic over node and edge types
- Customizable heuristic function

### Dijkstra

```go
dijkstra := NewDijkstra[GridNode, float32](graph)
path := dijkstra.Search(start, goal)
```

**Features:**
- Optimized priority queue implementation
- Generic over node and edge types

### BFS / DFS

```go
bfs := NewBFS[GridNode, float32](graph)
visited := bfs.Search(start)

dfs := NewDFS[GridNode, float32](graph)
visited := dfs.Search(start)
```

### Cycle Detection

```go
hasCycle := HasCycle[GridNode, float32](graph)
components := ConnectedComponents[GridNode, float32](graph)
```

## Path Representation

```go
type Path[N any, E any] []Node[N, E]
```

**Helper Functions:**
- `GridPathToVector2D[E](path Path[GridNode, E]) []vec.Vector2D`
- `MatrixPathToVector[E](path Path[MatrixNode, E]) []vec.Vector`

## Cost Calculation

Cost is **not stored** in edge records. Instead:

1. **Edge.Cost()**: Returns cost calculated from edge data or algorithm
2. **Node.Cost(toOther)**: Calculates cost from this node to another
3. **Cost Calculator**: `GenericGraph` supports setting a cost calculator function

**Rationale:**
- Cost is algorithm-dependent (Euclidean, Manhattan, custom)
- Storing cost would duplicate data and limit flexibility
- Dynamic calculation allows different algorithms to use different cost functions

## Iterator Pattern

All traversal methods return `iter.Seq[T]` for zero-allocation iteration:

```go
// Iterate over nodes
for node := range graph.Nodes() {
    // Process node
}

// Iterate over edges
for edge := range graph.Edges() {
    // Process edge
}

// Iterate over neighbors
for neighbor := range node.Neighbors() {
    // Process neighbor
}
```

**Benefits:**
- Zero allocations during iteration
- Early termination support (break from loop)
- Compatible with Go's range syntax

## Value-Based Storage

All graph implementations use value arrays instead of pointer-based structures:

**Benefits:**
- Better cache locality
- Reduced memory fragmentation
- Predictable memory layout
- Suitable for embedded systems

**Example:**
```go
type GenericGraph[N, E] struct {
    nodes []GenericNode[N, E]  // Value array
    edges []GenericEdge[N, E]  // Value array
}
```

## Type Safety

All interfaces and implementations are generic:

```go
// Graph with string nodes and float32 edges
g1 := NewGenericGraph[string, float32]()

// Graph with custom struct nodes
type MyNode struct { X, Y float32 }
g2 := NewGenericGraph[MyNode, float32]()

// Tree with int nodes
tree := NewGenericTree[int, float32](42)
```

## Robotics Use Cases

### Dynamic Map Building (SLAM)
- Use `GenericGraph` for pose graph
- Nodes: robot poses
- Edges: constraints/measurements
- Update graph dynamically as robot moves

### Decision / Expression Logic
- Use `GenericTree` or `GenericBinaryTree` plus `DecisionNode`/`DecisionEdge` interfaces for decision trees
- Evaluate logic via `DecisionTree.Decide`
- Use `ExpressionGraph` to evaluate expression DAGs such as `(x + 2) * 3`
- Store compiled decision/expr graphs in flash via the marshaller for deterministic behavior

### Visual Odometry
- Use `GenericGraph` for scene graph
- Nodes: keyframes/landmarks
- Edges: transformations
- Efficient nearest neighbor with `GenericKDTree`

### Path Planning
- Use `GridGraph` for occupancy grids
- A* or Dijkstra for pathfinding
- Dynamic obstacle avoidance

## Performance Considerations

1. **Cache Locality**: Value arrays provide better cache performance
2. **Zero-Copy Iteration**: `iter.Seq` avoids allocations
3. **Batch Operations**: Use transactions for multiple updates
4. **Cost Calculation**: Lazy evaluation, calculated on-demand
5. **Memory Efficiency**: Value-based storage reduces pointer overhead

## Embedded Systems Support

- **No Dynamic Allocations**: Iterators avoid allocations
- **Flash Storage**: Graph marshaller supports mmap for flash
- **Constrained Memory**: Value arrays are more memory-efficient
- **Static Decision Trees**: Can be stored in flash and memory-mapped

## Future Enhancements

- Graph marshaller for persistent storage
- Transaction log support
- Defragmentation for dynamic updates
- Network storage backends
- Virtual storage for testing

## File Organization

- `graph.go`: Core interfaces
- `generic.go`: GenericGraph implementation
- `grid_graph.go`: GridGraph adapter
- `matrix_graph.go`: MatrixGraph adapter
- `tree.go`: GenericTree implementation
- `binary_tree.go`: GenericBinaryTree implementation
- `tree_strategies.go`: Insertion/balancing strategies for trees
- `kd_tree.go`: GenericKDTree implementation
- `forest.go`: GenericForest implementation
- `decision_interfaces.go`, `decision_tree.go`: Decision abstractions and runtime wrappers
- `vector_decision_tree.go`: Vector-specialized decision tree helpers
- `expression_interfaces.go`, `expression_graph.go`: Expression graph support
- `astar.go`: A* algorithm
- `dijkstra.go`: Dijkstra algorithm
- `bfs.go`, `dfs.go`: Traversal algorithms
- `cycles.go`: Cycle detection
- `path.go`: Path utilities
- `DECISION_EXPRESSION_TREE_PLAN.md`: Design plan for decision/expression stack

## Testing

All implementations have comprehensive unit tests:
- `*_test.go`: Test files for each implementation
- Tests use `iter.Seq` for traversal
- Tests verify interface conformance
- Tests cover edge cases and error conditions

