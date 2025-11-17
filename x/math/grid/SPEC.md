# Grid Package Specification

## Overview

The `grid` package provides utilities for working with 2D grids and occupancy grids, including:
- Pathfinding algorithms (A*, Dijkstra)
- Ray casting and projection
- Matrix masking and extraction
- Shape extraction (rectangle, circle, ellipse)
- Occupancy queries

## Purpose

This package provides general-purpose grid operations that can be used by:
- SLAM algorithms
- Path planning
- Sensor fusion
- Map processing
- Any algorithm working with 2D occupancy grids

## Pathfinding

### A* Search

Finds optimal path using A* algorithm with customizable heuristics.

```go
func AStar(
    matrix mat.Matrix,
    startRow, startCol int,
    goalRow, goalCol int,
    opts *AStarOptions,
) []vec.Vector2D
```

**Options:**
- `AllowDiagonal`: Enable 8-directional movement (default: false)
- `Heuristic`: Heuristic function (default: Euclidean)
- `ObstacleValue`: Values <= this are obstacles (default: 0)

**Heuristics:**
- `EuclideanHeuristic`: Euclidean distance
- `ManhattanHeuristic`: Manhattan distance
- `ZeroHeuristic`: Returns 0 (equivalent to Dijkstra)

**Usage:**
```go
path := AStar(matrix, 0, 0, 10, 10, &AStarOptions{
    AllowDiagonal: true,
    Heuristic:     ManhattanHeuristic,
    ObstacleValue: 0,
})
```

### Dijkstra

Finds shortest path using Dijkstra's algorithm.

```go
func Dijkstra(
    matrix mat.Matrix,
    startRow, startCol int,
    goalRow, goalCol int,
    opts *DijkstraOptions,
) []vec.Vector2D
```

**Options:**
- `AllowDiagonal`: Enable 8-directional movement (default: false)
- `ObstacleValue`: Values <= this are obstacles (default: 0)

### FastAStar

Optimized A* implementation that reuses internal buffers.

```go
fastAStar := NewFastAStar(matrix, allowDiag, obstacle, heuristic)
path := fastAStar.Search(startRow, startCol, goalRow, goalCol)
fastAStar.Reset()  // Reuse for next search
```

**Benefits:**
- Reuses buffers between searches
- Avoids allocations
- Faster for multiple searches

### FastDijkstra

Optimized Dijkstra implementation with buffer reuse.

```go
fastDijkstra := NewFastDijkstra(matrix, allowDiag, obstacle)
path := fastDijkstra.Search(startRow, startCol, goalRow, goalCol)
fastDijkstra.Reset()  // Reuse for next search
```

## Integration with Graph Package

The grid package uses the `graph` package's generic types:

- **Graph Type**: `graph.GridGraph` - adapts matrix to graph interface
- **Node Type**: `graph.GridNode[graph.GridNode, float32]` - represents grid cells
- **Heuristic Type**: `graph.Heuristic[graph.GridNode, float32]` - generic heuristic function

**Node Creation:**
```go
g := &graph.GridGraph{
    Matrix:    matrix,
    AllowDiag: false,
    Obstacle:  0,
}
node := graph.NewGridNode(g, row, col)
```

**Path Conversion:**
```go
path := astar.Search(start, goal)
vectors := graph.GridPathToVector2D(path)  // Converts to []vec.Vector2D
```

## Ray Casting

### RayCast

Cast a single ray and return distance to obstacle.

```go
func RayCast(
    matrix mat.Matrix,
    startRow, startCol int,
    angle float32,
    maxDistance float32,
    obstacleValue float32,
) float32
```

### RayCastOptimized

Optimized ray casting using pre-computed directions.

```go
func RayCastOptimized(
    matrix mat.Matrix,
    startRow, startCol int,
    direction []int,  // Pre-computed direction vector
    maxDistance float32,
    obstacleValue float32,
) float32
```

### RayCastAll

Cast all rays and return distances.

```go
func RayCastAll(
    matrix mat.Matrix,
    startRow, startCol int,
    angles []float32,
    maxDistance float32,
    obstacleValue float32,
) []float32
```

### RayCastAllOptimized

Optimized batch ray casting.

```go
func RayCastAllOptimized(
    matrix mat.Matrix,
    startRow, startCol int,
    directions [][]int,  // Pre-computed direction vectors
    maxDistance float32,
    obstacleValue float32,
) []float32
```

### NewRayDirections

Pre-compute ray directions for efficiency.

```go
func NewRayDirections(angles []float32, resolution float32) [][]int
```

### RayProjection

Project a ray of specified length and check for intersection.

```go
func RayProjection(
    matrix mat.Matrix,
    startRow, startCol int,
    angle, length float32,
    obstacleValue float32,
) (endRow, endCol int, hit bool)
```

## Matrix Operations

### Mask

Mask one matrix with another (element-wise multiplication with mask).

```go
func Mask(matrix, mask mat.Matrix) mat.Matrix
```

### ExtractRectangle

Extract rectangular region from matrix.

```go
func ExtractRectangle(
    matrix mat.Matrix,
    startRow, startCol int,
    width, height int,
) mat.Matrix
```

### ExtractCircle

Extract circular region from matrix.

```go
func ExtractCircle(
    matrix mat.Matrix,
    centerRow, centerCol int,
    radius float32,
) mat.Matrix
```

### ExtractEllipse

Extract elliptical region from matrix.

```go
func ExtractEllipse(
    matrix mat.Matrix,
    centerRow, centerCol int,
    radiusX, radiusY float32,
    angle float32,
) mat.Matrix
```

## Occupancy Queries

### GetOccupancyReadings

Get occupancy readings for given angles using optimized raycasting.

```go
func GetOccupancyReadings(
    matrix mat.Matrix,
    startRow, startCol int,
    angles []float32,
    maxDistance float32,
    obstacleValue float32,
) []float32
```

## Design Principles

- **Embedded-friendly**: Optimized for embedded systems with minimal allocations
- **Pre-computation**: Where possible, pre-compute expensive operations
- **Reusable**: General-purpose functions usable across multiple algorithms
- **Efficient**: Use Bresenham algorithm and integer-only operations where possible
- **Generic Types**: Uses graph package's generic types for type safety
- **Buffer Reuse**: Fast implementations reuse buffers to avoid allocations

## Performance Considerations

1. **Fast Implementations**: Use `FastAStar` and `FastDijkstra` for multiple searches
2. **Pre-computed Directions**: Use `NewRayDirections` for repeated ray casting
3. **Optimized Ray Casting**: Use `RayCastOptimized` and `RayCastAllOptimized` when possible
4. **Matrix Operations**: Extract operations work on views when possible

## Usage Examples

### Path Planning

```go
// Create occupancy grid
matrix := mat.New(100, 100)
// ... populate matrix with obstacles ...

// Find path
path := AStar(matrix, 0, 0, 99, 99, &AStarOptions{
    AllowDiagonal: true,
    Heuristic:     ManhattanHeuristic,
})

// Use path for robot navigation
for _, point := range path {
    // Move robot to point
}
```

### Ray Casting for Sensors

```go
// Pre-compute directions for LIDAR
angles := []float32{0, 45, 90, 135, 180, 225, 270, 315}
directions := NewRayDirections(angles, 1.0)

// Cast rays from robot position
readings := RayCastAllOptimized(
    matrix,
    robotRow, robotCol,
    directions,
    10.0,  // max distance
    0,     // obstacle value
)
```

### Fast Repeated Searches

```go
// Create fast searcher
fastAStar := NewFastAStar(matrix, true, 0, ManhattanHeuristic)

// Multiple searches reuse buffers
for i := 0; i < 100; i++ {
    path := fastAStar.Search(startRow, startCol, goalRow, goalCol)
    // Process path
    fastAStar.Reset()  // Prepare for next search
}
```

## Integration with Graph Package

The grid package is built on top of the `graph` package:

- **GridGraph**: Adapts `mat.Matrix` to `graph.Graph[graph.GridNode, float32]`
- **GridNode**: Implements `graph.Node[graph.GridNode, float32]`
- **Heuristics**: Use `graph.Heuristic[graph.GridNode, float32]` type
- **Algorithms**: Use `graph.NewAStar` and `graph.NewDijkstra` internally

This allows:
- Type-safe pathfinding operations
- Consistent API with other graph types
- Reuse of graph algorithms
- Easy integration with graph marshaller for persistence
