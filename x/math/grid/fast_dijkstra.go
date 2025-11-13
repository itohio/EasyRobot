package grid

import (
	"container/heap"

	"github.com/itohio/EasyRobot/x/math/mat"
	"github.com/itohio/EasyRobot/x/math/vec"
)

// FastDijkstra is an optimized Dijkstra search that operates directly on matrices
// Reuses internal buffers to avoid allocations
type FastDijkstra struct {
	matrix    mat.Matrix
	allowDiag bool
	obstacle  float32

	// Reusable buffers
	openSet fastDijkstraQueue
	dist    map[cell]float32
	prev    map[cell]cell
	visited map[cell]bool
}

// NewFastDijkstra creates a new optimized Dijkstra searcher for matrices
func NewFastDijkstra(matrix mat.Matrix, allowDiag bool, obstacle float32) *FastDijkstra {
	return &FastDijkstra{
		matrix:    matrix,
		allowDiag: allowDiag,
		obstacle:  obstacle,
		openSet:   make(fastDijkstraQueue, 0, 64),
		dist:      make(map[cell]float32),
		prev:      make(map[cell]cell),
		visited:   make(map[cell]bool),
	}
}

// Search finds shortest path from start to goal using Dijkstra's algorithm
// Returns path as []vec.Vector2D, or nil if no path exists
func (f *FastDijkstra) Search(startRow, startCol, goalRow, goalCol int) []vec.Vector2D {
	if !f.validateInput(startRow, startCol, goalRow, goalCol) {
		return nil
	}

	// Check if start/goal are obstacles
	if f.matrix[startRow][startCol] <= f.obstacle ||
		f.matrix[goalRow][goalCol] <= f.obstacle {
		return nil
	}

	// Clear buffers for reuse
	f.clear()

	start := cell{row: startRow, col: startCol}
	goal := cell{row: goalRow, col: goalCol}

	heap.Init(&f.openSet)

	f.dist[start] = 0
	heap.Push(&f.openSet, &fastDijkstraNode{cell: start, dist: 0})

	for f.openSet.Len() > 0 {
		current := heap.Pop(&f.openSet).(*fastDijkstraNode).cell

		// Skip if already visited
		if f.visited[current] {
			continue
		}

		// Mark as visited before processing neighbors
		f.visited[current] = true

		// Check if we reached the goal
		if current == goal {
			return f.reconstructPath(current)
		}

		// Process neighbors
		neighbors := f.getNeighbors(current)
		for _, neighbor := range neighbors {
			// Skip already visited neighbors
			if f.visited[neighbor] {
				continue
			}

			// Calculate tentative distance
			cost := f.getCost(current, neighbor)
			newDist := f.dist[current] + cost

			// If we found a shorter path to neighbor, update it
			currentDist, exists := f.dist[neighbor]
			if !exists || newDist < currentDist {
				f.dist[neighbor] = newDist
				f.prev[neighbor] = current
				// Add to heap
				heap.Push(&f.openSet, &fastDijkstraNode{cell: neighbor, dist: newDist})
			}
		}
	}

	return nil
}

func (f *FastDijkstra) validateInput(startRow, startCol, goalRow, goalCol int) bool {
	if len(f.matrix) == 0 || len(f.matrix[0]) == 0 {
		return false
	}
	rows, cols := len(f.matrix), len(f.matrix[0])
	if startRow < 0 || startRow >= rows || startCol < 0 || startCol >= cols {
		return false
	}
	if goalRow < 0 || goalRow >= rows || goalCol < 0 || goalCol >= cols {
		return false
	}
	return true
}

func (f *FastDijkstra) clear() {
	f.openSet = f.openSet[:0] // Reset slice but keep capacity
	// Clear maps
	for k := range f.dist {
		delete(f.dist, k)
	}
	for k := range f.prev {
		delete(f.prev, k)
	}
	for k := range f.visited {
		delete(f.visited, k)
	}
}

func (f *FastDijkstra) getNeighbors(c cell) []cell {
	var neighbors []cell
	dirs := f.getDirections()

	for _, dir := range dirs {
		newRow := c.row + dir[0]
		newCol := c.col + dir[1]

		if newRow < 0 || newRow >= len(f.matrix) ||
			newCol < 0 || newCol >= len(f.matrix[0]) {
			continue
		}

		if f.matrix[newRow][newCol] <= f.obstacle {
			continue
		}

		neighbors = append(neighbors, cell{row: newRow, col: newCol})
	}

	return neighbors
}

func (f *FastDijkstra) getCost(from, to cell) float32 {
	cost := f.matrix[to.row][to.col]

	if f.allowDiag && from.row != to.row && from.col != to.col {
		cost *= 1.41421356237 // sqrt(2)
	}

	return cost
}

func (f *FastDijkstra) getDirections() [][]int {
	if f.allowDiag {
		return [][]int{
			{-1, 0}, {-1, 1}, {0, 1}, {1, 1},
			{1, 0}, {1, -1}, {0, -1}, {-1, -1},
		}
	}
	return [][]int{
		{-1, 0}, {0, 1}, {1, 0}, {0, -1},
	}
}

func (f *FastDijkstra) reconstructPath(current cell) []vec.Vector2D {
	var cells []cell
	for {
		cells = append(cells, current)
		next, exists := f.prev[current]
		if !exists {
			break
		}
		current = next
	}

	// Reverse to get forward path (start to goal)
	path := make([]vec.Vector2D, len(cells))
	for i := 0; i < len(cells); i++ {
		path[i] = cells[len(cells)-1-i].toVector2D()
	}

	return path
}

// fastDijkstraNode wraps cell with distance for priority queue
type fastDijkstraNode struct {
	cell  cell
	dist  float32
	index int
}

// fastDijkstraQueue implements heap.Interface for Dijkstra's algorithm
type fastDijkstraQueue []*fastDijkstraNode

func (dq fastDijkstraQueue) Len() int { return len(dq) }

func (dq fastDijkstraQueue) Less(i, j int) bool {
	return dq[i].dist < dq[j].dist
}

func (dq fastDijkstraQueue) Swap(i, j int) {
	dq[i], dq[j] = dq[j], dq[i]
	dq[i].index = i
	dq[j].index = j
}

func (dq *fastDijkstraQueue) Push(x any) {
	n := len(*dq)
	item := x.(*fastDijkstraNode)
	item.index = n
	*dq = append(*dq, item)
}

func (dq *fastDijkstraQueue) Pop() any {
	old := *dq
	n := len(old)
	item := old[n-1]
	old[n-1] = nil
	item.index = -1
	*dq = old[0 : n-1]
	return item
}
