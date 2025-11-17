package grid

import (
	"container/heap"

	"github.com/itohio/EasyRobot/x/math/graph"
	"github.com/itohio/EasyRobot/x/math/mat"
	"github.com/itohio/EasyRobot/x/math/vec"
)

// FastAStar is an optimized A* search that operates directly on matrices
// Reuses internal buffers to avoid allocations
type FastAStar struct {
	matrix    mat.Matrix
	allowDiag bool
	obstacle  float32
	heuristic graph.Heuristic[graph.GridNode, float32]

	// Reusable buffers
	openSet   fastPriorityQueue
	closedSet map[cell]bool
	gScore    map[cell]float32
	cameFrom  map[cell]cell
}

type cell struct {
	row, col int
}

func (c cell) toVector2D() vec.Vector2D {
	return vec.Vector2D{float32(c.col), float32(c.row)}
}

// NewFastAStar creates a new optimized A* searcher for matrices
func NewFastAStar(matrix mat.Matrix, allowDiag bool, obstacle float32, heuristic graph.Heuristic[graph.GridNode, float32]) *FastAStar {
	return &FastAStar{
		matrix:    matrix,
		allowDiag: allowDiag,
		obstacle:  obstacle,
		heuristic: heuristic,
		openSet:   make(fastPriorityQueue, 0, 64),
		closedSet: make(map[cell]bool),
		gScore:    make(map[cell]float32),
		cameFrom:  make(map[cell]cell),
	}
}

// Search finds a path from start to goal using A* algorithm
// Returns path as []vec.Vector2D, or nil if no path exists
func (f *FastAStar) Search(startRow, startCol, goalRow, goalCol int) []vec.Vector2D {
	if !f.validateInput(startRow, startCol, goalRow, goalCol) {
		return nil
	}

	if f.heuristic == nil {
		return nil
	}

	// Clear buffers for reuse
	f.clear()

	start := cell{row: startRow, col: startCol}
	goal := cell{row: goalRow, col: goalCol}

	heap.Init(&f.openSet)

	f.gScore[start] = 0
	startF := f.fastHeuristic(start, goal)
	heap.Push(&f.openSet, &fastNodeWrapper{cell: start, fScore: startF, gScore: 0})

	for f.openSet.Len() > 0 {
		current := heap.Pop(&f.openSet).(*fastNodeWrapper).cell

		if current == goal {
			return f.reconstructPath(current)
		}

		f.closedSet[current] = true

		neighbors := f.getNeighbors(current)
		for _, neighbor := range neighbors {
			if f.closedSet[neighbor] {
				continue
			}

			tentativeG := f.gScore[current] + f.getCost(current, neighbor)
			currentG, exists := f.gScore[neighbor]
			if exists && tentativeG >= currentG {
				continue
			}

			f.cameFrom[neighbor] = current
			f.gScore[neighbor] = tentativeG
			h := f.fastHeuristic(neighbor, goal)
			fScore := tentativeG + h

			heap.Push(&f.openSet, &fastNodeWrapper{cell: neighbor, fScore: fScore, gScore: tentativeG})
		}
	}

	return nil
}

func (f *FastAStar) validateInput(startRow, startCol, goalRow, goalCol int) bool {
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

func (f *FastAStar) clear() {
	f.openSet = f.openSet[:0] // Reset slice but keep capacity
	// Clear maps
	for k := range f.closedSet {
		delete(f.closedSet, k)
	}
	for k := range f.gScore {
		delete(f.gScore, k)
	}
	for k := range f.cameFrom {
		delete(f.cameFrom, k)
	}
}

func (f *FastAStar) getNeighbors(c cell) []cell {
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

func (f *FastAStar) getCost(from, to cell) float32 {
	cost := f.matrix[to.row][to.col]

	if f.allowDiag && from.row != to.row && from.col != to.col {
		cost *= 1.41421356237 // sqrt(2)
	}

	return cost
}

func (f *FastAStar) fastHeuristic(from, to cell) float32 {
	// Create temporary grid graph for node creation
	g := &graph.GridGraph{
		Matrix:    f.matrix,
		AllowDiag: f.allowDiag,
		Obstacle:  f.obstacle,
	}
	fromNode := graph.NewGridNode(g, from.row, from.col)
	toNode := graph.NewGridNode(g, to.row, to.col)
	return f.heuristic(fromNode, toNode)
}

func (f *FastAStar) getDirections() [][]int {
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

func (f *FastAStar) reconstructPath(current cell) []vec.Vector2D {
	var cells []cell
	for {
		cells = append(cells, current)
		next, exists := f.cameFrom[current]
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

// fastNodeWrapper wraps cell with f-score for priority queue
type fastNodeWrapper struct {
	cell   cell
	fScore float32
	gScore float32
	index  int
}

// fastPriorityQueue implements heap.Interface for efficient min-heap operations
type fastPriorityQueue []*fastNodeWrapper

func (pq fastPriorityQueue) Len() int { return len(pq) }

func (pq fastPriorityQueue) Less(i, j int) bool {
	return pq[i].fScore < pq[j].fScore
}

func (pq fastPriorityQueue) Swap(i, j int) {
	pq[i], pq[j] = pq[j], pq[i]
	pq[i].index = i
	pq[j].index = j
}

func (pq *fastPriorityQueue) Push(x any) {
	n := len(*pq)
	item := x.(*fastNodeWrapper)
	item.index = n
	*pq = append(*pq, item)
}

func (pq *fastPriorityQueue) Pop() any {
	old := *pq
	n := len(old)
	item := old[n-1]
	old[n-1] = nil
	item.index = -1
	*pq = old[0 : n-1]
	return item
}
