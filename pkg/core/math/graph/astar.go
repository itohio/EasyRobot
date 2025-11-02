package graph

import (
	"container/heap"
)

// Heuristic is a callback function that estimates the cost from a node to the goal
type Heuristic func(from, to Node) float32

// AStar is an A* search algorithm implementation with reusable buffers
type AStar struct {
	graph     Graph
	heuristic Heuristic

	// Reusable buffers to avoid allocations
	openSet   priorityQueue
	closedSet map[Node]bool
	gScore    map[Node]float32
	cameFrom  map[Node]Node
}

// NewAStar creates a new A* search instance
func NewAStar(g Graph, heuristic Heuristic) *AStar {
	return &AStar{
		graph:     g,
		heuristic: heuristic,
		openSet:   make(priorityQueue, 0, 64),
		closedSet: make(map[Node]bool),
		gScore:    make(map[Node]float32),
		cameFrom:  make(map[Node]Node),
	}
}

// Search finds a path from start to goal using A* algorithm
// Returns path as list of nodes, or nil if no path exists
func (a *AStar) Search(start, goal Node) Path {
	if start == nil || goal == nil || a.heuristic == nil {
		return nil
	}

	// Clear buffers for reuse
	a.clear()

	heap.Init(&a.openSet)

	a.gScore[start] = 0
	startF := a.heuristic(start, goal)
	heap.Push(&a.openSet, &nodeWrapper{node: start, fScore: startF, gScore: 0})

	for a.openSet.Len() > 0 {
		current := heap.Pop(&a.openSet).(*nodeWrapper).node

		if current.Equal(goal) {
			return a.reconstructPath(current)
		}

		a.closedSet[current] = true

		neighbors := a.graph.Neighbors(current)
		for _, neighbor := range neighbors {
			if a.closedSet[neighbor] {
				continue
			}

			tentativeG := a.gScore[current] + a.graph.Cost(current, neighbor)
			currentG, exists := a.gScore[neighbor]
			if exists && tentativeG >= currentG {
				continue
			}

			a.cameFrom[neighbor] = current
			a.gScore[neighbor] = tentativeG
			h := a.heuristic(neighbor, goal)
			f := tentativeG + h

			heap.Push(&a.openSet, &nodeWrapper{node: neighbor, fScore: f, gScore: tentativeG})
		}
	}

	return nil
}

func (a *AStar) clear() {
	a.openSet = a.openSet[:0] // Reset slice but keep capacity
	// Clear maps
	for k := range a.closedSet {
		delete(a.closedSet, k)
	}
	for k := range a.gScore {
		delete(a.gScore, k)
	}
	for k := range a.cameFrom {
		delete(a.cameFrom, k)
	}
}

func (a *AStar) reconstructPath(current Node) Path {
	var nodes []Node
	for {
		nodes = append(nodes, current)
		next, exists := a.cameFrom[current]
		if !exists {
			break
		}
		current = next
	}

	// Reverse to get forward path (start to goal)
	path := make(Path, len(nodes))
	for i := 0; i < len(nodes); i++ {
		path[i] = nodes[len(nodes)-1-i]
	}

	return path
}

// Path represents a path through the graph
type Path []Node

// nodeWrapper wraps node with f-score for priority queue
type nodeWrapper struct {
	node   Node
	fScore float32
	gScore float32
	index  int
}

// priorityQueue implements heap.Interface for efficient min-heap operations
type priorityQueue []*nodeWrapper

func (pq priorityQueue) Len() int { return len(pq) }

func (pq priorityQueue) Less(i, j int) bool {
	// Min-heap: lower fScore has higher priority
	return pq[i].fScore < pq[j].fScore
}

func (pq priorityQueue) Swap(i, j int) {
	pq[i], pq[j] = pq[j], pq[i]
	pq[i].index = i
	pq[j].index = j
}

func (pq *priorityQueue) Push(x any) {
	n := len(*pq)
	item := x.(*nodeWrapper)
	item.index = n
	*pq = append(*pq, item)
}

func (pq *priorityQueue) Pop() any {
	old := *pq
	n := len(old)
	item := old[n-1]
	old[n-1] = nil
	item.index = -1
	*pq = old[0 : n-1]
	return item
}

// AStar is a convenience function that creates a temporary AStar instance
// For multiple searches, use NewAStar() and reuse the instance
func AStarFunc(g Graph, heuristic Heuristic, start, goal Node) Path {
	astar := NewAStar(g, heuristic)
	return astar.Search(start, goal)
}
