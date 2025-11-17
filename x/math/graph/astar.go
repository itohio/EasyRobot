package graph

import (
	"container/heap"
)

// Heuristic is a callback function that estimates the cost from a node to the goal
type Heuristic[N any, E any] func(from, to Node[N, E]) float32

// AStar is an A* search algorithm implementation with reusable buffers
type AStar[N any, E any] struct {
	graph     Graph[N, E]
	heuristic Heuristic[N, E]

	// Reusable buffers to avoid allocations
	openSet   priorityQueue[N, E]
	closedSet map[int64]bool
	gScore    map[int64]float32
	cameFrom  map[int64]int64
	nodeCache map[int64]Node[N, E]
}

// NewAStar creates a new A* search instance
func NewAStar[N any, E any](g Graph[N, E], heuristic Heuristic[N, E]) *AStar[N, E] {
	return &AStar[N, E]{
		graph:     g,
		heuristic: heuristic,
		openSet:   make(priorityQueue[N, E], 0, 64),
		closedSet: make(map[int64]bool),
		gScore:    make(map[int64]float32),
		cameFrom:  make(map[int64]int64),
		nodeCache: make(map[int64]Node[N, E]),
	}
}

// Search finds a path from start to goal using A* algorithm
// Returns path as list of nodes, or nil if no path exists
func (a *AStar[N, E]) Search(start, goal Node[N, E]) Path[N, E] {
	if start == nil || goal == nil || a.heuristic == nil {
		return nil
	}

	// Clear buffers for reuse
	a.clear()

	heap.Init(&a.openSet)

	start = a.cacheNode(a.bindNode(start))
	goal = a.cacheNode(a.bindNode(goal))

	startID := start.ID()
	goalID := goal.ID()

	a.gScore[startID] = 0
	startF := a.heuristic(start, goal)
	heap.Push(&a.openSet, &nodeWrapper[N, E]{node: start, fScore: startF, gScore: 0})

	for a.openSet.Len() > 0 {
		current := a.cacheNode(heap.Pop(&a.openSet).(*nodeWrapper[N, E]).node)
		currentID := current.ID()

		if currentID == goalID {
			return a.reconstructPath(goalID)
		}

		a.closedSet[currentID] = true

		for neighbor := range current.Neighbors() {
			if neighbor == nil {
				continue
			}

			neighbor = a.cacheNode(a.bindNode(neighbor))
			neighborID := neighbor.ID()

			if a.closedSet[neighborID] {
				continue
			}

			tentativeG := a.gScore[currentID] + current.Cost(neighbor)
			if currentG, exists := a.gScore[neighborID]; exists && tentativeG >= currentG {
				continue
			}

			a.cameFrom[neighborID] = currentID
			a.gScore[neighborID] = tentativeG
			h := a.heuristic(neighbor, goal)
			f := tentativeG + h

			heap.Push(&a.openSet, &nodeWrapper[N, E]{node: neighbor, fScore: f, gScore: tentativeG})
		}
	}

	return nil
}

func (a *AStar[N, E]) clear() {
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
	for k := range a.nodeCache {
		delete(a.nodeCache, k)
	}
}

func (a *AStar[N, E]) reconstructPath(goalID int64) Path[N, E] {
	var nodes []Node[N, E]
	currentID := goalID
	for {
		node, ok := a.nodeCache[currentID]
		if !ok {
			break
		}
		nodes = append(nodes, node)
		nextID, exists := a.cameFrom[currentID]
		if !exists {
			break
		}
		currentID = nextID
	}

	path := make(Path[N, E], len(nodes))
	for i := 0; i < len(nodes); i++ {
		path[i] = nodes[len(nodes)-1-i]
	}

	return path
}

func (a *AStar[N, E]) cacheNode(node Node[N, E]) Node[N, E] {
	if node == nil {
		return nil
	}
	id := node.ID()
	if existing, ok := a.nodeCache[id]; ok {
		return existing
	}
	a.nodeCache[id] = node
	return node
}

func (a *AStar[N, E]) bindNode(node Node[N, E]) Node[N, E] {
	if node == nil {
		return nil
	}
	switch n := any(node).(type) {
	case GridNode:
		if n.graph == nil {
			if gg, ok := any(a.graph).(*GridGraph); ok {
				n.graph = gg
				if bound, ok := any(n).(Node[N, E]); ok {
					return bound
				}
			}
		}
	case MatrixNode:
		if n.Graph == nil {
			if mg, ok := any(a.graph).(*MatrixGraph); ok {
				n.Graph = mg
				if bound, ok := any(n).(Node[N, E]); ok {
					return bound
				}
			}
		}
	}
	return node
}

// nodeWrapper wraps node with f-score for priority queue
type nodeWrapper[N any, E any] struct {
	node   Node[N, E]
	fScore float32
	gScore float32
	index  int
}

// priorityQueue implements heap.Interface for efficient min-heap operations
type priorityQueue[N any, E any] []*nodeWrapper[N, E]

func (pq priorityQueue[N, E]) Len() int { return len(pq) }

func (pq priorityQueue[N, E]) Less(i, j int) bool {
	// Min-heap: lower fScore has higher priority
	return pq[i].fScore < pq[j].fScore
}

func (pq priorityQueue[N, E]) Swap(i, j int) {
	pq[i], pq[j] = pq[j], pq[i]
	pq[i].index = i
	pq[j].index = j
}

func (pq *priorityQueue[N, E]) Push(x any) {
	n := len(*pq)
	item := x.(*nodeWrapper[N, E])
	item.index = n
	*pq = append(*pq, item)
}

func (pq *priorityQueue[N, E]) Pop() any {
	old := *pq
	n := len(old)
	item := old[n-1]
	old[n-1] = nil
	item.index = -1
	*pq = old[0 : n-1]
	return item
}

// AStarFunc is a convenience function that creates a temporary AStar instance
// For multiple searches, use NewAStar() and reuse the instance
func AStarFunc[N any, E any](g Graph[N, E], heuristic Heuristic[N, E], start, goal Node[N, E]) Path[N, E] {
	astar := NewAStar(g, heuristic)
	return astar.Search(start, goal)
}
