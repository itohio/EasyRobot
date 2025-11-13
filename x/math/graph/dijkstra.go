package graph

import (
	"container/heap"
)

// Dijkstra is a Dijkstra search algorithm implementation with reusable buffers
type Dijkstra struct {
	graph Graph

	// Reusable buffers to avoid allocations
	openSet dijkstraQueue
	dist    map[Node]float32
	prev    map[Node]Node
	visited map[Node]bool
}

// NewDijkstra creates a new Dijkstra search instance
func NewDijkstra(g Graph) *Dijkstra {
	return &Dijkstra{
		graph:   g,
		openSet: make(dijkstraQueue, 0, 64),
		dist:    make(map[Node]float32),
		prev:    make(map[Node]Node),
		visited: make(map[Node]bool),
	}
}

// Search finds shortest path from start to goal using Dijkstra's algorithm
// Returns path as list of nodes, or nil if no path exists
func (d *Dijkstra) Search(start, goal Node) Path {
	if start == nil || goal == nil {
		return nil
	}

	// Clear buffers for reuse
	d.clear()

	heap.Init(&d.openSet)

	d.dist[start] = 0
	heap.Push(&d.openSet, &dijkstraNode{node: start, dist: 0})

	for d.openSet.Len() > 0 {
		current := heap.Pop(&d.openSet).(*dijkstraNode)

		// Skip if already visited (we may have added it multiple times with different distances)
		if d.visited[current.node] {
			continue
		}

		// Mark as visited before processing neighbors
		d.visited[current.node] = true

		// Check if we reached the goal
		if current.node.Equal(goal) {
			return d.reconstructPath(current.node)
		}

		// Process neighbors
		neighbors := d.graph.Neighbors(current.node)
		for _, neighbor := range neighbors {
			// Skip already visited neighbors
			if d.visited[neighbor] {
				continue
			}

			// Calculate tentative distance
			cost := d.graph.Cost(current.node, neighbor)
			newDist := d.dist[current.node] + cost

			// If we found a shorter path to neighbor, update it
			currentDist, exists := d.dist[neighbor]
			if !exists || newDist < currentDist {
				d.dist[neighbor] = newDist
				d.prev[neighbor] = current.node
				// Add to heap (even if already there - we'll skip when popped if visited)
				heap.Push(&d.openSet, &dijkstraNode{node: neighbor, dist: newDist})
			}
		}
	}

	return nil
}

func (d *Dijkstra) clear() {
	d.openSet = d.openSet[:0] // Reset slice but keep capacity
	// Clear maps
	for k := range d.dist {
		delete(d.dist, k)
	}
	for k := range d.prev {
		delete(d.prev, k)
	}
	for k := range d.visited {
		delete(d.visited, k)
	}
}

func (d *Dijkstra) reconstructPath(current Node) Path {
	var nodes []Node
	for {
		nodes = append(nodes, current)
		next, exists := d.prev[current]
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

// dijkstraNode wraps node with distance for priority queue
type dijkstraNode struct {
	node  Node
	dist  float32
	index int
}

// dijkstraQueue implements heap.Interface for Dijkstra's algorithm
type dijkstraQueue []*dijkstraNode

func (dq dijkstraQueue) Len() int { return len(dq) }

func (dq dijkstraQueue) Less(i, j int) bool {
	// Min-heap: lower distance has higher priority
	return dq[i].dist < dq[j].dist
}

func (dq dijkstraQueue) Swap(i, j int) {
	dq[i], dq[j] = dq[j], dq[i]
	dq[i].index = i
	dq[j].index = j
}

func (dq *dijkstraQueue) Push(x any) {
	n := len(*dq)
	item := x.(*dijkstraNode)
	item.index = n
	*dq = append(*dq, item)
}

func (dq *dijkstraQueue) Pop() any {
	old := *dq
	n := len(old)
	item := old[n-1]
	old[n-1] = nil
	item.index = -1
	*dq = old[0 : n-1]
	return item
}

// DijkstraFunc is a convenience function that creates a temporary Dijkstra instance
// For multiple searches, use NewDijkstra() and reuse the instance
func DijkstraFunc(g Graph, start, goal Node) Path {
	d := NewDijkstra(g)
	return d.Search(start, goal)
}
