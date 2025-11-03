package graph

// BFS is a Breadth-First Search algorithm implementation with reusable buffers
type BFS struct {
	graph Graph

	// Reusable buffers to avoid allocations
	queue   []Node
	visited map[Node]bool
	prev    map[Node]Node
}

// NewBFS creates a new BFS search instance
func NewBFS(g Graph) *BFS {
	return &BFS{
		graph:   g,
		queue:   make([]Node, 0, 64),
		visited: make(map[Node]bool),
		prev:    make(map[Node]Node),
	}
}

// Search finds a path from start to goal using BFS algorithm
// Returns path as list of nodes, or nil if no path exists
func (b *BFS) Search(start, goal Node) Path {
	if start == nil || goal == nil {
		return nil
	}

	// Clear buffers for reuse
	b.clear()

	b.queue = append(b.queue, start)
	b.visited[start] = true

	for len(b.queue) > 0 {
		current := b.queue[0]
		b.queue = b.queue[1:]

		if current.Equal(goal) {
			return b.reconstructPath(current)
		}

		neighbors := b.graph.Neighbors(current)
		for _, neighbor := range neighbors {
			if b.visited[neighbor] {
				continue
			}

			b.visited[neighbor] = true
			b.prev[neighbor] = current
			b.queue = append(b.queue, neighbor)
		}
	}

	return nil
}

func (b *BFS) clear() {
	b.queue = b.queue[:0] // Reset slice but keep capacity
	// Clear maps
	for k := range b.visited {
		delete(b.visited, k)
	}
	for k := range b.prev {
		delete(b.prev, k)
	}
}

func (b *BFS) reconstructPath(current Node) Path {
	var nodes []Node
	for {
		nodes = append(nodes, current)
		next, exists := b.prev[current]
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

// BFSFunc is a convenience function that creates a temporary BFS instance
// For multiple searches, use NewBFS() and reuse the instance
func BFSFunc(g Graph, start, goal Node) Path {
	bfs := NewBFS(g)
	return bfs.Search(start, goal)
}
