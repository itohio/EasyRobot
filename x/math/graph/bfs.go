package graph

// BFS is a Breadth-First Search algorithm implementation with reusable buffers
type BFS[N any, E any] struct {
	graph Graph[N, E]

	// Reusable buffers to avoid allocations
	queue   []Node[N, E]
	visited map[Node[N, E]]bool
	prev    map[Node[N, E]]Node[N, E]
}

// NewBFS creates a new BFS search instance
func NewBFS[N any, E any](g Graph[N, E]) *BFS[N, E] {
	return &BFS[N, E]{
		graph:   g,
		queue:   make([]Node[N, E], 0, 64),
		visited: make(map[Node[N, E]]bool),
		prev:    make(map[Node[N, E]]Node[N, E]),
	}
}

// Search finds a path from start to goal using BFS algorithm
// Returns path as list of nodes, or nil if no path exists
func (b *BFS[N, E]) Search(start, goal Node[N, E]) Path[N, E] {
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

		for neighbor := range current.Neighbors() {
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

func (b *BFS[N, E]) clear() {
	b.queue = b.queue[:0] // Reset slice but keep capacity
	// Clear maps
	for k := range b.visited {
		delete(b.visited, k)
	}
	for k := range b.prev {
		delete(b.prev, k)
	}
}

func (b *BFS[N, E]) reconstructPath(current Node[N, E]) Path[N, E] {
	var nodes []Node[N, E]
	for {
		nodes = append(nodes, current)
		next, exists := b.prev[current]
		if !exists {
			break
		}
		current = next
	}

	// Reverse to get forward path (start to goal)
	path := make(Path[N, E], len(nodes))
	for i := 0; i < len(nodes); i++ {
		path[i] = nodes[len(nodes)-1-i]
	}

	return path
}

// BFSFunc is a convenience function that creates a temporary BFS instance
// For multiple searches, use NewBFS() and reuse the instance
func BFSFunc[N any, E any](g Graph[N, E], start, goal Node[N, E]) Path[N, E] {
	bfs := NewBFS(g)
	return bfs.Search(start, goal)
}
