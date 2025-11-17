package graph

// DFS is a Depth-First Search algorithm implementation with reusable buffers
type DFS[N any, E any] struct {
	graph Graph[N, E]

	// Reusable buffers to avoid allocations
	stack   []Node[N, E]
	visited map[Node[N, E]]bool
	prev    map[Node[N, E]]Node[N, E]
}

// NewDFS creates a new DFS search instance
func NewDFS[N any, E any](g Graph[N, E]) *DFS[N, E] {
	return &DFS[N, E]{
		graph:   g,
		stack:   make([]Node[N, E], 0, 64),
		visited: make(map[Node[N, E]]bool),
		prev:    make(map[Node[N, E]]Node[N, E]),
	}
}

// Search finds a path from start to goal using DFS algorithm
// Returns path as list of nodes, or nil if no path exists
func (d *DFS[N, E]) Search(start, goal Node[N, E]) Path[N, E] {
	if start == nil || goal == nil {
		return nil
	}

	// Clear buffers for reuse
	d.clear()

	d.stack = append(d.stack, start)
	d.visited[start] = true

	for len(d.stack) > 0 {
		// Pop from stack (LIFO)
		stackLen := len(d.stack)
		current := d.stack[stackLen-1]
		d.stack = d.stack[:stackLen-1]

		if current.Equal(goal) {
			return d.reconstructPath(current)
		}

		// Collect neighbors first (since we need to iterate in reverse)
		var neighbors []Node[N, E]
		for neighbor := range current.Neighbors() {
			neighbors = append(neighbors, neighbor)
		}
		// Push neighbors in reverse order to process in original order
		for i := len(neighbors) - 1; i >= 0; i-- {
			neighbor := neighbors[i]
			if d.visited[neighbor] {
				continue
			}

			d.visited[neighbor] = true
			d.prev[neighbor] = current
			d.stack = append(d.stack, neighbor)
		}
	}

	return nil
}

func (d *DFS[N, E]) clear() {
	d.stack = d.stack[:0] // Reset slice but keep capacity
	// Clear maps
	for k := range d.visited {
		delete(d.visited, k)
	}
	for k := range d.prev {
		delete(d.prev, k)
	}
}

func (d *DFS[N, E]) reconstructPath(current Node[N, E]) Path[N, E] {
	var nodes []Node[N, E]
	for {
		nodes = append(nodes, current)
		next, exists := d.prev[current]
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

// DFSFunc is a convenience function that creates a temporary DFS instance
// For multiple searches, use NewDFS() and reuse the instance
func DFSFunc[N any, E any](g Graph[N, E], start, goal Node[N, E]) Path[N, E] {
	dfs := NewDFS(g)
	return dfs.Search(start, goal)
}
