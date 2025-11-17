package graph

// LoopDetection detects if there are cycles (loops) in the graph
// Returns true if cycle exists, false otherwise
func LoopDetection[N any, E any](g Graph[N, E], start Node[N, E]) bool {
	if start == nil {
		return false
	}

	visited := make(map[Node[N, E]]bool)
	recStack := make(map[Node[N, E]]bool)

	return loopDetectionDFS(g, start, visited, recStack)
}

func loopDetectionDFS[N any, E any](g Graph[N, E], node Node[N, E], visited, recStack map[Node[N, E]]bool) bool {
	visited[node] = true
	recStack[node] = true

	for neighbor := range node.Neighbors() {
		if !visited[neighbor] {
			if loopDetectionDFS(g, neighbor, visited, recStack) {
				return true
			}
		} else if recStack[neighbor] {
			// Found back edge - cycle detected
			return true
		}
	}

	recStack[node] = false // Remove from recursion stack
	return false
}

// ConnectedComponents finds all connected components in the graph
// Returns a slice of slices, where each inner slice contains nodes in one component
func ConnectedComponents[N any, E any](g Graph[N, E], nodes []Node[N, E]) [][]Node[N, E] {
	if len(nodes) == 0 {
		return nil
	}

	visited := make(map[Node[N, E]]bool)
	var components [][]Node[N, E]

	for _, node := range nodes {
		if visited[node] {
			continue
		}

		// Find all nodes reachable from this node
		component := findComponent(g, node, visited)
		if len(component) > 0 {
			components = append(components, component)
		}
	}

	return components
}

func findComponent[N any, E any](g Graph[N, E], start Node[N, E], visited map[Node[N, E]]bool) []Node[N, E] {
	var component []Node[N, E]
	queue := []Node[N, E]{start}
	visited[start] = true

	for len(queue) > 0 {
		current := queue[0]
		queue = queue[1:]
		component = append(component, current)

		for neighbor := range current.Neighbors() {
			if !visited[neighbor] {
				visited[neighbor] = true
				queue = append(queue, neighbor)
			}
		}
	}

	return component
}
