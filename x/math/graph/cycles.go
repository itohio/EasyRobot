package graph

// LoopDetection detects if there are cycles (loops) in the graph
// Returns true if cycle exists, false otherwise
func LoopDetection[N any, E any](g Graph[N, E], start Node[N, E]) bool {
	if start == nil {
		return false
	}

	state := make(map[int64]int) // 0 = unvisited, 1 = visiting, 2 = done
	return loopDetectionDFS(g, start, -1, state)
}

func loopDetectionDFS[N any, E any](g Graph[N, E], node Node[N, E], parentID int64, state map[int64]int) bool {
	nodeID := node.ID()
	if state[nodeID] == 1 {
		return true
	}
	if state[nodeID] == 2 {
		return false
	}

	state[nodeID] = 1
	for neighbor := range node.Neighbors() {
		if neighbor == nil {
			continue
		}
		neighborID := neighbor.ID()
		if neighborID == parentID {
			continue
		}
		if loopDetectionDFS(g, neighbor, nodeID, state) {
			return true
		}
	}

	state[nodeID] = 2
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
