package graph

// LoopDetection detects if there are cycles (loops) in the graph
// Returns true if cycle exists, false otherwise
func LoopDetection(g Graph, start Node) bool {
	if start == nil {
		return false
	}

	visited := make(map[Node]bool)
	recStack := make(map[Node]bool)

	return loopDetectionDFS(g, start, visited, recStack)
}

func loopDetectionDFS(g Graph, node Node, visited, recStack map[Node]bool) bool {
	visited[node] = true
	recStack[node] = true

	neighbors := g.Neighbors(node)
	for _, neighbor := range neighbors {
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
func ConnectedComponents(g Graph, nodes []Node) [][]Node {
	if len(nodes) == 0 {
		return nil
	}

	visited := make(map[Node]bool)
	var components [][]Node

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

func findComponent(g Graph, start Node, visited map[Node]bool) []Node {
	var component []Node
	queue := []Node{start}
	visited[start] = true

	for len(queue) > 0 {
		current := queue[0]
		queue = queue[1:]
		component = append(component, current)

		neighbors := g.Neighbors(current)
		for _, neighbor := range neighbors {
			if !visited[neighbor] {
				visited[neighbor] = true
				queue = append(queue, neighbor)
			}
		}
	}

	return component
}

