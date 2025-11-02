package graph

// Searcher is the interface for graph search algorithms
type Searcher interface {
	// Search finds a path from start to goal
	// Returns path as list of nodes, or nil if no path exists
	Search(start, goal Node) Path
}

