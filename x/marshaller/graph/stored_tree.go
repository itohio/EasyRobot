package graph

import (
	"fmt"

	"github.com/itohio/EasyRobot/x/math/graph"
)

var _ graph.Tree[any, any] = (*StoredTree)(nil)

// StoredTree provides tree-specific helpers on top of StoredGraph.
type StoredTree struct {
	*StoredGraph
	rootID   int64
	treeType string
}

func newStoredTree(base *StoredGraph, meta *graphMetadata) (*StoredTree, error) {
	if meta == nil {
		return nil, fmt.Errorf("tree metadata missing")
	}
	if meta.RootID == 0 {
		return nil, fmt.Errorf("tree root not specified")
	}
	return &StoredTree{
		StoredGraph: base,
		rootID:      meta.RootID,
		treeType:    meta.TreeType,
	}, nil
}

// Root returns the stored root node.
func (t *StoredTree) Root() graph.Node[any, any] {
	if t == nil || t.StoredGraph == nil {
		return nil
	}
	node, _ := t.nodeByID(t.rootID)
	return node
}

// GetHeight computes the longest path length from root to leaves.
func (t *StoredTree) GetHeight() int {
	if t == nil || t.StoredGraph == nil {
		return 0
	}
	visited := make(map[int64]bool)
	return t.heightFrom(t.rootID, visited)
}

// NodeCount returns the total number of nodes in the tree.
func (t *StoredTree) NodeCount() int {
	if t == nil || t.StoredGraph == nil {
		return 0
	}
	return t.NumNodes()
}

func (t *StoredTree) heightFrom(nodeID int64, visited map[int64]bool) int {
	if nodeID == 0 || visited[nodeID] {
		return 0
	}
	visited[nodeID] = true
	maxChild := 0
	for _, edgeIdx := range t.neighborsOf(nodeID) {
		edgeRec, ok := t.edgeRecord(edgeIdx)
		if !ok {
			continue
		}
		childID := edgeRec.ToID
		h := t.heightFrom(childID, visited)
		if h > maxChild {
			maxChild = h
		}
	}
	visited[nodeID] = false
	return maxChild + 1
}

func (t *StoredTree) resolveStartNode(start graph.Node[any, any]) (*storedNode, error) {
	if start == nil {
		node, ok := t.nodeByID(t.rootID)
		if !ok {
			return nil, fmt.Errorf("root node %d not found", t.rootID)
		}
		return node, nil
	}
	node, ok := t.nodeFromGraphNode(start)
	if !ok {
		return nil, fmt.Errorf("node %d not found", start.ID())
	}
	return node, nil
}
