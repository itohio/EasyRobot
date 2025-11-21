package graph

import (
	"math/rand"
	"testing"

	"github.com/itohio/EasyRobot/x/math/vec"
	vecTypes "github.com/itohio/EasyRobot/x/math/vec/types"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRRTStar_Simple2DGrid(t *testing.T) {
	// Use vecTypes.Vector (interface) as node type, but vec.Vector2D in callbacks
	graph := NewGenericGraph[vecTypes.Vector, float32]()

	// Define search space bounds
	minX, maxX := float32(0), float32(10)
	minY, maxY := float32(0), float32(10)

	// Sampler: generates random nodes with 2D coordinates
	sampler := func() Node[vecTypes.Vector, float32] {
		data := vec.Vector2D{
			minX + rand.Float32()*(maxX-minX),
			minY + rand.Float32()*(maxY-minY),
		}
		// Create a node for sampling (will be added to graph by RRT*)
		// Use ID 0 so AddNode will assign a unique ID automatically
		// Don't set graph - AddNode will set it when the node is added
		return &GenericNode[vecTypes.Vector, float32]{
			id:           0,
			data:         data,
			nodeIdx:      -1,
			edgeIdxs:     make([]int, 0),
			neighborIdxs: make([]int, 0),
			graph:        nil,
		}
	}

	// Connection threshold - edges will only be created within this distance
	// This is enforced by RRT* logic, not by a separate connector
	connectionThreshold := float32(2.0)

	// Goal checker: within small distance of goal
	goalTolerance := float32(1.0)
	goalChecker := func(node, goal Node[vecTypes.Vector, float32]) bool {
		nodeVec := node.(*GenericNode[vecTypes.Vector, float32]).Data()
		goalVec := goal.(*GenericNode[vecTypes.Vector, float32]).Data()
		dist := nodeVec.Distance(goalVec)
		return dist <= goalTolerance
	}

	// Create start and goal (as vec.Vector2D, but stored as vecTypes.Vector)
	startData := vec.Vector2D{1, 1}
	goalData := vec.Vector2D{9, 9}

	// Create nodes directly (vec.Vector2D implements vecTypes.Vector)
	startNode := &GenericNode[vecTypes.Vector, float32]{
		id:           graph.nextNodeID,
		data:         startData,
		nodeIdx:      -1,
		edgeIdxs:     make([]int, 0),
		neighborIdxs: make([]int, 0),
		graph:        graph,
	}
	graph.nextNodeID++

	goalNode := &GenericNode[vecTypes.Vector, float32]{
		id:           graph.nextNodeID,
		data:         goalData,
		nodeIdx:      -1,
		edgeIdxs:     make([]int, 0),
		neighborIdxs: make([]int, 0),
		graph:        graph,
	}
	graph.nextNodeID++

	// Set cost calculator - uses vecTypes.Vector's Distance method
	graph.SetCostCalculator(func(from, to Node[vecTypes.Vector, float32]) float32 {
		fromVec := from.(*GenericNode[vecTypes.Vector, float32]).Data()
		toVec := to.(*GenericNode[vecTypes.Vector, float32]).Data()
		return fromVec.Distance(toVec)
	})

	// Create RRT* searcher - no connector needed, edges are created directly
	rrtstar := NewRRTStar(
		graph,
		sampler,
		goalChecker,
		RRTStarOptions{
			RewireRadius:  3.0,
			MaxIterations: 500,
			Seed:          42, // Fixed seed for reproducibility
		},
	)

	// Search for path
	path := rrtstar.Search(startNode, goalNode)

	// Verify path exists
	require.NotNil(t, path, "Path should be found")
	assert.Greater(t, len(path), 0, "Path should have at least one node")

	// Verify path connectivity - each node should be reachable from the previous
	for i := 0; i < len(path)-1; i++ {
		current := path[i]
		next := path[i+1]
		currentVec := current.(*GenericNode[vecTypes.Vector, float32]).Data()
		nextVec := next.(*GenericNode[vecTypes.Vector, float32]).Data()
		dist := currentVec.Distance(nextVec)
		assert.LessOrEqual(t, dist, connectionThreshold*1.1, "Path nodes should be connected")
	}

	// Verify path ends near goal (within tolerance)
	lastNode := path[len(path)-1]
	lastVec := lastNode.(*GenericNode[vecTypes.Vector, float32]).Data()
	distToGoal := lastVec.Distance(goalData)
	assert.LessOrEqual(t, distToGoal, goalTolerance*2, "Path should end near goal")
}

func TestRRTStar_ImplementsSearcher(t *testing.T) {
	graph := NewGenericGraph[vecTypes.Vector, float32]()
	sampler := func() Node[vecTypes.Vector, float32] {
		data := vec.Vector2D{0, 0}
		return &GenericNode[vecTypes.Vector, float32]{
			id:           graph.nextNodeID,
			data:         data,
			nodeIdx:      -1,
			edgeIdxs:     make([]int, 0),
			neighborIdxs: make([]int, 0),
			graph:        graph,
		}
	}
	goalChecker := func(node, goal Node[vecTypes.Vector, float32]) bool { return false }

	rrtstar := NewRRTStar(graph, sampler, goalChecker)

	var searcher Searcher[vecTypes.Vector, float32] = rrtstar
	assert.NotNil(t, searcher, "RRTStar should implement Searcher interface")
}

// TestRRTStar_GridGraphNavigation is commented out because GridNode doesn't implement DistanceMeasurable
// To enable this test, GridNode would need to implement Distance(other vecTypes.Vector) float32 method
/*
func TestRRTStar_GridGraphNavigation(t *testing.T) {
	// Test RRT* on a grid graph with obstacles
	graph := NewGenericGraph[GridNode, float32]()

	// Define grid bounds
	gridWidth, gridHeight := 20, 20

	// Sampler: random grid coordinates
	sampler := func() GridNode {
		return GridNode{
			Row: rand.Intn(gridHeight),
			Col: rand.Intn(gridWidth),
		}
	}

	// Helper to calculate distance between grid nodes
	gridDistance := func(a, b GridNode) float32 {
		dr := float32(a.Row - b.Row)
		dc := float32(a.Col - b.Col)
		return float32(math.Sqrt(float64(dr*dr + dc*dc)))
	}

	// Goal checker: same cell
	goalChecker := func(node, goal Node[GridNode, float32]) bool {
		nodeData := node.(*GenericNode[GridNode, float32]).Data()
		goalData := goal.(*GenericNode[GridNode, float32]).Data()
		return nodeData.Row == goalData.Row && nodeData.Col == goalData.Col
	}

	// Create start and goal
	startData := GridNode{Row: 0, Col: 0}
	goalData := GridNode{Row: 19, Col: 19}

	startNode := &GenericNode[GridNode, float32]{
		id:           graph.nextNodeID,
		data:         startData,
		nodeIdx:      -1,
		edgeIdxs:     make([]int, 0),
		neighborIdxs: make([]int, 0),
		graph:        graph,
	}
	graph.nextNodeID++

	goalNode := &GenericNode[GridNode, float32]{
		id:           graph.nextNodeID,
		data:         goalData,
		nodeIdx:      -1,
		edgeIdxs:     make([]int, 0),
		neighborIdxs: make([]int, 0),
		graph:        graph,
	}
	graph.nextNodeID++

	// Set cost calculator for GridNode (uses Manhattan distance)
	graph.SetCostCalculator(func(from, to Node[GridNode, float32]) float32 {
		fromNode := from.(*GenericNode[GridNode, float32]).Data()
		toNode := to.(*GenericNode[GridNode, float32]).Data()
		dr := float32(fromNode.Row - toNode.Row)
		dc := float32(fromNode.Col - toNode.Col)
		return float32(math.Abs(float64(dr)) + math.Abs(float64(dc)))
	})

	// Create RRT* searcher
	rrtstar := NewRRTStar(
		graph,
		sampler,
		goalChecker,
		RRTStarOptions{
			RewireRadius:  5.0,
			MaxIterations: 1000,
			Seed:          42,
		},
	)

	// Search for path
	path := rrtstar.Search(startNode, goalNode)

	// Verify path exists
	require.NotNil(t, path, "Path should be found")
	assert.Greater(t, len(path), 0, "Path should have at least one node")

	// Verify path connectivity
	for i := 0; i < len(path)-1; i++ {
		current := path[i]
		next := path[i+1]
		currentData := current.(*GenericNode[GridNode, float32]).Data()
		nextData := next.(*GenericNode[GridNode, float32]).Data()
		dist := gridDistance(currentData, nextData)
		assert.LessOrEqual(t, dist, float32(3.0), "Path should have connected nodes")
	}
}
*/

func TestRRTStar_NoPath(t *testing.T) {
	// Test case where goal is unreachable
	graph := NewGenericGraph[vecTypes.Vector, float32]()

	sampler := func() Node[vecTypes.Vector, float32] {
		data := vec.Vector2D{rand.Float32() * 5, rand.Float32() * 5}
		return &GenericNode[vecTypes.Vector, float32]{
			id:           graph.nextNodeID,
			data:         data,
			nodeIdx:      -1,
			edgeIdxs:     make([]int, 0),
			neighborIdxs: make([]int, 0),
			graph:        graph,
		}
	}

	goalChecker := func(node, goal Node[vecTypes.Vector, float32]) bool {
		nodeVec := node.(*GenericNode[vecTypes.Vector, float32]).Data()
		goalVec := goal.(*GenericNode[vecTypes.Vector, float32]).Data()
		return nodeVec.Distance(goalVec) <= 0.1
	}

	startData := vec.Vector2D{0, 0}
	goalData := vec.Vector2D{100, 100} // Far away, unreachable

	startNode := &GenericNode[vecTypes.Vector, float32]{
		id:           graph.nextNodeID,
		data:         startData,
		nodeIdx:      -1,
		edgeIdxs:     make([]int, 0),
		neighborIdxs: make([]int, 0),
		graph:        graph,
	}
	graph.nextNodeID++

	goalNode := &GenericNode[vecTypes.Vector, float32]{
		id:           graph.nextNodeID,
		data:         goalData,
		nodeIdx:      -1,
		edgeIdxs:     make([]int, 0),
		neighborIdxs: make([]int, 0),
		graph:        graph,
	}
	graph.nextNodeID++

	rrtstar := NewRRTStar(
		graph,
		sampler,
		goalChecker,
		RRTStarOptions{
			MaxIterations: 100, // Low iterations to test timeout
			Seed:          42,
		},
	)

	_ = rrtstar.Search(startNode, goalNode)
	// Path may be nil if goal is unreachable within max iterations
	// This is expected behavior - we just verify it doesn't crash
}

func TestRRTStar_ReuseGraphForSecondTarget(t *testing.T) {
	// Test that a graph built during first search can be reused for a second target
	graph := NewGenericGraph[vecTypes.Vector, float32]()

	// Define search space bounds
	minX, maxX := float32(0), float32(10)
	minY, maxY := float32(0), float32(10)

	// Sampler: generates random nodes with 2D coordinates
	sampler := func() Node[vecTypes.Vector, float32] {
		data := vec.Vector2D{
			minX + rand.Float32()*(maxX-minX),
			minY + rand.Float32()*(maxY-minY),
		}
		return &GenericNode[vecTypes.Vector, float32]{
			id:           0,
			data:         data,
			nodeIdx:      -1,
			edgeIdxs:     make([]int, 0),
			neighborIdxs: make([]int, 0),
			graph:        nil,
		}
	}

	// Goal checker: within small distance of goal
	goalTolerance := float32(1.0)
	goalChecker := func(node, goal Node[vecTypes.Vector, float32]) bool {
		nodeVec := node.(*GenericNode[vecTypes.Vector, float32]).Data()
		goalVec := goal.(*GenericNode[vecTypes.Vector, float32]).Data()
		dist := nodeVec.Distance(goalVec)
		return dist <= goalTolerance
	}

	// Create start node
	startData := vec.Vector2D{1, 1}
	startNode := &GenericNode[vecTypes.Vector, float32]{
		id:           graph.nextNodeID,
		data:         startData,
		nodeIdx:      -1,
		edgeIdxs:     make([]int, 0),
		neighborIdxs: make([]int, 0),
		graph:        graph,
	}
	graph.nextNodeID++

	// Set cost calculator
	graph.SetCostCalculator(func(from, to Node[vecTypes.Vector, float32]) float32 {
		fromVec := from.(*GenericNode[vecTypes.Vector, float32]).Data()
		toVec := to.(*GenericNode[vecTypes.Vector, float32]).Data()
		return fromVec.Distance(toVec)
	})

	// Create RRT* searcher
	rrtstar := NewRRTStar(
		graph,
		sampler,
		goalChecker,
		RRTStarOptions{
			RewireRadius:  3.0,
			MaxIterations: 500,
			Seed:          42, // Fixed seed for reproducibility
		},
	)

	// First search: find path to goal1
	goal1Data := vec.Vector2D{9, 9}
	goal1Node := &GenericNode[vecTypes.Vector, float32]{
		id:           graph.nextNodeID,
		data:         goal1Data,
		nodeIdx:      -1,
		edgeIdxs:     make([]int, 0),
		neighborIdxs: make([]int, 0),
		graph:        graph,
	}
	graph.nextNodeID++

	path1 := rrtstar.Search(startNode, goal1Node)
	require.NotNil(t, path1, "First path should be found")
	assert.Greater(t, len(path1), 0, "First path should have at least one node")

	// Verify graph has nodes and edges from first search
	initialNodeCount := graph.NumNodes()
	initialEdgeCount := graph.NumEdges()
	assert.Greater(t, initialNodeCount, 0, "Graph should have nodes after first search")
	assert.Greater(t, initialEdgeCount, 0, "Graph should have edges after first search")

	// Second search: reuse the same graph to find path to goal2
	goal2Data := vec.Vector2D{8, 2}
	goal2Node := &GenericNode[vecTypes.Vector, float32]{
		id:           graph.nextNodeID,
		data:         goal2Data,
		nodeIdx:      -1,
		edgeIdxs:     make([]int, 0),
		neighborIdxs: make([]int, 0),
		graph:        graph,
	}
	graph.nextNodeID++

	path2 := rrtstar.Search(startNode, goal2Node)
	require.NotNil(t, path2, "Second path should be found")
	assert.Greater(t, len(path2), 0, "Second path should have at least one node")

	// Verify graph has grown (or at least maintained) from second search
	finalNodeCount := graph.NumNodes()
	finalEdgeCount := graph.NumEdges()
	assert.GreaterOrEqual(t, finalNodeCount, initialNodeCount, "Graph should have at least as many nodes after second search")
	assert.GreaterOrEqual(t, finalEdgeCount, initialEdgeCount, "Graph should have at least as many edges after second search")

	// Verify both paths are valid
	// Path 1 should end near goal1
	lastNode1 := path1[len(path1)-1]
	lastVec1 := lastNode1.(*GenericNode[vecTypes.Vector, float32]).Data()
	distToGoal1 := lastVec1.Distance(goal1Data)
	assert.LessOrEqual(t, distToGoal1, goalTolerance*2, "First path should end near goal1")

	// Path 2 should end near goal2
	lastNode2 := path2[len(path2)-1]
	lastVec2 := lastNode2.(*GenericNode[vecTypes.Vector, float32]).Data()
	distToGoal2 := lastVec2.Distance(goal2Data)
	assert.LessOrEqual(t, distToGoal2, goalTolerance*2, "Second path should end near goal2")

	// Verify both paths are valid and the graph was reused
	// The key test is that the graph grew/maintained nodes from the first search
	// Paths may start from different nodes in the graph, but both should be valid paths
	
	// Verify paths are non-empty and have valid connectivity
	assert.Greater(t, len(path1), 1, "First path should have multiple nodes")
	assert.Greater(t, len(path2), 1, "Second path should have multiple nodes")
	
	// Verify graph reuse: the graph should have nodes from both searches
	// The second search should be able to use nodes from the first search
	assert.Greater(t, finalNodeCount, 1, "Graph should have multiple nodes after both searches")
	assert.Greater(t, finalEdgeCount, 1, "Graph should have multiple edges after both searches")
}
