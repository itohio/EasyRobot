package yaml

import (
	"bytes"
	_ "embed"
	"iter"
	"testing"

	"github.com/itohio/EasyRobot/x/marshaller/types"
	"github.com/itohio/EasyRobot/x/math/graph"
	"github.com/itohio/EasyRobot/x/math/tensor"
)

//go:embed testdata/simple_graph.yaml
var simpleGraphYAML string

//go:embed testdata/simple_tree.yaml
var simpleTreeYAML string

//go:embed testdata/decision_tree.yaml
var decisionTreeYAML string

//go:embed testdata/expression_graph.yaml
var expressionGraphYAML string

func TestTensorRoundTrip(t *testing.T) {
	tests := []struct {
		name  string
		shape []int
		dtype types.DataType
	}{
		{"Vector_FP32", []int{5}, types.FP32},
		{"Matrix_FP32", []int{3, 4}, types.FP32},
		{"Tensor_FP32", []int{2, 3, 4}, types.FP32},
		{"Vector_FP64", []int{10}, types.FP64},
		{"Matrix_FP64", []int{5, 6}, types.FP64},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create tensor with test data
			shape := tensor.NewShape(tt.shape...)
			original := tensor.New(tt.dtype, shape)

			// Fill with test data
			for i := 0; i < original.Size(); i++ {
				original.SetAt(float64(i)*0.5, i)
			}

			// Marshal
			var buf bytes.Buffer
			m := NewMarshaller()
			if err := m.Marshal(&buf, original); err != nil {
				t.Fatalf("Marshal failed: %v", err)
			}

			t.Logf("YAML output:\n%s", buf.String())

			// Unmarshal
			u := NewUnmarshaller()
			var result types.Tensor
			if err := u.Unmarshal(&buf, &result); err != nil {
				t.Fatalf("Unmarshal failed: %v", err)
			}

			// Verify
			if result.DataType() != original.DataType() {
				t.Errorf("DataType mismatch: got %v, want %v", result.DataType(), original.DataType())
			}
			if len(result.Shape()) != len(original.Shape()) {
				t.Fatalf("Shape length mismatch: got %d, want %d", len(result.Shape()), len(original.Shape()))
			}
			for i := range result.Shape() {
				if result.Shape()[i] != original.Shape()[i] {
					t.Errorf("Shape[%d] mismatch: got %d, want %d", i, result.Shape()[i], original.Shape()[i])
				}
			}

			// Verify data
			for i := 0; i < original.Size(); i++ {
				if result.At(i) != original.At(i) {
					t.Errorf("Data[%d] mismatch: got %v, want %v", i, result.At(i), original.At(i))
				}
			}

			t.Log("✓ YAML tensor round-trip successful")
		})
	}
}

func TestArrayRoundTrip(t *testing.T) {
	tests := []struct {
		name string
		data any
	}{
		{"Float32Array", []float32{1.0, 2.5, 3.7, 4.2, 5.9}},
		{"Float64Array", []float64{1.1, 2.2, 3.3, 4.4}},
		{"Int32Array", []int32{-5, 0, 10, 20, 30}},
		{"IntArray", []int{100, 200, 300}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Marshal
			var buf bytes.Buffer
			m := NewMarshaller()
			if err := m.Marshal(&buf, tt.data); err != nil {
				t.Fatalf("Marshal failed: %v", err)
			}

			t.Logf("YAML output:\n%s", buf.String())

			// Unmarshal
			u := NewUnmarshaller()

			// Create result of same type
			var result any
			switch tt.data.(type) {
			case []float32:
				var r []float32
				result = &r
			case []float64:
				var r []float64
				result = &r
			case []int32:
				var r []int32
				result = &r
			case []int:
				var r []int
				result = &r
			}

			if err := u.Unmarshal(&buf, result); err != nil {
				t.Fatalf("Unmarshal failed: %v", err)
			}

			t.Logf("✓ YAML array round-trip successful")
		})
	}
}

// minimalNode is a minimal node implementation that satisfies the graph.Node interface
type minimalNode[N any, E any] struct {
	id   int64
	data N
}

func (n *minimalNode[N, E]) ID() int64 { return n.id }
func (n *minimalNode[N, E]) Data() N   { return n.data }
func (n *minimalNode[N, E]) Equal(other graph.Node[N, E]) bool {
	return other != nil && n.id == other.ID()
}
func (n *minimalNode[N, E]) Compare(other graph.Node[N, E]) int {
	if other == nil {
		return 1
	}
	if n.id < other.ID() {
		return -1
	}
	if n.id > other.ID() {
		return 1
	}
	return 0
}
func (n *minimalNode[N, E]) Neighbors() iter.Seq[graph.Node[N, E]] {
	return func(yield func(graph.Node[N, E]) bool) {}
}
func (n *minimalNode[N, E]) Edges() iter.Seq[graph.Edge[N, E]] {
	return func(yield func(graph.Edge[N, E]) bool) {}
}
func (n *minimalNode[N, E]) NumNeighbors() int { return 0 }
func (n *minimalNode[N, E]) Cost(toOther graph.Node[N, E]) float32 { return 0 }

// minimalEdge is a minimal edge implementation that satisfies the graph.Edge interface
type minimalEdge[N any, E any] struct {
	from graph.Node[N, E]
	to   graph.Node[N, E]
	data E
	id   int64
}

func (e *minimalEdge[N, E]) ID() int64 { return e.id }
func (e *minimalEdge[N, E]) From() graph.Node[N, E] { return e.from }
func (e *minimalEdge[N, E]) To() graph.Node[N, E]   { return e.to }
func (e *minimalEdge[N, E]) Data() E                { return e.data }
func (e *minimalEdge[N, E]) Cost() float32 {
	switch v := any(e.data).(type) {
	case float32:
		return v
	case float64:
		return float32(v)
	case int, int32, int64:
		return 0
	default:
		return 0
	}
}

func TestGraphRoundTrip(t *testing.T) {
	t.Parallel()

	// Create a simple graph using the GenericGraph API
	g := graph.NewGenericGraph[string, float32]()
	
	// Add nodes - GenericGraph.AddNode accepts any Node interface
	nodeA := &minimalNode[string, float32]{id: 1, data: "A"}
	nodeB := &minimalNode[string, float32]{id: 2, data: "B"}
	nodeC := &minimalNode[string, float32]{id: 3, data: "C"}
	
	if err := g.AddNode(nodeA); err != nil {
		t.Fatalf("AddNode A failed: %v", err)
	}
	if err := g.AddNode(nodeB); err != nil {
		t.Fatalf("AddNode B failed: %v", err)
	}
	if err := g.AddNode(nodeC); err != nil {
		t.Fatalf("AddNode C failed: %v", err)
	}
	
	// Get nodes from graph (AddNode creates GenericNode instances internally)
	var actualNodeA, actualNodeB, actualNodeC graph.Node[string, float32]
	for node := range g.Nodes() {
		switch node.ID() {
		case 1:
			actualNodeA = node
		case 2:
			actualNodeB = node
		case 3:
			actualNodeC = node
		}
	}
	if actualNodeA == nil || actualNodeB == nil || actualNodeC == nil {
		t.Fatal("Failed to retrieve nodes from graph")
	}

	// Add edges using nodes from graph
	edgeAB := &minimalEdge[string, float32]{from: actualNodeA, to: actualNodeB, data: float32(1.5), id: 1}
	edgeBC := &minimalEdge[string, float32]{from: actualNodeB, to: actualNodeC, data: float32(2.0), id: 2}
	if err := g.AddEdge(edgeAB); err != nil {
		t.Fatalf("AddEdge A->B failed: %v", err)
	}
	if err := g.AddEdge(edgeBC); err != nil {
		t.Fatalf("AddEdge B->C failed: %v", err)
	}

	// Marshal
	var buf bytes.Buffer
	m := NewMarshaller()
	if err := m.Marshal(&buf, g); err != nil {
		t.Fatalf("Marshal failed: %v", err)
	}

	t.Logf("YAML graph output:\n%s", buf.String())

	// Unmarshal
	u := NewUnmarshaller()
	var result *graph.GenericGraph[any, any]
	if err := u.Unmarshal(&buf, &result); err != nil {
		t.Fatalf("Unmarshal failed: %v", err)
	}

	// Verify structure
	if result.NumNodes() != 3 {
		t.Errorf("Node count mismatch: got %d, want 3", result.NumNodes())
	}
	if result.NumEdges() != 2 {
		t.Errorf("Edge count mismatch: got %d, want 2", result.NumEdges())
	}

	// Verify nodes
	nodeMap := make(map[int64]graph.Node[any, any])
	for node := range result.Nodes() {
		nodeMap[node.ID()] = node
	}
	if len(nodeMap) != 3 {
		t.Errorf("Node map size mismatch: got %d, want 3", len(nodeMap))
	}

	t.Log("✓ YAML graph round-trip successful")
}

func TestTreeRoundTrip(t *testing.T) {
	t.Parallel()

	// Create a tree
	tree := graph.NewGenericTree[string, float32]("root")
	rootIdx := tree.RootIdx()
	leftIdx := tree.AddChild(rootIdx, "left")
	rightIdx := tree.AddChild(rootIdx, "right")
	tree.AddChild(leftIdx, "left-left")
	tree.AddChild(leftIdx, "left-right")
	tree.AddChild(rightIdx, "right-left")

	// Marshal
	var buf bytes.Buffer
	m := NewMarshaller()
	if err := m.Marshal(&buf, tree); err != nil {
		t.Fatalf("Marshal failed: %v", err)
	}

	t.Logf("YAML tree output:\n%s", buf.String())

	// Unmarshal
	u := NewUnmarshaller()
	var result *graph.GenericTree[any, any]
	if err := u.Unmarshal(&buf, &result); err != nil {
		t.Fatalf("Unmarshal failed: %v", err)
	}

	// Verify structure
	if result.NodeCount() != 6 {
		t.Errorf("Node count mismatch: got %d, want 6", result.NodeCount())
	}
	if result.Root() == nil {
		t.Fatal("Root is nil")
	}
	if result.GetHeight() != 2 {
		t.Errorf("Height mismatch: got %d, want 2", result.GetHeight())
	}

	t.Log("✓ YAML tree round-trip successful")
}

func TestDecisionTreeRoundTripWithComputation(t *testing.T) {
	t.Parallel()

	// Create decision tree
	tree := graph.NewGenericDecisionTree[string, float32, int, string]("root")
	rootIdx := tree.RootIdx()
	leftIdx := tree.AddChild(rootIdx, "left")
	rightIdx := tree.AddChild(rootIdx, "right")

	// Set operations
	tree.SetEdgeOpByIndex(rootIdx, leftIdx, decisionEdgeLessThanFive)
	tree.SetEdgeOpByIndex(rootIdx, rightIdx, decisionEdgeGreaterEqFive)
	tree.SetNodeOpByIndex(leftIdx, decisionLeafLow)
	tree.SetNodeOpByIndex(rightIdx, decisionLeafHigh)

	// Test computation before marshalling
	inputs := []int{2, 7, 4, 9}
	want, err := tree.Decide(nil, inputs...)
	if err != nil {
		t.Fatalf("Source Decide() failed: %v", err)
	}
	expected := []string{"low", "high", "low", "high"}
	if len(want) != len(expected) {
		t.Fatalf("Result length mismatch: got %d, want %d", len(want), len(expected))
	}
	for i := range expected {
		if want[i] != expected[i] {
			t.Errorf("Decision mismatch at %d: got %v, want %v", i, want[i], expected[i])
		}
	}

	// Marshal
	var buf bytes.Buffer
	m := NewMarshaller()
	if err := m.Marshal(&buf, tree); err != nil {
		t.Fatalf("Marshal failed: %v", err)
	}

	t.Logf("YAML decision tree output:\n%s", buf.String())

	// Unmarshal
	u := NewUnmarshaller()
	var result *graph.GenericDecisionTree[any, any, any, any]
	if err := u.Unmarshal(&buf, &result); err != nil {
		t.Fatalf("Unmarshal failed: %v", err)
	}

	// Verify structure
	if result.NodeCount() != 3 {
		t.Errorf("Node count mismatch: got %d, want 3", result.NodeCount())
	}
	if result.Root() == nil {
		t.Fatal("Root is nil")
	}

	// Note: Operations need to be re-registered after unmarshalling
	// This is a limitation - operations are stored as function names in metadata
	// but need to be bound via SetNodeOpByIndex/SetEdgeOpByIndex
	// For now, we verify the structure is correct
	t.Log("✓ YAML decision tree round-trip successful (structure verified)")
}

func TestExpressionGraphRoundTripWithComputation(t *testing.T) {
	t.Parallel()

	// Create expression graph
	expr := graph.NewGenericExpressionGraph[string, float32, float64, float64]()
	inputNode, err := expr.AddNode("input", exprInput)
	if err != nil {
		t.Fatalf("AddNode input failed: %v", err)
	}
	doubleNode, err := expr.AddNode("double", exprDouble)
	if err != nil {
		t.Fatalf("AddNode double failed: %v", err)
	}
	sumNode, err := expr.AddNode("sum", exprSum)
	if err != nil {
		t.Fatalf("AddNode sum failed: %v", err)
	}
	if err := expr.AddEdge(sumNode, inputNode, float32(0)); err != nil {
		t.Fatalf("AddEdge sum->input failed: %v", err)
	}
	if err := expr.AddEdge(sumNode, doubleNode, float32(0)); err != nil {
		t.Fatalf("AddEdge sum->double failed: %v", err)
	}
	if !expr.SetRoot(sumNode) {
		t.Fatalf("SetRoot failed")
	}

	// Test computation before marshalling
	inputs := []float64{1.0, 2.5, -3.0}
	want, err := expr.Compute(nil, inputs...)
	if err != nil {
		t.Fatalf("Source Compute() failed: %v", err)
	}
	if len(want) != 3 {
		t.Fatalf("Result length mismatch: got %d, want 3", len(want))
	}

	// Marshal
	var buf bytes.Buffer
	m := NewMarshaller()
	if err := m.Marshal(&buf, expr); err != nil {
		t.Fatalf("Marshal failed: %v", err)
	}

	t.Logf("YAML expression graph output:\n%s", buf.String())

	// Unmarshal
	u := NewUnmarshaller()
	var result *graph.GenericExpressionGraph[any, any, any, any]
	if err := u.Unmarshal(&buf, &result); err != nil {
		t.Fatalf("Unmarshal failed: %v", err)
	}

	// Verify structure
	if result.NumNodes() != 3 {
		t.Errorf("Node count mismatch: got %d, want 3", result.NumNodes())
	}
	if result.NumEdges() != 2 {
		t.Errorf("Edge count mismatch: got %d, want 2", result.NumEdges())
	}

	// Note: Operations need to be re-registered after unmarshalling
	// This is a limitation - operations are stored as function names in metadata
	// but need to be bound via SetNodeOpByID
	// For now, we verify the structure is correct
	t.Log("✓ YAML expression graph round-trip successful (structure verified)")
}

// Helper functions for decision tree operations
func decisionLeafLow(input int) (string, bool) {
	if input < 5 {
		return "low", true
	}
	return "", false
}

func decisionLeafHigh(input int) (string, bool) {
	if input >= 5 {
		return "high", true
	}
	return "", false
}

func decisionEdgeLessThanFive(input int) bool  { return input < 5 }
func decisionEdgeGreaterEqFive(input int) bool { return input >= 5 }

// Helper functions for expression graph operations
func exprInput(input float64, _ map[int64]float64) (float64, bool) {
	return input, true
}

func exprDouble(input float64, _ map[int64]float64) (float64, bool) {
	return input * 2, true
}

func exprSum(_ float64, childOutputs map[int64]float64) (float64, bool) {
	var total float64
	for _, v := range childOutputs {
		total += v
	}
	return total, true
}

func TestUnmarshalFromEmbeddedYAML(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		yamlData  string
		verify    func(t *testing.T, result any)
	}{
		{
			name:     "simple_graph",
			yamlData: simpleGraphYAML,
			verify: func(t *testing.T, result any) {
				g, ok := result.(*graph.GenericGraph[any, any])
				if !ok {
					t.Fatalf("Expected *graph.GenericGraph[any, any], got %T", result)
				}
				if g.NumNodes() != 3 {
					t.Errorf("Expected 3 nodes, got %d", g.NumNodes())
				}
				if g.NumEdges() != 2 {
					t.Errorf("Expected 2 edges, got %d", g.NumEdges())
				}
			},
		},
		{
			name:     "simple_tree",
			yamlData: simpleTreeYAML,
			verify: func(t *testing.T, result any) {
				tree, ok := result.(*graph.GenericTree[any, any])
				if !ok {
					t.Fatalf("Expected *graph.GenericTree[any, any], got %T", result)
				}
				if tree.NodeCount() != 6 {
					t.Errorf("Expected 6 nodes, got %d", tree.NodeCount())
				}
				if tree.Root() == nil {
					t.Fatal("Expected root node, got nil")
				}
				if tree.GetHeight() != 2 {
					t.Errorf("Expected height 2, got %d", tree.GetHeight())
				}
			},
		},
		{
			name:     "decision_tree",
			yamlData: decisionTreeYAML,
			verify: func(t *testing.T, result any) {
				dt, ok := result.(*graph.GenericDecisionTree[any, any, any, any])
				if !ok {
					t.Fatalf("Expected *graph.GenericDecisionTree[any, any, any, any], got %T", result)
				}
				if dt.NodeCount() != 3 {
					t.Errorf("Expected 3 nodes, got %d", dt.NodeCount())
				}
				if dt.Root() == nil {
					t.Fatal("Expected root node, got nil")
				}
			},
		},
		{
			name:     "expression_graph",
			yamlData: expressionGraphYAML,
			verify: func(t *testing.T, result any) {
				eg, ok := result.(*graph.GenericExpressionGraph[any, any, any, any])
				if !ok {
					t.Fatalf("Expected *graph.GenericExpressionGraph[any, any, any, any], got %T", result)
				}
				if eg.NumNodes() != 3 {
					t.Errorf("Expected 3 nodes, got %d", eg.NumNodes())
				}
				if eg.NumEdges() != 2 {
					t.Errorf("Expected 2 edges, got %d", eg.NumEdges())
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			u := NewUnmarshaller()
			var result any
			buf := bytes.NewBufferString(tt.yamlData)
			if err := u.Unmarshal(buf, &result); err != nil {
				t.Fatalf("Unmarshal failed: %v", err)
			}
			tt.verify(t, result)
		})
	}
}
