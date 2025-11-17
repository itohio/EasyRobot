package graph

import (
	"path/filepath"
	"testing"

	graphlib "github.com/itohio/EasyRobot/x/math/graph"
)

func TestDecisionTreeRoundTrip(t *testing.T) {
	t.Parallel()

	tree := graphlib.NewGenericDecisionTree[string, float32, int, string]("root")
	rootIdx := tree.RootIdx()
	leftIdx := tree.AddChild(rootIdx, "left")
	rightIdx := tree.AddChild(rootIdx, "right")

	if !tree.SetEdgeOpByIndex(rootIdx, leftIdx, decisionEdgeLessThanFive) {
		t.Fatalf("failed to set left edge op")
	}
	if !tree.SetEdgeOpByIndex(rootIdx, rightIdx, decisionEdgeGreaterEqFive) {
		t.Fatalf("failed to set right edge op")
	}
	if !tree.SetNodeOpByIndex(leftIdx, decisionLeafLow) {
		t.Fatalf("failed to set left node op")
	}
	if !tree.SetNodeOpByIndex(rightIdx, decisionLeafHigh) {
		t.Fatalf("failed to set right node op")
	}

	inputs := []int{2, 7, 4, 9}
	want, err := tree.Decide(nil, inputs...)
	if err != nil {
		t.Fatalf("source Decide() failed: %v", err)
	}

	tmpDir := t.TempDir()
	nodePath := filepath.Join(tmpDir, "decision.nodes")
	edgePath := filepath.Join(tmpDir, "decision.edges")
	dataPath := filepath.Join(tmpDir, "decision.data")

	factory := NewFileMap()
	mar, err := NewMarshaller(factory,
		WithPath(nodePath),
		WithEdgesPath(edgePath),
		WithLabelsPath(dataPath),
		WithDecisionOp(decisionLeafLow, decisionLeafHigh),
		WithDecisionEdge(decisionEdgeLessThanFive, decisionEdgeGreaterEqFive),
	)
	if err != nil {
		t.Fatalf("NewMarshaller() failed: %v", err)
	}
	if err := mar.Marshal(nil, tree); err != nil {
		t.Fatalf("Marshal() failed: %v", err)
	}

	unmar, err := NewUnmarshaller(factory,
		WithPath(nodePath),
		WithEdgesPath(edgePath),
		WithLabelsPath(dataPath),
		WithDecisionOp(decisionLeafLow, decisionLeafHigh),
		WithDecisionEdge(decisionEdgeLessThanFive, decisionEdgeGreaterEqFive),
	)
	if err != nil {
		t.Fatalf("NewUnmarshaller() failed: %v", err)
	}

	var storedTree StoredDecisionTree
	if err := unmar.Unmarshal(nil, &storedTree); err != nil {
		t.Fatalf("Unmarshal() failed: %v", err)
	}
	t.Cleanup(func() { _ = storedTree.Close() })

	got, err := storedTree.Decide(nil, toAnySlice(inputs)...)
	if err != nil {
		t.Fatalf("Stored Decide() failed: %v", err)
	}
	if len(got) != len(want) {
		t.Fatalf("result length mismatch: want %d got %d", len(want), len(got))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("decision mismatch at %d: want %v got %v", i, want[i], got[i])
		}
	}
}

func TestExpressionGraphRoundTrip(t *testing.T) {
	t.Parallel()

	expr := graphlib.NewGenericExpressionGraph[string, float32, float64, float64]()
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
	if err := expr.AddEdge(sumNode, inputNode, 0); err != nil {
		t.Fatalf("AddEdge sum->input failed: %v", err)
	}
	if err := expr.AddEdge(sumNode, doubleNode, 0); err != nil {
		t.Fatalf("AddEdge sum->double failed: %v", err)
	}
	if !expr.SetRoot(sumNode) {
		t.Fatalf("SetRoot failed")
	}

	inputs := []float64{1, 2.5, -3}
	want, err := expr.Compute(nil, inputs...)
	if err != nil {
		t.Fatalf("source Compute() failed: %v", err)
	}

	tmpDir := t.TempDir()
	nodePath := filepath.Join(tmpDir, "expr.nodes")
	edgePath := filepath.Join(tmpDir, "expr.edges")
	dataPath := filepath.Join(tmpDir, "expr.data")

	factory := NewFileMap()
	mar, err := NewMarshaller(factory,
		WithPath(nodePath),
		WithEdgesPath(edgePath),
		WithLabelsPath(dataPath),
		WithExpressionOp(exprInput, exprDouble, exprSum),
	)
	if err != nil {
		t.Fatalf("NewMarshaller() failed: %v", err)
	}
	if err := mar.Marshal(nil, expr); err != nil {
		t.Fatalf("Marshal() failed: %v", err)
	}

	unmar, err := NewUnmarshaller(factory,
		WithPath(nodePath),
		WithEdgesPath(edgePath),
		WithLabelsPath(dataPath),
		WithExpressionOp(exprInput, exprDouble, exprSum),
	)
	if err != nil {
		t.Fatalf("NewUnmarshaller() failed: %v", err)
	}

	var storedExpr StoredExpressionGraph
	if err := unmar.Unmarshal(nil, &storedExpr); err != nil {
		t.Fatalf("Unmarshal() failed: %v", err)
	}
	t.Cleanup(func() { _ = storedExpr.Close() })

	got, err := storedExpr.Compute(nil, toAnySlice(inputs)...)
	if err != nil {
		t.Fatalf("Stored Compute() failed: %v", err)
	}
	if len(got) != len(want) {
		t.Fatalf("result length mismatch: want %d got %d", len(want), len(got))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("expression mismatch at %d: want %v got %v", i, want[i], got[i])
		}
	}
}

// Decision node operations
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

// Decision edge operations
func decisionEdgeLessThanFive(input int) bool  { return input < 5 }
func decisionEdgeGreaterEqFive(input int) bool { return input >= 5 }

// Expression operations
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

func toAnySlice[T any](vals []T) []any {
	result := make([]any, len(vals))
	for i, v := range vals {
		result[i] = v
	}
	return result
}
