package graph

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestGenericDecisionTree_Decide(t *testing.T) {
	tree := NewGenericDecisionTree[string, float32, int, string]("root")
	rootIdx := tree.RootIdx()
	leftIdx := tree.AddChild(rootIdx, "left")
	rightIdx := tree.AddChild(rootIdx, "right")

	require.NotEqual(t, -1, leftIdx)
	require.NotEqual(t, -1, rightIdx)

	require.True(t, tree.SetNodeOpByIndex(leftIdx, func(int) (string, bool) { return "A", true }))
	require.True(t, tree.SetNodeOpByIndex(rightIdx, func(int) (string, bool) { return "B", true }))

	require.True(t, tree.SetEdgeOpByIndex(rootIdx, leftIdx, func(x int) bool { return x > 5 }))
	require.True(t, tree.SetEdgeOpByIndex(rootIdx, rightIdx, func(x int) bool { return x <= 5 }))

	tests := []struct {
		name string
		x    int
		want string
	}{
		{"GreaterThanThreshold", 10, "A"},
		{"EqualThreshold", 5, "B"},
		{"LessThanThreshold", -3, "B"},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			results, err := tree.Decide(nil, tc.x)
			require.NoError(t, err)
			require.Len(t, results, 1)
			assert.Equal(t, tc.want, results[0])
		})
	}
}

func TestGenericDecisionTree_Interfaces(t *testing.T) {
	tree := NewGenericDecisionTree[string, float32, int, string]("root")
	rootIdx := tree.RootIdx()
	childIdx := tree.AddChild(rootIdx, "child")
	require.NotEqual(t, -1, childIdx)

	require.True(t, tree.SetNodeOpByIndex(childIdx, func(int) (string, bool) { return "child", true }))
	require.True(t, tree.SetEdgeOpByIndex(rootIdx, childIdx, func(int) bool { return true }))

	root := tree.Root()
	require.NotNil(t, root)

	if _, ok := root.(DecisionNode[string, float32, int, string]); !ok {
		t.Fatalf("root should implement DecisionNode")
	}

	var edge Edge[string, float32]
	for e := range tree.Edges() {
		edge = e
		break
	}
	require.NotNil(t, edge)

	if _, ok := edge.(DecisionEdge[string, float32, int]); !ok {
		t.Fatalf("edge should implement DecisionEdge")
	}
}
