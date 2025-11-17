package graph

import (
	"math"
	"math/rand"
	"path/filepath"
	"testing"

	graphlib "github.com/itohio/EasyRobot/x/math/graph"
	"github.com/itohio/EasyRobot/x/math/mat"
)

func TestGraphMarshallerMatrixRoundTrip(t *testing.T) {
	t.Parallel()

	rows, cols := 10, 10
	matrix := mat.New(rows, cols)
	rand.Seed(42)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			matrix[i][j] = rand.Float32()*10 + 0.5
		}
	}

	srcGraph := &graphlib.MatrixGraph{
		Matrix:   matrix,
		Obstacle: -1,
	}

	tmpDir := t.TempDir()
	nodePath := filepath.Join(tmpDir, "nodes.graph")
	edgePath := filepath.Join(tmpDir, "edges.graph")
	dataPath := filepath.Join(tmpDir, "data.graph")

	factory := NewFileMap()
	mar, err := NewMarshaller(factory, WithPath(nodePath), WithEdgesPath(edgePath), WithLabelsPath(dataPath))
	if err != nil {
		t.Fatalf("NewMarshaller() failed: %v", err)
	}

	if err := mar.Marshal(nil, srcGraph); err != nil {
		t.Fatalf("Marshal() failed: %v", err)
	}

	unmar, err := NewUnmarshaller(factory, WithPath(nodePath), WithEdgesPath(edgePath), WithLabelsPath(dataPath))
	if err != nil {
		t.Fatalf("NewUnmarshaller() failed: %v", err)
	}

	var stored StoredGraph
	if err := unmar.Unmarshal(nil, &stored); err != nil {
		t.Fatalf("Unmarshal() failed: %v", err)
	}
	t.Cleanup(func() { _ = stored.Close() })

	restored := mat.New(rows, cols)
	if err := graphlib.ToMatrix[any, any](&stored, restored); err != nil {
		t.Fatalf("ToMatrix() failed: %v", err)
	}

	const tol = 1e-5
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			if diff := math.Abs(float64(matrix[i][j] - restored[i][j])); diff > tol {
				t.Fatalf("matrix mismatch at (%d,%d): want %f got %f (diff %f)", i, j, matrix[i][j], restored[i][j], diff)
			}
		}
	}
}
