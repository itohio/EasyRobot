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

func TestGraphMarshallerComplex64ArrayRoundTrip(t *testing.T) {
	t.Parallel()

	// Test complex64 array serialization/deserialization directly
	srcData := []complex64{
		complex(1.0, 2.0),
		complex(3.5, 4.5),
		complex(-1.0, -2.0),
		complex(0.0, 1.0),
	}

	// Serialize
	serialized, dataType, typeName, err := serializeData(srcData)
	if err != nil {
		t.Fatalf("serializeData() failed: %v", err)
	}
	if dataType != DataTypeSlice {
		t.Fatalf("expected DataTypeSlice, got %d", dataType)
	}

	// Deserialize
	deserialized, err := deserializeData(serialized, dataType, typeName, nil)
	if err != nil {
		t.Fatalf("deserializeData() failed: %v", err)
	}

	// Verify the data
	restoredData, ok := deserialized.([]any)
	if !ok {
		t.Fatalf("deserialized data is not []any, got %T", deserialized)
	}

	if len(restoredData) != len(srcData) {
		t.Fatalf("data length mismatch: want %d, got %d", len(srcData), len(restoredData))
	}

	const tol = 1e-5
	for i := range srcData {
		restoredComplex, ok := restoredData[i].(complex64)
		if !ok {
			t.Fatalf("element %d is not complex64, got %T", i, restoredData[i])
		}
		if diff := math.Abs(float64(real(srcData[i]) - real(restoredComplex))); diff > tol {
			t.Errorf("real part mismatch at index %d: want %f, got %f (diff %f)", i, real(srcData[i]), real(restoredComplex), diff)
		}
		if diff := math.Abs(float64(imag(srcData[i]) - imag(restoredComplex))); diff > tol {
			t.Errorf("imag part mismatch at index %d: want %f, got %f (diff %f)", i, imag(srcData[i]), imag(restoredComplex), diff)
		}
	}
}