package marshaller_test

import (
	"bytes"
	"testing"

	"github.com/itohio/EasyRobot/x/marshaller/gob"
	"github.com/itohio/EasyRobot/x/marshaller/text"
	"github.com/itohio/EasyRobot/x/marshaller/types"
	"github.com/itohio/EasyRobot/x/math/tensor"
)

func TestTextMarshaller(t *testing.T) {
	// Create a text marshaller directly
	m := text.NewMarshaller()

	if m.Format() != "text" {
		t.Errorf("Expected format 'text', got %q", m.Format())
	}

	// Test with a tensor
	testTensor := tensor.New(tensor.DTFP32, tensor.NewShape(2, 3))
	testTensor.Fill(nil, 1.0)

	var buf bytes.Buffer
	err := m.Marshal(&buf, testTensor)
	if err != nil {
		t.Fatalf("Failed to marshal tensor: %v", err)
	}

	output := buf.String()
	if len(output) == 0 {
		t.Error("Expected non-empty output")
	}
	t.Logf("Tensor output:\n%s", output)

	// Test with a slice
	buf.Reset()
	testSlice := []float32{1.0, 2.0, 3.0}
	err = m.Marshal(&buf, testSlice)
	if err != nil {
		t.Fatalf("Failed to marshal slice: %v", err)
	}

	output = buf.String()
	if len(output) == 0 {
		t.Error("Expected non-empty output")
	}
	t.Logf("Slice output:\n%s", output)
}

func TestGobMarshallerTensor(t *testing.T) {
	// Create gob marshaller and unmarshaller directly
	m := gob.NewMarshaller()
	u := gob.NewUnmarshaller()

	if m.Format() != "gob" {
		t.Errorf("Expected marshaller format 'gob', got %q", m.Format())
	}
	if u.Format() != "gob" {
		t.Errorf("Expected unmarshaller format 'gob', got %q", u.Format())
	}

	// Create a test tensor
	testTensor := tensor.New(tensor.DTFP32, tensor.NewShape(2, 3))
	for i := 0; i < testTensor.Size(); i++ {
		testTensor.SetAt(float64(i), i)
	}

	// Marshal
	var buf bytes.Buffer
	err := m.Marshal(&buf, testTensor)
	if err != nil {
		t.Fatalf("Failed to marshal tensor: %v", err)
	}

	// Unmarshal
	var result types.Tensor
	err = u.Unmarshal(&buf, &result)
	if err != nil {
		t.Fatalf("Failed to unmarshal tensor: %v", err)
	}

	// Verify
	if result.Empty() {
		t.Fatal("Result tensor is empty")
	}

	if !result.Shape().Equal(testTensor.Shape()) {
		t.Errorf("Shape mismatch: expected %v, got %v", testTensor.Shape(), result.Shape())
	}

	if result.DataType() != testTensor.DataType() {
		t.Errorf("DataType mismatch: expected %v, got %v", testTensor.DataType(), result.DataType())
	}

	for i := 0; i < testTensor.Size(); i++ {
		expected := testTensor.At(i)
		actual := result.At(i)
		if expected != actual {
			t.Errorf("Value mismatch at index %d: expected %v, got %v", i, expected, actual)
		}
	}
}

func TestGobMarshallerSlice(t *testing.T) {
	// Create gob marshaller and unmarshaller directly
	m := gob.NewMarshaller()
	u := gob.NewUnmarshaller()

	// Create a test slice
	testSlice := []float32{1.0, 2.0, 3.0, 4.0, 5.0}

	// Marshal
	var buf bytes.Buffer
	err := m.Marshal(&buf, testSlice)
	if err != nil {
		t.Fatalf("Failed to marshal slice: %v", err)
	}

	// Unmarshal
	var result []float32
	err = u.Unmarshal(&buf, &result)
	if err != nil {
		t.Fatalf("Failed to unmarshal slice: %v", err)
	}

	// Verify
	if len(result) != len(testSlice) {
		t.Fatalf("Length mismatch: expected %d, got %d", len(testSlice), len(result))
	}

	for i, expected := range testSlice {
		if result[i] != expected {
			t.Errorf("Value mismatch at index %d: expected %v, got %v", i, expected, result[i])
		}
	}
}

func TestDirectCreation(t *testing.T) {
	// Test that we can create marshallers directly
	textMarshaller := text.NewMarshaller()
	if textMarshaller == nil {
		t.Error("Failed to create text marshaller")
	}
	if textMarshaller.Format() != "text" {
		t.Errorf("Expected format 'text', got %q", textMarshaller.Format())
	}

	gobMarshaller := gob.NewMarshaller()
	if gobMarshaller == nil {
		t.Error("Failed to create gob marshaller")
	}
	if gobMarshaller.Format() != "gob" {
		t.Errorf("Expected format 'gob', got %q", gobMarshaller.Format())
	}

	gobUnmarshaller := gob.NewUnmarshaller()
	if gobUnmarshaller == nil {
		t.Error("Failed to create gob unmarshaller")
	}
	if gobUnmarshaller.Format() != "gob" {
		t.Errorf("Expected format 'gob', got %q", gobUnmarshaller.Format())
	}
}
