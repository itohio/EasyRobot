package marshaller_test

import (
	"bytes"
	"fmt"

	"github.com/itohio/EasyRobot/x/marshaller"
	"github.com/itohio/EasyRobot/x/math/tensor"
)

// Example_textMarshaller demonstrates the text marshaller with a tensor.
func Example_textMarshaller() {
	// Create a text marshaller
	m, err := marshaller.NewMarshaller("text")
	if err != nil {
		panic(err)
	}

	// Create a test tensor
	t := tensor.New(tensor.DTFP32, tensor.NewShape(3, 4, 5))
	t.Fill(nil, 1.5)

	// Marshal to text
	var buf bytes.Buffer
	if err := m.Marshal(&buf, t); err != nil {
		panic(err)
	}

	fmt.Print(buf.String())
	// Output: Tensor(shape=[3 4 5], dtype=FP32, size=60)
}

// Example_gobMarshaller demonstrates round-trip marshalling with gob.
func Example_gobMarshaller() {
	// Create marshaller and unmarshaller
	m, err := marshaller.NewMarshaller("gob")
	if err != nil {
		panic(err)
	}

	u, err := marshaller.NewUnmarshaller("gob")
	if err != nil {
		panic(err)
	}

	// Create a test tensor with some data
	original := tensor.New(tensor.DTFP32, tensor.NewShape(2, 3))
	for i := 0; i < original.Size(); i++ {
		original.SetAt(float64(i)*0.5, i)
	}

	// Marshal
	var buf bytes.Buffer
	if err := m.Marshal(&buf, original); err != nil {
		panic(err)
	}

	// Unmarshal
	var restored tensor.Tensor
	if err := u.Unmarshal(&buf, &restored); err != nil {
		panic(err)
	}

	// Verify shapes match
	fmt.Printf("Original shape: %v\n", original.Shape())
	fmt.Printf("Restored shape: %v\n", restored.Shape())
	fmt.Printf("Data matches: %v\n", checkTensorsEqual(original, restored))

	// Output:
	// Original shape: [2 3]
	// Restored shape: [2 3]
	// Data matches: true
}

func checkTensorsEqual(a, b tensor.Tensor) bool {
	if !a.Shape().Equal(b.Shape()) {
		return false
	}
	for i := 0; i < a.Size(); i++ {
		if a.At(i) != b.At(i) {
			return false
		}
	}
	return true
}
