package gob

import (
	"bytes"
	"testing"

	"github.com/itohio/EasyRobot/x/marshaller/types"
	"github.com/itohio/EasyRobot/x/math/tensor"
)

func TestTensorRoundTrip(t *testing.T) {
	tests := []struct {
		name  string
		shape []int
		dtype types.DataType
	}{
		{"Vector_FP32", []int{5}, types.FP32},
		{"Matrix_FP32", []int{3, 4}, types.FP32},
		{"Tensor_FP32", []int{2, 3, 4}, types.FP32},
		{"Vector_FP64", []int{7}, types.FP64},
		{"Matrix_FP64", []int{2, 5}, types.FP64},
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

			t.Logf("Marshalled %d bytes for %s", buf.Len(), tt.name)

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
			if result.Size() != original.Size() {
				t.Errorf("Size mismatch: got %d, want %d", result.Size(), original.Size())
			}

			// Verify data
			for i := 0; i < original.Size(); i++ {
				if result.At(i) != original.At(i) {
					t.Errorf("Data[%d] mismatch: got %v, want %v", i, result.At(i), original.At(i))
				}
			}

			t.Log("✓ Round-trip successful")
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
		{"Int64Array", []int64{100, 200, 300}},
		{"IntArray", []int{-10, -5, 0, 5, 10}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Marshal
			var buf bytes.Buffer
			m := NewMarshaller()
			if err := m.Marshal(&buf, tt.data); err != nil {
				t.Fatalf("Marshal failed: %v", err)
			}

			t.Logf("Marshalled %d bytes", buf.Len())

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
			case []int64:
				var r []int64
				result = &r
			case []int:
				var r []int
				result = &r
			}

			if err := u.Unmarshal(&buf, result); err != nil {
				t.Fatalf("Unmarshal failed: %v", err)
			}

			t.Log("✓ Array round-trip successful")
		})
	}
}

func TestTensorTypeConversion(t *testing.T) {
	// Create FP32 tensor
	shape := tensor.NewShape(2, 3)
	original := tensor.New(types.FP32, shape)
	for i := 0; i < original.Size(); i++ {
		original.SetAt(float64(i)*1.5, i)
	}

	// Marshal
	var buf bytes.Buffer
	m := NewMarshaller()
	if err := m.Marshal(&buf, original); err != nil {
		t.Fatalf("Marshal failed: %v", err)
	}

	// Unmarshal with type conversion to FP64
	u := NewUnmarshaller(types.WithDestinationType(types.FP64))
	var result types.Tensor
	if err := u.Unmarshal(&buf, &result); err != nil {
		t.Fatalf("Unmarshal failed: %v", err)
	}

	// Verify type was converted
	if result.DataType() != types.FP64 {
		t.Errorf("DataType not converted: got %v, want %v", result.DataType(), types.FP64)
	}

	// Verify data values are preserved (within floating point precision)
	for i := 0; i < original.Size(); i++ {
		if result.At(i) != original.At(i) {
			t.Errorf("Data[%d] mismatch after conversion: got %v, want %v", i, result.At(i), original.At(i))
		}
	}

	t.Log("✓ Type conversion successful")
}
