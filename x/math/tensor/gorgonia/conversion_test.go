package gorgonia

import (
	"testing"

	"github.com/itohio/EasyRobot/x/math/tensor/eager_tensor"
	"github.com/itohio/EasyRobot/x/math/tensor/types"
	"github.com/stretchr/testify/assert"
)

func TestToEagerTensor(t *testing.T) {
	// Create a gorgonia tensor
	gTensor := New(types.FP32, 2, 3)
	for i := 0; i < gTensor.Size(); i++ {
		gTensor.SetAt(float64(i+1), i)
	}

	// Convert to eager tensor
	eTensor := gTensor.ToEagerTensor()

	// Verify shape matches
	assert.True(t, gTensor.Shape().Equal(eTensor.Shape()), "Shapes should match")

	// Verify data matches
	for i := 0; i < gTensor.Size(); i++ {
		assert.InDelta(t, gTensor.At(i), eTensor.At(i), 0.001, "Data should match")
	}

	// Verify data type
	assert.Equal(t, types.FP32, eTensor.DataType(), "Data type should be FP32")

	// Verify independence - modifying gorgonia tensor shouldn't affect eager tensor
	gTensor.SetAt(999.0, 0)
	assert.NotEqual(t, 999.0, eTensor.At(0), "Tensors should be independent")
}

func TestFromEagerTensor(t *testing.T) {
	// Create an eager tensor
	eTensor := eager_tensor.New(types.FP32, types.NewShape(2, 3))
	for i := 0; i < eTensor.Size(); i++ {
		eTensor.SetAt(float64(i+1), i)
	}

	// Convert to gorgonia tensor
	gTensor := FromEagerTensor(eTensor)

	// Verify shape matches
	assert.True(t, eTensor.Shape().Equal(gTensor.Shape()), "Shapes should match")

	// Verify data matches
	for i := 0; i < eTensor.Size(); i++ {
		assert.InDelta(t, eTensor.At(i), gTensor.At(i), 0.001, "Data should match")
	}

	// Verify data type
	assert.Equal(t, types.FP32, gTensor.DataType(), "Data type should be FP32")
}

func TestRoundtripConversion(t *testing.T) {
	// Create a gorgonia tensor
	original := New(types.FP32, 2, 3)
	for i := 0; i < original.Size(); i++ {
		original.SetAt(float64(i+1), i)
	}

	// Convert to eager and back
	eager := original.ToEagerTensor()
	converted := FromEagerTensor(eager)

	// Verify shape and data match
	assert.True(t, original.Shape().Equal(converted.Shape()), "Shapes should match after roundtrip")

	for i := 0; i < original.Size(); i++ {
		assert.InDelta(t, original.At(i), converted.At(i), 0.001, "Data should match after roundtrip")
	}
}

func TestConversionDifferentTypes(t *testing.T) {
	testCases := []struct {
		name string
		dt   types.DataType
	}{
		{"FP32", types.FP32},
		{"FP64", types.FP64},
		// INT types not yet fully supported in eager_tensor At() method
		// {"INT32", types.INT32},
		// {"INT64", types.INT64},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Create gorgonia tensor with specific type
			gTensor := New(tc.dt, 2, 3)
			gTensor.Fill(nil, 5.0)

			// Convert to eager
			eTensor := gTensor.ToEagerTensor()

			// Verify data type preserved
			assert.Equal(t, tc.dt, eTensor.DataType(), "Data type should be preserved")

			// Verify values
			for i := 0; i < gTensor.Size(); i++ {
				assert.InDelta(t, 5.0, eTensor.At(i), 0.001, "Values should match")
			}
		})
	}
}

func TestCopyFromEagerTensor(t *testing.T) {
	// Create an eager tensor
	eTensor := eager_tensor.New(types.FP32, types.NewShape(2, 3))
	for i := 0; i < eTensor.Size(); i++ {
		eTensor.SetAt(float64(i+1), i)
	}

	// Create a gorgonia tensor
	gTensor := New(types.FP32, 2, 3)

	// Copy from eager to gorgonia
	gTensor.Copy(eTensor)

	// Verify data matches
	for i := 0; i < gTensor.Size(); i++ {
		assert.InDelta(t, eTensor.At(i), gTensor.At(i), 0.001, "Data should match after Copy")
	}

	// Verify independence - modifying eager shouldn't affect gorgonia
	eTensor.SetAt(999.0, 0)
	assert.NotEqual(t, 999.0, gTensor.At(0), "Tensors should be independent after Copy")
}

func TestCopyBetweenGorgoniaTensors(t *testing.T) {
	// Create two gorgonia tensors
	src := New(types.FP32, 2, 3)
	for i := 0; i < src.Size(); i++ {
		src.SetAt(float64(i+10), i)
	}

	dst := New(types.FP32, 2, 3)

	// Copy
	dst.Copy(src)

	// Verify data matches
	for i := 0; i < dst.Size(); i++ {
		assert.InDelta(t, src.At(i), dst.At(i), 0.001, "Data should match after Copy")
	}

	// Verify independence
	src.SetAt(999.0, 0)
	assert.NotEqual(t, 999.0, dst.At(0), "Tensors should be independent after Copy")
}

func TestCopyWithTypeConversion(t *testing.T) {
	// Create an FP64 eager tensor
	eTensor := eager_tensor.New(types.FP64, types.NewShape(2, 3))
	for i := 0; i < eTensor.Size(); i++ {
		eTensor.SetAt(float64(i+1), i)
	}

	// Create an FP32 gorgonia tensor
	gTensor := New(types.FP32, 2, 3)

	// Copy with automatic type conversion
	gTensor.Copy(eTensor)

	// Verify data matches (with type conversion)
	for i := 0; i < gTensor.Size(); i++ {
		assert.InDelta(t, eTensor.At(i), gTensor.At(i), 0.001, "Data should match after Copy with conversion")
	}
}

func TestCopyShapeMismatch(t *testing.T) {
	// Create tensors with different shapes
	eTensor := eager_tensor.New(types.FP32, types.NewShape(2, 3))
	gTensor := New(types.FP32, 3, 2)

	// Copy should panic with shape mismatch
	assert.Panics(t, func() {
		gTensor.Copy(eTensor)
	}, "Copy should panic on shape mismatch")
}
