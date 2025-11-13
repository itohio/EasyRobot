package json

import (
	"bytes"
	"testing"

	"github.com/itohio/EasyRobot/x/marshaller/types"
	"github.com/itohio/EasyRobot/x/math/learn"
	"github.com/itohio/EasyRobot/x/math/nn"
	"github.com/itohio/EasyRobot/x/math/nn/layers"
	"github.com/itohio/EasyRobot/x/math/tensor"
)

func TestXORModelParametersRoundTrip(t *testing.T) {
	// Train a simple XOR model
	X := [][]float32{
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
	}
	Y := [][]float32{
		{0},
		{1},
		{1},
		{0},
	}

	// Build model using builder pattern
	dense1, err := layers.NewDense(2, 4, layers.UseBias(true))
	if err != nil {
		t.Fatalf("Failed to create dense1: %v", err)
	}

	dense2, err := layers.NewDense(4, 1, layers.UseBias(true))
	if err != nil {
		t.Fatalf("Failed to create dense2: %v", err)
	}

	model, err := nn.NewSequentialModelBuilder(tensor.NewShape(2)).
		AddLayer(dense1).
		AddLayer(layers.NewReLU("relu")).
		AddLayer(dense2).
		AddLayer(layers.NewSigmoid("sigmoid")).
		Build()
	if err != nil {
		t.Fatalf("Failed to build model: %v", err)
	}

	// Initialize model
	if err := model.Init(tensor.NewShape(2)); err != nil {
		t.Fatalf("Failed to initialize model: %v", err)
	}

	// Train briefly
	lossFunc := nn.NewMSE()
	opt := learn.NewSGD(0.1)

	epochs := 100
	for epoch := 0; epoch < epochs; epoch++ {
		for i := range X {
			xTensor := tensor.FromArray(tensor.NewShape(2), X[i])
			yTensor := tensor.FromArray(tensor.NewShape(1), Y[i])
			if _, err := learn.TrainStep(model, opt, lossFunc, xTensor, yTensor); err != nil {
				t.Fatalf("Training failed at epoch %d: %v", epoch, err)
			}
		}
	}

	// Get trained parameters
	params := model.Parameters()
	if len(params) == 0 {
		t.Fatal("Model has no parameters")
	}
	t.Logf("Model has %d parameters", len(params))

	// Marshal and unmarshal each parameter to verify data integrity
	for idx, param := range params {
		if param.Data.Empty() {
			continue
		}

		// Marshal parameter data
		var buf bytes.Buffer
		m := NewMarshaller()
		if err := m.Marshal(&buf, param.Data); err != nil {
			t.Fatalf("Marshal parameter %d failed: %v", idx, err)
		}

		// Unmarshal parameter data
		u := NewUnmarshaller()
		var restored types.Tensor
		if err := u.Unmarshal(&buf, &restored); err != nil {
			t.Fatalf("Unmarshal parameter %d failed: %v", idx, err)
		}

		// Verify shape and data integrity
		if len(restored.Shape()) != len(param.Data.Shape()) {
			t.Fatalf("Param %d: Shape length mismatch: got %d, want %d", idx, len(restored.Shape()), len(param.Data.Shape()))
		}
		for i := range restored.Shape() {
			if restored.Shape()[i] != param.Data.Shape()[i] {
				t.Errorf("Param %d: Shape[%d] mismatch: got %d, want %d", idx, i, restored.Shape()[i], param.Data.Shape()[i])
			}
		}

		// Verify all data values match exactly
		for i := 0; i < param.Data.Size(); i++ {
			if restored.At(i) != param.Data.At(i) {
				t.Errorf("Param %d: Data[%d] mismatch: got %v, want %v", idx, i, restored.At(i), param.Data.At(i))
			}
		}

		t.Logf("✓ Parameter %d successfully round-tripped with %d values", idx, param.Data.Size())
	}

	// Verify model still computes correctly after we've marshalled its parameters
	testInput := tensor.FromArray(tensor.NewShape(2), []float32{1, 0})
	output, err := model.Forward(testInput)
	if err != nil {
		t.Fatalf("Forward pass failed: %v", err)
	}

	result := output.At(0)
	t.Logf("XOR(1,0) = %v (expected ~1.0)", result)

	// Should be close to 1.0 for XOR(1,0)
	if result < 0.7 || result > 1.3 {
		t.Logf("Warning: Model prediction may not be fully converged: got %v, expected ~1.0", result)
	}
}
