package layers_test

import (
	"testing"

	"github.com/itohio/EasyRobot/x/math/nn/layers"
	"github.com/itohio/EasyRobot/x/math/primitive/fp32"
	"github.com/itohio/EasyRobot/x/math/tensor"
	"github.com/itohio/EasyRobot/x/math/tensor/eager_tensor"
	"github.com/itohio/EasyRobot/x/math/tensor/types"
)

func TestAvgPoolLayerBackwardMatchesExpected(t *testing.T) {
	conv, err := layers.NewConv2D(1, 1, 1, 1, 1, 1, 0, 0, layers.WithCanLearn(true), layers.UseBias(false))
	if err != nil {
		t.Fatalf("create conv: %v", err)
	}
	if err := conv.Init(tensor.NewShape(1, 1, 3, 3)); err != nil {
		t.Fatalf("init conv: %v", err)
	}

	avgPool, err := layers.NewAvgPool2D(2, 2, 2, 2, 1, 1)
	if err != nil {
		t.Fatalf("create avg pool: %v", err)
	}
	convOutShape, err := conv.OutputShape(tensor.NewShape(1, 1, 3, 3))
	if err != nil {
		t.Fatalf("conv output shape: %v", err)
	}
	if err := avgPool.Init(convOutShape); err != nil {
		t.Fatalf("init avg pool: %v", err)
	}

	weight := tensor.FromFloat32(tensor.NewShape(1, 1, 1, 1), []float32{1})
	if err := conv.SetWeight(weight); err != nil {
		t.Fatalf("set conv weight: %v", err)
	}

	input := tensor.FromFloat32(tensor.NewShape(1, 1, 3, 3), []float32{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	})

	convOutput, err := conv.Forward(input)
	if err != nil {
		t.Fatalf("conv forward: %v", err)
	}
	t.Logf("conv output type: %T", convOutput)
	t.Logf("conv output: %v", convOutput.Data())
	if _, err := avgPool.Forward(convOutput); err != nil {
		t.Fatalf("avgpool forward: %v", err)
	}
	t.Logf("avgpool output: %v", avgPool.Base.Output().Data())

	gradOutput := eager_tensor.FromFloat32(types.NewShape(1, 1, 2, 2), []float32{1, 1, 1, 1})
	baselineInput := eager_tensor.FromFloat32(types.NewShape(1, 1, 3, 3), []float32{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	})
	baselineGrad := baselineInput.AvgPool2DBackward(nil, gradOutput, []int{2, 2}, []int{2, 2}, []int{1, 1})
	t.Logf("baseline grad: %v", baselineGrad.Data())
	assertTensorClose(t, baselineGrad, []float32{
		1, 0.5, 0.5,
		0.5, 0.25, 0.25,
		0.5, 0.25, 0.25,
	}, 1e-6)

	manualInput := eager_tensor.FromFloat32(types.NewShape(1, 1, 3, 3), []float32{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	})
	t.Logf("manual input type: %T", manualInput)
	directGrad := convOutput.AvgPool2DBackward(nil, gradOutput, []int{2, 2}, []int{2, 2}, []int{1, 1})
	t.Logf("gradOutput data after direct call: %v", gradOutput.Data())
	manualGrad := manualInput.AvgPool2DBackward(nil, gradOutput, []int{2, 2}, []int{2, 2}, []int{1, 1})
	t.Logf("gradOutput data: %v", gradOutput.Data())
	t.Logf("direct grad: %v", directGrad.Data())
	t.Logf("manual grad: %v", manualGrad.Data())

	primitiveGrad := make([]float32, 9)
	fp32.AvgPool2DBackward(
		primitiveGrad,
		[]float32{1, 1, 1, 1},
		1, 1, 3, 3,
		2, 2,
		2, 2,
		2, 2,
		1, 1,
	)
	t.Logf("primitive grad: %v", primitiveGrad)

	gradAfterPool, err := avgPool.Backward(gradOutput)
	if err != nil {
		t.Fatalf("avgpool backward: %v", err)
	}

	expected := []float32{
		1, 0.5, 0.5,
		0.5, 0.25, 0.25,
		0.5, 0.25, 0.25,
	}
	assertTensorClose(t, directGrad, expected, 1e-6)
	assertTensorClose(t, gradAfterPool, expected, 1e-6)

	gradInput, err := conv.Backward(gradAfterPool)
	if err != nil {
		t.Fatalf("conv backward: %v", err)
	}
	assertTensorClose(t, gradInput, expected, 1e-6)
}

func TestMaxPoolLayerBackwardMatchesExpected(t *testing.T) {
	conv, err := layers.NewConv2D(1, 1, 1, 1, 1, 1, 0, 0, layers.WithCanLearn(true), layers.UseBias(false))
	if err != nil {
		t.Fatalf("create conv: %v", err)
	}
	if err := conv.Init(tensor.NewShape(1, 1, 4, 4)); err != nil {
		t.Fatalf("init conv: %v", err)
	}

	maxPool, err := layers.NewMaxPool2D(2, 2, 2, 2, 0, 0)
	if err != nil {
		t.Fatalf("create max pool: %v", err)
	}
	convOutShape, err := conv.OutputShape(tensor.NewShape(1, 1, 4, 4))
	if err != nil {
		t.Fatalf("conv output shape: %v", err)
	}
	if err := maxPool.Init(convOutShape); err != nil {
		t.Fatalf("init max pool: %v", err)
	}

	weight := tensor.FromFloat32(tensor.NewShape(1, 1, 1, 1), []float32{1})
	if err := conv.SetWeight(weight); err != nil {
		t.Fatalf("set conv weight: %v", err)
	}

	input := tensor.FromFloat32(tensor.NewShape(1, 1, 4, 4), []float32{
		1, 2, 3, 4,
		5, 9, 8, 7,
		0, 6, 5, 4,
		3, 2, 1, 0,
	})

	convOutput, err := conv.Forward(input)
	if err != nil {
		t.Fatalf("conv forward: %v", err)
	}
	if _, err := maxPool.Forward(convOutput); err != nil {
		t.Fatalf("maxpool forward: %v", err)
	}

	gradOutput := tensor.FromFloat32(tensor.NewShape(1, 1, 2, 2), []float32{1, 1, 1, 1})
	gradAfterPool, err := maxPool.Backward(gradOutput)
	if err != nil {
		t.Fatalf("maxpool backward: %v", err)
	}

	expected := []float32{
		0, 0, 0, 0,
		0, 1, 1, 0,
		0, 1, 1, 0,
		0, 0, 0, 0,
	}
	assertTensorClose(t, gradAfterPool, expected, 1e-6)

	gradInput, err := conv.Backward(gradAfterPool)
	if err != nil {
		t.Fatalf("conv backward: %v", err)
	}
	assertTensorClose(t, gradInput, expected, 1e-6)
}

func assertTensorClose(t *testing.T, tensor tensor.Tensor, expected []float32, tol float64) {
	t.Helper()
	data, ok := tensor.Data().([]float32)
	if !ok {
		t.Fatalf("unexpected tensor data type %T", tensor.Data())
	}
	if len(data) != len(expected) {
		t.Fatalf("data length %d differs from expected %d", len(data), len(expected))
	}
	for i, got := range data {
		want := expected[i]
		if diff := float64(got - want); diff > tol || diff < -tol {
			t.Fatalf("value mismatch at %d: got %f want %f", i, got, want)
		}
	}
}
