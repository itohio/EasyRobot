package learn_test

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/itohio/EasyRobot/x/math/learn"
	"github.com/itohio/EasyRobot/x/math/nn"
	"github.com/itohio/EasyRobot/x/math/nn/layers"
	"github.com/itohio/EasyRobot/x/math/nn/types"
	"github.com/itohio/EasyRobot/x/math/tensor"
)

// oneHot creates a one-hot encoded tensor for a label.
// Returns shape [1, numClasses] to match model output.
func oneHot(label int, numClasses int) tensor.Tensor {
	data := make([]float32, numClasses)
	data[label] = 1.0
	return tensor.FromFloat32(tensor.NewShape(1, numClasses), data)
}
func TestTrainStep_NilInputs(t *testing.T) {
	// Create a simple model for testing
	denseLayer, err := layers.NewDense(2, 1, layers.WithCanLearn(true))
	require.NoError(t, err)
	model, err := nn.NewSequentialModelBuilder(tensor.NewShape(2)).
		AddLayer(denseLayer).
		AddLayer(layers.NewSigmoid("sigmoid")).
		Build()
	require.NoError(t, err)
	require.NoError(t, model.Init(tensor.NewShape(2)))

	optimizer := learn.NewSGD(0.1)
	lossFn := nn.NewMSE()

	input := tensor.FromFloat32(tensor.NewShape(2), []float32{1, 0})
	target := tensor.FromFloat32(tensor.NewShape(1), []float32{1})

	// Test nil layer
	_, err = learn.TrainStep(nil, optimizer, lossFn, input, target)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "nil layer")

	// Test nil optimizer
	_, err = learn.TrainStep(model, nil, lossFn, input, target)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "nil optimizer")

	// Test nil loss function
	_, err = learn.TrainStep(model, optimizer, nil, input, target)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "nil loss function")
}

func TestTrainStep_EmptyTensors(t *testing.T) {
	// Create a simple model for testing
	denseLayer, err := layers.NewDense(2, 1, layers.WithCanLearn(true))
	require.NoError(t, err)
	model, err := nn.NewSequentialModelBuilder(tensor.NewShape(2)).
		AddLayer(denseLayer).
		AddLayer(layers.NewSigmoid("sigmoid")).
		Build()
	require.NoError(t, err)
	require.NoError(t, model.Init(tensor.NewShape(2)))

	optimizer := learn.NewSGD(0.1)
	lossFn := nn.NewMSE()

	emptyTensor := tensor.FromFloat32(tensor.NewShape(0), []float32{})
	validInput := tensor.FromFloat32(tensor.NewShape(2), []float32{1, 0})
	validTarget := tensor.FromFloat32(tensor.NewShape(1), []float32{1})

	// Test empty input
	_, err = learn.TrainStep(model, optimizer, lossFn, emptyTensor, validTarget)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "empty input")

	// Test empty target
	_, err = learn.TrainStep(model, optimizer, lossFn, validInput, emptyTensor)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "empty target")
}

func TestTrainStep_SimpleTraining(t *testing.T) {
	// Create a simple linear model: y = wx + b
	denseLayer, err := layers.NewDense(1, 1, layers.WithCanLearn(true))
	require.NoError(t, err)
	model, err := nn.NewSequentialModelBuilder(tensor.NewShape(1)).
		AddLayer(denseLayer).
		Build()
	require.NoError(t, err)
	require.NoError(t, model.Init(tensor.NewShape(1)))

	optimizer := learn.NewSGD(0.1)
	lossFn := nn.NewMSE()

	// Set initial weights to [2] and bias to [1], so y = 2x + 1
	params := model.Parameters()
	// For a dense layer, parameters are typically "0:0" for weights and "0:1" for bias
	var weightParam, biasParam types.Parameter
	for _, param := range params {
		if len(param.Data.Shape()) == 2 { // weights are 2D
			weightParam = param
		} else if len(param.Data.Shape()) == 1 { // bias is 1D
			biasParam = param
		}
	}

	// Dense layer has weight matrix [out_features, in_features] and bias [out_features]
	// For our case: weight [1, 1], bias [1]
	require.Greater(t, weightParam.Data.Size(), 0, "Weight param should have data")
	require.Greater(t, biasParam.Data.Size(), 0, "Bias param should have data")
	weightParam.Data.SetAt(2.0, 0, 0) // weight
	biasParam.Data.SetAt(1.0, 0)      // bias

	// Train on a simple example: x=1, y=3 (since 2*1 + 1 = 3)
	input := tensor.FromFloat32(tensor.NewShape(1), []float32{1})
	target := tensor.FromFloat32(tensor.NewShape(1), []float32{3})

	// Before training, get initial loss
	output, err := model.Forward(input)
	require.NoError(t, err)
	initialLoss, err := lossFn.Compute(output, target)
	require.NoError(t, err)

	// Perform one training step
	finalLoss, err := learn.TrainStep(model, optimizer, lossFn, input, target)
	require.NoError(t, err)

	// Loss should be the same as computed initially (convert to same type for comparison)
	assert.InDelta(t, float64(initialLoss), finalLoss, 1e-6)

	// After one training step, the model should have updated its parameters
	// Since it's a simple case with perfect initialization, loss should decrease
	// Let's do another forward pass to check if the output improved
	outputAfter, err := model.Forward(input)
	require.NoError(t, err)
	lossAfter, err := lossFn.Compute(outputAfter, target)
	require.NoError(t, err)

	// Loss should have decreased (or at least not increased significantly)
	assert.LessOrEqual(t, lossAfter, initialLoss)
}

func TestTrainStep_MultipleSteps(t *testing.T) {
	// Create a simple model
	denseLayer, err := layers.NewDense(1, 1, layers.WithCanLearn(true))
	require.NoError(t, err)
	model, err := nn.NewSequentialModelBuilder(tensor.NewShape(1)).
		AddLayer(denseLayer).
		Build()
	require.NoError(t, err)
	require.NoError(t, model.Init(tensor.NewShape(1)))

	optimizer := learn.NewSGD(0.01) // Small learning rate
	lossFn := nn.NewMSE()

	// Initialize with wrong weights
	params := model.Parameters()
	var weightParam, biasParam types.Parameter
	for _, param := range params {
		if len(param.Data.Shape()) == 2 { // weights are 2D
			weightParam = param
		} else if len(param.Data.Shape()) == 1 { // bias is 1D
			biasParam = param
		}
	}
	weightParam.Data.SetAt(0.0, 0, 0) // weight = 0 (should be 2)
	biasParam.Data.SetAt(0.0, 0)      // bias = 0 (should be 1)

	// Training data: y = 2x + 1
	trainData := []struct {
		input  tensor.Tensor
		target tensor.Tensor
	}{
		{tensor.FromFloat32(tensor.NewShape(1), []float32{0}), tensor.FromFloat32(tensor.NewShape(1), []float32{1})}, // 2*0 + 1 = 1
		{tensor.FromFloat32(tensor.NewShape(1), []float32{1}), tensor.FromFloat32(tensor.NewShape(1), []float32{3})}, // 2*1 + 1 = 3
		{tensor.FromFloat32(tensor.NewShape(1), []float32{2}), tensor.FromFloat32(tensor.NewShape(1), []float32{5})}, // 2*2 + 1 = 5
	}

	// Train for several steps
	initialLoss := float32(0)
	for _, data := range trainData {
		output, err := model.Forward(data.input)
		require.NoError(t, err)
		loss, err := lossFn.Compute(output, data.target)
		require.NoError(t, err)
		initialLoss += loss
	}
	initialLoss /= float32(len(trainData))

	// Train for 100 steps
	for step := 0; step < 100; step++ {
		for _, data := range trainData {
			_, err := learn.TrainStep(model, optimizer, lossFn, data.input, data.target)
			require.NoError(t, err)
		}
	}

	// Check final loss
	finalLoss := float32(0)
	for _, data := range trainData {
		output, err := model.Forward(data.input)
		require.NoError(t, err)
		loss, err := lossFn.Compute(output, data.target)
		require.NoError(t, err)
		finalLoss += loss
	}
	finalLoss /= float32(len(trainData))

	// Loss should have decreased significantly
	assert.Less(t, finalLoss, initialLoss*0.1, "Loss should decrease significantly after training")
}
