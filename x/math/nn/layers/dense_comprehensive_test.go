package layers

import (
	"math"
	"testing"

	"github.com/itohio/EasyRobot/x/math/nn/types"
	"github.com/itohio/EasyRobot/x/math/tensor"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestDense_ForwardAccuracy tests forward pass accuracy with known mathematical results
func TestDense_ForwardAccuracy(t *testing.T) {
	tests := []struct {
		name        string
		inFeatures  int
		outFeatures int
		weights     []float32
		biases      []float32
		input       []float32
		expected    []float32
	}{
		{
			name:        "simple_2x1",
			inFeatures:  2,
			outFeatures: 1,
			weights:     []float32{1.0, 2.0}, // [2, 1] matrix
			biases:      []float32{0.0},
			input:       []float32{3.0, 4.0},
			expected:    []float32{11.0}, // 3*1 + 4*2 = 11
		},
		{
			name:        "with_bias",
			inFeatures:  2,
			outFeatures: 1,
			weights:     []float32{1.0, 1.0},
			biases:      []float32{5.0},
			input:       []float32{2.0, 3.0},
			expected:    []float32{10.0}, // 2*1 + 3*1 + 5 = 10
		},
		{
			name:        "matrix_multiply_2x3",
			inFeatures:  2,
			outFeatures: 3,
			weights: []float32{
				1.0, 2.0, 3.0, // Row 0: weights for input 0
				4.0, 5.0, 6.0, // Row 1: weights for input 1
			},
			biases: []float32{0.0, 0.0, 0.0},
			input:  []float32{1.0, 2.0},
			expected: []float32{
				9.0,  // 1*1 + 2*4 = 9
				12.0, // 1*2 + 2*5 = 12
				15.0, // 1*3 + 2*6 = 15
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dense, err := NewDense(tt.inFeatures, tt.outFeatures, WithCanLearn(true))
			require.NoError(t, err)

			// Set weights
			weightTensor := tensor.FromFloat32(tensor.NewShape(tt.inFeatures, tt.outFeatures), tt.weights)
			err = dense.SetWeight(weightTensor)
			require.NoError(t, err)

			// Set biases
			biasTensor := tensor.FromFloat32(tensor.NewShape(tt.outFeatures), tt.biases)
			err = dense.SetBias(biasTensor)
			require.NoError(t, err)

			// Initialize
			err = dense.Init(tensor.NewShape(tt.inFeatures))
			require.NoError(t, err)

			// Forward pass
			inputTensor := tensor.FromFloat32(tensor.NewShape(tt.inFeatures), tt.input)
			output, err := dense.Forward(inputTensor)
			require.NoError(t, err)

			// Verify output
			outputData := output.Data().([]float32)
			require.Len(t, outputData, len(tt.expected), "Output length should match")
			for i := range tt.expected {
				assert.InDelta(t, tt.expected[i], outputData[i], 1e-5,
					"Output[%d] should match expected", i)
			}
		})
	}
}

// TestDense_ForwardBatch tests forward pass with batch processing
func TestDense_ForwardBatch(t *testing.T) {
	dense, err := NewDense(2, 3, WithCanLearn(true))
	require.NoError(t, err)

	// Set simple weights: [[1, 2, 3], [4, 5, 6]]^T (transposed)
	weights := tensor.FromFloat32(tensor.NewShape(2, 3), []float32{
		1.0, 2.0, 3.0,
		4.0, 5.0, 6.0,
	})
	err = dense.SetWeight(weights)
	require.NoError(t, err)

	biases := tensor.FromFloat32(tensor.NewShape(3), []float32{0.0, 0.0, 0.0})
	err = dense.SetBias(biases)
	require.NoError(t, err)

	// Initialize for batch input [batch=2, features=2]
	err = dense.Init(tensor.NewShape(2, 2))
	require.NoError(t, err)

	// Batch input: 2 samples, each with 2 features
	input := tensor.FromFloat32(tensor.NewShape(2, 2), []float32{
		1.0, 2.0, // Sample 1
		3.0, 4.0, // Sample 2
	})

	output, err := dense.Forward(input)
	require.NoError(t, err)

	// Expected output:
	// Sample 1: [1*1+2*4, 1*2+2*5, 1*3+2*6] = [9, 12, 15]
	// Sample 2: [3*1+4*4, 3*2+4*5, 3*3+4*6] = [19, 26, 33]
	expected := []float32{9.0, 12.0, 15.0, 19.0, 26.0, 33.0}
	outputData := output.Data().([]float32)
	require.Len(t, outputData, len(expected), "Output length should match")
	for i := range expected {
		assert.InDelta(t, expected[i], outputData[i], 1e-5,
			"Output[%d] should match expected", i)
	}
}

// TestDense_BackwardAccuracy_Comprehensive tests backward pass with comprehensive cases
func TestDense_BackwardAccuracy_Comprehensive(t *testing.T) {
	tests := []struct {
		name               string
		inFeatures         int
		outFeatures        int
		weights            []float32
		biases             []float32
		input              []float32
		gradOutput         []float32
		expectedWeightGrad []float32
		expectedBiasGrad   []float32
		expectedInputGrad  []float32
	}{
		{
			name:        "simple_gradient",
			inFeatures:  2,
			outFeatures: 1,
			weights:     []float32{1.0, 2.0},
			biases:      []float32{0.0},
			input:       []float32{3.0, 4.0},
			gradOutput:  []float32{1.0},
			// Weight grad = input @ gradOutput = [3, 4] @ [1] = [3, 4]
			expectedWeightGrad: []float32{3.0, 4.0},
			// Bias grad = gradOutput = [1]
			expectedBiasGrad: []float32{1.0},
			// Input grad = weight^T @ gradOutput = [1, 2]^T @ [1] = [1, 2]
			expectedInputGrad: []float32{1.0, 2.0},
		},
		{
			name:        "multi_output",
			inFeatures:  2,
			outFeatures: 3,
			weights: []float32{
				1.0, 2.0, 3.0,
				4.0, 5.0, 6.0,
			},
			biases:     []float32{0.0, 0.0, 0.0},
			input:      []float32{1.0, 2.0},
			gradOutput: []float32{1.0, 1.0, 1.0},
			// Weight grad = input @ gradOutput = [1, 2] @ [1, 1, 1] = [1, 1, 1; 2, 2, 2]
			expectedWeightGrad: []float32{1.0, 1.0, 1.0, 2.0, 2.0, 2.0},
			// Bias grad = gradOutput = [1, 1, 1]
			expectedBiasGrad: []float32{1.0, 1.0, 1.0},
			// Input grad = weight^T @ gradOutput
			// = [1, 4; 2, 5; 3, 6] @ [1, 1, 1] = [1+2+3, 4+5+6] = [6, 15]
			expectedInputGrad: []float32{6.0, 15.0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dense, err := NewDense(tt.inFeatures, tt.outFeatures, WithCanLearn(true))
			require.NoError(t, err)

			// Set weights
			weightTensor := tensor.FromFloat32(tensor.NewShape(tt.inFeatures, tt.outFeatures), tt.weights)
			err = dense.SetWeight(weightTensor)
			require.NoError(t, err)

			// Set biases
			biasTensor := tensor.FromFloat32(tensor.NewShape(tt.outFeatures), tt.biases)
			err = dense.SetBias(biasTensor)
			require.NoError(t, err)

			// Initialize
			err = dense.Init(tensor.NewShape(tt.inFeatures))
			require.NoError(t, err)

			// Forward pass
			inputTensor := tensor.FromFloat32(tensor.NewShape(tt.inFeatures), tt.input)
			_, err = dense.Forward(inputTensor)
			require.NoError(t, err)

			// Zero gradients
			dense.ZeroGrad()

			// Backward pass
			gradOutputTensor := tensor.FromFloat32(tensor.NewShape(tt.outFeatures), tt.gradOutput)
			gradInput, err := dense.Backward(gradOutputTensor)
			require.NoError(t, err)

			// Verify weight gradient
			weightParam, ok := dense.Base.Parameter(types.ParamWeights)
			require.True(t, ok, "Weight parameter should exist")
			require.NotNil(t, weightParam.Grad, "Weight grad should be allocated")
			weightGradData := weightParam.Grad.Data().([]float32)
			require.Len(t, weightGradData, len(tt.expectedWeightGrad), "Weight grad length should match")
			for i := range tt.expectedWeightGrad {
				assert.InDelta(t, tt.expectedWeightGrad[i], weightGradData[i], 1e-5,
					"Weight grad[%d] should match expected", i)
			}

			// Verify bias gradient
			biasParam, ok := dense.Base.Parameter(types.ParamBiases)
			require.True(t, ok, "Bias parameter should exist")
			require.NotNil(t, biasParam.Grad, "Bias grad should be allocated")
			biasGradData := biasParam.Grad.Data().([]float32)
			require.Len(t, biasGradData, len(tt.expectedBiasGrad), "Bias grad length should match")
			for i := range tt.expectedBiasGrad {
				assert.InDelta(t, tt.expectedBiasGrad[i], biasGradData[i], 1e-5,
					"Bias grad[%d] should match expected", i)
			}

			// Verify input gradient
			gradInputData := gradInput.Data().([]float32)
			require.Len(t, gradInputData, len(tt.expectedInputGrad), "GradInput length should match")
			for i := range tt.expectedInputGrad {
				assert.InDelta(t, tt.expectedInputGrad[i], gradInputData[i], 1e-5,
					"GradInput[%d] should match expected", i)
			}
		})
	}
}

// TestDense_NumericalGradientCheck performs numerical gradient checking
func TestDense_NumericalGradientCheck(t *testing.T) {
	// Create a simple dense layer
	dense, err := NewDense(3, 2, WithCanLearn(true))
	require.NoError(t, err)

	// Set random but fixed weights
	weights := tensor.FromFloat32(tensor.NewShape(3, 2), []float32{
		0.5, 0.3,
		0.2, 0.8,
		0.1, 0.9,
	})
	err = dense.SetWeight(weights)
	require.NoError(t, err)

	biases := tensor.FromFloat32(tensor.NewShape(2), []float32{0.1, 0.2})
	err = dense.SetBias(biases)
	require.NoError(t, err)

	// Initialize
	err = dense.Init(tensor.NewShape(3))
	require.NoError(t, err)

	// Fixed input
	input := tensor.FromFloat32(tensor.NewShape(3), []float32{1.0, 2.0, 3.0})

	// Forward pass
	_, err = dense.Forward(input)
	require.NoError(t, err)

	// Create gradOutput
	gradOutput := tensor.FromFloat32(tensor.NewShape(2), []float32{1.0, 1.0})

	// Zero gradients
	dense.ZeroGrad()

	// Backward pass to get analytical gradients
	_, err = dense.Backward(gradOutput)
	require.NoError(t, err)

	// Get analytical gradients
	weightParam, _ := dense.Base.Parameter(types.ParamWeights)
	biasParam, _ := dense.Base.Parameter(types.ParamBiases)

	// Compute numerical gradients for weights
	epsilon := 1e-5
	weightGradData := weightParam.Grad.Data().([]float32)
	numericalWeightGrad := make([]float32, len(weightGradData))

	for i := 0; i < weightParam.Data.Size(); i++ {
		originalVal := weightParam.Data.At(i)

		// Perturb weight
		weightParam.Data.SetAt(originalVal+float64(epsilon), i)
		dense.Base.SetParam(types.ParamWeights, weightParam)

		// Forward pass with perturbed weight
		outputPlus, err := dense.Forward(input)
		require.NoError(t, err)
		lossPlusTensor := outputPlus.Sum(nil, nil) // Sum over all dimensions
		lossPlus := lossPlusTensor.At(0)

		// Restore weight
		weightParam.Data.SetAt(originalVal, i)
		dense.Base.SetParam(types.ParamWeights, weightParam)

		// Forward pass with original weight
		outputOrig, err := dense.Forward(input)
		require.NoError(t, err)
		lossOrigTensor := outputOrig.Sum(nil, nil) // Sum over all dimensions
		lossOrig := lossOrigTensor.At(0)

		// Numerical gradient
		numericalWeightGrad[i] = float32((lossPlus - lossOrig) / epsilon)
	}

	// Compare analytical and numerical gradients for weights
	// Use a more lenient tolerance (5e-3) for numerical gradient checking
	// due to finite difference approximation errors
	for i := range weightGradData {
		analytical := float64(weightGradData[i])
		numerical := float64(numericalWeightGrad[i])
		diff := math.Abs(analytical - numerical)
		relDiff := diff / (math.Abs(numerical) + 1e-8)

		assert.Less(t, relDiff, 5e-3,
			"Weight grad[%d]: analytical=%.6f, numerical=%.6f, diff=%.6f",
			i, analytical, numerical, diff)
	}

	// Compute numerical gradients for biases
	biasGradData := biasParam.Grad.Data().([]float32)
	numericalBiasGrad := make([]float32, len(biasGradData))

	for i := 0; i < biasParam.Data.Size(); i++ {
		originalVal := biasParam.Data.At(i)

		// Perturb bias
		biasParam.Data.SetAt(originalVal+float64(epsilon), i)
		dense.Base.SetParam(types.ParamBiases, biasParam)

		// Forward pass with perturbed bias
		outputPlus, err := dense.Forward(input)
		require.NoError(t, err)
		lossPlusTensor := outputPlus.Sum(nil, nil) // Sum over all dimensions
		lossPlus := lossPlusTensor.At(0)

		// Restore bias
		biasParam.Data.SetAt(originalVal, i)
		dense.Base.SetParam(types.ParamBiases, biasParam)

		// Forward pass with original bias
		outputOrig, err := dense.Forward(input)
		require.NoError(t, err)
		lossOrigTensor := outputOrig.Sum(nil, nil) // Sum over all dimensions
		lossOrig := lossOrigTensor.At(0)

		// Numerical gradient
		numericalBiasGrad[i] = float32((lossPlus - lossOrig) / epsilon)
	}

	// Compare analytical and numerical gradients for biases
	// Use a more lenient tolerance (5e-3) for numerical gradient checking
	// due to finite difference approximation errors
	for i := range biasGradData {
		analytical := float64(biasGradData[i])
		numerical := float64(numericalBiasGrad[i])
		diff := math.Abs(analytical - numerical)
		relDiff := diff / (math.Abs(numerical) + 1e-8)

		assert.Less(t, relDiff, 5e-3,
			"Bias grad[%d]: analytical=%.6f, numerical=%.6f, diff=%.6f",
			i, analytical, numerical, diff)
	}
}

// TestDense_BackwardBatch tests backward pass with batch processing
func TestDense_BackwardBatch(t *testing.T) {
	dense, err := NewDense(2, 3, WithCanLearn(true))
	require.NoError(t, err)

	// Set weights
	weights := tensor.FromFloat32(tensor.NewShape(2, 3), []float32{
		1.0, 2.0, 3.0,
		4.0, 5.0, 6.0,
	})
	err = dense.SetWeight(weights)
	require.NoError(t, err)

	biases := tensor.FromFloat32(tensor.NewShape(3), []float32{0.0, 0.0, 0.0})
	err = dense.SetBias(biases)
	require.NoError(t, err)

	// Initialize for batch
	err = dense.Init(tensor.NewShape(2, 2))
	require.NoError(t, err)

	// Batch input: [2, 2]
	input := tensor.FromFloat32(tensor.NewShape(2, 2), []float32{
		1.0, 2.0,
		3.0, 4.0,
	})

	// Forward pass
	_, err = dense.Forward(input)
	require.NoError(t, err)

	// Zero gradients
	dense.ZeroGrad()

	// Batch gradOutput: [2, 3]
	gradOutput := tensor.FromFloat32(tensor.NewShape(2, 3), []float32{
		1.0, 1.0, 1.0,
		1.0, 1.0, 1.0,
	})

	// Backward pass
	gradInput, err := dense.Backward(gradOutput)
	require.NoError(t, err)

	// Verify weight gradient: should sum over batch dimension
	// For batch, weight grad = sum over batch of (input^T @ gradOutput)
	// Sample 1: [1, 2]^T @ [1, 1, 1] = [1, 1, 1; 2, 2, 2]
	// Sample 2: [3, 4]^T @ [1, 1, 1] = [3, 3, 3; 4, 4, 4]
	// Sum: [4, 4, 4; 6, 6, 6]
	expectedWeightGrad := []float32{4.0, 4.0, 4.0, 6.0, 6.0, 6.0}

	weightParam, _ := dense.Base.Parameter(types.ParamWeights)
	weightGradData := weightParam.Grad.Data().([]float32)
	require.Len(t, weightGradData, len(expectedWeightGrad), "Weight grad length should match")
	for i := range expectedWeightGrad {
		assert.InDelta(t, expectedWeightGrad[i], weightGradData[i], 1e-5,
			"Weight grad[%d] should match expected", i)
	}

	// Bias grad should sum over batch: [1+1, 1+1, 1+1] = [2, 2, 2]
	expectedBiasGrad := []float32{2.0, 2.0, 2.0}
	biasParam, _ := dense.Base.Parameter(types.ParamBiases)
	biasGradData := biasParam.Grad.Data().([]float32)
	require.Len(t, biasGradData, len(expectedBiasGrad), "Bias grad length should match")
	for i := range expectedBiasGrad {
		assert.InDelta(t, expectedBiasGrad[i], biasGradData[i], 1e-5,
			"Bias grad[%d] should match expected", i)
	}

	// Verify input gradient shape
	assert.Equal(t, []int{2, 2}, gradInput.Shape().ToSlice(), "GradInput shape should match input")
}

// TestDense_WeightBiasInitializationAndCloning tests the complete workflow:
// 1. Create layer
// 2. Initialize with specific biases (element by element)
// 3. Initialize with specific weights (element by element)
// 4. Check forward computes correctly
// 5. Clone weights and biases
// 6. Create layer with WithBiases and WithWeights using cloned values
// 7. Check forward computes correctly
// 8. Create another layer
// 9. Set weights and biases using SetWeight/SetBias with cloned values
// 10. Check forward computes correctly
func TestDense_WeightBiasInitializationAndCloning(t *testing.T) {
	inFeatures := 3
	outFeatures := 2
	inputData := []float32{1.0, 2.0, 3.0}
	// Expected output calculation:
	// Input: [1.0, 2.0, 3.0]
	// Weights: [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]] (shape [3, 2])
	// Biases: [1.0, 2.0]
	// Output[0] = 1*1 + 2*3 + 3*5 + 1 = 1 + 6 + 15 + 1 = 23
	// Output[1] = 1*2 + 2*4 + 3*6 + 2 = 2 + 8 + 18 + 2 = 30
	expectedOutput := []float32{23.0, 30.0}

	// Step 1: Create layer
	dense1, err := NewDense(inFeatures, outFeatures, WithCanLearn(true))
	require.NoError(t, err)

	// Step 2: Initialize with specific biases (element by element)
	biasParam1, ok := dense1.Base.Parameter(types.ParamBiases)
	require.True(t, ok, "Bias parameter should exist")
	require.NotNil(t, biasParam1.Data, "Bias data should exist")
	biasParam1.Data.SetAt(1.0, 0)
	biasParam1.Data.SetAt(2.0, 1)
	dense1.Base.SetParam(types.ParamBiases, biasParam1)

	// Step 3: Initialize with specific weights (element by element)
	// Weight matrix: [3, 2]
	// Weights: [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
	weightParam1, ok := dense1.Base.Parameter(types.ParamWeights)
	require.True(t, ok, "Weight parameter should exist")
	require.NotNil(t, weightParam1.Data, "Weight data should exist")
	// Set weights element by element
	weightParam1.Data.SetAt(1.0, 0, 0) // row 0, col 0
	weightParam1.Data.SetAt(2.0, 0, 1) // row 0, col 1
	weightParam1.Data.SetAt(3.0, 1, 0) // row 1, col 0
	weightParam1.Data.SetAt(4.0, 1, 1) // row 1, col 1
	weightParam1.Data.SetAt(5.0, 2, 0) // row 2, col 0
	weightParam1.Data.SetAt(6.0, 2, 1) // row 2, col 1
	dense1.Base.SetParam(types.ParamWeights, weightParam1)

	// Initialize
	err = dense1.Init(tensor.NewShape(inFeatures))
	require.NoError(t, err)

	// Step 4: Check if forward computes correctly
	inputTensor := tensor.FromFloat32(tensor.NewShape(inFeatures), inputData)
	output1, err := dense1.Forward(inputTensor)
	require.NoError(t, err)
	output1Data := output1.Data().([]float32)
	require.Len(t, output1Data, len(expectedOutput), "Output length should match")
	for i := range expectedOutput {
		assert.InDelta(t, expectedOutput[i], output1Data[i], 1e-5,
			"Dense1 Output[%d] should match expected", i)
	}

	// Step 5: Clone weights and biases
	clonedWeights := weightParam1.Data.Clone()
	clonedBiases := biasParam1.Data.Clone()

	// Verify cloned values match
	require.Equal(t, weightParam1.Data.Shape().ToSlice(), clonedWeights.Shape().ToSlice(),
		"Cloned weights shape should match")
	require.Equal(t, biasParam1.Data.Shape().ToSlice(), clonedBiases.Shape().ToSlice(),
		"Cloned biases shape should match")

	// Step 6: Create layer with WithBiases and WithWeights using cloned values
	dense2, err := NewDense(inFeatures, outFeatures,
		WithCanLearn(true),
		WithWeights(clonedWeights),
		WithBiases(clonedBiases))
	require.NoError(t, err)

	// Initialize
	err = dense2.Init(tensor.NewShape(inFeatures))
	require.NoError(t, err)

	// Step 7: Check if forward computes correctly
	output2, err := dense2.Forward(inputTensor)
	require.NoError(t, err)
	output2Data := output2.Data().([]float32)
	require.Len(t, output2Data, len(expectedOutput), "Output length should match")
	for i := range expectedOutput {
		assert.InDelta(t, expectedOutput[i], output2Data[i], 1e-5,
			"Dense2 Output[%d] should match expected", i)
	}

	// Verify outputs are identical
	for i := range output1Data {
		assert.InDelta(t, output1Data[i], output2Data[i], 1e-5,
			"Output[%d] should match between dense1 and dense2", i)
	}

	// Step 8: Create another layer
	dense3, err := NewDense(inFeatures, outFeatures, WithCanLearn(true))
	require.NoError(t, err)

	// Step 9: Set weights and biases using SetWeight/SetBias with cloned values
	err = dense3.SetWeight(clonedWeights)
	require.NoError(t, err)
	err = dense3.SetBias(clonedBiases)
	require.NoError(t, err)

	// Initialize
	err = dense3.Init(tensor.NewShape(inFeatures))
	require.NoError(t, err)

	// Step 10: Check if forward computes correctly
	output3, err := dense3.Forward(inputTensor)
	require.NoError(t, err)
	output3Data := output3.Data().([]float32)
	require.Len(t, output3Data, len(expectedOutput), "Output length should match")
	for i := range expectedOutput {
		assert.InDelta(t, expectedOutput[i], output3Data[i], 1e-5,
			"Dense3 Output[%d] should match expected", i)
	}

	// Verify all outputs are identical
	for i := range output1Data {
		assert.InDelta(t, output1Data[i], output3Data[i], 1e-5,
			"Output[%d] should match between dense1 and dense3", i)
	}
}

// TestDense_SetCanLearn_DisablesWeightBiasGradients tests that SetCanLearn(false)
// correctly disables weight and bias gradient computation while still computing input gradients.
func TestDense_SetCanLearn_DisablesWeightBiasGradients(t *testing.T) {
	inFeatures := 2
	outFeatures := 2
	inputData := []float32{1.0, 2.0}
	gradOutputData := []float32{1.0, 1.0}

	// Step 1: Create layer with learning enabled
	dense, err := NewDense(inFeatures, outFeatures, WithCanLearn(true))
	require.NoError(t, err)

	// Set specific weights and biases
	weights := tensor.FromFloat32(tensor.NewShape(inFeatures, outFeatures), []float32{
		1.0, 2.0,
		3.0, 4.0,
	})
	err = dense.SetWeight(weights)
	require.NoError(t, err)

	biases := tensor.FromFloat32(tensor.NewShape(outFeatures), []float32{0.1, 0.2})
	err = dense.SetBias(biases)
	require.NoError(t, err)

	// Initialize
	err = dense.Init(tensor.NewShape(inFeatures))
	require.NoError(t, err)

	// Verify CanLearn is true
	assert.True(t, dense.CanLearn(), "CanLearn should be true initially")

	// Step 2: Forward pass
	inputTensor := tensor.FromFloat32(tensor.NewShape(inFeatures), inputData)
	output, err := dense.Forward(inputTensor)
	require.NoError(t, err)
	require.NotNil(t, output, "Output should not be nil")

	// Step 3: Backward pass with learning enabled
	dense.ZeroGrad()
	gradOutput := tensor.FromFloat32(tensor.NewShape(outFeatures), gradOutputData)
	gradInput1, err := dense.Backward(gradOutput)
	require.NoError(t, err)
	require.NotNil(t, gradInput1, "GradInput should be computed")

	// Verify weight and bias gradients ARE computed
	weightParam1, ok := dense.Base.Parameter(types.ParamWeights)
	require.True(t, ok, "Weight parameter should exist")
	require.NotNil(t, weightParam1.Grad, "Weight grad should be computed when CanLearn=true")
	weightGradData1 := weightParam1.Grad.Data().([]float32)
	require.Len(t, weightGradData1, inFeatures*outFeatures, "Weight grad should have correct size")

	// Verify weight gradient is non-zero
	weightGradSum1 := float32(0.0)
	for _, val := range weightGradData1 {
		weightGradSum1 += val
	}
	assert.NotEqual(t, float32(0.0), weightGradSum1, "Weight grad should be non-zero when CanLearn=true")

	biasParam1, ok := dense.Base.Parameter(types.ParamBiases)
	require.True(t, ok, "Bias parameter should exist")
	require.NotNil(t, biasParam1.Grad, "Bias grad should be computed when CanLearn=true")
	biasGradData1 := biasParam1.Grad.Data().([]float32)
	require.Len(t, biasGradData1, outFeatures, "Bias grad should have correct size")

	// Verify bias gradient is non-zero
	biasGradSum1 := float32(0.0)
	for _, val := range biasGradData1 {
		biasGradSum1 += val
	}
	assert.NotEqual(t, float32(0.0), biasGradSum1, "Bias grad should be non-zero when CanLearn=true")

	// Step 4: Disable learning
	dense.SetCanLearn(false)
	assert.False(t, dense.CanLearn(), "CanLearn should be false after SetCanLearn(false)")

	// Step 5: Forward pass again (needed for backward)
	output2, err := dense.Forward(inputTensor)
	require.NoError(t, err)
	require.NotNil(t, output2, "Output should not be nil")

	// Step 6: Backward pass with learning disabled
	// Zero gradients first to clear previous gradients
	dense.ZeroGrad()
	gradInput2, err := dense.Backward(gradOutput)
	require.NoError(t, err)
	require.NotNil(t, gradInput2, "GradInput should still be computed when CanLearn=false")

	// Verify input gradient is still computed and correct
	gradInput1Data := gradInput1.Data().([]float32)
	gradInput2Data := gradInput2.Data().([]float32)
	require.Len(t, gradInput1Data, len(gradInput2Data), "GradInput lengths should match")
	for i := range gradInput1Data {
		assert.InDelta(t, gradInput1Data[i], gradInput2Data[i], 1e-5,
			"GradInput[%d] should match (input gradients should still be computed)", i)
	}

	// Verify weight gradient is NOT computed (should be nil or zero)
	weightParam2, ok := dense.Base.Parameter(types.ParamWeights)
	require.True(t, ok, "Weight parameter should still exist")
	// When CanLearn=false, weight gradient should not be allocated
	if weightParam2.Grad != nil && !tensor.IsNil(weightParam2.Grad) {
		// If gradient tensor exists, it should be zero (not updated)
		weightGradData2 := weightParam2.Grad.Data().([]float32)
		weightGradSum2 := float32(0.0)
		for _, val := range weightGradData2 {
			weightGradSum2 += val
		}
		assert.InDelta(t, float32(0.0), weightGradSum2, 1e-5,
			"Weight grad should be zero when CanLearn=false")
	} else {
		// Gradient tensor should be nil/empty when CanLearn=false
		assert.True(t, tensor.IsNil(weightParam2.Grad),
			"Weight grad should be nil/empty when CanLearn=false")
	}

	// Verify bias gradient is NOT computed (should be nil or zero)
	biasParam2, ok := dense.Base.Parameter(types.ParamBiases)
	require.True(t, ok, "Bias parameter should still exist")
	// When CanLearn=false, bias gradient should not be allocated
	if biasParam2.Grad != nil && !tensor.IsNil(biasParam2.Grad) {
		// If gradient tensor exists, it should be zero (not updated)
		biasGradData2 := biasParam2.Grad.Data().([]float32)
		biasGradSum2 := float32(0.0)
		for _, val := range biasGradData2 {
			biasGradSum2 += val
		}
		assert.InDelta(t, float32(0.0), biasGradSum2, 1e-5,
			"Bias grad should be zero when CanLearn=false")
	} else {
		// Gradient tensor should be nil/empty when CanLearn=false
		assert.True(t, tensor.IsNil(biasParam2.Grad),
			"Bias grad should be nil/empty when CanLearn=false")
	}
}
