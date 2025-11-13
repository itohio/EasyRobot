package layers

import (
	"testing"

	"github.com/itohio/EasyRobot/pkg/core/math/nn/types"
	"github.com/itohio/EasyRobot/pkg/core/math/tensor"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestConv2D_WeightBiasInitializationAndCloning tests the complete workflow:
// 1. Create layer
// 2. Initialize with specific biases (element by element)
// 3. Initialize with specific weights (element by element)
// 4. Check forward computes correctly
// 5. Clone weights and biases
// 6. Create layer with WithKernels and WithBiases using cloned values
// 7. Check forward computes correctly
// 8. Create another layer
// 9. Set weights and biases using SetWeight/SetBias with cloned values
// 10. Check forward computes correctly
func TestConv2D_WeightBiasInitializationAndCloning(t *testing.T) {
	inChannels := 2
	outChannels := 2
	kernelH := 2
	kernelW := 2
	strideH := 1
	strideW := 1
	padH := 0
	padW := 0
	inputShape := []int{1, inChannels, 4, 4} // [batch, channels, height, width]
	inputData := []float32{
		// Channel 0: 4x4
		1.0, 2.0, 3.0, 4.0,
		5.0, 6.0, 7.0, 8.0,
		9.0, 10.0, 11.0, 12.0,
		13.0, 14.0, 15.0, 16.0,
		// Channel 1: 4x4
		17.0, 18.0, 19.0, 20.0,
		21.0, 22.0, 23.0, 24.0,
		25.0, 26.0, 27.0, 28.0,
		29.0, 30.0, 31.0, 32.0,
	}

	// Expected output shape: [1, 2, 3, 3] (batch=1, outChannels=2, height=3, width=3)

	// Step 1: Create layer
	conv1, err := NewConv2D(inChannels, outChannels, kernelH, kernelW, strideH, strideW, padH, padW,
		WithCanLearn(true), UseBias(true))
	require.NoError(t, err)

	// Step 2: Initialize with specific biases (element by element)
	bias1 := conv1.Bias()
	require.NotNil(t, bias1, "Bias should exist")
	biasParam1, ok := conv1.Base.Parameter(types.ParamBiases)
	require.True(t, ok, "Bias parameter should exist")
	biasParam1.Data.SetAt(0.1, 0)
	biasParam1.Data.SetAt(0.2, 1)
	conv1.Base.SetParam(types.ParamBiases, biasParam1)

	// Step 3: Initialize with specific weights (element by element)
	// Weight shape: [outChannels, inChannels, kernelH, kernelW] = [2, 2, 2, 2]
	weightParam1, ok := conv1.Base.Parameter(types.ParamKernels)
	require.True(t, ok, "Weight parameter should exist")
	// Set weights element by element
	// Output channel 0, Input channel 0
	weightParam1.Data.SetAt(1.0, 0, 0, 0, 0) // outCh=0, inCh=0, h=0, w=0
	weightParam1.Data.SetAt(2.0, 0, 0, 0, 1) // outCh=0, inCh=0, h=0, w=1
	weightParam1.Data.SetAt(3.0, 0, 0, 1, 0) // outCh=0, inCh=0, h=1, w=0
	weightParam1.Data.SetAt(4.0, 0, 0, 1, 1) // outCh=0, inCh=0, h=1, w=1
	// Output channel 0, Input channel 1
	weightParam1.Data.SetAt(5.0, 0, 1, 0, 0) // outCh=0, inCh=1, h=0, w=0
	weightParam1.Data.SetAt(6.0, 0, 1, 0, 1) // outCh=0, inCh=1, h=0, w=1
	weightParam1.Data.SetAt(7.0, 0, 1, 1, 0) // outCh=0, inCh=1, h=1, w=0
	weightParam1.Data.SetAt(8.0, 0, 1, 1, 1) // outCh=0, inCh=1, h=1, w=1
	// Output channel 1, Input channel 0
	weightParam1.Data.SetAt(9.0, 1, 0, 0, 0) // outCh=1, inCh=0, h=0, w=0
	weightParam1.Data.SetAt(10.0, 1, 0, 0, 1) // outCh=1, inCh=0, h=0, w=1
	weightParam1.Data.SetAt(11.0, 1, 0, 1, 0) // outCh=1, inCh=0, h=1, w=0
	weightParam1.Data.SetAt(12.0, 1, 0, 1, 1) // outCh=1, inCh=0, h=1, w=1
	// Output channel 1, Input channel 1
	weightParam1.Data.SetAt(13.0, 1, 1, 0, 0) // outCh=1, inCh=1, h=0, w=0
	weightParam1.Data.SetAt(14.0, 1, 1, 0, 1) // outCh=1, inCh=1, h=0, w=1
	weightParam1.Data.SetAt(15.0, 1, 1, 1, 0) // outCh=1, inCh=1, h=1, w=0
	weightParam1.Data.SetAt(16.0, 1, 1, 1, 1) // outCh=1, inCh=1, h=1, w=1
	conv1.Base.SetParam(types.ParamKernels, weightParam1)

	// Initialize
	err = conv1.Init(inputShape)
	require.NoError(t, err)

	// Step 4: Check if forward computes correctly
	inputTensor := tensor.FromFloat32(tensor.NewShape(inputShape...), inputData)
	output1, err := conv1.Forward(inputTensor)
	require.NoError(t, err)
	require.NotNil(t, output1, "Output should not be nil")
	output1Shape := output1.Shape().ToSlice()
	assert.Equal(t, []int{1, outChannels, 3, 3}, output1Shape, "Output shape should match")

	// Step 5: Clone weights and biases
	clonedWeights := weightParam1.Data.Clone()
	clonedBiases := biasParam1.Data.Clone()

	// Verify cloned values match
	require.Equal(t, weightParam1.Data.Shape().ToSlice(), clonedWeights.Shape().ToSlice(),
		"Cloned weights shape should match")
	require.Equal(t, biasParam1.Data.Shape().ToSlice(), clonedBiases.Shape().ToSlice(),
		"Cloned biases shape should match")

	// Step 6: Create layer with WithKernels and WithBiases using cloned values
	conv2, err := NewConv2D(inChannels, outChannels, kernelH, kernelW, strideH, strideW, padH, padW,
		WithCanLearn(true),
		UseBias(true),
		WithKernels(clonedWeights),
		WithBiases(clonedBiases))
	require.NoError(t, err)

	// Initialize
	err = conv2.Init(inputShape)
	require.NoError(t, err)

	// Step 7: Check if forward computes correctly
	output2, err := conv2.Forward(inputTensor)
	require.NoError(t, err)
	require.NotNil(t, output2, "Output should not be nil")
	output2Shape := output2.Shape().ToSlice()
	assert.Equal(t, []int{1, outChannels, 3, 3}, output2Shape, "Output shape should match")

	// Verify outputs are identical
	output1Data := output1.Data().([]float32)
	output2Data := output2.Data().([]float32)
	require.Len(t, output1Data, len(output2Data), "Output lengths should match")
	for i := range output1Data {
		assert.InDelta(t, output1Data[i], output2Data[i], 1e-5,
			"Output[%d] should match between conv1 and conv2", i)
	}

	// Step 8: Create another layer
	conv3, err := NewConv2D(inChannels, outChannels, kernelH, kernelW, strideH, strideW, padH, padW,
		WithCanLearn(true), UseBias(true))
	require.NoError(t, err)

	// Step 9: Set weights and biases using SetWeight/SetBias with cloned values
	err = conv3.SetWeight(clonedWeights)
	require.NoError(t, err)
	err = conv3.SetBias(clonedBiases)
	require.NoError(t, err)

	// Initialize
	err = conv3.Init(inputShape)
	require.NoError(t, err)

	// Step 10: Check if forward computes correctly
	output3, err := conv3.Forward(inputTensor)
	require.NoError(t, err)
	require.NotNil(t, output3, "Output should not be nil")
	output3Shape := output3.Shape().ToSlice()
	assert.Equal(t, []int{1, outChannels, 3, 3}, output3Shape, "Output shape should match")

	// Verify all outputs are identical
	output3Data := output3.Data().([]float32)
	require.Len(t, output1Data, len(output3Data), "Output lengths should match")
	for i := range output1Data {
		assert.InDelta(t, output1Data[i], output3Data[i], 1e-5,
			"Output[%d] should match between conv1 and conv3", i)
	}
}

// TestConv2D_SetCanLearn_DisablesWeightBiasGradients tests that SetCanLearn(false)
// correctly disables weight and bias gradient computation while still computing input gradients.
func TestConv2D_SetCanLearn_DisablesWeightBiasGradients(t *testing.T) {
	inChannels := 2
	outChannels := 2
	kernelH := 2
	kernelW := 2
	strideH := 1
	strideW := 1
	padH := 0
	padW := 0
	inputShape := []int{1, inChannels, 4, 4}
	inputData := []float32{
		// Channel 0: 4x4
		1.0, 2.0, 3.0, 4.0,
		5.0, 6.0, 7.0, 8.0,
		9.0, 10.0, 11.0, 12.0,
		13.0, 14.0, 15.0, 16.0,
		// Channel 1: 4x4
		17.0, 18.0, 19.0, 20.0,
		21.0, 22.0, 23.0, 24.0,
		25.0, 26.0, 27.0, 28.0,
		29.0, 30.0, 31.0, 32.0,
	}
	gradOutputShape := []int{1, outChannels, 3, 3}
	gradOutputData := []float32{
		// Output channel 0: 3x3
		1.0, 1.0, 1.0,
		1.0, 1.0, 1.0,
		1.0, 1.0, 1.0,
		// Output channel 1: 3x3
		1.0, 1.0, 1.0,
		1.0, 1.0, 1.0,
		1.0, 1.0, 1.0,
	}

	// Step 1: Create layer with learning enabled
	conv, err := NewConv2D(inChannels, outChannels, kernelH, kernelW, strideH, strideW, padH, padW,
		WithCanLearn(true), UseBias(true))
	require.NoError(t, err)

	// Set specific weights and biases (using element-by-element)
	weightParam, ok := conv.Base.Parameter(types.ParamKernels)
	require.True(t, ok, "Weight parameter should exist")
	// Set all weights to 1.0 for simplicity
	for oc := 0; oc < outChannels; oc++ {
		for ic := 0; ic < inChannels; ic++ {
			for h := 0; h < kernelH; h++ {
				for w := 0; w < kernelW; w++ {
					weightParam.Data.SetAt(1.0, oc, ic, h, w)
				}
			}
		}
	}
	conv.Base.SetParam(types.ParamKernels, weightParam)

	biasParam, ok := conv.Base.Parameter(types.ParamBiases)
	require.True(t, ok, "Bias parameter should exist")
	biasParam.Data.SetAt(0.1, 0)
	biasParam.Data.SetAt(0.2, 1)
	conv.Base.SetParam(types.ParamBiases, biasParam)

	// Initialize
	err = conv.Init(inputShape)
	require.NoError(t, err)

	// Verify CanLearn is true
	assert.True(t, conv.CanLearn(), "CanLearn should be true initially")

	// Step 2: Forward pass
	inputTensor := tensor.FromFloat32(tensor.NewShape(inputShape...), inputData)
	output, err := conv.Forward(inputTensor)
	require.NoError(t, err)
	require.NotNil(t, output, "Output should not be nil")

	// Step 3: Backward pass with learning enabled
	conv.ZeroGrad()
	gradOutput := tensor.FromFloat32(tensor.NewShape(gradOutputShape...), gradOutputData)
	gradInput1, err := conv.Backward(gradOutput)
	require.NoError(t, err)
	require.NotNil(t, gradInput1, "GradInput should be computed")

	// Verify weight and bias gradients ARE computed
	weightParam1, ok := conv.Base.Parameter(types.ParamKernels)
	require.True(t, ok, "Weight parameter should exist")
	require.NotNil(t, weightParam1.Grad, "Weight grad should be computed when CanLearn=true")
	weightGradData1 := weightParam1.Grad.Data().([]float32)
	require.Greater(t, len(weightGradData1), 0, "Weight grad should have elements")
	
	// Verify weight gradient is non-zero
	weightGradSum1 := float32(0.0)
	for _, val := range weightGradData1 {
		weightGradSum1 += val
	}
	assert.NotEqual(t, float32(0.0), weightGradSum1, "Weight grad should be non-zero when CanLearn=true")

	biasParam1, ok := conv.Base.Parameter(types.ParamBiases)
	require.True(t, ok, "Bias parameter should exist")
	require.NotNil(t, biasParam1.Grad, "Bias grad should be computed when CanLearn=true")
	biasGradData1 := biasParam1.Grad.Data().([]float32)
	require.Len(t, biasGradData1, outChannels, "Bias grad should have correct size")
	
	// Verify bias gradient is non-zero
	biasGradSum1 := float32(0.0)
	for _, val := range biasGradData1 {
		biasGradSum1 += val
	}
	assert.NotEqual(t, float32(0.0), biasGradSum1, "Bias grad should be non-zero when CanLearn=true")

	// Step 4: Disable learning
	conv.SetCanLearn(false)
	assert.False(t, conv.CanLearn(), "CanLearn should be false after SetCanLearn(false)")

	// Step 5: Forward pass again (needed for backward)
	output2, err := conv.Forward(inputTensor)
	require.NoError(t, err)
	require.NotNil(t, output2, "Output should not be nil")

	// Step 6: Backward pass with learning disabled
	conv.ZeroGrad()
	gradInput2, err := conv.Backward(gradOutput)
	require.NoError(t, err)
	require.NotNil(t, gradInput2, "GradInput should still be computed when CanLearn=false")

	// Verify input gradient is still computed
	gradInput1Data := gradInput1.Data().([]float32)
	gradInput2Data := gradInput2.Data().([]float32)
	require.Len(t, gradInput1Data, len(gradInput2Data), "GradInput lengths should match")
	for i := range gradInput1Data {
		assert.InDelta(t, gradInput1Data[i], gradInput2Data[i], 1e-5,
			"GradInput[%d] should match (input gradients should still be computed)", i)
	}

	// Verify weight gradient is NOT computed (should be nil or zero)
	weightParam2, ok := conv.Base.Parameter(types.ParamKernels)
	require.True(t, ok, "Weight parameter should still exist")
	if weightParam2.Grad != nil && !tensor.IsNil(weightParam2.Grad) {
		weightGradData2 := weightParam2.Grad.Data().([]float32)
		weightGradSum2 := float32(0.0)
		for _, val := range weightGradData2 {
			weightGradSum2 += val
		}
		assert.InDelta(t, float32(0.0), weightGradSum2, 1e-5,
			"Weight grad should be zero when CanLearn=false")
	} else {
		assert.True(t, tensor.IsNil(weightParam2.Grad),
			"Weight grad should be nil/empty when CanLearn=false")
	}

	// Verify bias gradient is NOT computed (should be nil or zero)
	biasParam2, ok := conv.Base.Parameter(types.ParamBiases)
	require.True(t, ok, "Bias parameter should still exist")
	if biasParam2.Grad != nil && !tensor.IsNil(biasParam2.Grad) {
		biasGradData2 := biasParam2.Grad.Data().([]float32)
		biasGradSum2 := float32(0.0)
		for _, val := range biasGradData2 {
			biasGradSum2 += val
		}
		assert.InDelta(t, float32(0.0), biasGradSum2, 1e-5,
			"Bias grad should be zero when CanLearn=false")
	} else {
		assert.True(t, tensor.IsNil(biasParam2.Grad),
			"Bias grad should be nil/empty when CanLearn=false")
	}
}

