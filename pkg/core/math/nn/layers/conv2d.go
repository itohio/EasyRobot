package layers

import (
	"fmt"

	"github.com/itohio/EasyRobot/pkg/core/math/tensor"
	"github.com/itohio/EasyRobot/pkg/core/math/tensor/types"
)

// Conv2D represents a 2D convolution layer.
type Conv2D struct {
	Base
	inChannels  int
	outChannels int
	kernelH     int
	kernelW     int
	strideH     int
	strideW     int
	padH        int
	padW        int
	hasBias     bool
}

// NewConv2D creates a new Conv2D layer.
func NewConv2D(
	inChannels, outChannels, kernelH, kernelW, strideH, strideW, padH, padW int,
	opts ...Option,
) (*Conv2D, error) {
	if inChannels <= 0 {
		return nil, fmt.Errorf("Conv2D: inChannels must be positive, got %d", inChannels)
	}
	if outChannels <= 0 {
		return nil, fmt.Errorf("Conv2D: outChannels must be positive, got %d", outChannels)
	}
	if kernelH <= 0 {
		return nil, fmt.Errorf("Conv2D: kernelH must be positive, got %d", kernelH)
	}
	if kernelW <= 0 {
		return nil, fmt.Errorf("Conv2D: kernelW must be positive, got %d", kernelW)
	}
	if strideH <= 0 {
		return nil, fmt.Errorf("Conv2D: strideH must be positive, got %d", strideH)
	}
	if strideW <= 0 {
		return nil, fmt.Errorf("Conv2D: strideW must be positive, got %d", strideW)
	}

	// Create Base without options first
	base := NewBase("conv2d")

	// Parse options first to get configuration hints (like BiasHint)
	base.ParseOptions(opts...)

	// Determine if bias should be created based on hint
	hasBias := false // Default to no bias
	if hint := base.BiasHint(); hint != nil {
		hasBias = *hint
	}

	conv := &Conv2D{
		Base:        base,
		inChannels:  inChannels,
		outChannels: outChannels,
		kernelH:     kernelH,
		kernelW:     kernelW,
		strideH:     strideH,
		strideW:     strideW,
		padH:        padH,
		padW:        padW,
		hasBias:     hasBias,
	}

	// Set defaults: create kernel parameter
	// Default to FP32, but can be overridden if set via options
	_, hasKernel := conv.Base.Parameter(ParamKernels)
	if !hasKernel {
		kernelData := tensor.New(tensor.DTFP32, tensor.NewShape(outChannels, inChannels, kernelH, kernelW))
		conv.Base.SetParam(ParamKernels, Parameter{
			Data:         kernelData,
			RequiresGrad: conv.Base.CanLearn(),
		})
	}

	// Create bias parameter if needed and not already set via options
	if conv.hasBias {
		_, hasBiasParam := conv.Base.Parameter(ParamBiases)
		if !hasBiasParam {
			// Use kernel's data type for bias
			kernelParam, _ := conv.Base.Parameter(ParamKernels)
			kernelDtype := kernelParam.Data.DataType()
			biasData := tensor.New(kernelDtype, tensor.NewShape(outChannels))
			conv.Base.SetParam(ParamBiases, Parameter{
				Data:         biasData,
				RequiresGrad: conv.Base.CanLearn(),
			})
		}
	}

	// Parse options again after defaults to allow overriding
	conv.Base.ParseOptions(opts...)

	return conv, nil
}

// Init initializes the layer, creating internal computation tensors.
func (c *Conv2D) Init(inputShape []int) error {
	if c == nil {
		return fmt.Errorf("Conv2D.Init: nil layer")
	}

	// Validate input shape: [batch, inChannels, height, width]
	if len(inputShape) != 4 {
		return fmt.Errorf("Conv2D.Init: input must be 4D [batch, inChannels, height, width], got %dD: %v", len(inputShape), inputShape)
	}

	batchSize := inputShape[0]
	if batchSize <= 0 {
		return fmt.Errorf("Conv2D.Init: batch size must be positive, got %d", batchSize)
	}

	inChannels := inputShape[1]
	if inChannels != c.inChannels {
		return fmt.Errorf("Conv2D.Init: input channels %d don't match layer inChannels %d", inChannels, c.inChannels)
	}

	inHeight := inputShape[2]
	inWidth := inputShape[3]

	// Compute output shape: [batch, outChannels, outHeight, outWidth]
	outHeight := (inHeight+2*c.padH-c.kernelH)/c.strideH + 1
	outWidth := (inWidth+2*c.padW-c.kernelW)/c.strideW + 1

	if outHeight <= 0 {
		return fmt.Errorf("Conv2D.Init: invalid output height %d (input height %d, kernel %d, pad %d, stride %d)",
			outHeight, inHeight, c.kernelH, c.padH, c.strideH)
	}
	if outWidth <= 0 {
		return fmt.Errorf("Conv2D.Init: invalid output width %d (input width %d, kernel %d, pad %d, stride %d)",
			outWidth, inWidth, c.kernelW, c.padW, c.strideW)
	}

	outputShape := []int{batchSize, c.outChannels, outHeight, outWidth}
	outputSize := batchSize * c.outChannels * outHeight * outWidth
	c.Base.AllocOutput(outputShape, outputSize)

	return nil
}

// Forward computes the forward pass using tensor.Conv2D.
func (c *Conv2D) Forward(input types.Tensor) (types.Tensor, error) {
	if c == nil {
		return nil, fmt.Errorf("Conv2D.Forward: nil layer")
	}

	if tensor.IsNil(input) {
		return nil, fmt.Errorf("Conv2D.Forward: empty input")
	}

	// Store input
	c.Base.StoreInput(input)

	// Get pre-allocated output tensor
	output := c.Base.Output()
	if tensor.IsNil(output) {
		return nil, fmt.Errorf("Conv2D.Forward: output not allocated, must call Init first")
	}

	// Compute convolution using tensor.Conv2DTo with pre-allocated output
	kernelParam, ok := c.Base.Parameter(ParamKernels)
	if !ok {
		return nil, fmt.Errorf("Conv2D.Forward: kernel parameter not initialized")
	}
	var biasTensor types.Tensor
	if c.hasBias {
		biasParamVal, ok := c.Base.Parameter(ParamBiases)
		if ok {
			biasTensor = biasParamVal.Data
		}
	}
	output = input.Conv2DTo(kernelParam.Data, biasTensor, output, []int{c.strideH, c.strideW}, []int{c.padH, c.padW})

	// Store output
	c.Base.StoreOutput(output)
	return output, nil
}

// Backward computes gradients w.r.t. input, weight, and bias.
// TODO: Replace manual gradient computation with efficient Tensor API methods.
// For now, this implements gradients using existing Tensor operations where possible.
func (c *Conv2D) Backward(gradOutput types.Tensor) (types.Tensor, error) {
	if c == nil {
		return nil, fmt.Errorf("Conv2D.Backward: nil layer")
	}

	if tensor.IsNil(gradOutput) {
		return nil, fmt.Errorf("Conv2D.Backward: empty gradOutput")
	}

	input := c.Base.Input()
	if tensor.IsNil(input) {
		return nil, fmt.Errorf("Conv2D.Backward: input not stored, must call Forward first")
	}

	// Get kernel parameter
	kernelParam, ok := c.Base.Parameter(ParamKernels)
	if !ok {
		return nil, fmt.Errorf("Conv2D.Backward: kernel parameter not initialized")
	}

	// Compute input gradient using transposed convolution
	// Input gradient = Conv2DTransposed(gradOutput, kernel, stride, padding)
	// Note: For backprop, we need to use the kernel as-is for transposed conv
	var emptyBias types.Tensor
	inputGrad := gradOutput.Conv2DTransposed(kernelParam.Data, emptyBias, []int{c.strideH, c.strideW}, []int{c.padH, c.padW})

	// Compute bias gradient: sum gradOutput over spatial dimensions and batch
	if c.hasBias && c.Base.CanLearn() {
		biasParam, ok := c.Base.Parameter(ParamBiases)
		if ok && biasParam.RequiresGrad {
			if tensor.IsNil(biasParam.Grad) {
				biasParam.Grad = tensor.New(tensor.DTFP32, tensor.NewShape(c.outChannels))
			}

			// Sum over batch, height, and width dimensions for each output channel
			// gradOutput shape: [batch, outChannels, outHeight, outWidth]
			summed := gradOutput.Sum(0, 2, 3) // Sum over batch, height, width -> [outChannels]
			// Copy summed values to bias gradient using optimized Tensor.Copy method
			biasParam.Grad.Copy(summed)
			c.Base.SetParam(ParamBiases, biasParam)
		}
	}

	// Compute kernel gradient using primitive composition
	if c.Base.CanLearn() && kernelParam.RequiresGrad {
		if tensor.IsNil(kernelParam.Grad) {
			kernelParam.Grad = tensor.New(tensor.DTFP32, kernelParam.Data.Shape())
		}

		// Kernel gradient computation using primitives:
		// kernelGrad = (gradOutput ⊛ input) where ⊛ is correlation
		// This is equivalent to: kernelGrad = Im2Col(gradOutput)^T @ Im2Col(input)

		// Convert input and gradOutput to column format
		inputCols := input.Im2Col([]int{c.kernelH, c.kernelW}, []int{c.strideH, c.strideW}, []int{c.padH, c.padW})
		gradOutputCols := gradOutput.Im2Col([]int{c.kernelH, c.kernelW}, []int{c.strideH, c.strideW}, []int{c.padH, c.padW})

		// Compute: kernelGrad = gradOutputCols^T @ inputCols
		// Reshape to proper matrix dimensions for MatMul
		// inputCols shape: [batch*outHeight*outWidth, inChannels*kernelH*kernelW]
		// gradOutputCols shape: [batch*outHeight*outWidth, outChannels]
		// We want: [outChannels, inChannels*kernelH*kernelW] = gradOutputCols^T @ inputCols

		kernelGradMatrix := gradOutputCols.Transpose().MatMul(inputCols)

		// Reshape result to kernel shape: [outChannels, inChannels, kernelH, kernelW]
		kernelGradReshaped := kernelGradMatrix.Reshape(tensor.NewShape(c.outChannels, c.inChannels, c.kernelH, c.kernelW))

		// Copy using optimized Tensor.Copy method
		kernelParam.Grad.Copy(kernelGradReshaped)
		c.Base.SetParam(ParamKernels, kernelParam)
	}

	// Store and return the input gradient computed via transposed convolution
	c.Base.StoreGrad(inputGrad)
	return inputGrad, nil
}

// OutputShape returns the output shape for given input shape.
func (c *Conv2D) OutputShape(inputShape []int) ([]int, error) {
	if c == nil {
		return nil, fmt.Errorf("Conv2D.OutputShape: nil layer")
	}

	if len(inputShape) != 4 {
		return nil, fmt.Errorf("Conv2D.OutputShape: input must be 4D [batch, inChannels, height, width], got %dD: %v", len(inputShape), inputShape)
	}

	batchSize := inputShape[0]
	inChannels := inputShape[1]
	inHeight := inputShape[2]
	inWidth := inputShape[3]

	if inChannels != c.inChannels {
		return nil, fmt.Errorf("Conv2D.OutputShape: input channels %d don't match layer inChannels %d", inChannels, c.inChannels)
	}

	outHeight := (inHeight+2*c.padH-c.kernelH)/c.strideH + 1
	outWidth := (inWidth+2*c.padW-c.kernelW)/c.strideW + 1

	return []int{batchSize, c.outChannels, outHeight, outWidth}, nil
}

// Weight returns the kernel parameter tensor.
func (c *Conv2D) Weight() types.Tensor {
	if c == nil {
		// Return empty tensor instead of nil to match test expectations
		return tensor.Empty(tensor.DTFP32)
	}
	kernelParam := c.Base.Kernels()
	if tensor.IsNil(kernelParam.Data) {
		return tensor.Empty(tensor.DTFP32)
	}
	return kernelParam.Data
}

// Bias returns the bias parameter tensor.
func (c *Conv2D) Bias() types.Tensor {
	if c == nil || !c.hasBias {
		// Return empty tensor instead of nil to match test expectations
		return tensor.Empty(tensor.DTFP32)
	}
	biasParam := c.Base.Biases()
	if tensor.IsNil(biasParam.Data) {
		return tensor.Empty(tensor.DTFP32)
	}
	return biasParam.Data
}

// SetWeight sets the weight parameter tensor.
func (c *Conv2D) SetWeight(weight types.Tensor) error {
	if c == nil {
		return fmt.Errorf("Conv2D.SetWeight: nil layer")
	}
	if tensor.IsNil(weight) {
		return fmt.Errorf("Conv2D.SetWeight: empty weight tensor")
	}
	// Validate shape matches expected
	weightShape := weight.Shape()
	if weightShape.Rank() != 4 {
		return fmt.Errorf("Conv2D.SetWeight: weight must be 4D, got %dD", weightShape.Rank())
	}
	expectedShape := tensor.NewShape(c.outChannels, c.inChannels, c.kernelH, c.kernelW)
	for i, dim := range expectedShape {
		if i >= len(weightShape) || weightShape[i] != dim {
			return fmt.Errorf("Conv2D.SetWeight: weight shape %v doesn't match expected %v",
				weightShape, expectedShape)
		}
	}
	kernelParam, ok := c.Base.Parameter(ParamKernels)
	if !ok {
		return fmt.Errorf("Conv2D.SetWeight: kernel parameter not initialized")
	}
	kernelParam.Data = weight
	c.Base.SetParam(ParamKernels, kernelParam)
	return nil
}

// SetBias sets the bias parameter tensor.
func (c *Conv2D) SetBias(bias types.Tensor) error {
	if c == nil {
		return fmt.Errorf("Conv2D.SetBias: nil layer")
	}
	if !c.hasBias {
		return fmt.Errorf("Conv2D.SetBias: layer has no bias")
	}
	if tensor.IsNil(bias) {
		return fmt.Errorf("Conv2D.SetBias: empty bias tensor")
	}
	// Validate shape matches expected
	biasShape := bias.Shape()
	if biasShape.Rank() != 1 || biasShape[0] != c.outChannels {
		return fmt.Errorf("Conv2D.SetBias: bias shape %v doesn't match expected [%d]",
			biasShape, c.outChannels)
	}
	biasParam, ok := c.Base.Parameter(ParamBiases)
	if !ok {
		return fmt.Errorf("Conv2D.SetBias: bias parameter not initialized")
	}
	biasParam.Data = bias
	c.Base.SetParam(ParamBiases, biasParam)
	return nil
}
