package layers

import (
	"fmt"

	"github.com/itohio/EasyRobot/pkg/core/math/nn/types"
	"github.com/itohio/EasyRobot/pkg/core/math/tensor"
	tensorTypes "github.com/itohio/EasyRobot/pkg/core/math/tensor/types"
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
	// Pre-allocated scratch tensors for backward pass optimization
	gradOutputT        tensorTypes.Tensor // Scratch tensor for gradOutput transpose [outChannels, batch*outHeight*outWidth]
	kernelGradMatrix   tensorTypes.Tensor // Scratch tensor for kernel gradient matrix [outChannels, inChannels*kernelH*kernelW]
	inputGradTmpTensor tensorTypes.Tensor // Scratch tensor for input gradient temporary [batch, inChannels, inHeight, inWidth]
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

	// Parse options again after defaults to allow overriding
	conv.Base.ParseOptions(opts...)

	// Initialize parameters using Parameter.Init()
	// This will use provided kernels/biases if set via options, or initialize with XavierUniform
	dtype := conv.Base.DataType()

	// Initialize kernel parameter
	conv.Base.initParam(types.ParamKernels)
	kernelParam, _ := conv.Base.Parameter(types.ParamKernels)
	// For Conv2D kernels: shape [outChannels, inChannels, kernelH, kernelW]
	// fanIn = inChannels * kernelH * kernelW, fanOut = outChannels
	kernelParam.Init(dtype, tensor.NewShape(outChannels, inChannels, kernelH, kernelW), types.ParamKernels, 0, 0, conv.Base.rng, conv.Base.CanLearn())
	conv.Base.SetParam(types.ParamKernels, kernelParam)

	// Initialize bias parameter if needed
	if conv.hasBias {
		conv.Base.initParam(types.ParamBiases)
		biasParam, _ := conv.Base.Parameter(types.ParamBiases)
		biasParam.Init(dtype, tensor.NewShape(outChannels), types.ParamBiases, 1, outChannels, conv.Base.rng, conv.Base.CanLearn())
		conv.Base.SetParam(types.ParamBiases, biasParam)
	}

	return conv, nil
}

// Init initializes the layer, creating internal computation tensors.
func (c *Conv2D) Init(inputShape tensor.Shape) error {
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

	outputShape := tensor.NewShape(batchSize, c.outChannels, outHeight, outWidth)
	outputSize := batchSize * c.outChannels * outHeight * outWidth
	c.Base.AllocOutput(outputShape, outputSize)

	// Pre-allocate scratch tensors for backward pass to avoid allocations
	// These tensors will be reused across backward passes
	dtype := c.Base.DataType()
	// gradOutputT shape: [outChannels, batch*outHeight*outWidth]
	gradOutputTShape := tensor.NewShape(c.outChannels, batchSize*outHeight*outWidth)
	c.gradOutputT = tensor.New(dtype, gradOutputTShape)
	// kernelGradMatrix shape: [outChannels, inChannels*kernelH*kernelW]
	kernelGradMatrixShape := tensor.NewShape(c.outChannels, c.inChannels*c.kernelH*c.kernelW)
	c.kernelGradMatrix = tensor.New(dtype, kernelGradMatrixShape)

	// Pre-allocate inputGradTmpTensor for backward pass (matches input shape)
	c.inputGradTmpTensor = tensor.New(dtype, inputShape)

	return nil
}

// Forward computes the forward pass using tensor.Conv2D.
func (c *Conv2D) Forward(input tensorTypes.Tensor) (tensorTypes.Tensor, error) {
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
	kernelParam, ok := c.Base.Parameter(types.ParamKernels)
	if !ok || tensor.IsNil(kernelParam.Data) {
		return nil, fmt.Errorf("Conv2D.Forward: kernel parameter not initialized")
	}
	var biasTensor tensorTypes.Tensor
	if c.hasBias {
		biasParamVal, ok := c.Base.Parameter(types.ParamBiases)
		if ok && !tensor.IsNil(biasParamVal.Data) {
			biasTensor = biasParamVal.Data
		}
	}
	output = input.Conv2D(output, kernelParam.Data, biasTensor, []int{c.strideH, c.strideW}, []int{c.padH, c.padW})

	// Store output
	c.Base.StoreOutput(output)
	return output, nil
}

// Backward computes gradients w.r.t. input, weight, and bias.
// TODO: Replace manual gradient computation with efficient Tensor API methods.
// For now, this implements gradients using existing Tensor operations where possible.
func (c *Conv2D) Backward(gradOutput tensorTypes.Tensor) (tensorTypes.Tensor, error) {
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
	kernelParam, ok := c.Base.Parameter(types.ParamKernels)
	if !ok || tensor.IsNil(kernelParam.Data) {
		return nil, fmt.Errorf("Conv2D.Backward: kernel parameter not initialized")
	}

	// Compute input gradient using transposed convolution
	// Input gradient = Conv2DTransposed(gradOutput, kernel, stride, padding)
	// For backward: gradOutput has c.outChannels, we want gradInput with c.inChannels
	// Conv2DTransposed expects kernel [inChannels, outChannels, ...] where:
	//   - inChannels matches gradOutput channels (c.outChannels)
	//   - outChannels is gradInput channels (c.inChannels)
	// So we need kernel [c.outChannels, c.inChannels, ...] which matches our kernel format!
	inputShape := input.Shape()

	// Kernel format: [outChannels, inChannels, kernelH, kernelW]
	// For Conv2DTransposed backward pass:
	//   - Input is gradOutput: [batch, outChannels, ...]
	//   - Output is gradInput: [batch, inChannels, ...]
	//   - Conv2DTransposed expects kernel [inChannels, outChannels, ...] where inChannels matches input
	//   - Since input has outChannels, we need kernel [outChannels, inChannels, ...]
	//   - Our kernel format [outChannels, inChannels, ...] already matches, so no permutation needed!
	kernelTransposed := kernelParam.Data

	var emptyBias tensorTypes.Tensor
	// Use pre-allocated inputGradTmpTensor if available and shape matches
	if tensor.IsNil(c.inputGradTmpTensor) || !c.inputGradTmpTensor.Shape().Equal(inputShape) {
		c.inputGradTmpTensor = tensor.New(gradOutput.DataType(), inputShape)
	}
	inputGradTmp := gradOutput.Conv2DTransposed(c.inputGradTmpTensor, kernelTransposed, emptyBias, []int{c.strideH, c.strideW}, []int{c.padH, c.padW})

	// Reshape to match input shape (transposed conv might not perfectly restore size)
	// Use input's data type for input gradient (for correctness in backward pass)
	// Use Base.Grad() if available, otherwise create new
	inputGrad := c.Base.Grad()
	if tensor.IsNil(inputGrad) || !inputGrad.Shape().Equal(inputShape) {
		inputGrad = tensor.New(input.DataType(), inputShape)
	}
	if inputGradTmp.Size() == inputGrad.Size() {
		// Reshape and copy if sizes match
		inputGradTmpReshaped := inputGradTmp.Reshape(nil, inputShape)
		inputGrad.Copy(inputGradTmpReshaped)
	} else {
		// If sizes don't match, reshape and copy what we can
		// This handles cases where transposed conv output is slightly different
		// Use Reshape to match input shape, then copy
		inputGradTmpReshaped := inputGradTmp.Reshape(nil, inputShape)
		inputGrad.Copy(inputGradTmpReshaped)
	}

	// Compute bias gradient: sum gradOutput over spatial dimensions and batch
	if c.hasBias && c.Base.CanLearn() {
		biasParam, ok := c.Base.Parameter(types.ParamBiases)
		if ok && !tensor.IsNil(biasParam.Data) && biasParam.RequiresGrad {
			if tensor.IsNil(biasParam.Grad) {
				// Use parameter's data type for gradient (matches layer's data type)
				biasParam.Grad = tensor.New(biasParam.Data.DataType(), tensor.NewShape(c.outChannels))
			}

			// Sum over batch, height, and width dimensions for each output channel
			// gradOutput shape: [batch, outChannels, outHeight, outWidth]
			// Sum over dimensions 0 (batch), 2 (height), and 3 (width), keeping dimension 1 (channels)
			summed := gradOutput.Sum(biasParam.Grad, []int{0, 2, 3}) // Sum over batch, height, width -> [outChannels]
			// Copy summed values to bias gradient
			biasParam.Grad = summed
			c.Base.SetParam(types.ParamBiases, biasParam)
		}
	}

	// Compute kernel gradient using primitive composition
	if c.Base.CanLearn() && kernelParam.RequiresGrad {
		if tensor.IsNil(kernelParam.Grad) {
			// Use parameter's data type for gradient (matches layer's data type)
			kernelParam.Grad = tensor.New(kernelParam.Data.DataType(), kernelParam.Data.Shape())
		}

		// Kernel gradient computation using primitives:
		// kernelGrad = (gradOutput ⊛ input) where ⊛ is correlation
		// This is equivalent to: kernelGrad = gradOutput_reshaped^T @ Im2Col(input)
		// where gradOutput_reshaped: [batch*outHeight*outWidth, outChannels]

		// Convert input to column format
		inputCols := input.Im2Col(nil, []int{c.kernelH, c.kernelW}, []int{c.strideH, c.strideW}, []int{c.padH, c.padW})
		// inputCols shape: [batch*outHeight*outWidth, inChannels*kernelH*kernelW]

		// Reshape gradOutput from [batch, outChannels, outHeight, outWidth] to [batch*outHeight*outWidth, outChannels]
		gradOutputShape := gradOutput.Shape()
		batchSize := gradOutputShape[0]
		outHeight := gradOutputShape[2]
		outWidth := gradOutputShape[3]
		gradOutputReshaped := gradOutput.Reshape(nil, tensor.NewShape(batchSize*outHeight*outWidth, c.outChannels))

		// Compute: kernelGrad = gradOutputReshaped^T @ inputCols
		// gradOutputReshaped^T shape: [outChannels, batch*outHeight*outWidth]
		// inputCols shape: [batch*outHeight*outWidth, inChannels*kernelH*kernelW]
		// Result: [outChannels, inChannels*kernelH*kernelW]
		// Use pre-allocated scratch tensors
		// Check if gradOutputT needs resizing (batch size might change)
		expectedGradOutputTShape := tensor.NewShape(c.outChannels, batchSize*outHeight*outWidth)
		if tensor.IsNil(c.gradOutputT) || !c.gradOutputT.Shape().Equal(expectedGradOutputTShape) {
			c.gradOutputT = tensor.New(gradOutputReshaped.DataType(), expectedGradOutputTShape)
		}
		gradOutputT := gradOutputReshaped.Transpose(c.gradOutputT, []int{1, 0})
		kernelGradMatrix := gradOutputT.MatMul(c.kernelGradMatrix, inputCols)

		// Reshape result to kernel shape: [outChannels, inChannels, kernelH, kernelW]
		kernelGradReshaped := kernelGradMatrix.Reshape(nil, tensor.NewShape(c.outChannels, c.inChannels, c.kernelH, c.kernelW))

		// Copy using optimized Tensor.Copy method
		kernelParam.Grad.Copy(kernelGradReshaped)
		c.Base.SetParam(types.ParamKernels, kernelParam)
	}

	// Store and return the input gradient computed via transposed convolution
	c.Base.StoreGrad(inputGrad)
	return inputGrad, nil
}

// OutputShape returns the output shape for given input shape.
func (c *Conv2D) OutputShape(inputShape tensor.Shape) (tensor.Shape, error) {
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

	return tensor.NewShape(batchSize, c.outChannels, outHeight, outWidth), nil
}

// Weight returns the kernel parameter tensor.
func (c *Conv2D) Weight() tensorTypes.Tensor {
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
func (c *Conv2D) Bias() tensorTypes.Tensor {
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
func (c *Conv2D) SetWeight(weight tensorTypes.Tensor) error {
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
	kernelParam, ok := c.Base.Parameter(types.ParamKernels)
	if !ok || tensor.IsNil(kernelParam.Data) {
		return fmt.Errorf("Conv2D.SetWeight: kernel parameter not initialized")
	}
	kernelParam.Data = weight
	c.Base.SetParam(types.ParamKernels, kernelParam)
	return nil
}

// SetBias sets the bias parameter tensor.
func (c *Conv2D) SetBias(bias tensorTypes.Tensor) error {
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
	biasParam, ok := c.Base.Parameter(types.ParamBiases)
	if !ok || tensor.IsNil(biasParam.Data) {
		return fmt.Errorf("Conv2D.SetBias: bias parameter not initialized")
	}
	biasParam.Data = bias
	c.Base.SetParam(types.ParamBiases, biasParam)
	return nil
}
