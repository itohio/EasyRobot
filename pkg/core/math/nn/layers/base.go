package layers

import (
	"fmt"

	"github.com/itohio/EasyRobot/pkg/core/math/nn"
	"github.com/itohio/EasyRobot/pkg/core/math/tensor"
)

// ParamIndex represents a typed parameter index for layers
type ParamIndex int

// Standard parameter indices
const (
	ParamWeights ParamIndex = 1
	ParamBiases  ParamIndex = 2
	ParamKernels ParamIndex = 3
	ParamCustom  ParamIndex = 100
)

// Option represents a configuration option for layers.
type Option func(*Base)

// Base provides common layer functionality that can be embedded by all layers.
// Layers embedding Base should implement Forward and Backward directly.
type Base struct {
	name      string
	canLearn  bool
	input     tensor.Tensor
	output    tensor.Tensor
	grad      tensor.Tensor                // Gradient tensor for backward pass
	params    map[ParamIndex]*nn.Parameter // Parameters map (new)
	paramsOld []nn.Parameter               // Old slice-based params (for backward compatibility)
}

// NewBase creates a new Base layer with options.
func NewBase(opts ...Option) Base {
	b := Base{
		name:      "",
		canLearn:  false,
		params:    make(map[ParamIndex]*nn.Parameter),
		paramsOld: []nn.Parameter{},
	}

	// Apply options
	for _, opt := range opts {
		opt(&b)
	}

	return b
}

// Option implementations

// WithName returns an Option that sets the layer name.
func WithName(name string) Option {
	return func(b *Base) {
		b.name = name
	}
}

// WithCanLearn returns an Option that sets whether the layer computes gradients.
func WithCanLearn(canLearn bool) Option {
	return func(b *Base) {
		b.canLearn = canLearn
	}
}

// WithWeights returns an Option that sets the weight parameter at ParamWeights index.
func WithWeights(weight tensor.Tensor) Option {
	return func(b *Base) {
		b.initParam(ParamWeights)
		b.params[ParamWeights].Data = weight
		b.params[ParamWeights].RequiresGrad = b.canLearn
	}
}

// WithBiases returns an Option that sets the bias parameter at ParamBiases index.
func WithBiases(bias tensor.Tensor) Option {
	return func(b *Base) {
		b.initParam(ParamBiases)
		b.params[ParamBiases].Data = bias
		b.params[ParamBiases].RequiresGrad = b.canLearn
	}
}

// WithKernels returns an Option that sets the kernel parameter at ParamKernels index.
func WithKernels(kernel tensor.Tensor) Option {
	return func(b *Base) {
		b.initParam(ParamKernels)
		b.params[ParamKernels].Data = kernel
		b.params[ParamKernels].RequiresGrad = b.canLearn
	}
}

// WithParameter returns an Option that sets a parameter at the given index.
func WithParameter(idx ParamIndex, param nn.Parameter) Option {
	return func(b *Base) {
		b.initParam(idx)
		*b.params[idx] = param
		b.params[idx].RequiresGrad = b.canLearn
	}
}

// WithParameters returns an Option that sets multiple parameters.
// Copies as many parameters as fit into Base.params.
func WithParameters(params map[ParamIndex]nn.Parameter) Option {
	return func(b *Base) {
		// Copy as many parameters as fit
		for idx, param := range params {
			b.initParam(idx)
			*b.params[idx] = param
			b.params[idx].RequiresGrad = b.canLearn
		}
	}
}

// Helper to initialize a parameter if it doesn't exist
func (b *Base) initParam(idx ParamIndex) {
	if b.params[idx] == nil {
		b.params[idx] = &nn.Parameter{}
	}
}

// Name returns the name of this layer.
func (b *Base) Name() string {
	if b == nil {
		return ""
	}
	return b.name
}

// SetName sets the name of this layer.
func (b *Base) SetName(name string) {
	if b == nil {
		return
	}
	b.name = name
}

// SetDefaultName sets a default name based on layer type and shape.
func (b *Base) SetDefaultName(layerType string, shape []int) {
	if b == nil {
		return
	}
	if b.name != "" {
		return // Don't override explicitly set name
	}
	// Generate name from layer type and shape
	if len(shape) > 0 {
		shapeStr := fmt.Sprintf("%v", shape)
		b.name = fmt.Sprintf("%s_%s", layerType, shapeStr)
	} else {
		b.name = layerType
	}
}

// CanLearn returns whether this layer computes gradients.
func (b *Base) CanLearn() bool {
	if b == nil {
		return false
	}
	return b.canLearn
}

// SetCanLearn sets whether this layer computes gradients.
func (b *Base) SetCanLearn(canLearn bool) {
	if b == nil {
		return
	}
	b.canLearn = canLearn
}

// Input returns the input tensor from the last Forward pass.
func (b *Base) Input() tensor.Tensor {
	if b == nil {
		return tensor.Tensor{}
	}
	return b.input
}

// Output returns the output tensor from the last Forward pass.
func (b *Base) Output() tensor.Tensor {
	if b == nil {
		return tensor.Tensor{}
	}
	return b.output
}

// Grad returns the gradient tensor from the last Backward pass.
func (b *Base) Grad() tensor.Tensor {
	if b == nil {
		return tensor.Tensor{}
	}
	return b.grad
}

// setInput stores the input tensor (internal method).
func (b *Base) setInput(input tensor.Tensor) {
	if b == nil {
		return
	}
	b.input = input
}

// setOutput stores the output tensor (internal method).
func (b *Base) setOutput(output tensor.Tensor) {
	if b == nil {
		return
	}
	b.output = output
}

// setGrad stores the gradient tensor (internal method).
func (b *Base) setGrad(grad tensor.Tensor) {
	if b == nil {
		return
	}
	b.grad = grad
}

// AllocOutput allocates the output tensor with the given shape and size.
func (b *Base) AllocOutput(shape []int, size int) {
	if b == nil {
		return
	}
	b.output = tensor.Tensor{
		Dim:  make([]int, len(shape)),
		Data: make([]float32, size),
	}
	copy(b.output.Dim, shape)
}

// AllocGrad allocates the gradient tensor with the given shape and size.
func (b *Base) AllocGrad(shape []int, size int) {
	if b == nil {
		return
	}
	b.grad = tensor.Tensor{
		Dim:  make([]int, len(shape)),
		Data: make([]float32, size),
	}
	copy(b.grad.Dim, shape)
}

// Helper method for layers to store input during Forward.
func (b *Base) StoreInput(input tensor.Tensor) {
	b.setInput(input)
}

// Helper method for layers to store output during Forward.
func (b *Base) StoreOutput(output tensor.Tensor) {
	b.setOutput(output)
}

// Helper method for layers to store grad during Backward.
func (b *Base) StoreGrad(grad tensor.Tensor) {
	b.setGrad(grad)
}

// InitParams initializes the old parameter array with the given number of parameters.
// This is for backward compatibility. Layers should migrate to using params map.
func (b *Base) InitParams(numParams int) {
	if b == nil {
		return
	}
	b.paramsOld = make([]nn.Parameter, numParams)
}

// Parameter returns the parameter at the given index from old slice (backward compatibility).
func (b *Base) Parameter(idx int) *nn.Parameter {
	if b == nil || idx < 0 || idx >= len(b.paramsOld) {
		return nil
	}
	return &b.paramsOld[idx]
}

// Parameters returns all parameters from old slice (backward compatibility).
// Also includes parameters from params map.
func (b *Base) Parameters() []*nn.Parameter {
	if b == nil {
		return nil
	}

	// Start with old slice params
	result := make([]*nn.Parameter, len(b.paramsOld))
	for i := range b.paramsOld {
		result[i] = &b.paramsOld[i]
	}

	// Add map params (for indices >= ParamCustom, or if not in slice)
	for idx, param := range b.params {
		if param != nil {
			if int(idx) >= int(ParamCustom) || int(idx) >= len(b.paramsOld) {
				// Add new params that don't fit in slice
				result = append(result, param)
			} else if int(idx) < len(b.paramsOld) {
				// Override slice param with map param if both exist
				result[int(idx)] = param
			}
		}
	}

	if len(result) == 0 {
		return nil
	}
	return result
}

// SetParameters sets all parameters from old slice (backward compatibility).
func (b *Base) SetParameters(params []nn.Parameter) error {
	if b == nil {
		return fmt.Errorf("Base.SetParameters: nil layer")
	}
	if len(params) != len(b.paramsOld) {
		return fmt.Errorf("Base.SetParameters: expected %d parameters, got %d", len(b.paramsOld), len(params))
	}
	b.paramsOld = make([]nn.Parameter, len(params))
	copy(b.paramsOld, params)
	return nil
}

// ZeroGrad zeros all parameter gradients (both old slice and new map).
func (b *Base) ZeroGrad() {
	if b == nil {
		return
	}
	// Zero old slice params
	for i := range b.paramsOld {
		b.paramsOld[i].ZeroGrad()
	}
	// Zero map params
	for _, param := range b.params {
		if param != nil {
			param.ZeroGrad()
		}
	}
}

// GetParam returns a parameter from the map at the given index.
func (b *Base) GetParam(idx ParamIndex) *nn.Parameter {
	if b == nil {
		return nil
	}
	return b.params[idx]
}

// SetParam sets a parameter in the map at the given index.
func (b *Base) SetParam(idx ParamIndex, param *nn.Parameter) {
	if b == nil {
		return
	}
	b.initParam(idx)
	b.params[idx] = param
}

// Weights returns the weights parameter.
func (b *Base) Weights() *nn.Parameter {
	return b.GetParam(ParamWeights)
}

// Biases returns the biases parameter.
func (b *Base) Biases() *nn.Parameter {
	return b.GetParam(ParamBiases)
}

// Kernels returns the kernels parameter.
func (b *Base) Kernels() *nn.Parameter {
	return b.GetParam(ParamKernels)
}
