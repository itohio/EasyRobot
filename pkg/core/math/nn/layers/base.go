package layers

import (
	"fmt"
	"sync/atomic"

	"github.com/itohio/EasyRobot/pkg/core/math/tensor"
)

var layerCounter int64

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
	name     string
	nameSet  bool // Whether name was explicitly set
	prefix   string
	canLearn bool
	biasHint *bool // Optional bias hint for layers that support it
	input    tensor.Tensor
	output   tensor.Tensor
	grad     tensor.Tensor            // Gradient tensor for backward pass
	params   map[ParamIndex]Parameter // Parameters map
	layerIdx int64                    // Unique layer index assigned at creation
}

// NewBase creates a new Base layer with options.
func NewBase(prefix string, opts ...Option) Base {
	// Increment and capture global layer counter value for this layer
	layerIdx := atomic.AddInt64(&layerCounter, 1)

	b := Base{
		name:     "",
		prefix:   prefix,
		canLearn: false,
		params:   make(map[ParamIndex]Parameter),
		layerIdx: layerIdx,
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
		b.nameSet = true
	}
}

// WithCanLearn returns an Option that sets whether the layer computes gradients.
func WithCanLearn(canLearn bool) Option {
	return func(b *Base) {
		b.canLearn = canLearn
	}
}

// WithBias returns an Option that sets whether the layer uses bias.
// This is a hint for layers that support optional bias (like Conv1D, Conv2D).
// Layers that don't support optional bias will ignore this option.
func WithBias(hasBias bool) Option {
	return func(b *Base) {
		b.biasHint = &hasBias
	}
}

// WithWeights returns an Option that sets the weight parameter at ParamWeights index.
func WithWeights(weight tensor.Tensor) Option {
	return func(b *Base) {
		b.initParam(ParamWeights)
		param := b.params[ParamWeights]
		param.Data = weight
		param.RequiresGrad = b.canLearn
		b.params[ParamWeights] = param
	}
}

// WithBiases returns an Option that sets the bias parameter at ParamBiases index.
func WithBiases(bias tensor.Tensor) Option {
	return func(b *Base) {
		b.initParam(ParamBiases)
		param := b.params[ParamBiases]
		param.Data = bias
		param.RequiresGrad = b.canLearn
		b.params[ParamBiases] = param
	}
}

// WithKernels returns an Option that sets the kernel parameter at ParamKernels index.
func WithKernels(kernel tensor.Tensor) Option {
	return func(b *Base) {
		b.initParam(ParamKernels)
		param := b.params[ParamKernels]
		param.Data = kernel
		param.RequiresGrad = b.canLearn
		b.params[ParamKernels] = param
	}
}

// WithParameter returns an Option that sets a parameter at the given index.
func WithParameter(idx ParamIndex, param Parameter) Option {
	return func(b *Base) {
		b.initParam(idx)
		param.RequiresGrad = b.canLearn
		b.params[idx] = param
	}
}

// WithParameters returns an Option that sets multiple parameters.
// Copies as many parameters as fit into Base.params.
func WithParameters(params map[ParamIndex]Parameter) Option {
	return func(b *Base) {
		// Copy as many parameters as fit
		for idx, param := range params {
			b.initParam(idx)
			param.RequiresGrad = b.canLearn
			b.params[idx] = param
		}
	}
}

// Helper to initialize a parameter if it doesn't exist
func (b *Base) initParam(idx ParamIndex) {
	if _, ok := b.params[idx]; !ok {
		b.params[idx] = Parameter{}
	}
}

// Name returns the name of this layer.
// If name was explicitly set (even to empty), returns that name.
// Otherwise, automatically generates a default name as {prefix}_{layer_idx}_{shape}
// if prefix is set, or {layer_idx}_{shape} if prefix is empty.
func (b *Base) Name() string {
	if b == nil {
		return ""
	}
	if b.nameSet {
		return b.name
	}
	// Generate default name from prefix, layer_idx and shape
	var nameFormat string
	if b.prefix != "" {
		if b.output.Shape().Rank() > 0 {
			shapeStr := fmt.Sprintf("%v", b.output.Shape().ToSlice())
			nameFormat = fmt.Sprintf("%s_%d_%s", b.prefix, b.layerIdx, shapeStr)
		} else {
			nameFormat = fmt.Sprintf("%s_%d", b.prefix, b.layerIdx)
		}
	} else {
		if b.output.Shape().Rank() > 0 {
			shapeStr := fmt.Sprintf("%v", b.output.Shape().ToSlice())
			nameFormat = fmt.Sprintf("%d_%s", b.layerIdx, shapeStr)
		} else {
			nameFormat = fmt.Sprintf("%d", b.layerIdx)
		}
	}
	return nameFormat
}

// SetName sets the name of this layer.
func (b *Base) SetName(name string) {
	if b == nil {
		return
	}
	b.name = name
	b.nameSet = name != ""
}

// BiasHint returns the bias hint if set, otherwise nil.
func (b *Base) BiasHint() *bool {
	if b == nil {
		return nil
	}
	return b.biasHint
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
	b.output = *tensor.FromFloat32(tensor.NewShape(shape...), make([]float32, size))
}

// AllocGrad allocates the gradient tensor with the given shape and size.
func (b *Base) AllocGrad(shape []int, size int) {
	if b == nil {
		return
	}
	b.grad = *tensor.FromFloat32(tensor.NewShape(shape...), make([]float32, size))
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

// Parameter returns the parameter at the given index and whether it exists.
func (b *Base) Parameter(idx ParamIndex) (Parameter, bool) {
	if b == nil {
		return Parameter{}, false
	}
	param, ok := b.params[idx]
	return param, ok
}

// Parameters returns all parameters from the params map.
// Returns map[ParamIndex]Parameter where values are Parameter structs (not pointers).
func (b *Base) Parameters() map[ParamIndex]Parameter {
	if b == nil {
		return nil
	}

	result := make(map[ParamIndex]Parameter)
	for idx, param := range b.params {
		result[idx] = param
	}
	if len(result) == 0 {
		return nil
	}
	return result
}

// SetParameters sets all parameters in the map.
func (b *Base) SetParameters(params map[ParamIndex]Parameter) error {
	if b == nil {
		return fmt.Errorf("Base.SetParameters: nil layer")
	}
	for idx, param := range params {
		b.params[idx] = param
	}
	return nil
}

// ZeroGrad zeros all parameter gradients.
func (b *Base) ZeroGrad() {
	if b == nil {
		return
	}
	for idx := range b.params {
		param := b.params[idx]
		param.ZeroGrad()
		b.params[idx] = param
	}
}

// SetParam sets a parameter in the map at the given index.
func (b *Base) SetParam(idx ParamIndex, param Parameter) {
	if b == nil {
		return
	}
	b.initParam(idx)
	b.params[idx] = param
}

// Weights returns the weights parameter.
func (b *Base) Weights() Parameter {
	if b == nil {
		return Parameter{}
	}
	return b.params[ParamWeights]
}

// Biases returns the biases parameter.
func (b *Base) Biases() Parameter {
	if b == nil {
		return Parameter{}
	}
	return b.params[ParamBiases]
}

// Kernels returns the kernels parameter.
func (b *Base) Kernels() Parameter {
	if b == nil {
		return Parameter{}
	}
	return b.params[ParamKernels]
}
