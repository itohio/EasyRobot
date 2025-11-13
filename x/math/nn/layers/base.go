package layers

import (
	"fmt"
	"math/rand"
	"sync/atomic"

	"github.com/itohio/EasyRobot/pkg/core/math/nn/types"
	"github.com/itohio/EasyRobot/pkg/core/math/tensor"
	tensorTypes "github.com/itohio/EasyRobot/pkg/core/math/tensor/types"
)

var layerCounter int64

// Option represents a configuration option for layers.
type Option func(*Base)

// Base provides common layer functionality that can be embedded by all layers.
// Layers embedding Base should implement Forward and Backward directly.
type Base struct {
	name     string
	nameSet  bool // Whether name was explicitly set
	prefix   string
	canLearn bool
	biasHint *bool           // Optional bias hint for layers that support it
	dataType tensor.DataType // Data type for layer tensors (default: DTFP32)
	rng      *rand.Rand      // Random number generator for initialization (optional)
	input    tensorTypes.Tensor
	output   tensorTypes.Tensor
	grad     tensorTypes.Tensor                   // Gradient tensor for backward pass
	params   map[types.ParamIndex]types.Parameter // Parameters map
	layerIdx int64                                // Unique layer index assigned at creation
}

// NewBase creates a new Base layer without options.
// Options should be applied via ParseOptions after layer-specific defaults are set.
func NewBase(prefix string) Base {
	// Increment and capture global layer counter value for this layer
	layerIdx := atomic.AddInt64(&layerCounter, 1)

	return Base{
		name:     "",
		prefix:   prefix,
		canLearn: false,
		dataType: tensor.DTFP32,                          // Default to FP32
		rng:      rand.New(rand.NewSource(rand.Int63())), // Default RNG
		params:   make(map[types.ParamIndex]types.Parameter),
		layerIdx: layerIdx,
	}
}

// ParseOptions applies the given options to the Base layer.
// This should be called after layer-specific defaults are set, allowing options to override defaults.
func (b *Base) ParseOptions(opts ...Option) {
	if b == nil {
		return
	}
	// Apply options
	for _, opt := range opts {
		opt(b)
	}
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

// UseBias returns an Option that sets whether the layer uses bias.
// This is a hint for layers that support optional bias (like Conv1D, Conv2D).
// Layers that don't support optional bias will ignore this option.
func UseBias(hasBias bool) Option {
	return func(b *Base) {
		b.biasHint = &hasBias
	}
}

// WithDataType returns an Option that sets the data type for the layer.
// This data type will be used for creating weights, biases, and intermediate tensors.
// Default is DTFP32.
func WithDataType(dt tensor.DataType) Option {
	return func(b *Base) {
		b.dataType = dt
	}
}

// WithRNG returns an Option that sets the random number generator for the layer.
// This RNG is used for initializing parameters with XavierUniform.
// If not set, a default RNG is created.
func WithRNG(rng *rand.Rand) Option {
	return func(b *Base) {
		if rng != nil {
			b.rng = rng
		}
	}
}

// WithWeights returns an Option that sets the weight parameter at ParamWeights index.
func WithWeights(weight tensorTypes.Tensor) Option {
	return func(b *Base) {
		b.initParam(types.ParamWeights)
		param := b.params[types.ParamWeights]
		param.Data = weight
		param.RequiresGrad = b.canLearn
		b.params[types.ParamWeights] = param
	}
}

// WithBiases returns an Option that sets the bias parameter at ParamBiases index.
func WithBiases(bias tensorTypes.Tensor) Option {
	return func(b *Base) {
		b.initParam(types.ParamBiases)
		param := b.params[types.ParamBiases]
		param.Data = bias
		param.RequiresGrad = b.canLearn
		b.params[types.ParamBiases] = param
	}
}

// WithKernels returns an Option that sets the kernel parameter at ParamKernels index.
func WithKernels(kernel tensorTypes.Tensor) Option {
	return func(b *Base) {
		b.initParam(types.ParamKernels)
		param := b.params[types.ParamKernels]
		param.Data = kernel
		param.RequiresGrad = b.canLearn
		b.params[types.ParamKernels] = param
	}
}

// WithParameter returns an Option that sets a parameter at the given index.
func WithParameter(idx types.ParamIndex, param types.Parameter) Option {
	return func(b *Base) {
		b.initParam(idx)
		param.RequiresGrad = b.canLearn
		b.params[idx] = param
	}
}

// WithParameters returns an Option that sets multiple parameters.
// Copies as many parameters as fit into Base.params.
func WithParameters(params map[types.ParamIndex]types.Parameter) Option {
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
func (b *Base) initParam(idx types.ParamIndex) {
	if _, ok := b.params[idx]; !ok {
		b.params[idx] = types.Parameter{}
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
	// Generate default name from prefix and layer_idx
	// Don't use output shape in name generation as output may not be initialized yet
	var nameFormat string
	if b.prefix != "" {
		nameFormat = fmt.Sprintf("%s_%d", b.prefix, b.layerIdx)
	} else {
		nameFormat = fmt.Sprintf("layer_%d", b.layerIdx)
	}
	b.name = nameFormat
	b.nameSet = true
	return b.name
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

// DataType returns the data type for this layer.
func (b *Base) DataType() tensor.DataType {
	if b == nil {
		return tensor.DTFP32
	}
	return b.dataType
}

// Input returns the input tensor from the last Forward pass.
func (b *Base) Input() tensorTypes.Tensor {
	if b == nil {
		return nil
	}
	return b.input
}

// Output returns the output tensor from the last Forward pass.
func (b *Base) Output() tensorTypes.Tensor {
	if b == nil {
		return nil
	}
	return b.output
}

// Grad returns the gradient tensor from the last Backward pass.
func (b *Base) Grad() tensorTypes.Tensor {
	if b == nil {
		return nil
	}
	return b.grad
}

// setInput stores the input tensor (internal method).
func (b *Base) setInput(input tensorTypes.Tensor) {
	if b == nil {
		return
	}
	b.input = input
}

// setOutput stores the output tensor (internal method).
func (b *Base) setOutput(output tensorTypes.Tensor) {
	if b == nil {
		return
	}
	b.output = output
}

// setGrad stores the gradient tensor (internal method).
func (b *Base) setGrad(grad tensorTypes.Tensor) {
	if b == nil {
		return
	}
	b.grad = grad
}

// AllocOutput allocates the output tensor with the given shape and size.
// Uses the layer's data type (from Base.DataType()).
func (b *Base) AllocOutput(shape []int, size int) {
	if b == nil {
		return
	}
	b.output = tensor.New(b.dataType, tensor.NewShape(shape...))
}

// AllocGrad allocates the gradient tensor with the given shape and size.
// Uses the layer's data type (from Base.DataType()).
func (b *Base) AllocGrad(shape []int, size int) {
	if b == nil {
		return
	}
	b.grad = tensor.New(b.dataType, tensor.NewShape(shape...))
}

// Helper method for layers to store input during Forward.
func (b *Base) StoreInput(input tensorTypes.Tensor) {
	b.setInput(input)
}

// Helper method for layers to store output during Forward.
func (b *Base) StoreOutput(output tensorTypes.Tensor) {
	b.setOutput(output)
}

// Helper method for layers to store grad during Backward.
func (b *Base) StoreGrad(grad tensorTypes.Tensor) {
	b.setGrad(grad)
}

// Parameter returns the parameter at the given index and whether it exists.
func (b *Base) Parameter(idx types.ParamIndex) (types.Parameter, bool) {
	if b == nil {
		return types.Parameter{}, false
	}
	param, ok := b.params[idx]
	return param, ok
}

// ParametersByIndex returns all parameters from the params map.
// Returns map[ParamIndex]Parameter where values are Parameter structs (not pointers).
// This is the internal method used by layers.
func (b *Base) ParametersByIndex() map[types.ParamIndex]types.Parameter {
	if b == nil {
		return nil
	}

	result := make(map[types.ParamIndex]types.Parameter)
	for idx, param := range b.params {
		result[idx] = param
	}
	if len(result) == 0 {
		return nil
	}
	return result
}

// SetParameters sets all parameters in the map.
func (b *Base) SetParameters(params map[types.ParamIndex]types.Parameter) error {
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
		// Get parameter value and call ZeroGrad (which has pointer receiver)
		param := b.params[idx]
		paramPtr := &param
		paramPtr.ZeroGrad()
		// ZeroGrad modifies the gradient tensor in place, so we need to store it back
		b.params[idx] = param
	}
}

// SetParam sets a parameter in the map at the given index.
func (b *Base) SetParam(idx types.ParamIndex, param types.Parameter) {
	if b == nil {
		return
	}
	b.initParam(idx)
	b.params[idx] = param
}

// Weights returns the weights parameter.
func (b *Base) Weights() types.Parameter {
	if b == nil {
		return types.Parameter{}
	}
	return b.params[types.ParamWeights]
}

// Biases returns the biases parameter.
func (b *Base) Biases() types.Parameter {
	if b == nil {
		return types.Parameter{}
	}
	return b.params[types.ParamBiases]
}

// Kernels returns the kernels parameter.
func (b *Base) Kernels() types.Parameter {
	if b == nil {
		return types.Parameter{}
	}
	return b.params[types.ParamKernels]
}

// Parameters returns all parameters as a map with ParamIndex keys.
// This implements the types.Model interface.
func (b *Base) Parameters() map[types.ParamIndex]types.Parameter {
	if b == nil {
		return nil
	}

	result := make(map[types.ParamIndex]types.Parameter)
	for idx, param := range b.params {
		result[idx] = param
	}

	if len(result) == 0 {
		return nil
	}
	return result
}

// Update updates all parameters using the given optimizer.
// This allows individual layers to be trained directly.
// This implements the types.Layer interface.
func (b *Base) Update(optimizer types.Optimizer) error {
	if b == nil {
		return fmt.Errorf("Base.Update: nil layer")
	}

	if optimizer == nil {
		return fmt.Errorf("Base.Update: nil optimizer")
	}

	// Update each parameter using the optimizer
	// We iterate over indices to ensure we can modify the map entries
	for idx := range b.params {
		param := b.params[idx]
		if !tensor.IsNil(param.Data) && param.RequiresGrad && !tensor.IsNil(param.Grad) {
			// Optimizer.Update takes parameter by value
			// Since tensor operations like Sub() modify in place,
			// the optimizer modifies param.Data directly through the tensor reference
			// The tensor data is modified in place, so the parameter in b.params[idx].Data is already updated
			// We don't need to reassign because param.Data is a reference to the tensor
			if err := optimizer.Update(param); err != nil {
				return fmt.Errorf("Base.Update: failed to update parameter %v: %w", idx, err)
			}
			// Note: param.Data is a tensor interface reference, and Sub() modifies the underlying data
			// Since we're working with the parameter from b.params[idx], the tensor reference is already
			// the same one stored in the map, so the update is reflected automatically
		}
	}

	return nil
}
