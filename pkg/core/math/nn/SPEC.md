# Neural Network Framework Specification

## Overview

This document describes the implemented neural network framework in `pkg/core/math/nn`. The framework provides high-level abstractions for building, training, and running neural networks with TensorFlow Lite compatibility.

**Note**: Optimizers (SGD, Adam, etc.) are in the `math/learn` package, not `math/nn`.

## Implemented Components

### Core Interfaces

#### Layer Interface
```go
type Layer interface {
    Forward(ctx context.Context, input *tensor.Tensor) (*tensor.Tensor, error)
    Backward(ctx context.Context, gradOutput *tensor.Tensor) (*tensor.Tensor, error)
    OutputShape(inputShape []int) ([]int, error)
    CanLearn() bool
}
```

**Features**:
- Layers store their own input/output state from Forward pass
- Backward uses stored input/output - no need to pass them
- `CanLearn()` controls whether gradients are computed (default: false for inference-only)

#### ParameterLayer Interface
```go
type ParameterLayer interface {
    Layer
    Parameters() []*Parameter
    ZeroGrad()
}
```

**File**: `layer.go`

#### Parameter Type
```go
type Parameter struct {
    Data        *tensor.Tensor
    Grad        *tensor.Tensor
    RequiresGrad bool
}

func (p *Parameter) ZeroGrad()
func (p *Parameter) Update(optimizer Optimizer) error
```

**File**: `parameter.go`

#### Activation Interface
```go
type Activation interface {
    Forward(input *tensor.Tensor) *tensor.Tensor
    Backward(gradOutput *tensor.Tensor, input, output *tensor.Tensor) *tensor.Tensor
}
```

**File**: `activation.go`

#### LossFunction Interface
```go
type LossFunction interface {
    Compute(pred, target *tensor.Tensor) (float32, error)
    Gradient(pred, target *tensor.Tensor) (*tensor.Tensor, error)
}
```

**File**: `loss.go`

#### Optimizer Interface

**Note**: Optimizer interface is in `math/learn` package, not `math/nn`.

```go
// In math/learn package
type Optimizer interface {
    Update(param *nn.Parameter) error
}
```

**File**: `math/learn/optimizer.go`

### Implemented Layers

#### Dense Layer
Fully connected (linear) layer.

```go
type Dense struct {
    weight      *Parameter
    bias        *Parameter
    activation  Activation
    inFeatures  int
    outFeatures int
    canLearn    bool
    input       *tensor.Tensor // Stored input from Forward
    output      *tensor.Tensor // Stored output from Forward
}

func NewDense(inFeatures, outFeatures int, opts ...DenseOption) (*Dense, error)
func WithActivation(activation Activation) DenseOption
func WithBias(useBias bool) DenseOption
func WithCanLearn(canLearn bool) DenseOption
```

**Features**:
- Forward pass: `output = input @ weight + bias`
- Optional activation function
- Optional bias (can be disabled)
- Stores input/output during Forward for Backward
- Backward pass with gradient computation (only if CanLearn is true)
- Shape validation
- CanLearn defaults to false (inference-only by default)

**File**: `dense.go`

#### ActivationLayer
Wrapper that converts an Activation to a Layer.

```go
type ActivationLayer struct {
    activation Activation
}

func NewActivationLayer(activation Activation) *ActivationLayer
```

**File**: `activation_layer.go`

### Implemented Activation Functions

All activation functions implement the `Activation` interface.

#### ReLU
```go
type ReLUActivation struct{}
func NewReLU() *ReLUActivation
```

**Features**:
- Forward: `max(0, input)` (in-place)
- Backward: gradient passed through where input > 0

**File**: `activations.go`

#### Sigmoid
```go
type SigmoidActivation struct{}
func NewSigmoid() *SigmoidActivation
```

**Features**:
- Forward: `1 / (1 + exp(-input))`
- Backward: `gradOutput * output * (1 - output)`

**File**: `activations.go`

#### Tanh
```go
type TanhActivationType struct{}
func NewTanh() *TanhActivationType
```

**Features**:
- Forward: `tanh(input)`
- Backward: `gradOutput * (1 - output^2)`

**File**: `activations.go`

#### Softmax
```go
type SoftmaxActivationType struct {
    dim int
}
func NewSoftmax(dim int) *SoftmaxActivationType
```

**Features**:
- Forward: softmax along specified dimension
- Backward: handles normalization properly
- Supports 1D and 2D tensors

**File**: `activations.go`

### Implemented Loss Functions

All loss functions implement the `LossFunction` interface.

#### MSE (Mean Squared Error)
```go
type MSELoss struct{}
func NewMSE() *MSELoss
```

**Features**:
- Compute: `mean((pred - target)^2)`
- Gradient: `2 * (pred - target) / size`

**File**: `losses.go`

#### CrossEntropy
```go
type CrossEntropyLoss struct{}
func NewCrossEntropy() *CrossEntropyLoss
```

**Features**:
- Compute: `-sum(target * log(pred + epsilon))`
- Gradient: `-target / (pred + epsilon)`

**File**: `losses.go`

#### CategoricalCrossEntropy
```go
type CategoricalCrossEntropy struct {
    fromLogits bool
}
func NewCategoricalCrossEntropy(fromLogits bool) *CategoricalCrossEntropy
```

**Features**:
- If `fromLogits=true`: applies softmax then cross-entropy
- If `fromLogits=false`: assumes predictions are probabilities
- Gradient: `pred - target` (if softmax applied) or standard cross-entropy gradient

**File**: `losses.go`

### Optimizers

**Note**: Optimizers are in the `math/learn` package, not `math/nn`.

#### SGD (Stochastic Gradient Descent)
```go
// In math/learn package
type SGD struct {
    lr float32
}
func NewSGD(lr float32) *SGD
```

**Update Rule**: `param.Data = param.Data - lr * param.Grad`

**File**: `math/learn/optimizer.go`

### Model

#### Model Type
```go
type Model struct {
    layers     []Layer
    inputShape []int
    Input      *tensor.Tensor // Input tensor (set by user)
    Output     *tensor.Tensor // Output tensor (set after Forward)
}

func (m *Model) Forward(ctx context.Context, input *tensor.Tensor) (*tensor.Tensor, error)
func (m *Model) Backward(ctx context.Context, gradOutput *tensor.Tensor) error
func (m *Model) Parameters() []*Parameter
func (m *Model) ZeroGrad()
func (m *Model) Update(optimizer interface{}) error // Optimizer from math/learn package
```

**Features**:
- Forward pass through all layers, stores result in `m.Output`
- Users can set `m.Input` before forward or it's set automatically
- Backward pass through all layers in reverse order
- Layers manage their own input/output state - Model doesn't store per-layer values
- Parameter management (get all parameters, zero gradients, update)
- Shape validation
- Optimizer is from `math/learn` package (not `math/nn`)

**File**: `model.go`

#### ModelBuilder
Helper for constructing models sequentially.

```go
type ModelBuilder struct {
    layers     []Layer
    inputShape []int
}

func NewModelBuilder(inputShape []int) *ModelBuilder
func (b *ModelBuilder) AddLayer(layer Layer) *ModelBuilder
func (b *ModelBuilder) AddDense(inFeatures, outFeatures int, opts ...DenseOption) *ModelBuilder
func (b *ModelBuilder) AddActivation(activation Activation) *ModelBuilder
func (b *ModelBuilder) Build() (*Model, error)
```

**Features**:
- Sequential layer construction
- Automatic shape inference for Dense layers
- Shape validation during build
- Support for options pattern

**File**: `builder.go`

### Training

#### TrainStep
Performs a complete training step: forward pass, loss computation, backward pass, and weight update.

```go
func TrainStep(model *Model, optimizer interface{}, lossFn LossFunction, input, target *tensor.Tensor) (float32, error)
```

**Note**: `optimizer` parameter should be from `math/learn` package (e.g., `learn.NewSGD(0.01)`).

**File**: `training.go`

### Utilities

#### Validation
```go
func shapesEqual(a, b []int) bool
```

**File**: `validation.go`

## Integration with Existing Code

### Legacy Functions
The following functions from `nn.go` are still available for backward compatibility:
- `Linear(t, weight, bias)`
- `Relu(t)`
- `Sigmoid(t)`
- `Tanh(t)`
- `Softmax(t, dim)`
- `MSE(pred, target)`
- `CrossEntropy(pred, target)`

These are wrapped by the new layer-based implementations.

## Usage Examples

### Building a Model

```go
model, err := nn.NewModelBuilder([]int{784}).
    AddDense(784, 256, nn.WithActivation(nn.NewReLU()), nn.WithCanLearn(true)).
    AddDense(256, 128, nn.WithActivation(nn.NewReLU()), nn.WithCanLearn(true)).
    AddDense(128, 10, nn.WithActivation(nn.NewSoftmax(0))).
    Build()
```

### Forward Pass (Inference)

```go
ctx := context.Background()

// Option 1: Set input before forward
model.Input = input
output, err := model.Forward(ctx, input)

// Option 2: Let Forward set Input
output, err := model.Forward(ctx, input)

// After forward, output is available
result := model.Output
```

### Training

```go
import "github.com/itohio/EasyRobot/pkg/core/math/learn"

lossFn := nn.NewCategoricalCrossEntropy(true)
optimizer := learn.NewSGD(0.01)

loss, err := nn.TrainStep(model, optimizer, lossFn, input, target)
```

## File Structure

```
pkg/core/math/nn/
├── activation.go           # Activation interface
├── activation_layer.go     # ActivationLayer wrapper
├── activations.go          # ReLU, Sigmoid, Tanh, Softmax implementations
├── builder.go              # ModelBuilder for constructing models
├── dense.go                # Dense/Linear layer
├── layer.go                # Layer and ParameterLayer interfaces
├── loss.go                 # LossFunction interface
├── losses.go               # MSE, CrossEntropy, CategoricalCrossEntropy
├── model.go                # Model type and operations
├── nn.go                   # Legacy functions (Linear, activations, losses)
├── parameter.go            # Parameter type
├── training.go             # TrainStep function
├── validation.go           # Shape validation utilities
└── SPEC.md                 # This file

Note: Optimizers are in `math/learn/optimizer.go`, not in `math/nn`.
```

## Design Principles

1. **Composition over Inheritance**: Layers compose other layers
2. **Accept Interfaces, Return Structs**: Layer interface, concrete layer types
3. **Small Interfaces**: 1-3 methods per interface
4. **Explicit Dependencies**: Dependencies passed in constructors
5. **Package by Feature**: Organized by domain (layers, activations, losses, etc.)
6. **Tensor Integration**: Uses `math/tensor` throughout

## Error Handling

All functions return errors with context:
```go
return nil, fmt.Errorf("Dense.Forward: input shape %v incompatible with inFeatures %d: %w", inputShape, d.inFeatures, err)
```

## Shape Validation

- All layers validate input shapes in `Forward()`
- `OutputShape()` returns expected output shape for validation
- ModelBuilder validates shapes during construction
- Model validates input shape before forward pass

## Memory Management

- Parameters hold tensors, gradients allocated lazily
- Buffer reuse through tensor operations
- In-place operations where possible (ReLU)

## Current Limitations

1. **CanLearn Default**: Layers default to `CanLearn=false` (inference-only). Must explicitly enable with `WithCanLearn(true)` for training.
2. **Weight Initialization**: Dense layer creates zero-initialized weights - no initialization strategies yet
3. **Multiple Inputs**: Model only supports single input - no multi-input support yet
4. **Composite Layers**: No Sequential or Composite layer containers yet
5. **TensorFlow Lite**: No TFLite integration yet
6. **Utility Layers**: No Reshape, Concatenate, ElementwiseAdd, ElementwiseMul, Pad, Transpose layers yet

## Future Enhancements

See `NN_IMPLEMENTATION_PLAN.md` for remaining work.

