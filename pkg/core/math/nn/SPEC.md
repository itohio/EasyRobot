# Neural Network Framework Specification

## Overview

This document describes the implemented neural network framework in `pkg/core/math/nn`. The framework provides high-level abstractions for building, training, and running neural networks with TensorFlow Lite compatibility.

**Note**: Optimizers (SGD, Adam, etc.) are in the `math/learn` package, not `math/nn`.

## Implemented Components

### Core Interfaces

#### Layer Interface
```go
type Layer interface {
    Name() string
    Init(inputShape []int) error
    Forward(input tensor.Tensor) (tensor.Tensor, error)
    Backward(gradOutput tensor.Tensor) (tensor.Tensor, error)
    OutputShape(inputShape []int) ([]int, error)
    CanLearn() bool
    SetCanLearn(canLearn bool)
    Input() tensor.Tensor
    Output() tensor.Tensor
}
```

**Features**:
- Layers store their own input/output state from Forward pass
- Backward uses stored input/output - no need to pass them
- `CanLearn()` controls whether gradients are computed (default: false for inference-only)
- All layers are in `pkg/core/math/nn/layers/` subdirectory

**File**: `layer.go`

#### LossFunction Interface
```go
type LossFunction interface {
    Compute(pred, target tensor.Tensor) (float32, error)
    Gradient(pred, target tensor.Tensor) (tensor.Tensor, error)
}
```

**File**: `loss.go`

#### Parameter Type

**Note**: Parameter type is in `math/nn/layers` package.

```go
// In pkg/core/math/nn/layers package
type Parameter struct {
    Data         tensor.Tensor  // Value type, not pointer!
    Grad         tensor.Tensor
    RequiresGrad bool
}

func (p Parameter) ZeroGrad()
func (p Parameter) InitXavier(fanIn, fanOut int)
func (p Parameter) InitXavierNormal(fanIn, fanOut int)
```

**File**: `pkg/core/math/nn/layers/parameter.go`

#### Optimizer Interface

```go
type Optimizer interface {
    Update(param *layers.Parameter) error
}
```

**File**: `model.go`

**Note**: The interface is in the `nn` package because it's used by `Model.Update()`. Implementations (like `SGD`) are in `math/learn`.

### Implemented Layers

All layers are in `pkg/core/math/nn/layers/` subdirectory. See `layers/SPEC.md` for detailed documentation.

**Implemented Layers**:
- **Dense**: Fully connected layer
- **Conv1D**: 1D convolution
- **Conv2D**: 2D convolution
- **MaxPool2D**: Max pooling
- **AvgPool2D**: Average pooling
- **GlobalAvgPool2D**: Global average pooling
- **Flatten**: Flattens tensor
- **Reshape**: Reshapes tensor
- **ReLU**: ReLU activation
- **Sigmoid**: Sigmoid activation
- **Tanh**: Tanh activation
- **Softmax**: Softmax activation

All layers embed `Base` which provides parameter management, gradient tracking, and common functionality.

**File**: `layers/`

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

func (m *Model) Forward(input tensor.Tensor) (tensor.Tensor, error)
func (m *Model) Backward(gradOutput tensor.Tensor) error
func (m *Model) Parameters() map[string]layers.Parameter
func (m *Model) ZeroGrad()
func (m *Model) Update(optimizer Optimizer) error
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
func TrainStep(model *Model, optimizer Optimizer, lossFn LossFunction, input, target tensor.Tensor) (float32, error)
```

**Note**: This function is now in `math/learn` package. `optimizer` must implement `nn.Optimizer` interface (e.g., `learn.NewSGD(0.01)`).

**File**: `math/learn/training.go`

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

loss, err := learn.TrainStep(model, optimizer, lossFn, input, target)
```

## File Structure

```
pkg/core/math/nn/
├── builder.go              # ModelBuilder for constructing models
├── layer.go                # Layer interface
├── loss.go                 # LossFunction interface
├── losses.go               # MSE, CrossEntropy, CategoricalCrossEntropy
├── model.go                # Model type and operations
├── nn.go                   # Legacy functions (Linear, activations, losses)
├── validation.go           # Shape validation utilities
├── layers/                 # Layer implementations
│   ├── base.go             # Base layer (parameter management)
│   ├── dense.go            # Dense/Linear layer
│   ├── activations.go      # ReLU, Sigmoid, Tanh, Softmax
│   ├── conv1d.go           # Conv1D layer
│   ├── conv2d.go           # Conv2D layer
│   ├── pooling.go          # MaxPool2D, AvgPool2D, GlobalAvgPool2D
│   ├── reshape.go          # Reshape, Flatten layers
│   ├── parameter.go        # Parameter type
│   └── SPEC.md             # Layers package documentation
└── SPEC.md                 # This file

pkg/core/math/learn/
├── optimizer.go            # Optimizer interface and SGD implementation
├── training.go             # TrainStep function
├── xor_test.go             # XOR training test
└── LEARN_SPEC.md           # Learn package documentation
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

