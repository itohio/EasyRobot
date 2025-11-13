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
- **Unsqueeze**: Adds dimensions
- **Squeeze**: Removes size-1 dimensions
- **Transpose**: Transposes 2D tensors
- **Pad**: Pads tensor with constant/reflect values
- **Concatenate**: Concatenates tensors along dimension
- **ReLU**: ReLU activation
- **Sigmoid**: Sigmoid activation
- **Tanh**: Tanh activation
- **Softmax**: Softmax activation
- **Dropout**: Dropout regularization

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
    layerNames map[string]int // Map from layer name to index
    inputShape []int
    Input      tensor.Tensor // Input tensor (set during Forward)
    Output     tensor.Tensor // Output tensor (set after Forward)
}

func (m *Model) Init() error
func (m *Model) GetLayer(index int) Layer
func (m *Model) GetLayerByName(name string) Layer
func (m *Model) LayerCount() int
func (m *Model) Forward(input tensor.Tensor) (tensor.Tensor, error)
func (m *Model) Backward(gradOutput tensor.Tensor) error
func (m *Model) Parameters() map[string]layers.Parameter
func (m *Model) ZeroGrad()
func (m *Model) Update(optimizer Optimizer) error
```

**Features**:
- **Init()**: Must be called after building to initialize all layers with correct shapes
- **Layer access**: Get layers by index or name (for introspection/debugging)
- Forward pass through all layers, stores result in `m.Output`
- Backward pass through all layers in reverse order
- Layers manage their own input/output state internally
- Parameter management (get all parameters, zero gradients, update)
- Shape validation during Init and Forward
- Optimizer implements `nn.Optimizer` interface (typically from `math/learn` package)

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
func (b *ModelBuilder) Build() (*Model, error)
```

**Features**:
- Sequential layer construction by adding Layer instances
- Shape validation during build
- Automatic layer naming and duplicate name detection
- Creates layer name-to-index mapping for introspection

**File**: `builder.go`

### Training

Training functionality is in the `math/learn` package to separate concerns.

#### TrainStep
Performs a complete training step: forward pass, loss computation, backward pass, and weight update.

```go
// In math/learn package
func TrainStep(model *nn.Model, optimizer nn.Optimizer, lossFn nn.LossFunction, input, target tensor.Tensor) (float32, error)
```

**Features**:
- Forward pass through model
- Loss computation using provided loss function
- Backward pass through model
- Parameter updates using provided optimizer
- Returns loss value for monitoring

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
// Create layers
dense1 := layers.NewDense(784, 256, layers.WithCanLearn(true))
relu1 := layers.NewReLU("relu1")
dense2 := layers.NewDense(256, 128, layers.WithCanLearn(true))
relu2 := layers.NewReLU("relu2")
dense3 := layers.NewDense(128, 10, layers.WithCanLearn(true))
softmax := layers.NewSoftmax("softmax", 0)

// Build model
model, err := nn.NewModelBuilder([]int{784}).
    AddLayer(dense1).
    AddLayer(relu1).
    AddLayer(dense2).
    AddLayer(relu2).
    AddLayer(dense3).
    AddLayer(softmax).
    Build()

// Initialize all layers
err = model.Init()
```

### Forward Pass (Inference)

```go
// Forward pass
output, err := model.Forward(input)

// Output is also stored in model.Output
result := model.Output
```

### Training

```go
import "github.com/itohio/EasyRobot/x/math/learn"

lossFn := nn.NewCategoricalCrossEntropy(true)
optimizer := learn.NewSGD(0.01)

// Single training step
loss, err := learn.TrainStep(model, optimizer, lossFn, batchInput, batchTarget)
```

## File Structure

```
pkg/core/math/nn/
├── builder.go              # ModelBuilder for constructing models
├── layer.go                # Layer interface
├── loss.go                 # LossFunction interface
├── losses.go               # Loss function implementations
├── model.go                # Model type and operations
├── nn.go                   # Legacy functions (Linear, activations, losses)
├── nn_test.go              # Package tests
├── validation.go           # Shape validation utilities
├── layers/                 # Layer implementations
│   ├── activations.go      # Activation layers (ReLU, Sigmoid, Tanh, Softmax, Dropout)
│   ├── base.go             # Base layer (parameter management)
│   ├── conv1d.go           # Conv1D layer
│   ├── conv2d.go           # Conv2D layer
│   ├── dense.go            # Dense/Linear layer
│   ├── parameter.go        # Parameter type and initialization
│   ├── pooling.go          # Pooling layers (MaxPool2D, AvgPool2D, GlobalAvgPool2D)
│   ├── utility.go          # Utility layers (Flatten, Reshape, Unsqueeze, Squeeze, Transpose, Pad, Concatenate)
│   ├── activations_test.go # Tests for activation layers
│   ├── base_test.go        # Tests for base layer
│   ├── conv1d_test.go      # Tests for Conv1D layer
│   ├── conv2d_test.go      # Tests for Conv2D layer
│   ├── dense_test.go       # Tests for Dense layer
│   ├── parameter_test.go   # Tests for parameter type
│   ├── pooling_test.go     # Tests for pooling layers
│   ├── utility_test.go     # Tests for utility layers
│   └── SPEC.md             # Layers package documentation
├── models/                 # (Empty directory for future use)
└── SPEC.md                 # This file

pkg/core/math/learn/
├── optimizer.go            # Optimizer implementations
├── training.go             # Training utilities
├── quantization.go         # Quantization utilities
├── mnist_test.go           # MNIST training tests
├── mnist_large_test.go     # Large MNIST training tests
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
2. **Weight Initialization**: Layers create zero-initialized weights - Xavier initialization available but not used by default
3. **Multiple Inputs**: Model only supports single input - Concatenate layer supports multiple inputs but Model does not
4. **Composite Layers**: No Sequential or Composite layer containers yet
5. **TensorFlow Lite**: No TFLite integration yet
6. **Advanced Layers**: Missing BatchNorm, LayerNorm, GroupNorm, Conv2DTransposed, DepthwiseConv2D, Adaptive pooling layers
7. **Multi-Input Models**: Model interface only supports single input tensor
8. **Context Cancellation**: No context.Context support for cancellation/timeouts

## Future Enhancements

See `NN_IMPLEMENTATION_PLAN.md` for remaining work.

