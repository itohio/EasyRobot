# Neural Network Layers Specification

## Overview

This document describes the neural network layers implementation in `pkg/core/math/nn/layers`. The layers are built on top of the `nn.Layer` interface and use the Base layer for common functionality.

## Design Principles

1. **Base Layer**: All layers embed `*Base` for common functionality (input/output storage, gradient management, CanLearn flag)
2. **Separation of Concerns**: Activation layers are separate from computational layers
3. **Pre-allocated Tensors**: Layers allocate output/gradient tensors during Init
4. **Inference-First**: Layers default to `CanLearn=false` (inference-only) for efficiency
5. **Tensor Integration**: Uses `math/tensor` throughout with row-major storage

## Base Layer

The Base layer provides common functionality that all layers share:

```go
type Base struct {
    name     string
    canLearn bool
    input    tensor.Tensor  // Stored during Forward
    output   tensor.Tensor  // Pre-allocated during Init
    grad     tensor.Tensor  // Gradients stored during Backward
}
```

**Responsibilities:**
- Manages layer state (input/output/gradients)
- Provides helper methods for storing tensors
- Controls gradient computation via CanLearn flag
- Allocates output/gradient tensors

**File**: `base.go`

## Implemented Layers

### Dense Layer

Fully connected (linear) layer: `output = input @ weight + bias`

**Features:**
- Input: 1D or 2D tensors
- Weight shape: `[inFeatures, outFeatures]`
- Optional bias: `[outFeatures]`
- Supports batch processing
- Options pattern for configuration

**File**: `dense.go`

**Constructor:**
```go
func NewDense(inFeatures, outFeatures int, opts ...DenseOption) (*Dense, error)
```

**Options:**
- Base options (via `Option` type): `WithName(name string)`, `WithCanLearn(canLearn bool)`
- Dense-specific options (via `DenseOption` type): `WithDenseBias(useBias bool)`

**Note**: The constructor accepts both `Option` and `DenseOption` types through `...interface{}`. Type safety is enforced at runtime.

**Example:**
```go
layer, err := layers.NewDense(
    784, 256,
    layers.WithCanLearn(true),
)
```

**Parameter Access:**
```go
// Get parameters
weight := layer.Weight()
bias := layer.Bias()

// Set parameters (with validation)
err := layer.SetWeight(newWeight)
err := layer.SetBias(newBias)
```

### Activation Layers

Activation layers implement element-wise transformations. They preserve input shape.

#### ReLU
- **Forward**: `output = max(0, input)`
- **Backward**: Gradient passed through where `input > 0`
- **File**: `activations.go`

```go
func NewReLU(name string) *ReLU
```

#### Sigmoid
- **Forward**: `output = 1 / (1 + exp(-input))`
- **Backward**: `gradInput = gradOutput * output * (1 - output)`
- **File**: `activations.go`

```go
func NewSigmoid(name string) *Sigmoid
```

#### Tanh
- **Forward**: `output = tanh(input)`
- **Backward**: `gradInput = gradOutput * (1 - output^2)`
- **File**: `activations.go`

```go
func NewTanh(name string) *Tanh
```

#### Softmax
- **Forward**: Softmax along specified dimension
- **Backward**: Handles normalization properly
- **Supports**: 1D and 2D tensors
- **File**: `activations.go`

```go
func NewSoftmax(name string, dim int) *Softmax
```

### Conv2D Layer

2D Convolution layer: `output = conv2d(input, weight) + bias`

**Features:**
- Input: 4D tensors `[batch, inChannels, height, width]`
- Weight shape: `[outChannels, inChannels, kernelH, kernelW]`
- Optional bias: `[outChannels]`
- Supports stride, padding
- Options pattern for configuration

**File**: `conv2d.go`

**Constructor:**
```go
func NewConv2D(
    inChannels, outChannels, kernelH, kernelW, strideH, strideW, padH, padW int,
    opts ...Conv2DOption,
) (*Conv2D, error)
```

**Note**: Backward pass not yet implemented.

**Parameter Access:**
```go
// Get parameters
weight := layer.Weight()
bias := layer.Bias()

// Set parameters (with validation)
err := layer.SetWeight(newWeight)
err := layer.SetBias(newBias)
```

### Conv1D Layer

1D Convolution layer: `output = conv1d(input, weight) + bias`

**Features:**
- Input: 3D tensors `[batch, inChannels, length]`
- Weight shape: `[outChannels, inChannels, kernelLen]`
- Optional bias: `[outChannels]`
- Supports stride, padding
- Options pattern for configuration

**File**: `conv1d.go`

**Constructor:**
```go
func NewConv1D(
    inChannels, outChannels, kernelLen, stride, pad int,
    opts ...Conv1DOption,
) (*Conv1D, error)
```

**Note**: Backward pass not yet implemented.

**Parameter Access:**
```go
// Get parameters
weight := layer.Weight()
bias := layer.Bias()

// Set parameters (with validation)
err := layer.SetWeight(newWeight)
err := layer.SetBias(newBias)
```

### Pooling Layers

#### MaxPool2D
- **Forward**: Max pooling over spatial dimensions
- **Input**: `[batch, channels, height, width]`
- **Output**: `[batch, channels, outHeight, outWidth]`
- **Features**: Kernel size, stride, padding
- **File**: `pooling.go`

```go
func NewMaxPool2D(kernelH, kernelW, strideH, strideW, padH, padW int) (*MaxPool2D, error)
```

**Note**: Backward pass not yet implemented.

#### AvgPool2D
- **Forward**: Average pooling over spatial dimensions
- **Input**: `[batch, channels, height, width]`
- **Output**: `[batch, channels, outHeight, outWidth]`
- **Features**: Kernel size, stride, padding
- **File**: `pooling.go`

```go
func NewAvgPool2D(kernelH, kernelW, strideH, strideW, padH, padW int) (*AvgPool2D, error)
```

**Note**: Backward pass not yet implemented.

#### GlobalAvgPool2D
- **Forward**: Global average pooling (reduces to channels only)
- **Input**: `[batch, channels, height, width]`
- **Output**: `[batch, channels]`
- **File**: `pooling.go`

```go
func NewGlobalAvgPool2D() *GlobalAvgPool2D
```

**Note**: Backward pass not yet implemented.

### Utility Layers

#### Flatten
- **Forward**: Flattens multi-dimensional input
- **Features**: Custom start/end dimension range
- **File**: `utility.go`

```go
func NewFlatten(startDim, endDim int) *Flatten
```

#### Reshape
- **Forward**: Reshapes tensor without changing data
- **Features**: Target shape validation
- **File**: `utility.go`

```go
func NewReshape(targetShape []int) *Reshape
```

## Additional Planned Layers

The following layers are planned to be implemented based on the availability of underlying tensor operations:

### Convolution Layers

#### Conv2DTransposed
- **Type**: Transposed 2D Convolution (Deconvolution)
- **Input**: `[batch, inChannels, height, width]`
- **Kernel**: `[inChannels, outChannels, kernelH, kernelW]`
- **Output**: `[batch, outChannels, outHeight, outWidth]`
- **Tensor Support**: ✅ Available in `tensor.Conv2DTransposed()`

#### DepthwiseConv2D
- **Type**: Depthwise Separable Convolution
- **Input**: `[batch, inChannels, height, width]`
- **Kernel**: `[inChannels, kernelH, kernelW]`
- **Output**: `[batch, inChannels, outHeight, outWidth]`
- **Tensor Support**: ✅ Available in `tensor.DepthwiseConv2D()`

### Pooling Layers

#### AdaptiveAvgPool2D
- **Type**: Adaptive Average Pooling
- **Input**: `[batch, channels, height, width]`
- **Output**: `[batch, channels, targetH, targetW]`
- **Features**: Target output size
- **Tensor Support**: ✅ Available in `tensor.AdaptiveAvgPool2D()`

### Normalization Layers

#### BatchNorm2D
- **Type**: Batch Normalization for 2D inputs
- **Input**: `[batch, channels, height, width]`
- **Output**: Same as input
- **Parameters**: Weight (scale), bias (shift), running_mean, running_var
- **Features**: Training/inference mode, momentum
- **Tensor Support**: ❌ Not yet available

#### LayerNorm
- **Type**: Layer Normalization
- **Input**: `[batch, *dims]`
- **Output**: Same as input
- **Parameters**: Weight, bias
- **Features**: Normalize over specified dimensions
- **Tensor Support**: ❌ Not yet available

#### GroupNorm
- **Type**: Group Normalization
- **Input**: `[batch, channels, *dims]`
- **Output**: Same as input
- **Parameters**: Weight, bias
- **Features**: Normalize over groups of channels
- **Tensor Support**: ❌ Not yet available

### Regularization Layers

#### Dropout
- **Type**: Dropout regularization
- **Input**: Any shape
- **Output**: Same as input
- **Features**: Dropout rate, training/inference mode
- **Tensor Support**: ❌ Not yet available

### Utility Layers

#### Concatenate
- **Type**: Concatenate multiple inputs
- **Input**: Multiple tensors of compatible shapes
- **Output**: Concatenated tensor
- **Features**: Concatenate along specified dimension
- **Tensor Support**: ❌ Not yet available

#### Add
- **Type**: Element-wise addition
- **Input**: Two tensors of same shape
- **Output**: Element-wise sum
- **Tensor Support**: ✅ Available in `tensor.Add()`

#### Multiply
- **Type**: Element-wise multiplication
- **Input**: Two tensors of same shape
- **Output**: Element-wise product
- **Tensor Support**: ✅ Available in `tensor.Mul()`

#### Unsqueeze
- **Type**: Add dimension
- **Input**: Any shape
- **Output**: Shape with added dimension
- **Tensor Support**: ❌ Not yet available

#### Squeeze
- **Type**: Remove dimension of size 1
- **Input**: Any shape
- **Output**: Shape with removed dimensions
- **Tensor Support**: ❌ Not yet available

#### Pad
- **Type**: Padding
- **Input**: Any shape
- **Output**: Padded tensor
- **Features**: Padding mode, padding values
- **Tensor Support**: ❌ Not yet available

#### Transpose
- **Type**: Transpose dimensions
- **Input**: Any shape
- **Output**: Transposed tensor
- **Tensor Support**: ❌ Not yet available

### Residual Layers

#### ResidualBlock
- **Type**: Residual connection
- **Features**: Skip connection with optional projection
- **Components**: Two conv layers + optional projection layer

#### SEBlock
- **Type**: Squeeze-and-Excitation block
- **Features**: Channel attention mechanism

### Attention Layers

#### MultiHeadAttention
- **Type**: Multi-head self-attention
- **Features**: Query, key, value projections, attention mechanism

#### SelfAttention
- **Type**: Self-attention layer
- **Features**: Simplified multi-head attention

## Implementation Status

| Layer | Status | Priority | Dependencies |
|-------|--------|----------|--------------|
| Dense | ✅ Implemented | High | - |
| ReLU | ✅ Implemented | High | - |
| Sigmoid | ✅ Implemented | High | - |
| Tanh | ✅ Implemented | High | - |
| Softmax | ✅ Implemented | High | - |
| Conv2D | ✅ Implemented | High | tensor.Conv2D |
| Conv1D | ✅ Implemented | High | tensor.Conv1D |
| MaxPool2D | ✅ Implemented | High | tensor.MaxPool2D |
| AvgPool2D | ✅ Implemented | High | tensor.AvgPool2D |
| GlobalAvgPool2D | ✅ Implemented | High | tensor.GlobalAvgPool2D |
| Flatten | ✅ Implemented | Medium | tensor.Reshape |
| Reshape | ✅ Implemented | Medium | tensor.Reshape |
| BatchNorm2D | ⏳ Need tensor ops | Medium | - |
| Dropout | ⏳ Need implementation | Medium | - |

**Legend:**
- ✅ Implemented
- 🔄 Ready to implement (tensor ops available)
- ⏳ Needs underlying operations

## Layer Interface

All layers implement the `nn.Layer` interface:

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

**Note:** Forward and Backward methods do not take `context.Context` as a parameter. The interface does not require context for cancellation or timeouts.

**Parameter Management:**

Layers with trainable parameters expose them via the `Parameters()` method:
- `Base.Parameters()` returns `map[ParamIndex]Parameter` where values are Parameter structs (NOT pointers)
- `Parameter` type is defined in the `layers` package (not `nn` package to avoid import cycles)
- Layers embedding Base inherit this method via embedding
- Parameters are stored as values in the map for efficiency and clarity
- To update parameters, use `SetParam()` method on the layer

## Usage Examples

### Building a Simple Network

```go
// Create layers
layer1, _ := layers.NewDense(784, 256, layers.WithCanLearn(true))
relu := layers.NewReLU("relu")
layer2, _ := layers.NewDense(256, 10, layers.WithCanLearn(true))
```

### Forward Pass

```go
// Initialize layers
layer1.Init([]int{784})
relu.Init([]int{256})
layer2.Init([]int{256})

// Forward pass
input := tensor.Tensor{Dim: []int{784}, Data: data}
hidden1, _ := layer1.Forward(input)
activated, _ := relu.Forward(hidden1)
output, _ := layer2.Forward(activated)
```

### Training

```go
// Zero gradients
layer1.ZeroGrad()
layer2.ZeroGrad()

// Forward pass
input := tensor.Tensor{Dim: []int{1, 784}, Data: batchData}
hidden1, _ := layer1.Forward(input)
activated, _ := relu.Forward(hidden1)
output, _ := layer2.Forward(activated)

// Compute loss and backward pass
loss := computeLoss(output, target)
gradOutput := computeGrad(loss)

// Backward pass
grad2, _ := layer2.Backward(gradOutput)
gradRelu, _ := relu.Backward(grad2)
grad1, _ := layer1.Backward(gradRelu)

// Update parameters
optimizer := learn.NewSGD(0.01)
// Model.Parameters() collects all parameters from all layers as map[string]Parameter
// Model.Update() handles optimizer updates internally
err := model.Update(optimizer)
```

## Testing

All layers should have comprehensive tests covering:
- Forward pass correctness
- Backward pass gradients
- Shape validation
- Boundary conditions
- Integration with optimizer

See `xor_test.go` for an example integration test.

## Integration with Tensor Operations

Layers are built on top of optimized tensor operations:

- **Dense**: Uses `primitive.Gemm_NN`, `primitive.Gemv_T`, `primitive.Axpy`
- **Conv2D**: Uses `primitive.Conv2D`
- **Pooling**: Uses tensor pool operations
- **Normalization**: Uses tensor statistics operations (to be added)

## Memory Management

- Output tensors are pre-allocated during `Init()`
- Gradients are allocated lazily during `Backward()` (only if `CanLearn=true`)
- Tensor operations are optimized to minimize allocations
- Base layer manages tensor lifecycle

## Known Issues and Technical Debt

### Code Quality Issues

1. **Function Length**: Several backward pass functions exceed the 30-line guideline:
   - `Dense.Backward()`: ~142 lines (should be split into helper functions)
   - `Conv2D.Backward()`: ~162 lines (should be split into helper functions)
   - `Conv1D.Backward()`: ~145 lines (should be split into helper functions)

   **Recommendation**: Extract gradient computation logic into separate helper functions:
   - `computeWeightGrad()`
   - `computeBiasGrad()`
   - `computeInputGrad()`

2. **Dense Options Pattern**: `NewDense()` uses `...interface{}` for options, with runtime type checking:
   ```go
   func NewDense(inFeatures, outFeatures int, opts ...interface{}) (*Dense, error)
   ```
   
   **Status**: ✅ Improved - Error message now includes the invalid type. The pattern works but loses compile-time type safety. This is acceptable for now as it allows both Option and DenseOption types.

3. **AvgPool2D Prefix Bug**: `AvgPool2D` incorrectly uses "maxpool2d" prefix instead of "avgpool2d":
   ```go:221:pooling.go
   Base:    NewBase("maxpool2d"),
   ```
   
   **Fix Required**: Change to `NewBase("avgpool2d")`

### Design Issues

4. **ParameterLayer Interface Removed**: 
   - ✅ Fixed - ParameterLayer interface has been removed
   - `Base.Parameters()` returns `map[ParamIndex]Parameter` (values, not pointers)
   - `Parameter` type moved to `layers` package to avoid import cycles between `nn` and `layers`
   - Parameters are stored and returned as values (not pointers) for clarity and efficiency
   - `Model.Parameters()` collects parameters from layers via `Parameters()` method returning `map[ParamIndex]Parameter`
   - `Model.Update()` works with parameter values and writes back via `SetParam()` method
   - This design avoids the complexity of pointer management while maintaining functionality

5. **GlobalAvgPool2D Prefix**: Uses "avgpool2d" prefix instead of "globalavgpool2d":
   ```go:367:pooling.go
   Base: NewBase("avgpool2d"),
   ```
   
   **Recommendation**: Change to `NewBase("globalavgpool2d")` for consistency.

### Documentation Issues

6. **SPEC.md Outdated**: SPEC.md incorrectly states that Forward/Backward take `context.Context`, but the actual Layer interface does not.

7. **Missing Parameter Access Documentation**: SPEC.md shows `Parameters()` returning a slice, but Base returns a map.

## Future Enhancements

1. **More Layers**: Conv3D, RNNs, LSTMs, Transformers
2. **Fused Layers**: Conv+BatchNorm+ReLU fusion
3. **Quantization**: INT8 support for deployment (see [tensor/QUANTIZATION_PLAN.md](../../tensor/QUANTIZATION_PLAN.md))
4. **Mixed Precision**: FP16 training
5. **TensorFlow Lite Compatibility**: Layer-level TFLite export
6. **Layer Pruning**: Remove redundant layers
7. **Layer Fusion**: Combine multiple layers into one
8. **Refactoring**: Split long backward pass functions into smaller helper functions
9. **Type Safety**: Improve options pattern in Dense layer constructor
