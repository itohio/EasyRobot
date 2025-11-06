# Tensor API Migration Plan - TensorFlow Alignment

## Overview

This document outlines the migration plan for the Tensor interface to align more closely with TensorFlow's API design while maintaining efficiency through destination-parameter operations. The plan focuses on organizing operations into logical interface groups for better readability and maintainability.

**Goal**: Create a modular interface structure where operations are grouped by category, and the main `Tensor` interface embeds all category interfaces for comprehensive functionality.

**Key Principles**:
1. **Destination-based operations**: All operations should support destination parameters for efficient memory reuse
2. **TensorFlow alignment**: API should align with TensorFlow's operation naming and semantics where applicable
3. **Interface composition**: Main `Tensor` interface should embed all category interfaces for readability
4. **Zero allocations**: Operations should support destination parameters to avoid unnecessary allocations
5. **Backward compatibility**: Migration should maintain existing functionality while improving structure

## Current State Analysis

### Current Tensor Interface Structure

The current `types.Tensor` interface contains **80+ methods** organized by operation type:
- Core Properties and Access (13 methods)
- Element-wise Operations (17 methods)
- Comparison Operations (4 methods)
- Conditional Operations (1 method)
- Reduction Operations (5 methods)
- Broadcasting (1 method)
- Linear Algebra Operations (11 methods)
- Convolution Operations (10 methods)
- Pooling Operations (7 methods)
- Image/Column Conversion (2 methods)
- Gradient Routing (2 methods)
- Activation Functions (4 methods)
- Dropout Operations (2 methods)

### Issues with Current Structure

1. **Monolithic interface**: All operations in one large interface (80+ methods)
2. **Mixed patterns**: Some operations use destination parameters, others don't
3. **Inconsistent naming**: Not fully aligned with TensorFlow conventions
4. **Hard to navigate**: Large interface makes it difficult to find specific operations
5. **Limited extensibility**: Adding new operation categories requires modifying the main interface

## Proposed Interface Structure

### Interface Composition Strategy

The main `Tensor` interface will embed all category interfaces, providing a single comprehensive interface while maintaining logical organization:

```go
type Tensor interface {
    // Core properties and metadata
    TensorCore
    
    // Operation category interfaces
    TensorManipulation
    TensorElementWise
    TensorMath
    TensorActivations
    TensorConvolutions
    TensorPooling
    TensorDropout
    TensorActivationGradients
    TensorConvolutionGradients
    TensorPoolingGradients
    TensorDropoutGradients
}
```

## Interface Definitions

### 1. TensorCore Interface

**Purpose**: Core tensor properties and metadata access.

```go
type TensorCore interface {
    // Identity and metadata
    ID() uintptr
    DataType() DataType
    Data() any
    Shape() Shape
    Rank() int
    Size() int
    Empty() bool
    
    // Element access
    At(indices ...int) float64
    SetAt(value float64, indices ...int)
    Elements(fixedAxisValuePairs ...int) func(func(Element) bool)
}
```

**Operations**: 9 methods
- Core properties: ID, DataType, Data, Shape, Rank, Size, Empty
- Element access: At, SetAt, Elements

### 2. TensorManipulation Interface

**Purpose**: Copy, clone, fill, reshape, and tensor manipulation operations.

```go
type TensorManipulation interface {
    // Copy and clone
    Clone() Tensor
    Copy(src Tensor) Tensor
    
    // Shape manipulation
    Reshape(newShape Shape) Tensor
    Slice(dim int, start int, length int) Tensor
    Transpose(dims ...int) Tensor
    TransposeTo(dst Tensor, dims ...int) Tensor
    Permute(dims []int) Tensor
    
    // Broadcasting
    BroadcastTo(shape Shape) (Tensor, error)
    
    // Filling
    Fill(value float64) Tensor
    FillTo(dst Tensor, value float64) Tensor
    
    // Padding and unpadding
    Pad(padding []int, value float64) Tensor
    PadTo(dst Tensor, padding []int, value float64) Tensor
    Unpad(padding []int) Tensor
}
```

**Operations**: 15 methods
- Copy/clone: Clone, Copy
- Shape: Reshape, Slice, Transpose, TransposeTo, Permute
- Broadcasting: BroadcastTo
- Filling: Fill, FillTo
- Padding: Pad, PadTo, Unpad

**TensorFlow Alignment**:
- `Reshape` → `tf.reshape`
- `Transpose` → `tf.transpose`
- `Slice` → `tf.slice`
- `BroadcastTo` → `tf.broadcast_to`
- `Pad` → `tf.pad`
- `Unpad` → custom (no direct TF equivalent)

### 3. TensorElementWise Interface

**Purpose**: Element-wise operations on tensors.

```go
type TensorElementWise interface {
    // Binary operations (in-place)
    Add(other Tensor) Tensor
    Sub(other Tensor) Tensor
    Mul(other Tensor) Tensor
    Div(other Tensor) Tensor
    
    // Binary operations (destination-based)
    AddTo(other Tensor, dst Tensor) Tensor
    SubTo(other Tensor, dst Tensor) Tensor
    MulTo(other Tensor, dst Tensor) Tensor
    DivTo(other Tensor, dst Tensor) Tensor
    
    // Scalar operations (in-place)
    Scale(scalar float64) Tensor
    AddScalar(scalar float64) Tensor
    SubScalar(scalar float64) Tensor
    MulScalar(scalar float64) Tensor
    DivScalar(scalar float64) Tensor
    
    // Scalar operations (destination-based)
    ScaleTo(dst Tensor, scalar float64) Tensor
    AddScalarTo(dst Tensor, scalar float64) Tensor
    SubScalarTo(dst Tensor, scalar float64) Tensor
    MulScalarTo(dst Tensor, scalar float64) Tensor
    DivScalarTo(dst Tensor, scalar float64) Tensor
    
    // Unary operations (in-place)
    Square(dst Tensor) Tensor
    Sqrt(dst Tensor) Tensor
    Exp(dst Tensor) Tensor
    Log(dst Tensor) Tensor
    Pow(dst Tensor, power float64) Tensor
    Abs(dst Tensor) Tensor
    Sign(dst Tensor) Tensor
    Cos(dst Tensor) Tensor
    Sin(dst Tensor) Tensor
    Negative(dst Tensor) Tensor
    
    // Unary operations (destination-based)
    SquareTo(dst Tensor) Tensor
    SqrtTo(dst Tensor) Tensor
    ExpTo(dst Tensor) Tensor
    LogTo(dst Tensor) Tensor
    PowTo(dst Tensor, power float64) Tensor
    AbsTo(dst Tensor) Tensor
    SignTo(dst Tensor) Tensor
    CosTo(dst Tensor) Tensor
    SinTo(dst Tensor) Tensor
    NegativeTo(dst Tensor) Tensor
    
    // Comparison operations
    Equal(other Tensor) Tensor
    NotEqual(other Tensor) Tensor
    GreaterThan(other Tensor) Tensor
    GreaterEqual(other Tensor) Tensor
    Less(other Tensor) Tensor
    LessEqual(other Tensor) Tensor
    
    // Conditional operations
    Where(condition, a, b Tensor) Tensor
    WhereTo(dst Tensor, condition, a, b Tensor) Tensor
}
```

**Operations**: 50+ methods
- Binary: Add, Sub, Mul, Div (in-place and destination-based)
- Scalar: Scale, AddScalar, SubScalar, MulScalar, DivScalar (in-place and destination-based)
- Unary: Square, Sqrt, Exp, Log, Pow, Abs, Sign, Cos, Sin, Negative (in-place and destination-based)
- Comparison: Equal, NotEqual, GreaterThan, GreaterEqual, Less, LessEqual
- Conditional: Where, WhereTo

**TensorFlow Alignment**:
- `Add` → `tf.add`
- `Mul` → `tf.multiply`
- `Sub` → `tf.subtract`
- `Div` → `tf.divide`
- `Scale` → `tf.scalar_mul`
- `Square` → `tf.square`
- `Sqrt` → `tf.sqrt`
- `Exp` → `tf.exp`
- `Log` → `tf.log`
- `Pow` → `tf.pow`
- `Abs` → `tf.abs`
- `Sign` → `tf.sign`
- `Cos` → `tf.cos`
- `Sin` → `tf.sin`
- `Negative` → `tf.negative`
- `Equal` → `tf.equal`
- `NotEqual` → `tf.not_equal`
- `GreaterThan` → `tf.greater`
- `GreaterEqual` → `tf.greater_equal`
- `Less` → `tf.less`
- `LessEqual` → `tf.less_equal`
- `Where` → `tf.where`

### 4. TensorMath Interface

**Purpose**: Common tensor math operations (reductions, linear algebra, etc.).

```go
type TensorMath interface {
    // Reduction operations
    Sum(dims ...int) Tensor
    SumTo(dst Tensor, dims ...int) Tensor
    Mean(dims ...int) Tensor
    MeanTo(dst Tensor, dims ...int) Tensor
    Max(dims ...int) Tensor
    MaxTo(dst Tensor, dims ...int) Tensor
    Min(dims ...int) Tensor
    MinTo(dst Tensor, dims ...int) Tensor
    ArgMax(dim int) Tensor
    ArgMin(dim int) Tensor
    
    // Linear algebra operations
    MatMul(other Tensor) Tensor
    MatMulTo(other Tensor, dst Tensor) Tensor
    MatMulTransposed(other Tensor, transposeA, transposeB bool, dst Tensor) Tensor
    MatVecMulTransposed(matrix, vector Tensor, alpha, beta float64) Tensor
    Dot(other Tensor) float64
    Norm(ord int) float64
    Normalize(dim int) Tensor
    NormalizeTo(dst Tensor, dim int) Tensor
    AddScaled(other Tensor, alpha float64) Tensor
    AddScaledTo(dst Tensor, other Tensor, alpha float64) Tensor
    
    // Gradient routing
    ScatterAdd(dst, index, value Tensor) Tensor
}
```

**Operations**: 25+ methods
- Reductions: Sum, Mean, Max, Min, ArgMax, ArgMin (with destination variants)
- Linear algebra: MatMul, MatMulTo, MatMulTransposed, MatVecMulTransposed, Dot, Norm, Normalize, AddScaled
- Gradient routing: ScatterAdd

**TensorFlow Alignment**:
- `Sum` → `tf.reduce_sum`
- `Mean` → `tf.reduce_mean`
- `Max` → `tf.reduce_max`
- `Min` → `tf.reduce_min`
- `ArgMax` → `tf.argmax`
- `ArgMin` → `tf.argmin`
- `MatMul` → `tf.matmul`
- `Dot` → `tf.tensordot` (for vectors/matrices)
- `Norm` → `tf.norm`
- `Normalize` → `tf.nn.l2_normalize`
- `ScatterAdd` → `tf.scatter_nd_add`

### 5. TensorActivations Interface

**Purpose**: Activation functions.

```go
type TensorActivations interface {
    // Activation functions (destination-based)
    ReLU(dst Tensor) Tensor
    ReLU6(dst Tensor) Tensor
    LeakyReLU(dst Tensor, alpha float64) Tensor
    ELU(dst Tensor, alpha float64) Tensor
    Sigmoid(dst Tensor) Tensor
    Tanh(dst Tensor) Tensor
    Softmax(dim int, dst Tensor) Tensor
    Softplus(dst Tensor) Tensor
    Swish(dst Tensor) Tensor
    GELU(dst Tensor) Tensor
}
```

**Operations**: 10 methods
- ReLU variants: ReLU, ReLU6, LeakyReLU, ELU
- Sigmoid/Tanh: Sigmoid, Tanh
- Softmax: Softmax
- Other: Softplus, Swish, GELU

**TensorFlow Alignment**:
- `ReLU` → `tf.nn.relu`
- `ReLU6` → `tf.nn.relu6`
- `LeakyReLU` → `tf.nn.leaky_relu`
- `ELU` → `tf.nn.elu`
- `Sigmoid` → `tf.nn.sigmoid`
- `Tanh` → `tf.nn.tanh`
- `Softmax` → `tf.nn.softmax`
- `Softplus` → `tf.nn.softplus`
- `Swish` → `tf.nn.swish`
- `GELU` → `tf.nn.gelu`

### 6. TensorConvolutions Interface

**Purpose**: Convolution operations.

```go
type TensorConvolutions interface {
    // 1D Convolutions
    Conv1D(kernel, bias Tensor, stride, padding int) Tensor
    Conv1DTo(kernel, bias Tensor, dst Tensor, stride, padding int) Tensor
    Conv1DTransposed(kernel, bias Tensor, stride, padding int) Tensor
    Conv1DTransposedTo(kernel, bias Tensor, dst Tensor, stride, padding int) Tensor
    
    // 2D Convolutions
    Conv2D(kernel, bias Tensor, stride, padding []int) Tensor
    Conv2DTo(kernel, bias Tensor, dst Tensor, stride, padding []int) Tensor
    Conv2DTransposed(kernel, bias Tensor, stride, padding []int) Tensor
    Conv2DTransposedTo(kernel, bias Tensor, dst Tensor, stride, padding []int) Tensor
    DepthwiseConv2D(kernel, bias Tensor, stride, padding []int) Tensor
    DepthwiseConv2DTo(kernel, bias Tensor, dst Tensor, stride, padding []int) Tensor
    GroupConv2D(kernel, bias Tensor, stride, padding []int, groups int) Tensor
    GroupConv2DTo(kernel, bias Tensor, dst Tensor, stride, padding []int, groups int) Tensor
    DilatedConv2D(kernel, bias Tensor, stride, padding, dilation []int) Tensor
    DilatedConv2DTo(kernel, bias Tensor, dst Tensor, stride, padding, dilation []int) Tensor
    SeparableConv2D(depthwiseKernel, pointwiseKernel, bias Tensor, stride, padding []int) Tensor
    SeparableConv2DTo(depthwiseKernel, pointwiseKernel, bias Tensor, dst Tensor, stride, padding []int) Tensor
    
    // 3D Convolutions
    Conv3D(kernel, bias Tensor, stride, padding []int) Tensor
    Conv3DTo(kernel, bias Tensor, dst Tensor, stride, padding []int) Tensor
    Conv3DTransposed(kernel, bias Tensor, stride, padding []int) Tensor
    Conv3DTransposedTo(kernel, bias Tensor, dst Tensor, stride, padding []int) Tensor
    
    // Image/Column conversion (for efficient convolution)
    Im2Col(kernelSize, stride, padding []int) Tensor
    Im2ColTo(dst Tensor, kernelSize, stride, padding []int) Tensor
    Col2Im(outputShape, kernelSize, stride, padding []int) Tensor
    Col2ImTo(dst Tensor, outputShape, kernelSize, stride, padding []int) Tensor
}
```

**Operations**: 28 methods
- 1D: Conv1D, Conv1DTo, Conv1DTransposed, Conv1DTransposedTo
- 2D: Conv2D, Conv2DTo, Conv2DTransposed, Conv2DTransposedTo, DepthwiseConv2D, GroupConv2D, DilatedConv2D, SeparableConv2D (with To variants)
- 3D: Conv3D, Conv3DTo, Conv3DTransposed, Conv3DTransposedTo
- Conversion: Im2Col, Im2ColTo, Col2Im, Col2ImTo

**TensorFlow Alignment**:
- `Conv1D` → `tf.nn.conv1d`
- `Conv2D` → `tf.nn.conv2d`
- `Conv2DTransposed` → `tf.nn.conv2d_transpose`
- `DepthwiseConv2D` → `tf.nn.depthwise_conv2d`
- `GroupConv2D` → `tf.nn.conv2d` with groups
- `DilatedConv2D` → `tf.nn.atrous_conv2d`
- `SeparableConv2D` → `tf.nn.separable_conv2d`
- `Conv3D` → `tf.nn.conv3d`
- `Conv3DTransposed` → `tf.nn.conv3d_transpose`

### 7. TensorPooling Interface

**Purpose**: Pooling operations.

```go
type TensorPooling interface {
    // 1D Pooling
    MaxPool1D(kernelSize, stride, padding []int) Tensor
    MaxPool1DTo(dst Tensor, kernelSize, stride, padding []int) Tensor
    AvgPool1D(kernelSize, stride, padding []int) Tensor
    AvgPool1DTo(dst Tensor, kernelSize, stride, padding []int) Tensor
    
    // 2D Pooling
    MaxPool2D(kernelSize, stride, padding []int) Tensor
    MaxPool2DTo(dst Tensor, kernelSize, stride, padding []int) Tensor
    MaxPool2DWithIndices(kernelSize, stride, padding []int) (Tensor, Tensor)
    AvgPool2D(kernelSize, stride, padding []int) Tensor
    AvgPool2DTo(dst Tensor, kernelSize, stride, padding []int) Tensor
    
    // 3D Pooling
    MaxPool3D(kernelSize, stride, padding []int) Tensor
    MaxPool3DTo(dst Tensor, kernelSize, stride, padding []int) Tensor
    AvgPool3D(kernelSize, stride, padding []int) Tensor
    AvgPool3DTo(dst Tensor, kernelSize, stride, padding []int) Tensor
    
    // Global Pooling
    GlobalMaxPool1D() Tensor
    GlobalMaxPool1DTo(dst Tensor) Tensor
    GlobalMaxPool2D() Tensor
    GlobalMaxPool2DTo(dst Tensor) Tensor
    GlobalMaxPool3D() Tensor
    GlobalMaxPool3DTo(dst Tensor) Tensor
    GlobalAvgPool2D() Tensor
    GlobalAvgPool2DTo(dst Tensor) Tensor
    
    // Adaptive Pooling
    AdaptiveMaxPool1D(outputSize []int) Tensor
    AdaptiveMaxPool1DTo(dst Tensor, outputSize []int) Tensor
    AdaptiveMaxPool2D(outputSize []int) Tensor
    AdaptiveMaxPool2DTo(dst Tensor, outputSize []int) Tensor
    AdaptiveMaxPool3D(outputSize []int) Tensor
    AdaptiveMaxPool3DTo(dst Tensor, outputSize []int) Tensor
    AdaptiveAvgPool2D(outputSize []int) Tensor
    AdaptiveAvgPool2DTo(dst Tensor, outputSize []int) Tensor
}
```

**Operations**: 30+ methods
- 1D: MaxPool1D, AvgPool1D (with To variants)
- 2D: MaxPool2D, MaxPool2DWithIndices, AvgPool2D (with To variants)
- 3D: MaxPool3D, AvgPool3D (with To variants)
- Global: GlobalMaxPool1D/2D/3D, GlobalAvgPool2D (with To variants)
- Adaptive: AdaptiveMaxPool1D/2D/3D, AdaptiveAvgPool2D (with To variants)

**TensorFlow Alignment**:
- `MaxPool1D` → `tf.nn.max_pool1d`
- `MaxPool2D` → `tf.nn.max_pool2d`
- `MaxPool3D` → `tf.nn.max_pool3d`
- `AvgPool1D` → `tf.nn.avg_pool1d`
- `AvgPool2D` → `tf.nn.avg_pool2d`
- `AvgPool3D` → `tf.nn.avg_pool3d`
- `GlobalMaxPool2D` → `tf.reduce_max` with spatial dims
- `GlobalAvgPool2D` → `tf.reduce_mean` with spatial dims
- `AdaptiveMaxPool2D` → `tf.nn.adaptive_max_pool2d`
- `AdaptiveAvgPool2D` → `tf.nn.adaptive_avg_pool2d`

### 8. TensorDropout Interface

**Purpose**: Dropout operations.

```go
type TensorDropout interface {
    // Dropout operations
    DropoutForward(mask Tensor) Tensor
    DropoutForwardTo(dst Tensor, mask Tensor) Tensor
    DropoutMask(p, scale float64, rng RNG) Tensor
}
```

**Operations**: 3 methods
- Forward: DropoutForward, DropoutForwardTo
- Mask: DropoutMask

**TensorFlow Alignment**:
- `DropoutForward` → `tf.nn.dropout` (forward pass)
- `DropoutMask` → internal mask generation

### 9. TensorActivationGradients Interface

**Purpose**: Gradient computation for activation functions.

```go
type TensorActivationGradients interface {
    // ReLU gradients
    ReLUGrad(gradOutput Tensor) Tensor
    ReLUGradTo(dst Tensor, gradOutput Tensor) Tensor
    ReLU6Grad(gradOutput Tensor) Tensor
    ReLU6GradTo(dst Tensor, gradOutput Tensor) Tensor
    LeakyReLUGrad(gradOutput Tensor, alpha float64) Tensor
    LeakyReLUGradTo(dst Tensor, gradOutput Tensor, alpha float64) Tensor
    ELUGrad(gradOutput Tensor, alpha float64) Tensor
    ELUGradTo(dst Tensor, gradOutput Tensor, alpha float64) Tensor
    
    // Sigmoid/Tanh gradients
    SigmoidGrad(gradOutput Tensor) Tensor
    SigmoidGradTo(dst Tensor, gradOutput Tensor) Tensor
    TanhGrad(gradOutput Tensor) Tensor
    TanhGradTo(dst Tensor, gradOutput Tensor) Tensor
    
    // Softmax gradients
    SoftmaxGrad(gradOutput Tensor, dim int) Tensor
    SoftmaxGradTo(dst Tensor, gradOutput Tensor, dim int) Tensor
    
    // Other activation gradients
    SoftplusGrad(gradOutput Tensor) Tensor
    SoftplusGradTo(dst Tensor, gradOutput Tensor) Tensor
    SwishGrad(gradOutput Tensor) Tensor
    SwishGradTo(dst Tensor, gradOutput Tensor) Tensor
    GELUGrad(gradOutput Tensor) Tensor
    GELUGradTo(dst Tensor, gradOutput Tensor) Tensor
}
```

**Operations**: 20+ methods
- ReLU variants: ReLUGrad, ReLU6Grad, LeakyReLUGrad, ELUGrad (with To variants)
- Sigmoid/Tanh: SigmoidGrad, TanhGrad (with To variants)
- Softmax: SoftmaxGrad (with To variant)
- Other: SoftplusGrad, SwishGrad, GELUGrad (with To variants)

**TensorFlow Alignment**:
- Gradients align with TensorFlow's automatic differentiation
- `ReLUGrad` → gradient of `tf.nn.relu`
- `SigmoidGrad` → gradient of `tf.nn.sigmoid`
- `TanhGrad` → gradient of `tf.nn.tanh`
- `SoftmaxGrad` → gradient of `tf.nn.softmax`

### 10. TensorConvolutionGradients Interface

**Purpose**: Gradient computation for convolution operations.

```go
type TensorConvolutionGradients interface {
    // 1D Convolution gradients
    Conv1DInputGrad(gradOutput, kernel Tensor, stride, padding int) Tensor
    Conv1DInputGradTo(dst Tensor, gradOutput, kernel Tensor, stride, padding int) Tensor
    Conv1DKernelGrad(gradOutput, kernel Tensor, stride, padding int) Tensor
    Conv1DKernelGradTo(dst Tensor, gradOutput, kernel Tensor, stride, padding int) Tensor
    Conv1DBiasGrad(gradOutput Tensor) Tensor
    Conv1DBiasGradTo(dst Tensor, gradOutput Tensor) Tensor
    
    // 2D Convolution gradients
    Conv2DInputGrad(gradOutput, kernel Tensor, stride, padding []int) Tensor
    Conv2DInputGradTo(dst Tensor, gradOutput, kernel Tensor, stride, padding []int) Tensor
    Conv2DKernelGrad(gradOutput, kernel Tensor, stride, padding []int) Tensor
    Conv2DKernelGradTo(dst Tensor, gradOutput, kernel Tensor, stride, padding []int) Tensor
    Conv2DBiasGrad(gradOutput Tensor) Tensor
    Conv2DBiasGradTo(dst Tensor, gradOutput Tensor) Tensor
    
    // 2D Transposed convolution gradients
    Conv2DTransposedInputGrad(gradOutput, kernel Tensor, stride, padding []int) Tensor
    Conv2DTransposedInputGradTo(dst Tensor, gradOutput, kernel Tensor, stride, padding []int) Tensor
    Conv2DTransposedKernelGrad(gradOutput, kernel Tensor, stride, padding []int) Tensor
    Conv2DTransposedKernelGradTo(dst Tensor, gradOutput, kernel Tensor, stride, padding []int) Tensor
    
    // Depthwise convolution gradients
    DepthwiseConv2DInputGrad(gradOutput, kernel Tensor, stride, padding []int) Tensor
    DepthwiseConv2DInputGradTo(dst Tensor, gradOutput, kernel Tensor, stride, padding []int) Tensor
    DepthwiseConv2DKernelGrad(gradOutput, kernel Tensor, stride, padding []int) Tensor
    DepthwiseConv2DKernelGradTo(dst Tensor, gradOutput, kernel Tensor, stride, padding []int) Tensor
    
    // 3D Convolution gradients
    Conv3DInputGrad(gradOutput, kernel Tensor, stride, padding []int) Tensor
    Conv3DInputGradTo(dst Tensor, gradOutput, kernel Tensor, stride, padding []int) Tensor
    Conv3DKernelGrad(gradOutput, kernel Tensor, stride, padding []int) Tensor
    Conv3DKernelGradTo(dst Tensor, gradOutput, kernel Tensor, stride, padding []int) Tensor
    Conv3DBiasGrad(gradOutput Tensor) Tensor
    Conv3DBiasGradTo(dst Tensor, gradOutput Tensor) Tensor
}
```

**Operations**: 30+ methods
- 1D: Conv1DInputGrad, Conv1DKernelGrad, Conv1DBiasGrad (with To variants)
- 2D: Conv2DInputGrad, Conv2DKernelGrad, Conv2DBiasGrad (with To variants)
- 2D Transposed: Conv2DTransposedInputGrad, Conv2DTransposedKernelGrad (with To variants)
- Depthwise: DepthwiseConv2DInputGrad, DepthwiseConv2DKernelGrad (with To variants)
- 3D: Conv3DInputGrad, Conv3DKernelGrad, Conv3DBiasGrad (with To variants)

**TensorFlow Alignment**:
- Gradients align with TensorFlow's automatic differentiation
- `Conv2DInputGrad` → gradient w.r.t. input of `tf.nn.conv2d`
- `Conv2DKernelGrad` → gradient w.r.t. kernel of `tf.nn.conv2d`
- `Conv2DBiasGrad` → gradient w.r.t. bias of `tf.nn.conv2d`

### 11. TensorPoolingGradients Interface

**Purpose**: Gradient computation for pooling operations.

```go
type TensorPoolingGradients interface {
    // 1D Pooling gradients
    MaxPool1DBackward(gradOutput, indices Tensor, kernelSize, stride, padding []int) Tensor
    MaxPool1DBackwardTo(dst Tensor, gradOutput, indices Tensor, kernelSize, stride, padding []int) Tensor
    AvgPool1DBackward(gradOutput Tensor, kernelSize, stride, padding []int) Tensor
    AvgPool1DBackwardTo(dst Tensor, gradOutput Tensor, kernelSize, stride, padding []int) Tensor
    
    // 2D Pooling gradients
    MaxPool2DBackward(gradOutput, indices Tensor, kernelSize, stride, padding []int) Tensor
    MaxPool2DBackwardTo(dst Tensor, gradOutput, indices Tensor, kernelSize, stride, padding []int) Tensor
    AvgPool2DBackward(gradOutput Tensor, kernelSize, stride, padding []int) Tensor
    AvgPool2DBackwardTo(dst Tensor, gradOutput Tensor, kernelSize, stride, padding []int) Tensor
    
    // 3D Pooling gradients
    MaxPool3DBackward(gradOutput, indices Tensor, kernelSize, stride, padding []int) Tensor
    MaxPool3DBackwardTo(dst Tensor, gradOutput, indices Tensor, kernelSize, stride, padding []int) Tensor
    AvgPool3DBackward(gradOutput Tensor, kernelSize, stride, padding []int) Tensor
    AvgPool3DBackwardTo(dst Tensor, gradOutput Tensor, kernelSize, stride, padding []int) Tensor
}
```

**Operations**: 12 methods
- 1D: MaxPool1DBackward, AvgPool1DBackward (with To variants)
- 2D: MaxPool2DBackward, AvgPool2DBackward (with To variants)
- 3D: MaxPool3DBackward, AvgPool3DBackward (with To variants)

**TensorFlow Alignment**:
- `MaxPool2DBackward` → gradient of `tf.nn.max_pool2d`
- `AvgPool2DBackward` → gradient of `tf.nn.avg_pool2d`

### 12. TensorDropoutGradients Interface

**Purpose**: Gradient computation for dropout operations.

```go
type TensorDropoutGradients interface {
    // Dropout gradients
    DropoutBackward(gradOutput, mask Tensor) Tensor
    DropoutBackwardTo(dst Tensor, gradOutput, mask Tensor) Tensor
}
```

**Operations**: 2 methods
- Backward: DropoutBackward, DropoutBackwardTo

**TensorFlow Alignment**:
- `DropoutBackward` → gradient of `tf.nn.dropout`

## Complete Tensor Interface

The main `Tensor` interface embeds all category interfaces:

```go
type Tensor interface {
    // Core properties and metadata
    TensorCore
    
    // Operation category interfaces
    TensorManipulation
    TensorElementWise
    TensorMath
    TensorActivations
    TensorConvolutions
    TensorPooling
    TensorDropout
    TensorActivationGradients
    TensorConvolutionGradients
    TensorPoolingGradients
    TensorDropoutGradients
}
```

**Total Operations**: ~250+ methods organized into 12 logical interfaces

## Migration Strategy

**⚠️ IMPORTANT: This migration follows a granular, operation-by-operation approach. Each operation is migrated independently with tests and benchmarks before moving to the next operation.**

### Phase 1: Split Existing Interface into Multiple Interfaces ✅ **COMPLETED**

**Goal**: Split the monolithic `Tensor` interface into category interfaces, one operation at a time.

**Status**: ✅ **COMPLETED** - All operations have been successfully moved to category interfaces.

**Completed Work**:

1. **TensorCore interface** ✅:
   - Created `types/tensor_core.go` with `TensorCore` interface
   - Moved 10 operations: `ID()`, `DataType()`, `Data()`, `Shape()`, `Rank()`, `Size()`, `Empty()`, `At()`, `SetAt()`, `Elements()`
   - Updated `Tensor` interface to embed `TensorCore`
   - Verified `eager_tensor.Tensor` still satisfies interface
   - All tests pass

2. **TensorManipulation interface** ✅:
   - Created `types/tensor_manipulation.go` with `TensorManipulation` interface
   - Moved 10 operations: `Clone()`, `Copy()`, `Reshape()`, `Slice()`, `Transpose()`, `TransposeTo()`, `Permute()`, `BroadcastTo()`, `Fill()`, `Unpad()`
   - Updated `Tensor` interface to embed `TensorManipulation`
   - Verified `eager_tensor.Tensor` still satisfies interface
   - All tests pass

3. **TensorElementWise interface** ✅:
   - Created `types/tensor_elementwise.go` with `TensorElementWise` interface
   - Moved 22 operations: `Add()`, `Sub()`, `Mul()`, `Div()`, `Scale()`, `Square()`, `Sqrt()`, `Exp()`, `Log()`, `Pow()`, `Abs()`, `Sign()`, `Cos()`, `Sin()`, `Negative()`, `AddTo()`, `MulTo()`, `Equal()`, `GreaterThan()`, `Greater()`, `Less()`, `Where()`
   - Updated `Tensor` interface to embed `TensorElementWise`
   - Verified `eager_tensor.Tensor` still satisfies interface
   - All tests pass

4. **TensorMath interface** ✅:
   - Created `types/tensor_math.go` with `TensorMath` interface
   - Moved 14 operations: `Sum()`, `Mean()`, `Max()`, `Min()`, `ArgMax()`, `MatMul()`, `MatMulTo()`, `MatMulTransposed()`, `MatVecMulTransposed()`, `Dot()`, `Norm()`, `Normalize()`, `AddScaled()`, `ScatterAdd()`
   - Updated `Tensor` interface to embed `TensorMath`
   - Verified `eager_tensor.Tensor` still satisfies interface
   - All tests pass

5. **TensorActivations interface** ✅:
   - Created `types/tensor_activations.go` with `TensorActivations` interface
   - Moved 4 operations: `ReLU()`, `Sigmoid()`, `Tanh()`, `Softmax()`
   - Updated `Tensor` interface to embed `TensorActivations`
   - Verified `eager_tensor.Tensor` still satisfies interface
   - All tests pass

6. **TensorConvolutions interface** ✅:
   - Created `types/tensor_convolutions.go` with `TensorConvolutions` interface
   - Moved 9 operations: `Conv1D()`, `Conv1DTo()`, `Conv2D()`, `Conv2DTo()`, `Conv2DTransposed()`, `Conv2DKernelGrad()`, `Conv1DKernelGrad()`, `Im2Col()`, `Col2Im()`
   - Updated `Tensor` interface to embed `TensorConvolutions`
   - Verified `eager_tensor.Tensor` still satisfies interface
   - All tests pass

7. **TensorPooling interface** ✅:
   - Created `types/tensor_pooling.go` with `TensorPooling` interface
   - Moved 8 operations: `MaxPool2D()`, `MaxPool2DWithIndices()`, `MaxPool2DBackward()`, `AvgPool2D()`, `AvgPool2DBackward()`, `GlobalAvgPool2D()`, `AdaptiveAvgPool2D()`
   - Updated `Tensor` interface to embed `TensorPooling`
   - Verified `eager_tensor.Tensor` still satisfies interface
   - All tests pass

8. **TensorDropout interface** ✅:
   - Created `types/tensor_dropout.go` with `TensorDropout` interface
   - Moved 2 operations: `DropoutForward()`, `DropoutMask()`
   - Updated `Tensor` interface to embed `TensorDropout`
   - Verified `eager_tensor.Tensor` still satisfies interface
   - All tests pass

**Summary**:
- **Total operations moved**: 79 operations
- **Interfaces created**: 8 category interfaces
- **Main Tensor interface**: Reduced from 388 lines to 72 lines (81% reduction)
- **Files created**: 8 new interface files
- **Files modified**: 1 file (`types/tensor.go`)
- **Status**: All code compiles, all tests pass, no linter errors

**Note**: Gradient interfaces (TensorActivationGradients, TensorConvolutionGradients, TensorPoolingGradients, TensorDropoutGradients) will be created in Phase 4 when missing gradient operations are added.

**Actual Effort**: Completed in single session (all operations moved efficiently)

### Phase 2: Migrate Existing Operations to Support Destination

**Goal**: Add destination-based variants (`*To` methods) for existing operations, one operation at a time.

**Approach**: For each operation that doesn't have a destination variant, add it incrementally.

**Operation List** (process one at a time):

**Element-wise operations**:
1. `SubTo(other Tensor, dst Tensor) Tensor` - Add destination variant for `Sub`
2. `DivTo(other Tensor, dst Tensor) Tensor` - Add destination variant for `Div`
3. `ScaleTo(dst Tensor, scalar float64) Tensor` - Add destination variant for `Scale`
4. `AddScalarTo(dst Tensor, scalar float64) Tensor` - Add destination variant for `AddScalar`
5. `SubScalarTo(dst Tensor, scalar float64) Tensor` - Add destination variant for `SubScalar`
6. `MulScalarTo(dst Tensor, scalar float64) Tensor` - Add destination variant for `MulScalar`
7. `DivScalarTo(dst Tensor, scalar float64) Tensor` - Add destination variant for `DivScalar`
8. `SquareTo(dst Tensor) Tensor` - Add destination variant for `Square`
9. `SqrtTo(dst Tensor) Tensor` - Add destination variant for `Sqrt`
10. `ExpTo(dst Tensor) Tensor` - Add destination variant for `Exp`
11. `LogTo(dst Tensor) Tensor` - Add destination variant for `Log`
12. `PowTo(dst Tensor, power float64) Tensor` - Add destination variant for `Pow`
13. `AbsTo(dst Tensor) Tensor` - Add destination variant for `Abs`
14. `SignTo(dst Tensor) Tensor` - Add destination variant for `Sign`
15. `CosTo(dst Tensor) Tensor` - Add destination variant for `Cos`
16. `SinTo(dst Tensor) Tensor` - Add destination variant for `Sin`
17. `NegativeTo(dst Tensor) Tensor` - Add destination variant for `Negative`
18. `WhereTo(dst Tensor, condition, a, b Tensor) Tensor` - Add destination variant for `Where`

**Math operations**:
19. `SumTo(dst Tensor, dims ...int) Tensor` - Add destination variant for `Sum`
20. `MeanTo(dst Tensor, dims ...int) Tensor` - Add destination variant for `Mean`
21. `MaxTo(dst Tensor, dims ...int) Tensor` - Add destination variant for `Max`
22. `MinTo(dst Tensor, dims ...int) Tensor` - Add destination variant for `Min`
23. `NormalizeTo(dst Tensor, dim int) Tensor` - Add destination variant for `Normalize`
24. `AddScaledTo(dst Tensor, other Tensor, alpha float64) Tensor` - Add destination variant for `AddScaled`

**Manipulation operations**:
25. `FillTo(dst Tensor, value float64) Tensor` - Add destination variant for `Fill`
26. `PadTo(dst Tensor, padding []int, value float64) Tensor` - Add destination variant for `Pad` (if Pad exists)

**Convolution operations** (if missing):
27. `Conv1DTransposedTo(kernel, bias Tensor, dst Tensor, stride, padding int) Tensor`
28. `Conv2DTransposedTo(kernel, bias Tensor, dst Tensor, stride, padding []int) Tensor`
29. `DepthwiseConv2DTo(kernel, bias Tensor, dst Tensor, stride, padding []int) Tensor`
30. `GroupConv2DTo(kernel, bias Tensor, dst Tensor, stride, padding []int, groups int) Tensor`
31. `DilatedConv2DTo(kernel, bias Tensor, dst Tensor, stride, padding, dilation []int) Tensor`
32. `SeparableConv2DTo(depthwiseKernel, pointwiseKernel, bias Tensor, dst Tensor, stride, padding []int) Tensor`
33. `Conv3DTo(kernel, bias Tensor, dst Tensor, stride, padding []int) Tensor`
34. `Conv3DTransposedTo(kernel, bias Tensor, dst Tensor, stride, padding []int) Tensor`

**Pooling operations** (if missing):
35. `MaxPool1DTo(dst Tensor, kernelSize, stride, padding []int) Tensor`
36. `MaxPool2DTo(dst Tensor, kernelSize, stride, padding []int) Tensor`
37. `AvgPool1DTo(dst Tensor, kernelSize, stride, padding []int) Tensor`
38. `AvgPool2DTo(dst Tensor, kernelSize, stride, padding []int) Tensor`
39. `MaxPool3DTo(dst Tensor, kernelSize, stride, padding []int) Tensor`
40. `AvgPool3DTo(dst Tensor, kernelSize, stride, padding []int) Tensor`
41. `GlobalMaxPool1DTo(dst Tensor) Tensor`
42. `GlobalMaxPool2DTo(dst Tensor) Tensor`
43. `GlobalMaxPool3DTo(dst Tensor) Tensor`
44. `GlobalAvgPool2DTo(dst Tensor) Tensor`
45. `AdaptiveMaxPool1DTo(dst Tensor, outputSize []int) Tensor`
46. `AdaptiveMaxPool2DTo(dst Tensor, outputSize []int) Tensor`
47. `AdaptiveMaxPool3DTo(dst Tensor, outputSize []int) Tensor`
48. `AdaptiveAvgPool2DTo(dst Tensor, outputSize []int) Tensor`

**Gradient operations** (all need `To` variants):
49-100+. All gradient operations need `To` variants (see gradient interfaces above)

**Workflow per operation**:
1. Add `OperationTo` method signature to appropriate category interface
2. Implement `OperationTo` in `eager_tensor/` package
3. Add unit test for `OperationTo`
4. Add benchmark for `OperationTo`
5. Verify tests pass
6. Verify benchmark shows expected performance
7. Commit if all checks pass
8. Move to next operation

**Estimated Effort**: 10-15 days (100+ operations, ~1-2 hours per operation including tests)

### Phase 3: Align Existing Operation Names with TensorFlow

**Goal**: Rename existing operations to align with TensorFlow naming, one operation at a time.

**Approach**: For each operation that needs renaming, add TensorFlow-aligned name while keeping old name for backward compatibility.

**Operation Rename List** (process one at a time):

1. `Greater` → Keep as alias, ensure `GreaterThan` is primary (matches `tf.greater`)
2. `Mul` → Add alias `Multiply` (matches `tf.multiply`), keep `Mul` for backward compatibility
3. `Sub` → Add alias `Subtract` (matches `tf.subtract`), keep `Sub` for backward compatibility
4. `Div` → Add alias `Divide` (matches `tf.divide`), keep `Div` for backward compatibility
5. `Scale` → Add alias `ScalarMul` (matches `tf.scalar_mul`), keep `Scale` for backward compatibility
6. `Log` → Already matches `tf.log` ✓
7. `Pow` → Already matches `tf.pow` ✓
8. `Abs` → Already matches `tf.abs` ✓
9. `Sign` → Already matches `tf.sign` ✓
10. `Cos` → Already matches `tf.cos` ✓
11. `Sin` → Already matches `tf.sin` ✓
12. `Negative` → Already matches `tf.negative` ✓
13. `Sum` → Add alias `ReduceSum` (matches `tf.reduce_sum`), keep `Sum` for backward compatibility
14. `Mean` → Add alias `ReduceMean` (matches `tf.reduce_mean`), keep `Mean` for backward compatibility
15. `Max` → Add alias `ReduceMax` (matches `tf.reduce_max`), keep `Max` for backward compatibility
16. `Min` → Add alias `ReduceMin` (matches `tf.reduce_min`), keep `Min` for backward compatibility
17. `ArgMax` → Already matches `tf.argmax` ✓
18. `MatMul` → Already matches `tf.matmul` ✓
19. `Dot` → Add alias `Tensordot` (matches `tf.tensordot`), keep `Dot` for backward compatibility
20. `Norm` → Already matches `tf.norm` ✓
21. `Normalize` → Add alias `L2Normalize` (matches `tf.nn.l2_normalize`), keep `Normalize` for backward compatibility

**Workflow per operation**:
1. Add TensorFlow-aligned method name to interface (as alias or new method)
2. Implement alias/wrapper in `eager_tensor/` that calls existing implementation
3. Add unit test for new name
4. Update documentation
5. Verify tests pass
6. Commit if all checks pass
7. Move to next operation

**Estimated Effort**: 3-5 days (20+ operations, ~30-60 minutes per operation)

### Phase 4: Add Missing Operations

**Goal**: Add operations that exist in TensorFlow but are missing from current implementation, one operation at a time.

**Approach**: Implement missing operations incrementally, following existing patterns.

**Missing Operations List** (process one at a time):

**Comparison operations**:
1. `NotEqual(other Tensor) Tensor` - Element-wise not equal (matches `tf.not_equal`)
2. `GreaterEqual(other Tensor) Tensor` - Element-wise greater than or equal (matches `tf.greater_equal`)
3. `LessEqual(other Tensor) Tensor` - Element-wise less than or equal (matches `tf.less_equal`)

**Reduction operations**:
4. `ArgMin(dim int) Tensor` - Index of minimum element (matches `tf.argmin`)

**Activation functions**:
5. `ReLU6(dst Tensor) Tensor` - ReLU6 activation (matches `tf.nn.relu6`)
6. `LeakyReLU(dst Tensor, alpha float64) Tensor` - Leaky ReLU (matches `tf.nn.leaky_relu`)
7. `ELU(dst Tensor, alpha float64) Tensor` - ELU activation (matches `tf.nn.elu`)
8. `Softplus(dst Tensor) Tensor` - Softplus activation (matches `tf.nn.softplus`)
9. `Swish(dst Tensor) Tensor` - Swish activation (matches `tf.nn.swish`)
10. `GELU(dst Tensor) Tensor` - GELU activation (matches `tf.nn.gelu`)

**Manipulation operations**:
11. `Pad(padding []int, value float64) Tensor` - Pad tensor (matches `tf.pad`)
12. `PadTo(dst Tensor, padding []int, value float64) Tensor` - Pad to destination

**Pooling operations** (if missing):
13. `MaxPool1D(kernelSize, stride, padding []int) Tensor` - 1D max pooling
14. `MaxPool3D(kernelSize, stride, padding []int) Tensor` - 3D max pooling
15. `AvgPool1D(kernelSize, stride, padding []int) Tensor` - 1D average pooling
16. `AvgPool3D(kernelSize, stride, padding []int) Tensor` - 3D average pooling
17. `GlobalMaxPool1D() Tensor` - Global max pooling 1D
18. `GlobalMaxPool2D() Tensor` - Global max pooling 2D
19. `GlobalMaxPool3D() Tensor` - Global max pooling 3D
20. `AdaptiveMaxPool1D(outputSize []int) Tensor` - Adaptive max pooling 1D
21. `AdaptiveMaxPool2D(outputSize []int) Tensor` - Adaptive max pooling 2D
22. `AdaptiveMaxPool3D(outputSize []int) Tensor` - Adaptive max pooling 3D

**Gradient operations** (for new activations):
23. `ReLU6Grad(gradOutput Tensor) Tensor` - ReLU6 gradient
24. `ReLU6GradTo(dst Tensor, gradOutput Tensor) Tensor` - ReLU6 gradient to destination
25. `LeakyReLUGrad(gradOutput Tensor, alpha float64) Tensor` - Leaky ReLU gradient
26. `LeakyReLUGradTo(dst Tensor, gradOutput Tensor, alpha float64) Tensor` - Leaky ReLU gradient to destination
27. `ELUGrad(gradOutput Tensor, alpha float64) Tensor` - ELU gradient
28. `ELUGradTo(dst Tensor, gradOutput Tensor, alpha float64) Tensor` - ELU gradient to destination
29. `SoftmaxGrad(gradOutput Tensor, dim int) Tensor` - Softmax gradient (if missing)
30. `SoftmaxGradTo(dst Tensor, gradOutput Tensor, dim int) Tensor` - Softmax gradient to destination
31. `SoftplusGrad(gradOutput Tensor) Tensor` - Softplus gradient
32. `SoftplusGradTo(dst Tensor, gradOutput Tensor) Tensor` - Softplus gradient to destination
33. `SwishGrad(gradOutput Tensor) Tensor` - Swish gradient
34. `SwishGradTo(dst Tensor, gradOutput Tensor) Tensor` - Swish gradient to destination
35. `GELUGrad(gradOutput Tensor) Tensor` - GELU gradient
36. `GELUGradTo(dst Tensor, gradOutput Tensor) Tensor` - GELU gradient to destination

**Convolution gradient operations** (if missing):
37-60+. All convolution gradient operations (input, kernel, bias gradients for all convolution types)

**Pooling gradient operations** (if missing):
61-72+. All pooling gradient operations (1D, 2D, 3D max/avg pooling gradients)

**Dropout gradient operations**:
73. `DropoutBackward(gradOutput, mask Tensor) Tensor` - Dropout backward pass
74. `DropoutBackwardTo(dst Tensor, gradOutput, mask Tensor) Tensor` - Dropout backward to destination

**Workflow per operation**:
1. Add operation signature to appropriate category interface
2. Implement operation in `eager_tensor/` package
3. Use primitive operations from `fp32` package
4. Add unit test for operation
5. Add benchmark for operation
6. Verify tests pass
7. Verify benchmark shows expected performance
8. Update documentation
9. Commit if all checks pass
10. Move to next operation

**Estimated Effort**: 15-20 days (70+ operations, ~2-3 hours per operation including tests)

### Phase 5: Add Unit Tests and Benchmarks

**Goal**: Ensure comprehensive test coverage and benchmarks for all operations, one operation at a time.

**Approach**: Review each operation and add/update tests and benchmarks incrementally.

**Test Coverage Checklist** (process one operation at a time):

For each operation:
1. **Unit Tests**:
   - Basic functionality test
   - Edge cases (empty tensors, zero dimensions, NaN, Inf)
   - Shape validation tests
   - Numerical accuracy tests (compare with reference implementation)
   - Type conversion tests (if applicable)
   - Destination parameter tests (if operation has `To` variant)

2. **Benchmarks**:
   - Benchmark for in-place operation (if applicable)
   - Benchmark for destination-based operation (if applicable)
   - Compare performance with naive implementation
   - Compare performance with TensorFlow (if possible)

**Workflow per operation**:
1. Review existing tests for operation
2. Identify missing test cases
3. Add missing unit tests
4. Add/update benchmarks
5. Verify tests pass
6. Verify benchmarks show expected performance
7. Update test documentation
8. Commit if all checks pass
9. Move to next operation

**Estimated Effort**: 10-15 days (250+ operations, ~30-60 minutes per operation)

## Implementation Guidelines

### Operation-by-Operation Workflow

For each operation migration:

1. **Interface Update**:
   - Add method to appropriate category interface
   - Update main `Tensor` interface if needed
   - Verify interface compiles

2. **Implementation**:
   - Implement in `eager_tensor/` package
   - Follow existing patterns
   - Use primitive operations from `fp32` package
   - Handle edge cases

3. **Testing**:
   - Add unit tests
   - Add benchmarks
   - Verify tests pass
   - Verify benchmarks show expected performance

4. **Documentation**:
   - Update method documentation
   - Update `SPEC.md` if needed
   - Add examples if helpful

5. **Commit**:
   - Commit after each operation is complete
   - Use descriptive commit messages
   - Reference operation name in commit

### Destination Parameter Pattern

All destination-based operations should follow this pattern:

```go
// Destination-based operation (writes to dst)
func (t Tensor) OperationTo(dst Tensor, ...) Tensor {
    if dst == nil || dst.Empty() {
        dst = NewAs(t) // Create new tensor with same shape/type
    }
    // Validate dst shape matches expected output shape
    // Perform operation writing to dst
    return dst
}
```

### Testing Requirements

Each operation must have:
- **Unit tests**: Basic functionality, edge cases, shape validation, numerical accuracy
- **Benchmarks**: Performance comparison with naive implementation and/or TensorFlow
- **Documentation**: Clear method documentation with examples

### Commit Strategy

- **One operation per commit**: Each operation migration is a separate commit
- **Descriptive messages**: Commit messages should clearly indicate which operation was migrated
- **Test-driven**: Only commit after tests pass and benchmarks show expected performance

### TensorFlow Alignment Guidelines

1. **Naming**: Use TensorFlow operation names where applicable
2. **Semantics**: Match TensorFlow behavior (e.g., comparison operations return 1.0/0.0)
3. **Parameters**: Match TensorFlow parameter order and types where possible
4. **Edge cases**: Handle edge cases the same way TensorFlow does

### Interface Design Principles

1. **Composition over inheritance**: Use interface embedding
2. **Single responsibility**: Each interface has a clear purpose
3. **Readability**: Main `Tensor` interface is readable by embedding category interfaces
4. **Extensibility**: Easy to add new operation categories
5. **Backward compatibility**: Existing code continues to work

## Benefits of New Structure

1. **Better Organization**: Operations grouped by logical category
2. **Improved Readability**: Main interface shows structure at a glance
3. **Easier Navigation**: Find operations by category
4. **Better Documentation**: Each interface can be documented separately
5. **Extensibility**: Easy to add new operation categories
6. **TensorFlow Alignment**: API aligns with TensorFlow for familiarity
7. **Efficiency**: Destination-based operations enable memory reuse
8. **Type Safety**: Interface structure provides better type checking

## Success Criteria

1. ✅ All operations organized into logical interfaces - **COMPLETED (Phase 1)**
2. ✅ Main `Tensor` interface embeds all category interfaces - **COMPLETED (Phase 1)**
3. ⏳ All operations have destination-based variants - **IN PROGRESS (Phase 2)**
4. ⏳ API aligns with TensorFlow naming and semantics - **PENDING (Phase 3)**
5. ✅ All existing functionality preserved - **COMPLETED (Phase 1)**
6. ⏳ Comprehensive test coverage - **PENDING (Phase 5)**
7. ⏳ Documentation updated - **PENDING (Phase 5)**
8. ⏳ Migration guide provided - **PENDING (Phase 5)**

## Timeline Estimate

- **Phase 1**: ✅ **COMPLETED** (Split interface into multiple interfaces) - Completed in single session
- **Phase 2**: ⏳ **IN PROGRESS** (Migrate existing operations to support destination, one operation at a time) - Estimated 10-15 days
- **Phase 3**: ⏳ **PENDING** (Align existing operation names with TensorFlow, one operation at a time) - Estimated 3-5 days
- **Phase 4**: ⏳ **PENDING** (Add missing operations, one operation at a time) - Estimated 15-20 days
- **Phase 5**: ⏳ **PENDING** (Add unit tests and benchmarks, one operation at a time) - Estimated 10-15 days

**Total Remaining**: 38-55 days (~7.5-11 weeks)

**Note**: Timeline assumes one operation at a time with full test coverage and benchmarks for each operation. This ensures high quality and incremental progress.

## Progress Tracking

### Phase 1: Interface Splitting ✅ **COMPLETED**

**Completion Date**: Current session

**Results**:
- ✅ 8 category interfaces created
- ✅ 79 operations successfully moved from monolithic interface
- ✅ Main Tensor interface reduced from 388 lines to 72 lines
- ✅ All code compiles successfully
- ✅ All tests pass
- ✅ No breaking changes to existing API

**Files Created**:
- `types/tensor_core.go` (10 operations)
- `types/tensor_manipulation.go` (10 operations)
- `types/tensor_elementwise.go` (22 operations)
- `types/tensor_math.go` (14 operations)
- `types/tensor_activations.go` (4 operations)
- `types/tensor_convolutions.go` (9 operations)
- `types/tensor_pooling.go` (8 operations)
- `types/tensor_dropout.go` (2 operations)

**Files Modified**:
- `types/tensor.go` (simplified to embed interfaces)

### Phase 2: Add Destination Support ⏳ **NEXT**

**Status**: Ready to begin

**Operations to add**: 100+ destination-based variants (`*To` methods)

## Notes

- This plan focuses on **interface structure and organization**, not implementation details
- Implementation will follow existing patterns in `eager_tensor/` package
- Primitive operations from `fp32` package will be used for all implementations
- Destination-based operations enable zero-allocation patterns when reusing buffers
- TensorFlow alignment improves API familiarity for users coming from TensorFlow
- Interface composition provides better organization while maintaining single comprehensive interface

