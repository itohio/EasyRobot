# Gorgonia Tensor Wrapper

A lightweight wrapper around [gorgonia.org/tensor](https://pkg.go.dev/gorgonia.org/tensor) that implements the EasyRobot `types.Tensor` interface.

## Overview

This package provides a bridge between gorgonia's tensor operations and EasyRobot's tensor abstraction layer, enabling interoperability between the two systems while maintaining the interface contract.

## Features

- **Value Receivers**: All methods use value receivers to match the `types.Tensor` interface requirements
- **Linear Indexing**: Supports linear indexing for multi-dimensional tensors
- **Type Safety**: Automatic type conversion for different data types
- **Interface Compatible**: Implements the full `types.Tensor` interface

## Installation

```bash
go get gorgonia.org/tensor
```

## Usage

### Creating Tensors

```go
import "github.com/itohio/EasyRobot/pkg/core/math/tensor/gorgonia"
import "github.com/itohio/EasyRobot/pkg/core/math/tensor/types"

// Create a new FP32 tensor with shape [2, 3, 4]
t := gorgonia.New(types.FP32, 2, 3, 4)

// Create a tensor with a different data type
t64 := gorgonia.New(types.FP64, 2, 3)

// All elements are initialized to zero
```

### Basic Operations

```go
// Fill tensor with a value
t.Fill(nil, 1.0)

// Element-wise operations
a := gorgonia.New(types.FP32, 2, 3)
a.Fill(nil, 2.0)

b := gorgonia.New(types.FP32, 2, 3)
b.Fill(nil, 3.0)

// Addition
result := a.Add(nil, b) // result[i] = a[i] + b[i]

// Multiplication
result = a.Multiply(nil, b) // result[i] = a[i] * b[i]

// Scalar operations
result = a.ScalarMul(nil, 2.0) // result[i] = a[i] * 2.0
```

### Converting Between Gorgonia and Eager Tensors

```go
import "github.com/itohio/EasyRobot/pkg/core/math/tensor/eager_tensor"

// Explicit conversion using ToEagerTensor/FromEagerTensor
gTensor := gorgonia.New(types.FP32, 2, 3)
gTensor.Fill(nil, 5.0)

// Convert to eager tensor (creates a copy)
eTensor := gTensor.ToEagerTensor()

// Convert from eager tensor
eagerTensor := eager_tensor.New(types.FP32, types.NewShape(2, 3))
gTensor = gorgonia.FromEagerTensor(eagerTensor)

// Automatic conversion using Copy (recommended)
// Copy automatically handles conversion between tensor types
eagerSrc := eager_tensor.New(types.FP32, types.NewShape(2, 3))
eagerSrc.Fill(nil, 42.0)

gorgoniaDst := gorgonia.New(types.FP32, 2, 3)
gorgoniaDst.Copy(eagerSrc) // Automatically converts eager -> gorgonia

// Copy also supports automatic type conversion
eagerFP64 := eager_tensor.New(types.FP64, types.NewShape(2, 3))
gorgoniaFP32 := gorgonia.New(types.FP32, 2, 3)
gorgoniaFP32.Copy(eagerFP64) // Converts FP64 -> FP32 automatically

// All tensors are independent - modifications don't affect each other
```

### Accessing Elements

```go
t := gorgonia.New(types.FP32, 2, 3)

// Multi-dimensional indexing
value := t.At(0, 1) // Get element at row 0, column 1

// Linear indexing
value = t.At(3) // Get element at linear index 3

// Setting values
t.SetAt(5.0, 0, 1) // Set element at row 0, column 1
t.SetAt(7.0, 3)    // Set element at linear index 3
```

### Matrix Operations

```go
// Matrix multiplication
a := gorgonia.New(types.FP32, 2, 3)
b := gorgonia.New(types.FP32, 3, 4)
result := a.MatMul(nil, b) // Shape: [2, 4]

// Transpose
t := gorgonia.New(types.FP32, 2, 3)
transposed := t.Transpose(nil, nil) // Shape: [3, 2]

// Reshape
t = gorgonia.New(types.FP32, 2, 3, 4)
reshaped := t.Reshape(nil, []int{6, 4}) // Shape: [6, 4]
```

### Reductions

```go
t := gorgonia.New(types.FP32, 2, 3)
t.Fill(nil, 2.0)

// Sum all elements
sum := t.Sum(nil, nil) // Scalar: 12.0

// Sum along dimension
sum = t.Sum(nil, []int{0}) // Sum along axis 0

// Mean
mean := t.Mean(nil, nil) // Mean of all elements

// ArgMax
indices := t.ArgMax(nil, 0) // Indices of max along axis 0
```

### Cloning

```go
original := gorgonia.New(types.FP32, 2, 3)
original.Fill(nil, 1.0)

// Deep copy
cloned := original.Clone()

// Modify original - clone is unaffected
original.SetAt(99.0, 0, 0)
```

## Implementation Status

### Fully Implemented
- Core operations (Shape, Rank, Size, At, SetAt, etc.)
- Element-wise operations (Add, Subtract, Multiply, Divide)
- Scalar operations (AddScalar, MulScalar, etc.)
- Unary operations (Square, Sqrt, Exp, Log, Pow, Abs, Sign, Negative, Cos, Sin)
- Comparison operations (Equal, NotEqual, Greater, Less, GreaterEqual, LessEqual)
- Matrix operations (MatMul, Transpose, Reshape)
- Reductions (Sum, Mean, ArgMax, ArgMin)
- Activations (ReLU, Sigmoid, Tanh)
- Utility operations (Clone, Copy, Fill, FillFunc)
- Conversion (ToEagerTensor, FromEagerTensor)

### Partially Implemented
- Max/Min reductions (not yet implemented - gorgonia doesn't have these)

### Not Yet Implemented
Many advanced operations are not yet implemented and will panic if called:
- Batch/Layer/RMS/Instance/Group normalization
- Advanced activation functions (Softmax, ReLU6, LeakyReLU, ELU, Softplus, Swish, GELU)
- Activation gradients (ReLUGrad, SigmoidGrad, TanhGrad, SoftmaxGrad)
- Convolution operations (Conv1D, Conv2D, Conv2DTransposed)
- Pooling operations (MaxPool2D, AvgPool2D, GlobalAvgPool2D)
- Dropout operations
- Broadcast, Pad, Unpad
- Some comparison with scalars (EqualScalar, NotEqualScalar, etc.)
- Where (conditional operation)
- Norm, L2Normalize
- AddScaled, ScatterAdd

## Notes

- The package now requires a data type parameter in `New()`: `gorgonia.New(types.FP32, shape...)`
- Supported data types: FP32, FP64, INT, INT32, INT64, INT16, INT8
- Gorgonia operations are often in-place; this wrapper creates clones when necessary to maintain immutability
- **Conversion between tensor types**:
  - `Copy()` method automatically handles conversion between gorgonia and eager tensors
  - `Copy()` also performs automatic type conversion (e.g., FP64 → FP32)
  - All conversions create independent copies (data is not shared)
  - Use `Copy()` for seamless interoperability between tensor implementations
- For Go 1.24+, set `ASSUME_NO_MOVING_GC_UNSAFE_RISK_IT_WITH=go1.24` when running tests due to gorgonia dependencies

## Example

```go
package main

import (
    "fmt"
    "github.com/itohio/EasyRobot/pkg/core/math/tensor/gorgonia"
    "github.com/itohio/EasyRobot/pkg/core/math/tensor/types"
)

func main() {
    // Create a 2x3 matrix
    a := gorgonia.New(types.FP32, 2, 3)

    // Fill with values
    for i := 0; i < 2; i++ {
        for j := 0; j < 3; j++ {
            a.SetAt(float64(i*3+j+1), i, j)
        }
    }

    // Create another matrix
    b := gorgonia.New(types.FP32, 2, 3)
    b.Fill(nil, 2.0)

    // Add them
    result := a.Add(nil, b)

    // Print result
    for i := 0; i < 2; i++ {
        for j := 0; j < 3; j++ {
            fmt.Printf("%.1f ", result.At(i, j))
        }
        fmt.Println()
    }
}
```

## Benchmarks

Comprehensive benchmarks are available comparing gorgonia and eager_tensor implementations. See [BENCHMARKS.md](../BENCHMARKS.md) for detailed performance comparisons.

### Key Performance Highlights

- **MatMul**: Gorgonia is 2-43x faster (larger matrices = bigger speedup)
- **Add/Multiply**: Eager tensor is 2-4x faster
- **ReLU**: Eager tensor is 2x faster
- **Sum**: Gorgonia is 3.8x faster
- **Transpose**: Eager tensor is 234x faster (view vs copy)

Run benchmarks:

```bash
# Comparative benchmarks (eager vs gorgonia)
ASSUME_NO_MOVING_GC_UNSAFE_RISK_IT_WITH=go1.24 go test -bench=. -benchmem ./pkg/core/math/tensor/bench_test.go

# Gorgonia-specific benchmarks
ASSUME_NO_MOVING_GC_UNSAFE_RISK_IT_WITH=go1.24 go test -bench=. -benchmem ./pkg/core/math/tensor/gorgonia/
```

## Testing

The package includes comprehensive tests:

```bash
ASSUME_NO_MOVING_GC_UNSAFE_RISK_IT_WITH=go1.24 go test ./pkg/core/math/tensor/gorgonia/...
```

## Contributing

When adding new operations:
1. Ensure methods use value receivers
2. Handle type conversions appropriately
3. Support linear indexing where applicable
4. Add tests for new functionality

## License

This package follows the same license as the EasyRobot project.

