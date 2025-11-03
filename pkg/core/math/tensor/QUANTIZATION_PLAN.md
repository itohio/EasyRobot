# INT8 Quantized Tensor Computation - Design Document

## Overview

This document describes the design and architecture for generalizing the `math/tensor` package to support INT8 quantized computations for neural network inference. The goal is to enable efficient low-precision inference on embedded systems while maintaining compatibility with the existing float32-based architecture.

**For the detailed step-by-step implementation roadmap, see [QUANTIZATION_IMPLEMENTATION.md](./QUANTIZATION_IMPLEMENTATION.md).**

## Motivation

- **Memory Reduction**: INT8 uses 4x less memory than float32 (1 byte vs 4 bytes)
- **Performance**: Many embedded accelerators (ARM NEON, DSPs) have INT8 SIMD instructions
- **Power Efficiency**: Integer operations consume less power than floating-point
- **Deployment**: INT8 quantization is standard for edge AI inference (TensorFlow Lite, CoreML)

## Design Goals

1. **Backward Compatibility**: All existing float32 code continues to work unchanged
2. **Zero Runtime Overhead**: Quantization support adds no cost to float32 operations
3. **Type Safety**: Quantization parameters are explicit and validated
4. **Composability**: Quantized tensors work seamlessly with existing tensor operations
5. **Performance**: Optimized implementations for quantized operations

## Architecture Overview

### Approach: Generic Tensor with Type Parameter

Instead of creating separate types for each data type, we use Go generics (introduced in Go 1.18) to create a generic `Tensor[T]` structure that works with any numeric type.

### Key Components

```
┌─────────────────────────────────────────────────────────────────┐
│                        Generic Tensor[T]                         │
│  - Dim: []int                                                   │
│  - Data: []T                                                    │
│  - Operations: Add, Mul, MatMul, Conv2D, etc.                  │
└─────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
         ┌─────────────────┐          ┌─────────────────────────┐
         │  Tensor[float32]│          │   Tensor[QuantizedI8]   │
         │                 │          │                         │
         │  Existing code  │          │  Quantized values       │
         │  Works as-is    │          │  with scale/zero_point  │
         └─────────────────┘          └─────────────────────────┘
```

## Implementation Phases

### Phase 1: Generic Tensor Infrastructure

**Goal**: Create generic tensor type without breaking existing code

#### 1.1 Define Generic Tensor Type

```go
// File: tensor/generic.go

package tensor

import (
    "fmt"
    "math"
)

// Tensor is a generic multi-dimensional array
type Tensor[T Number] struct {
    Dim  []int
    Data []T
}

// Number is a constraint for numeric types
type Number interface {
    ~int8 | ~int16 | ~int32 | ~int64 |
    ~uint8 | ~uint16 | ~uint32 | ~uint64 |
    ~float32 | ~float64
}

// FloatTensor is a convenience type alias for float32 tensors
type FloatTensor = Tensor[float32]
```

#### 1.2 Create Type-Specific Tensor Alias

```go
// For backward compatibility and convenience
type Tensor = FloatTensor  // Default to float32

// Quantized tensor with scale and zero point
type QuantizedTensor struct {
    Tensor[uint8]  // Use uint8 to represent [-128, 127] range
    Scale     float32
    ZeroPoint int32  // Offset to center quantization around 0
}
```

**Rationale**: Using `uint8` instead of `int8`:
- Easier memory alignment
- Standard convention (many frameworks use uint8)
- Range [-128, 127] can be represented as uint8 with zero_point = 128

#### 1.3 Migrate Existing Code

Strategy: Create generic implementations alongside existing code, test thoroughly, then switch default.

```go
// Step 1: Create generic versions
func (t *Tensor[T]) genericAdd(other *Tensor[T]) *Tensor[T]
func (t *Tensor[T]) genericMul(other *Tensor[T]) *Tensor[T]
// ... etc

// Step 2: Update existing methods to call generics
func (t *Tensor) Add(other *Tensor) *Tensor {
    return (*genericTensor)(t).genericAdd((*genericTensor)(other))
}

// Step 3: Eventually deprecate old type, make generic default
```

**File Organization**:
```
tensor/
├── dense.go              # Existing Tensor struct (float32)
├── generic.go            # New generic Tensor[T]
├── quantized.go          # QuantizedTensor + quantization ops
├── tensor_math.go        # Float32 operations (existing)
├── generic_math.go       # Generic operations
├── quantized_ops.go      # Quantized-specific operations
└── ...
```

### Phase 2: Quantization Infrastructure

**Goal**: Implement quantization/dequantization and parameter management

#### 2.1 Quantization Parameters

```go
// QuantizationParams holds scale and zero point
type QuantizationParams struct {
    Scale     float32  // Scale factor: real = scale * (quantized - zero_point)
    ZeroPoint int32    // Zero point in quantized space
}

// Min/Max quantization (symmetric or asymmetric)
type QuantizationScheme int

const (
    QuantSymmetric QuantizationScheme = iota  // [-max(abs(min), abs(max)), max(abs(min), abs(max))]
    QuantAsymmetric                           // [min, max]
    QuantPerChannel                           // Per-channel quantization
    QuantPerTensor                            // Single scale/zero_point for entire tensor
)
```

#### 2.2 Quantization Functions

```go
// QuantizeFloat32 converts float32 tensor to quantized
func QuantizeFloat32[T Number](input *Tensor[float32], scheme QuantizationScheme) (*QuantizedTensor, error)

// Dequantize converts quantized tensor back to float32
func (qt *QuantizedTensor) Dequantize() *Tensor[float32]

// Requantize converts between quantization schemes
func (qt *QuantizedTensor) Requantize(targetScale float32, targetZeroPoint int32) (*QuantizedTensor, error)
```

**Implementation Details**:

```go
func QuantizeFloat32(input *Tensor[float32], scheme QuantizationScheme) (*QuantizedTensor, error) {
    // Find min/max
    min, max := findMinMax(input)
    
    // Calculate scale and zero point
    var scale float32
    var zeroPoint int32
    
    switch scheme {
    case QuantSymmetric:
        // Symmetric: range is [-r, r] where r = max(|min|, |max|)
        r := max(abs(min), abs(max))
        scale = r / 127.0  // [-127, 127] range in int8
        zeroPoint = 128    // uint8 representation
    case QuantAsymmetric:
        // Asymmetric: range is [min, max]
        scale = (max - min) / 255.0
        zeroPoint = int32(-min / scale)
    }
    
    // Quantize values
    quantized := make([]uint8, len(input.Data))
    for i, val := range input.Data {
        q := int32(math.Round(float64(val/scale))) + zeroPoint
        q = clamp(q, 0, 255)
        quantized[i] = uint8(q)
    }
    
    return &QuantizedTensor{
        Tensor: Tensor[uint8]{Dim: input.Dim, Data: quantized},
        Scale: scale,
        ZeroPoint: zeroPoint,
    }, nil
}
```

### Phase 3: Quantized Operations

**Goal**: Implement quantized versions of critical operations

#### 3.1 Matrix Multiplication (GEMM)

For quantized GEMM, we need **zero-point-based kernels** (like TensorFlow Lite):

**Mathematical Formula**:

```
For matrices A (m×k) and B (k×n), both quantized:
C[i,j] = sum(A[i,k] * B[k,j]) * scale_A * scale_B / scale_C

With zero points:
C_int[i,j] = (sum(A_int[i,k] * B_int[k,j]) 
             - zero_A * sum(B_int[k,j])
             - zero_B * sum(A_int[i,k])
             + zero_A * zero_B * k) * scale_A * scale_B / scale_C
```

**Implementation Strategy**:

```go
// File: tensor/quantized_gemm.go

// GEMM_U8 computes quantized matrix multiplication
func GEMM_U8(
    output, input, weight *QuantizedTensor,
    bias *Tensor[int32],  // Int32 accumulator
) error {
    // Step 1: Compute zero-point corrections
    sumB := computeRowSums(weight)
    sumA := computeColSums(input)
    
    // Step 2: Perform integer GEMM with corrections
    for i := 0; i < output.Rows(); i++ {
        for j := 0; j < output.Cols(); j++ {
            sum := int32(0)
            
            // Standard GEMM
            for k := 0; k < weight.Rows(); k++ {
                sum += int32(input.Data[i*weight.Rows()+k]) *
                       int32(weight.Data[k*output.Cols()+j])
            }
            
            // Apply zero-point corrections
            sum -= input.ZeroPoint * sumB[j]
            sum -= weight.ZeroPoint * sumA[i]
            sum += input.ZeroPoint * weight.ZeroPoint * int32(weight.Rows())
            
            // Add bias if provided
            if bias != nil {
                sum += bias.Data[i*output.Cols()+j]
            }
            
            // Store result
            output.Data[i*output.Cols()+j] = uint8(sum)
        }
    }
    
    // Step 3: Update output scale
    output.Scale = input.Scale * weight.Scale
    
    return nil
}
```

#### 3.2 Convolution Operations

**Strategy**: Use Im2Col + quantized GEMM

```go
// Conv2DQuantized performs quantized convolution
func Conv2DQuantized(
    input, kernel *QuantizedTensor,
    bias *Tensor[int32],
    stride, padding []int,
) *QuantizedTensor {
    // Step 1: Im2Col (unchanged)
    inputCols := input.Im2Col(kernel.KernelSize(), stride, padding)
    
    // Step 2: Quantized GEMM
    outputCols := quantizedGEMM(inputCols, kernel)
    
    // Step 3: Add bias and requantize
    output := addBiasAndRequantize(outputCols, bias, ...)
    
    return output
}
```

#### 3.3 Activation Functions

**Challenge**: ReLU, Sigmoid, Tanh are non-linear

**Solution**: Use lookup tables or approximate functions

```go
// ReLUQuantized applies quantized ReLU
func ReLUQuantized(input *QuantizedTensor) *QuantizedTensor {
    output := &QuantizedTensor{
        Tensor: Tensor[uint8]{Dim: input.Dim, Data: make([]uint8, len(input.Data))},
        Scale: input.Scale,
        ZeroPoint: input.ZeroPoint,
    }
    
    // ReLU: max(0, x)
    // In quantized space: max(zero_point, x)
    threshold := input.ZeroPoint
    
    for i, val := range input.Data {
        if int32(val) > threshold {
            output.Data[i] = val
        } else {
            output.Data[i] = uint8(threshold)
        }
    }
    
    return output
}

// SigmoidQuantized uses lookup table
func SigmoidQuantized(input *QuantizedTensor) *QuantizedTensor {
    // Pre-compute lookup table
    lut := buildSigmoidLUT(input.Scale, input.ZeroPoint)
    
    output := &QuantizedTensor{
        Tensor: Tensor[uint8]{Dim: input.Dim, Data: make([]uint8, len(input.Data))},
        Scale: 1.0/255.0,  // Output range [0, 1]
        ZeroPoint: 0,
    }
    
    for i, val := range input.Data {
        output.Data[i] = lut[val]
    }
    
    return output
}
```

### Phase 4: Primitive Layer Integration

**Goal**: Add quantized operations to primitive layer for optimization

#### 4.1 Primitive Quantized Operations

```go
// File: primitive/quantized.go

// Gemm_U8 performs quantized GEMM
func Gemm_U8(
    output, input, weight []uint8,
    outRows, outCols, k int,
    inputScale, weightScale, outputScale float32,
    inputZero, weightZero, outputZero int32,
) {
    // Optimized implementation
    // - Use SIMD when available
    // - Pre-compute zero-point corrections
    // - Cache-friendly memory access
}

// Conv2D_U8 performs quantized convolution
func Conv2D_U8(
    output, input, kernel []uint8,
    batchSize, inChannels, outChannels int,
    inH, inW, outH, outW int,
    kernelH, kernelW int,
    strideH, strideW int,
    padH, padW int,
    inputScale, kernelScale, outputScale float32,
    inputZero, kernelZero, outputZero int32,
)
```

### Phase 5: Neural Network Layer Integration

**Goal**: Add quantization support to nn.Layer interface

#### 5.1 Quantized Layer Interface

```go
// Layer interface remains the same
type Layer interface {
    Init(inputShape []int) error
    Forward(ctx context.Context, input tensor.Tensor) (tensor.Tensor, error)
    // ...
}

// QuantizedLayer is a separate interface for quantized layers
type QuantizedLayer interface {
    Layer
    QuantizedForward(ctx context.Context, input *QuantizedTensor) (*QuantizedTensor, error)
}

// DenseQuantized implements quantized dense layer
type DenseQuantized struct {
    weight *QuantizedTensor
    bias   *Tensor[int32]
}

func (d *DenseQuantized) QuantizedForward(
    ctx context.Context, 
    input *QuantizedTensor,
) (*QuantizedTensor, error) {
    // Use quantized GEMM
    output := quantizedGEMM(input, d.weight)
    
    // Add bias
    if d.bias != nil {
        addQuantizedBias(output, d.bias)
    }
    
    return output, nil
}
```

#### 5.2 Dynamic Quantization Wrapper

```go
// Wrapper that can work with both float32 and quantized
type AdaptiveLayer struct {
    floatLayer  Layer
    quantLayer  QuantizedLayer
    useQuantized bool
}

func (a *AdaptiveLayer) Forward(ctx context.Context, input tensor.Tensor) (tensor.Tensor, error) {
    if a.useQuantized {
        // Convert to quantized, run, convert back
        quantInput := QuantizeFloat32(input, QuantSymmetric)
        quantOutput, _ := a.quantLayer.QuantizedForward(ctx, quantInput)
        return quantOutput.Dequantize(), nil
    }
    return a.floatLayer.Forward(ctx, input)
}
```

### Phase 6: Calibration and Post-Training Quantization

**Goal**: Implement quantization-aware calibration

#### 6.1 Calibration Dataset

```go
// Calibrator manages quantization calibration
type Calibrator struct {
    datasets []*Tensor[float32]
    scheme   QuantizationScheme
}

// Calibrate computes optimal scale and zero point
func (c *Calibrator) Calibrate(targetLayer Layer) (*QuantizationParams, error) {
    // Collect activation statistics
    stats := c.collectActivationStats(targetLayer)
    
    // Compute optimal quantization
    params := c.computeOptimalQuantization(stats)
    
    return params, nil
}
```

#### 6.2 Quantization-Aware Training (Future)

- Gradually quantize during training
- Use fake quantization ops
- Fine-tune with quantized weights

## Data Flow Examples

### Example 1: Pure Float32 (Current)

```
Input (float32) → Dense(weights: float32) → ReLU → Output (float32)
```

### Example 2: Quantized Inference

```
Input (float32) 
    ↓ Quantize
Input (uint8, scale=0.01, zp=128)
    ↓ QuantizedForward
Dense(weights: uint8, scale=0.005, zp=100, bias: int32)
    ↓ QuantizedReLU
Output (uint8, scale=0.01, zp=128)
    ↓ Dequantize (optional)
Output (float32)
```

### Example 3: Mixed Precision

```
Input (float32)
    ↓
Conv2D(weights: quantized)
    ↓ Output still float32
BatchNorm(float32)
    ↓
ReLU(float32)
```

## Implementation Files

```
pkg/core/math/tensor/
├── dense.go                   # Existing Tensor (float32)
├── generic.go                 # NEW: Generic Tensor[T]
├── quantized.go               # NEW: QuantizedTensor + quantization
├── generic_math.go            # NEW: Generic operations
├── quantized_ops.go           # NEW: Quantized-specific ops
├── quantization.go            # NEW: Quantization utilities
├── tensor_math.go             # Keep: Float32 operations
├── tensor_linalg.go           # Keep: Float32 linear algebra
├── tensor_conv.go             # Keep: Float32 convolution
├── QUANTIZATION_PLAN.md       # This file
└── SPEC.md                    # Update: Add quantization section

pkg/core/math/primitive/
├── quantized.go               # NEW: Quantized BLAS operations
├── quantized_test.go          # NEW: Tests
└── ... (existing files unchanged)

pkg/core/math/nn/
├── quantized.go               # NEW: Quantized layer interfaces
├── quantized_dense.go         # NEW: Quantized Dense layer
├── calibrator.go              # NEW: Quantization calibration
└── ... (existing files unchanged)
```

## Testing Strategy

### 1. Unit Tests

```go
func TestQuantizeDequantize(t *testing.T) {
    input := &Tensor[float32]{Dim: []int{100}, Data: generateRandomData(100)}
    quantized, err := QuantizeFloat32(input, QuantSymmetric)
    assert.NoError(t, err)
    
    output := quantized.Dequantize()
    
    // Check reconstruction error
    for i := range input.Data {
        diff := abs(input.Data[i] - output.Data[i])
        assert.True(t, diff < quantized.Scale/2, "Reconstruction error too large")
    }
}
```

### 2. Integration Tests

```go
func TestQuantizedGEMMAccuracy(t *testing.T) {
    // Create float32 matrices
    a := createRandomMatrix(100, 50)
    b := createRandomMatrix(50, 75)
    
    // Compute reference
    cRef := a.MatMul(b)
    
    // Quantize and compute quantized
    aQuant, _ := QuantizeFloat32(a, QuantSymmetric)
    bQuant, _ := QuantizeFloat32(b, QuantSymmetric)
    cQuant := quantizedGEMM(aQuant, bQuant)
    cTest := cQuant.Dequantize()
    
    // Compare
    mse := computeMSE(cRef, cTest)
    assert.True(t, mse < 0.01, "Quantized GEMM error too large")
}
```

### 3. Performance Benchmarks

```go
func BenchmarkGEMM_Float32(b *testing.B) {
    a := createRandomMatrix(1024, 512)
    c := createRandomMatrix(512, 256)
    b.ResetTimer()
    
    for i := 0; i < b.N; i++ {
        a.MatMul(c)
    }
}

func BenchmarkGEMM_Quantized(b *testing.B) {
    aQuant, _ := QuantizeFloat32(createRandomMatrix(1024, 512), QuantSymmetric)
    cQuant, _ := QuantizeFloat32(createRandomMatrix(512, 256), QuantSymmetric)
    b.ResetTimer()
    
    for i := 0; i < b.N; i++ {
        quantizedGEMM(aQuant, cQuant)
    }
}
```

## Migration Strategy

### Backward Compatibility

1. **Type Alias**: `type Tensor = Tensor[float32]`
2. **No Breaking Changes**: All existing code continues to work
3. **Gradual Adoption**: Users opt-in to quantization features

### Rollout Plan

**Week 1-2: Phase 1**
- Implement generic `Tensor[T]` infrastructure
- Create parallel implementations
- Extensive testing

**Week 3-4: Phase 2**
- Implement quantization/dequantization
- Add quantization utilities
- Calibration infrastructure

**Week 5-6: Phase 3**
- Implement quantized GEMM
- Implement quantized convolution
- Activation function quantization

**Week 7-8: Phase 4**
- Add primitive quantized operations
- SIMD optimizations (if applicable)
- Performance tuning

**Week 9-10: Phase 5**
- Integrate with neural network layers
- Update layer interfaces
- End-to-end testing

**Week 11-12: Phase 6**
- Calibration and validation tools
- Documentation
- Examples and tutorials

## Performance Considerations

### Expected Improvements

- **Memory**: 4x reduction (uint8 vs float32)
- **Speed**: 2-4x on ARM with NEON, 1-2x on x86 without SIMD
- **Accuracy**: < 1% degradation for typical networks

### Optimization Strategies

1. **SIMD**: Use ARM NEON for uint8 operations
2. **Zero-Point Optimization**: Pre-compute zero-point corrections
3. **Parallel Processing**: Batched operations
4. **Memory Layout**: Cache-friendly access patterns
5. **Lookup Tables**: Pre-compute activation functions

## Open Questions

1. **Should we support INT16 quantization?**
   - Higher precision, 2x memory vs INT8
   - Recommendation: Start with INT8, add INT16 if needed

2. **Should quantization be compile-time or runtime?**
   - Compile-time: Better performance, less flexibility
   - Runtime: More flexible, slight overhead
   - Recommendation: Support both

3. **Should we support per-channel quantization?**
   - Better accuracy for convolutions
   - More complex implementation
   - Recommendation: Support it in Phase 3+

4. **How to handle mixed precision?**
   - Some layers quantized, some not
   - Need conversion ops at boundaries
   - Recommendation: Automatic insertion in Phase 5

5. **Should we integrate quantization-aware training?**
   - Better accuracy than post-training quantization
   - Requires training infrastructure
   - Recommendation: Future work

## Success Criteria

✅ All existing tests pass  
✅ Quantized operations produce < 1% accuracy loss  
✅ Memory usage reduced by 4x for quantized models  
✅ Quantized inference is faster than float32 on target hardware  
✅ API is easy to use and well-documented  
✅ No performance regression for float32 operations  

## References

- TensorFlow Lite Quantization: https://www.tensorflow.org/lite/performance/quantization_spec
- PyTorch Quantization: https://pytorch.org/docs/stable/quantization.html
- ONNX Quantization: https://onnxruntime.ai/docs/performance/quantization.html
- "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference" (Jacob et al., 2018)

