# Quantization Algorithms - Design Document

## Overview

This document describes the quantization algorithms implemented in the `learn` package for post-training quantization of neural network models. Quantization reduces model size and can improve inference speed on embedded devices by using lower precision (typically INT8) instead of float32.

## Purpose

- **Memory Reduction**: INT8 uses 4x less memory than float32 (1 byte vs 4 bytes)
- **Performance**: Many embedded accelerators have INT8 SIMD instructions
- **Power Efficiency**: Integer operations consume less power than floating-point
- **Deployment**: Standard technique for edge AI inference (TensorFlow Lite, CoreML, ONNX Runtime)

## Quantization Schemes

### Symmetric Quantization

Maps the range `[-max(|min|, |max|), max(|min|, |max|)]` to `[-127, 127]` (for 8-bit signed).

**Formula**:
```
scale = max(|min|, |max|) / 127
zero_point = 0
quantized = round(real / scale)
real = scale * quantized
```

**Use Cases**: 
- Weights (often have symmetric distribution)
- Simpler implementation (zero point is always 0)

### Asymmetric Quantization

Maps the range `[min, max]` to `[0, 255]` (for 8-bit unsigned) or `[-128, 127]` (for 8-bit signed).

**Formula**:
```
scale = (max - min) / (quantMax - quantMin)
zero_point = quantMin - round(min / scale)
quantized = round(real / scale) + zero_point
real = scale * (quantized - zero_point)
```

**Use Cases**:
- Activations (often non-symmetric, e.g., ReLU outputs [0, max])
- Better precision when distribution is not centered at zero

### Per-Channel Quantization

Each channel (e.g., each output channel in a convolution) gets its own scale and zero point.

**Use Cases**:
- Better accuracy when channels have different value ranges
- Common for weights in convolution layers

### Per-Tensor Quantization

Single scale and zero point for the entire tensor.

**Use Cases**:
- Simpler and faster
- Sufficient when tensor values have uniform distribution

## Calibration Methods

### Min-Max Calibration

Uses the minimum and maximum values from calibration data.

**Pros**:
- Fast and simple
- No data storage required

**Cons**:
- Sensitive to outliers
- May waste quantization range on rare extreme values

### Percentile Calibration

Uses percentile-based range (e.g., 99.9th percentile) to ignore outliers.

**Pros**:
- Robust to outliers
- Better utilization of quantization range

**Cons**:
- Requires storing calibration samples
- Slightly more computation

**Algorithm**:
1. Collect all calibration samples
2. Sort samples
3. Use lower and upper percentile bounds (e.g., 0.05th and 99.95th)
4. Compute quantization params from percentile range

### KL Divergence Calibration

Minimizes KL divergence between float32 and quantized distributions.

**Pros**:
- Optimal accuracy for given bit-width
- Used by TensorFlow Lite

**Cons**:
- Requires storing all calibration samples
- More computation (iterative optimization)

**Algorithm** (simplified):
1. Collect calibration samples
2. Try different bin boundaries
3. Compute KL divergence: D_KL(P_float || P_quantized)
4. Select bin boundaries that minimize KL divergence

## API Design

### Basic Usage

```go
import "github.com/itohio/EasyRobot/x/math/learn"

// Create calibrator
calibrator := learn.NewCalibrator(
    learn.CalibMinMax,           // Calibration method
    learn.QuantSymmetric,       // Quantization scheme
    8,                           // Bits (typically 8)
)

// Collect statistics from calibration data
for _, sample := range calibrationData {
    calibrator.AddTensor(sample.Tensor)
}

// Compute quantization parameters
params, err := calibrator.ComputeParams()
if err != nil {
    log.Fatal(err)
}

// Quantize a tensor
quantized, qp, err := learn.QuantizeTensor(
    floatTensor,
    params,
    learn.QuantSymmetric,
    8,
)

// Dequantize back to float32
dequantized, err := learn.DequantizeTensor(quantized, params)
```

### Model Quantization

```go
// Quantize all parameters in a model
quantParams, err := learn.QuantizeModel(
    model,
    learn.WithScheme(learn.QuantSymmetric),
    learn.WithBits(8),
    learn.WithCalibrationMethod(learn.CalibPercentile),
)

// quantParams is a map from parameter key to QuantizationParams
for key, params := range quantParams {
    fmt.Printf("Parameter %v: scale=%.6f, zero_point=%d\n", key, params.Scale, params.ZeroPoint)
}
```

## Implementation Details

### Quantization Formula

The fundamental quantization equation is:

```
quantized_value = round(real_value / scale) + zero_point
real_value = scale * (quantized_value - zero_point)
```

### Clamping

Values are clamped to the quantized range to prevent overflow:
- For 8-bit signed: `[-128, 127]`
- For 8-bit unsigned: `[0, 255]`

### Rounding

Uses `round()` function (round to nearest integer) for quantization. This minimizes quantization error.

### Error Analysis

Quantization error per value is at most `scale / 2` (half a quantization step).

## Examples

### Example 1: Quantize Weights

```go
// Load trained model
model := loadModel()

// Quantize all weights
quantParams, err := learn.QuantizeModel(
    model,
    learn.WithScheme(learn.QuantSymmetric), // Good for weights
    learn.WithBits(8),
    learn.WithCalibrationMethod(learn.CalibMinMax),
)

// Use quantized weights for inference
// (This would require quantized layer implementations)
```

### Example 2: Calibrate on Representative Dataset

```go
// Create calibrator for activations
calibrator := learn.NewCalibrator(
    learn.CalibPercentile,
    learn.QuantAsymmetric, // Often better for activations
    8,
)
calibrator.SetPercentile(0.999) // Use 99.9th percentile

// Run inference on representative dataset
for _, sample := range calibrationDataset {
    output := model.Forward(sample.Input)
    calibrator.AddTensor(&output)
}

// Compute quantization parameters
params, err := calibrator.ComputeParams()
```

## Future Enhancements

1. **Quantization-Aware Training (QAT)**: Train with fake quantization to improve accuracy
2. **Mixed Precision**: Different bit-widths for different layers
3. **Dynamic Quantization**: Quantize activations at runtime
4. **Per-Channel Support**: Full per-channel quantization for all layer types
5. **INT4 Quantization**: Ultra-low precision for very constrained devices
6. **Block-wise Quantization**: Quantize in blocks for better accuracy

## References

- TensorFlow Lite Quantization: https://www.tensorflow.org/lite/performance/quantization_spec
- PyTorch Quantization: https://pytorch.org/docs/stable/quantization.html
- ONNX Quantization: https://onnxruntime.ai/docs/performance/quantization.html

