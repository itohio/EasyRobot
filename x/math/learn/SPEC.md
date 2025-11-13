# Learn Package Specification

## Overview

The `learn` package provides machine learning training utilities for neural networks, including training loops, optimizers, and post-training quantization for model compression and deployment on embedded devices.

**Target Platform**: Embedded and low-power devices
- Primary focus on ARM Cortex-M, ESP32, Raspberry Pi Zero
- Float32 precision throughout for memory efficiency
- Emphasis on memory-efficient operations
- Support for gradient-based training and inference optimization

## Components

### 1. Training Loop (`training.go`)

**Purpose**: Execute a single training step for neural network models.

**Algorithm**:
- Forward pass through the model
- Loss computation between predictions and targets
- Backward pass to compute gradients
- Parameter update using the provided optimizer

**Use Cases**:
- Training neural networks on embedded devices
- Custom training loops
- Integration with existing model architectures

**Key Features**:
- Single training step execution
- Error handling for all phases
- Compatible with any nn.Model and nn.Optimizer
- Memory efficient - no additional allocations

**API Design**:

```go
// TrainStep performs a single training step: forward pass, loss computation, backward pass, and weight update.
// Optimizer must implement the nn.Optimizer interface.
func TrainStep(model *nn.Model, optimizer nn.Optimizer, lossFn nn.LossFunction, input, target tensor.Tensor) (float32, error)
```

**Implementation Details**:
- Validates all inputs are non-nil and non-empty
- Executes forward pass: `output = model.Forward(input)`
- Computes loss: `loss = lossFn.Compute(output, target)`
- Computes gradients: `gradOutput = lossFn.Gradient(output, target)`
- Zeroes gradients: `model.ZeroGrad()`
- Backward pass: `model.Backward(gradOutput)`
- Updates parameters: `model.Update(optimizer)`

**Error Handling**:
- Returns descriptive errors for nil inputs
- Propagates errors from model and loss function operations
- Validates tensor sizes before processing

**Memory Considerations**:
- Reuses existing tensor buffers
- No heap allocations during execution
- Minimal stack usage for embedded systems

### 2. Optimizers (`optimizer.go`)

**Purpose**: Update neural network parameters using gradient-based optimization algorithms.

#### SGD (Stochastic Gradient Descent)

**Algorithm**: `param = param - learning_rate * gradient`

**Use Cases**:
- Simple and memory-efficient optimization
- Baseline optimizer for comparison
- When computational resources are limited

**Key Features**:
- Single learning rate parameter
- Momentum support (planned)
- Memory efficient implementation

**API Design**:

```go
type SGD struct {
    lr float32 // Learning rate
}

func NewSGD(lr float32) *SGD
func (s *SGD) Update(param *layers.Parameter) error
```

#### Adam (Adaptive Moment Estimation)

**Algorithm**:
- Maintains first moment (mean) and second moment (variance) of gradients
- Bias-corrected estimates with exponential moving averages
- Adaptive learning rate per parameter

**Use Cases**:
- Training deep neural networks
- When faster convergence is needed
- Robust to different gradient scales

**Key Features**:
- Configurable beta1, beta2, epsilon parameters
- Per-parameter state management
- Thread-safe implementation using mutexes

**API Design**:

```go
type Adam struct {
    lr      float32
    beta1   float32
    beta2   float32
    epsilon float32
    mu      sync.Mutex
    state   map[uintptr]*adamState
}

func NewAdam(lr, beta1, beta2, epsilon float32) *Adam
func (a *Adam) Update(param *layers.Parameter) error
```

**Implementation Details**:
- Uses pointer-based state tracking for parameter reuse
- Bias correction: `m_hat = m / (1 - beta1^t)`
- Parameter update: `param -= lr * m_hat / (sqrt(v_hat) + epsilon)`
- Handles parameter shape validation
- Supports gradient skipping for non-trainable parameters

**Memory Considerations**:
- SGD: O(1) additional memory per parameter
- Adam: O(parameter_size) additional memory for moment estimates
- State tracking uses pointer-based keys for efficiency

### 3. Quantization (`quantization.go`)

**Purpose**: Compress neural network models through post-training quantization for deployment on memory-constrained devices.

**Algorithm**: Convert float32 weights and activations to lower precision (typically INT8) using calibration data.

#### Quantization Schemes

**Symmetric Quantization**:
- Maps `[-max(|min|, |max|), max(|min|, |max|)]` to `[-127, 127]`
- Zero point fixed at 0
- Best for weights with symmetric distributions

**Asymmetric Quantization**:
- Maps `[min, max]` to `[0, 255]` or `[-128, 127]`
- Dynamic zero point calculation
- Best for activations with asymmetric distributions

**Per-Tensor vs Per-Channel**:
- Per-tensor: Single scale/zero_point for entire tensor
- Per-channel: Individual parameters per channel (e.g., output channels in conv layers)

**API Design**:

```go
type QuantizationScheme int
const (
    QuantSymmetric QuantizationScheme = iota
    QuantAsymmetric
    QuantPerChannel
    QuantPerTensor
)

type QuantizationParams struct {
    Scale     float32
    ZeroPoint int32
}

type CalibrationMethod int
const (
    CalibMinMax CalibrationMethod = iota
    CalibPercentile
    CalibKLDivergence
)

// Core quantization functions
func QuantizeTensor(t *tensor.Tensor, params *QuantizationParams, scheme QuantizationScheme, bits int) (*tensor.Tensor, *QuantizationParams, error)
func DequantizeTensor(quantized *tensor.Tensor, params *QuantizationParams) (*tensor.Tensor, error)

// Calibration for computing quantization parameters
type Calibrator struct {
    // Implementation for collecting statistics
}

func NewCalibrator(method CalibrationMethod, scheme QuantizationScheme, bits int) *Calibrator
func (c *Calibrator) AddSample(val float32)
func (c *Calibrator) AddTensor(t *tensor.Tensor)
func (c *Calibrator) ComputeParams() (*QuantizationParams, error)
```

**Implementation Details**:
- Calibration collects min/max statistics from representative data
- Quantization formula: `quantized = round(real / scale) + zero_point`
- Dequantization formula: `real = scale * (quantized - zero_point)`
- Handles different bit depths (8-bit typical for embedded)
- Supports both signed and unsigned quantization

**Memory Considerations**:
- Quantization reduces memory usage by 4x (float32 → int8)
- Calibration requires storing representative dataset
- Dequantization may be needed for certain operations

**Performance Benefits**:
- **Memory**: 75% reduction in model size
- **Speed**: Integer operations faster on embedded CPUs
- **Power**: Lower precision reduces computational requirements
- **Cache**: Smaller models fit better in cache hierarchies

## Design Principles

### Memory Efficiency

All components designed for embedded constraints:
- No heap allocations in hot paths
- Reusable buffers and state
- Minimal stack usage
- Float32 precision throughout

### Error Handling

Comprehensive error checking:
- Nil pointer validation
- Shape compatibility verification
- Numerical stability checks
- Descriptive error messages

### Interface Compatibility

Works with existing neural network components:
- Compatible with `nn.Model` interface
- Uses `nn.LossFunction` for loss computation
- Integrates with `layers.Parameter` for optimization

### Thread Safety

Adam optimizer uses mutexes for concurrent access to parameter state. SGD is inherently thread-safe.

## Testing Strategy

### Unit Tests

1. **Training Loop**
   - Nil input validation
   - Empty tensor handling
   - Error propagation from model/loss functions
   - Gradient flow verification

2. **Optimizers**
   - Parameter update correctness (SGD: `param -= lr * grad`)
   - Adam convergence properties
   - Shape validation
   - Gradient skipping for non-trainable parameters

3. **Quantization**
   - Quantization/dequantization round-trip accuracy
   - Different schemes (symmetric/asymmetric)
   - Calibration method correctness
   - Edge case handling (zero range, outliers)

### Integration Tests

- End-to-end training loops with simple models (XOR, linear regression)
- MNIST training convergence
- Quantization accuracy preservation

### Benchmarks

- Training step execution time
- Memory usage during training
- Optimizer convergence rates
- Quantization overhead

## Example Usage

### Training Loop

```go
// Create model, optimizer, and loss function
model := nn.NewModelBuilder([]int{784}).
    AddLayer(layers.NewDense(784, 128, layers.WithCanLearn(true))).
    AddLayer(layers.NewReLU("relu")).
    AddLayer(layers.NewDense(128, 10, layers.WithCanLearn(true))).
    Build()

optimizer := learn.NewSGD(0.01)
lossFn := nn.NewCategoricalCrossEntropy(true)

// Training loop
for epoch := 0; epoch < numEpochs; epoch++ {
    for _, batch := range trainingData {
        loss, err := learn.TrainStep(model, optimizer, lossFn, batch.Input, batch.Target)
        if err != nil {
            log.Printf("Training error: %v", err)
            continue
        }
        // Log loss, update progress, etc.
    }
}
```

### Quantization

```go
// Calibrate quantization parameters
calibrator := learn.NewCalibrator(learn.CalibMinMax, learn.QuantSymmetric, 8)
for _, sample := range calibrationData {
    calibrator.AddTensor(sample)
}
params, err := calibrator.ComputeParams()

// Quantize model parameters
for _, param := range model.Parameters() {
    quantized, quantParams, err := learn.QuantizeTensor(param.Data, params, learn.QuantSymmetric, 8)
    // Store quantized parameters and quantization metadata
}
```

## Dependencies

- `github.com/itohio/EasyRobot/x/math/nn`: Neural network interfaces
- `github.com/itohio/EasyRobot/x/math/tensor`: Tensor operations
- Standard library: `fmt`, `math`, `sync`, `unsafe`

## Performance Considerations for Embedded Systems

- **Memory**: Pre-allocated buffers, no dynamic allocation in training loops
- **Cache**: Sequential memory access, small working sets
- **Power**: Minimal data movement, efficient algorithms
- **Determinism**: Predictable execution times for real-time constraints
- **Compatibility**: Works on ARM Cortex-M, ESP32, and similar constrained platforms

## Future Enhancements

1. **Additional Optimizers**: RMSProp, Adagrad, LAMB
2. **Learning Rate Schedulers**: Exponential decay, cosine annealing
3. **Regularization**: Weight decay, dropout integration
4. **Advanced Quantization**: Dynamic quantization, mixed precision
5. **Training Utilities**: Data loaders, metrics, checkpointing
6. **Hardware Acceleration**: SIMD optimizations for supported platforms
7. **Model Compression**: Pruning, knowledge distillation

**Out of Scope**:
- GPU acceleration
- Multi-threading (complexity vs. benefit)
- Large-scale distributed training

