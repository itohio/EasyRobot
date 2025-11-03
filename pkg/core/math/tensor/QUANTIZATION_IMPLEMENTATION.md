# Quantization Implementation Roadmap

## Status Overview

### ✅ Completed
- **Phase 0**: Primitive layer quantized operations (`primitive/quantized.go`)
  - `Copy_Q8`: Vector copy for quantized data
  - `Gemm_NN_Q8`: Quantized matrix multiplication with zero-point corrections
  - `Gemm_NN_Q8_Accum`: GEMM with int32 accumulator (for higher precision)
  - `Conv2D_Q8`: Quantized 2D convolution (Im2Col + GEMM)
  - `Im2Col_Q8` / `Col2Im_Q8`: Image format conversion
  - `GemmBatched_Q8`: Batched quantized operations
  - Comprehensive tests and documentation

### 🔄 In Progress
- **Phase 1**: Generic tensor infrastructure

### 📋 Planned
- **Phase 2**: Quantization utilities
- **Phase 3**: Quantized tensor operations
- **Phase 4**: Neural network layer integration
- **Phase 5**: Calibration and deployment tools

---

## Phase 1: Generic Tensor Infrastructure

**Goal**: Introduce generic tensor types without breaking existing code

**Duration**: 2-3 weeks  
**Dependencies**: None  
**Status**: Ready to start

### Step 1.1: Create Generic Tensor Type

**File**: `tensor/generic.go`

```go
package tensor

// Number constraint for numeric types
type Number interface {
    ~int8 | ~int16 | ~int32 | ~int64 |
    ~uint8 | ~uint16 | ~uint32 | ~uint64 |
    ~float32 | ~float64
}

// Tensor is a generic multi-dimensional array
type Tensor[T Number] struct {
    Dim  []int
    Data []T
}

// FloatTensor is a convenience type alias
type FloatTensor = Tensor[float32]

// Default Tensor type (for backward compatibility)
type Tensor = FloatTensor
```

**Tasks**:
- [ ] Create `tensor/generic.go` with generic type definitions
- [ ] Add type constraints and interfaces
- [ ] Create type aliases for backward compatibility
- [ ] Write basic tests for type system

**Acceptance Criteria**:
- Existing code compiles without changes
- Type inference works correctly
- Tests verify type constraints

### Step 1.2: Generic Shape and Size Operations

**File**: `tensor/generic.go` (extend)

```go
// Generic versions of shape/size operations
func (t *Tensor[T]) Shape() []int { ... }
func (t *Tensor[T]) Size() int { ... }
func (t *Tensor[T]) Clone() *Tensor[T] { ... }
func (t *Tensor[T]) Reshape(newShape []int) *Tensor[T] { ... }
```

**Tasks**:
- [ ] Port `Shape()`, `Size()`, `Clone()`, `Reshape()` to generic versions
- [ ] Update existing methods to call generics internally
- [ ] Add tests for all numeric types

**Acceptance Criteria**:
- All shape operations work with generic types
- No performance regression
- Tests pass for float32, int8, uint8, int32

### Step 1.3: Generic Element Access

**File**: `tensor/generic.go` (extend)

```go
func (t *Tensor[T]) At(indices ...int) T { ... }
func (t *Tensor[T]) SetAt(indices []int, value T) { ... }
```

**Tasks**:
- [ ] Port element access to generics
- [ ] Handle type-specific bounds checking
- [ ] Add comprehensive tests

**Acceptance Criteria**:
- Element access works for all types
- Bounds checking is correct
- Tests cover edge cases

---

## Phase 2: Quantization Utilities

**Goal**: Implement quantization/dequantization and parameter management

**Duration**: 1-2 weeks  
**Dependencies**: Phase 1.1-1.2  
**Status**: Blocked

### Step 2.1: Quantization Parameters

**File**: `tensor/quantization.go`

```go
// QuantizationParams holds scale and zero point
type QuantizationParams struct {
    Scale     float32
    ZeroPoint int32
}

// QuantizationScheme defines how to quantize
type QuantizationScheme int

const (
    QuantSymmetric QuantizationScheme = iota
    QuantAsymmetric
    QuantPerChannel
    QuantPerTensor
)
```

**Tasks**:
- [ ] Define quantization parameter structures
- [ ] Create quantization scheme enum
- [ ] Add validation functions
- [ ] Write tests for parameter validation

**Acceptance Criteria**:
- Parameters are well-typed and validated
- Schemes are clearly documented
- Tests verify all schemes

### Step 2.2: Quantization Functions

**File**: `tensor/quantization.go` (extend)

```go
// QuantizeFloat32 converts float32 tensor to quantized
func QuantizeFloat32(input *Tensor[float32], scheme QuantizationScheme) (*QuantizedTensor, error)

// Dequantize converts quantized tensor back to float32
func (qt *QuantizedTensor) Dequantize() *Tensor[float32]

// Requantize converts between quantization schemes
func (qt *QuantizedTensor) Requantize(targetScale float32, targetZeroPoint int32) (*QuantizedTensor, error)
```

**Tasks**:
- [ ] Implement symmetric quantization
- [ ] Implement asymmetric quantization
- [ ] Implement dequantization
- [ ] Implement requantization
- [ ] Add comprehensive tests

**Acceptance Criteria**:
- Quantization/dequantization preserves values within tolerance
- Symmetric and asymmetric schemes work correctly
- Tests verify accuracy (<1% error for typical ranges)

### Step 2.3: QuantizedTensor Type

**File**: `tensor/quantized.go`

```go
// QuantizedTensor wraps Tensor[uint8] with quantization parameters
type QuantizedTensor struct {
    Tensor[uint8]
    Scale     float32
    ZeroPoint int32
}

// Methods for QuantizedTensor
func (qt *QuantizedTensor) GetParams() QuantizationParams
func (qt *QuantizedTensor) Validate() error
```

**Tasks**:
- [ ] Define QuantizedTensor structure
- [ ] Implement parameter getters/setters
- [ ] Add validation methods
- [ ] Write tests

**Acceptance Criteria**:
- QuantizedTensor properly wraps Tensor[uint8]
- Parameters are accessible and validated
- Tests verify integration

---

## Phase 3: Quantized Tensor Operations

**Goal**: Implement quantized versions of critical tensor operations

**Duration**: 3-4 weeks  
**Dependencies**: Phase 1, Phase 2  
**Status**: Blocked

### Step 3.1: Quantized Element-Wise Operations

**File**: `tensor/quantized_math.go`

```go
// Quantized element-wise operations
func (qt *QuantizedTensor) Add(other *QuantizedTensor) *QuantizedTensor
func (qt *QuantizedTensor) Mul(other *QuantizedTensor) *QuantizedTensor
func (qt *QuantizedTensor) Scale(scalar float32) *QuantizedTensor
```

**Tasks**:
- [ ] Implement quantized Add (requires scale matching or requantization)
- [ ] Implement quantized Mul
- [ ] Implement quantized Scale
- [ ] Handle scale mismatches gracefully
- [ ] Add tests

**Acceptance Criteria**:
- Operations match float32 behavior within tolerance
- Scale mismatches are handled correctly
- Tests verify accuracy

### Step 3.2: Quantized Matrix Operations

**File**: `tensor/quantized_linalg.go`

```go
// Quantized matrix multiplication
func (qt *QuantizedTensor) MatMul(other *QuantizedTensor) *QuantizedTensor
```

**Tasks**:
- [ ] Implement quantized MatMul using `primitive.Gemm_NN_Q8`
- [ ] Handle output requantization
- [ ] Support batched operations
- [ ] Add tests comparing to float32 MatMul

**Acceptance Criteria**:
- MatMul produces correct results (<1% error)
- Batched operations work correctly
- Performance is better than dequantize→multiply→quantize

### Step 3.3: Quantized Convolution Operations

**File**: `tensor/quantized_conv.go`

```go
// Quantized convolution
func (qt *QuantizedTensor) Conv2D(kernel, bias *QuantizedTensor, stride, padding []int) *QuantizedTensor
```

**Tasks**:
- [ ] Implement quantized Conv2D using `primitive.Conv2D_Q8`
- [ ] Handle bias addition (int32 accumulator)
- [ ] Support all convolution variants
- [ ] Add tests

**Acceptance Criteria**:
- Conv2D matches float32 results (<1% error)
- All convolution variants work
- Performance is significantly better than float32

### Step 3.4: Quantized Activation Functions

**File**: `tensor/quantized_activations.go`

```go
// Quantized activations
func (qt *QuantizedTensor) ReLU() *QuantizedTensor
func (qt *QuantizedTensor) Sigmoid() *QuantizedTensor  // Uses lookup table
func (qt *QuantizedTensor) Tanh() *QuantizedTensor      // Uses lookup table
```

**Tasks**:
- [ ] Implement ReLU (simple threshold)
- [ ] Implement Sigmoid with lookup table
- [ ] Implement Tanh with lookup table
- [ ] Add tests for accuracy

**Acceptance Criteria**:
- Activations match float32 behavior
- Lookup tables are accurate
- Tests verify correctness

---

## Phase 4: Neural Network Layer Integration

**Goal**: Add quantization support to nn.Layer interface

**Duration**: 3-4 weeks  
**Dependencies**: Phase 3  
**Status**: Blocked

### Step 4.1: Quantized Layer Interface

**File**: `nn/quantized.go`

```go
// QuantizedLayer extends Layer for quantized operations
type QuantizedLayer interface {
    Layer
    QuantizedForward(ctx context.Context, input *QuantizedTensor) (*QuantizedTensor, error)
}
```

**Tasks**:
- [ ] Define QuantizedLayer interface
- [ ] Create base quantized layer structure
- [ ] Add quantization parameter storage
- [ ] Write interface tests

**Acceptance Criteria**:
- Interface is clean and extensible
- Backward compatible with existing Layer interface
- Tests verify interface compliance

### Step 4.2: Quantized Dense Layer

**File**: `nn/layers/quantized_dense.go`

```go
type DenseQuantized struct {
    *Base
    weight *QuantizedTensor
    bias   *Tensor[int32]
    params QuantizationParams
}

func (d *DenseQuantized) QuantizedForward(ctx context.Context, input *QuantizedTensor) (*QuantizedTensor, error)
```

**Tasks**:
- [ ] Implement quantized Dense layer
- [ ] Use `primitive.Gemm_NN_Q8` for computation
- [ ] Handle bias addition correctly
- [ ] Add integration tests

**Acceptance Criteria**:
- Quantized Dense matches float32 Dense output
- Performance is improved
- Tests verify end-to-end correctness

### Step 4.3: Quantized Activation Layers

**File**: `nn/layers/quantized_activations.go`

```go
// Quantized activation layers
type ReLUQuantized struct { ... }
type SigmoidQuantized struct { ... }
type TanhQuantized struct { ... }
```

**Tasks**:
- [ ] Implement quantized ReLU layer
- [ ] Implement quantized Sigmoid layer (with LUT)
- [ ] Implement quantized Tanh layer (with LUT)
- [ ] Add tests

**Acceptance Criteria**:
- Activations match float32 behavior
- Lookup tables are accurate
- Tests verify correctness

### Step 4.4: Quantized Convolution Layers

**File**: `nn/layers/quantized_conv.go`

```go
type Conv2DQuantized struct {
    *Base
    weight *QuantizedTensor
    bias   *Tensor[int32]
    // ... stride, padding, etc.
}
```

**Tasks**:
- [ ] Implement quantized Conv2D layer
- [ ] Use `primitive.Conv2D_Q8`
- [ ] Support all Conv2D features
- [ ] Add comprehensive tests

**Acceptance Criteria**:
- Conv2D matches float32 behavior
- All features work correctly
- Performance is improved

### Step 4.5: Adaptive Layer Wrapper

**File**: `nn/layers/adaptive.go`

```go
// AdaptiveLayer can work with both float32 and quantized
type AdaptiveLayer struct {
    floatLayer  Layer
    quantLayer  QuantizedLayer
    useQuantized bool
}
```

**Tasks**:
- [ ] Create adaptive wrapper
- [ ] Auto-detect tensor type
- [ ] Convert between types as needed
- [ ] Add tests

**Acceptance Criteria**:
- Seamless switching between float32 and quantized
- Conversion overhead is minimal
- Tests verify correctness

---

## Phase 5: Calibration and Deployment Tools

**Goal**: Implement quantization calibration and model conversion

**Duration**: 2-3 weeks  
**Dependencies**: Phase 4  
**Status**: Blocked

### Step 5.1: Calibration Infrastructure

**File**: `nn/calibration.go`

```go
// Calibrator manages quantization calibration
type Calibrator struct {
    datasets []*Tensor[float32]
    scheme   QuantizationScheme
}

func (c *Calibrator) Calibrate(layer Layer) (*QuantizationParams, error)
```

**Tasks**:
- [ ] Implement calibration dataset collection
- [ ] Implement min/max statistics collection
- [ ] Implement optimal scale/zero-point calculation
- [ ] Add tests

**Acceptance Criteria**:
- Calibration produces accurate parameters
- Supports multiple calibration schemes
- Tests verify calibration quality

### Step 5.2: Post-Training Quantization

**File**: `nn/quantization_aware.go`

```go
// QuantizeModel converts float32 model to quantized
func QuantizeModel(model *Model, scheme QuantizationScheme, calibrationData []*Tensor[float32]) (*QuantizedModel, error)
```

**Tasks**:
- [ ] Implement model quantization pass
- [ ] Quantize all layer weights
- [ ] Calculate activation scales
- [ ] Add tests

**Acceptance Criteria**:
- Model quantization preserves accuracy
- All layers are correctly quantized
- Tests verify model correctness

### Step 5.3: Validation Tools

**File**: `nn/quantization_test.go`

```go
// CompareQuantizedModel compares quantized vs float32 model
func CompareQuantizedModel(quantized *QuantizedModel, float32 *Model, testData []*Tensor[float32]) (accuracy float32, err error)
```

**Tasks**:
- [ ] Implement model comparison utilities
- [ ] Add accuracy measurement
- [ ] Add performance benchmarks
- [ ] Create validation reports

**Acceptance Criteria**:
- Tools accurately measure quantization impact
- Reports are comprehensive
- Benchmarks show performance gains

---

## Testing Strategy

### Unit Tests
- Each quantization operation tested independently
- Compare quantized vs float32 results
- Verify quantization parameters

### Integration Tests
- End-to-end quantized inference
- Model quantization round-trip
- Mixed precision scenarios

### Performance Tests
- Benchmark quantized vs float32
- Measure memory usage
- Profile hot paths

### Accuracy Tests
- Compare quantized model accuracy
- Measure quantization error
- Validate against reference implementations

---

## Success Metrics

### Accuracy
- ✅ Quantized model accuracy within 1% of float32
- ✅ Individual operations within quantization tolerance

### Performance
- ✅ 2-4x speedup on quantized operations (on target hardware)
- ✅ 4x memory reduction

### Code Quality
- ✅ All tests pass
- ✅ No performance regression for float32
- ✅ Documentation complete
- ✅ Code reviews passed

---

## Risk Mitigation

### Risk: Breaking existing code
**Mitigation**: Use type aliases, maintain backward compatibility, extensive testing

### Risk: Accuracy degradation
**Mitigation**: Comprehensive validation, calibration tools, fallback to float32

### Risk: Performance not meeting targets
**Mitigation**: Profiling, optimization passes, SIMD where available

### Risk: Complexity explosion
**Mitigation**: Clear interfaces, good documentation, gradual rollout

---

## Timeline Estimate

| Phase | Duration | Start | End |
|-------|----------|-------|-----|
| Phase 0 (Primitives) | ✅ Complete | - | - |
| Phase 1 (Generic Tensors) | 2-3 weeks | Week 1 | Week 3 |
| Phase 2 (Quantization Utils) | 1-2 weeks | Week 4 | Week 5 |
| Phase 3 (Quantized Ops) | 3-4 weeks | Week 6 | Week 9 |
| Phase 4 (NN Integration) | 3-4 weeks | Week 10 | Week 13 |
| Phase 5 (Calibration Tools) | 2-3 weeks | Week 14 | Week 16 |

**Total Estimated Duration**: 14-16 weeks (3.5-4 months)

---

## Next Steps

1. **Review and approve this plan**
2. **Start Phase 1.1**: Create generic tensor type
3. **Set up CI/CD**: Add quantized tests to CI
4. **Create tracking issues**: Break down into GitHub issues/tasks
5. **Schedule reviews**: Weekly progress reviews

---

## References

- [QUANTIZATION_PLAN.md](./QUANTIZATION_PLAN.md) - Design document
- [primitive/quantized.go](../primitive/quantized.go) - Implemented primitives
- "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference" (Jacob et al., 2018)
- TensorFlow Lite Quantization: https://www.tensorflow.org/lite/performance/quantization_spec

