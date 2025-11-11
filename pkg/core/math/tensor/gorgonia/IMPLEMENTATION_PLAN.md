# Gorgonia Tensor Implementation Plan

## Approach

All operations must use Gorgonia's native tensor operations (`gorgonia.org/tensor`) or manual implementations using Gorgonia tensor data access. **NO conversion to eager_tensor and back.**

## Implementation Strategy

### 1. Use Native Gorgonia Tensor Operations Where Available
- `tensor.Tanh()` ✓ (already used)
- `tensor.Add()`, `tensor.Mul()`, etc. ✓ (already used)
- Check for: Exp, Log, Pow, Sqrt operations

### 2. Manual Element-wise Operations
For operations not in `gorgonia.org/tensor`, implement manually by:
- Clone/copy to target tensor
- Get underlying data slice `target.dense.Data().([]float32)`
- Iterate and apply operation
- Return result

### 3. Complex Operations (Conv, Pool)
- **Convolutions**: Implement using im2col + MatMul (already have MatMul)
- **Pooling**: Manual iteration with stride/kernel logic
- These are computationally intensive - consider noting performance implications

## Operations to Implement

### Activations (Manual - element-wise)
- [x] ReLU - done
- [x] Sigmoid - done  
- [x] Tanh - done (uses native)
- [ ] Softmax - needs reduction + exp
- [ ] ReLU6 - clamp(x, 0, 6)
- [ ] LeakyReLU - max(alpha*x, x)
- [ ] ELU - x if x>0 else alpha*(exp(x)-1)
- [ ] Softplus - log(1 + exp(x))
- [ ] Swish - x * sigmoid(x)
- [ ] GELU - 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))

### Activation Gradients
- [ ] ReLUGrad - multiply gradOutput by (input > 0)
- [ ] SigmoidGrad - sigmoid * (1 - sigmoid) * gradOutput
- [ ] TanhGrad - (1 - tanh^2) * gradOutput
- [ ] SoftmaxGrad - more complex, needs jacobian

### Helper Operations
- [ ] BroadcastTo - use `tensor.Repeat` or manual
- [ ] AddScaled - a + scale*b (element-wise)
- [ ] Where - conditional selection
- [ ] Pad/Unpad - manual with index mapping

### Math Operations
- [ ] Max/Min reduction - iterate with comparison
- [ ] Norm - sqrt(sum(x^2))
- [ ] L2Normalize - x / norm(x)
- [ ] MatMulTransposed - MatMul with transposed input

### Convolutions (Complex - im2col approach)
- [ ] Conv1D - use im2col + matmul
- [ ] Conv2D - use im2col + matmul  
- [ ] Conv2DTransposed - col2im approach
- [ ] Im2Col - unfold patches into columns
- [ ] Col2Im - fold columns back into image

### Pooling (Manual iteration)
- [ ] MaxPool2D - sliding window max
- [ ] AvgPool2D - sliding window mean
- [ ] GlobalAvgPool2D - mean over spatial dimensions
- [ ] AdaptiveAvgPool2D - adaptive kernel size
- [ ] Pooling gradients - route gradients back

### Normalizations (Complex)
- [ ] BatchNorm - mean/var normalization
- [ ] LayerNorm - normalize over features
- [ ] RMSNorm - root mean square normalization
- [ ] GroupNorm - group-wise normalization

### Dropout
- [ ] DropoutForward - element-wise multiply with mask
- [ ] DropoutMask - random mask generation
- [ ] DropoutBackward - multiply grad with mask

## Priority Order

1. **High Priority** (needed for basic CNNs):
   - Activations: Softmax, LeakyReLU, GELU
   - Helpers: BroadcastTo, AddScaled
   - Pooling: MaxPool2D, AvgPool2D, GlobalAvgPool2D
   - Conv: Conv2D, Im2Col

2. **Medium Priority** (training):
   - Activation gradients
   - Pooling gradients
   - Dropout operations

3. **Lower Priority** (advanced features):
   - Normalizations (BatchNorm, LayerNorm)
   - Transposed convolutions
   - Adaptive pooling

## Notes

- For inference-only use cases, gradient operations can be lower priority
- Convolutions are the most complex - may need significant implementation effort
- Consider referencing eager_tensor implementation for algorithm logic, but use Gorgonia tensors

