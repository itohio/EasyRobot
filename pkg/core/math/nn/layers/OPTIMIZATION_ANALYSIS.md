# Forward Pass Optimization Analysis

## Overview
This document analyzes forward passes in `nn/layers` to identify opportunities for reducing copying, allocation, and cloning operations **using only existing tensor API operations** from `tensor/types/SPEC.md`.

## Key Findings

### 1. Pooling Operations - Unnecessary Copy Operations

**Issue**: Pooling operations are called without destination parameters, creating unnecessary temporary tensors that are then copied to pre-allocated output tensors.

**Current Pattern** (Inefficient):
```go
// pooling.go lines 114, 282, 393
result := input.MaxPool2D(...)  // Creates new tensor (dst is nil)
output.Copy(result)              // Copies to pre-allocated output
```

**Impact**: 
- Creates temporary tensor allocation
- Extra memory copy operation
- Higher memory bandwidth usage

**Optimization Using Existing API**:
According to `tensor/types/SPEC.md`, all pooling operations now support destination parameters:
- `MaxPool2D(dst Tensor, kernelSize, stride, padding []int) Tensor` (line 241) - "If dst is provided, writes result to dst and returns dst"
- `MaxPool2DWithIndices(dst Tensor, indicesDst Tensor, kernelSize, stride, padding []int) (Tensor, Tensor)` (line 242) - Accepts dst for both output and indices
- `AvgPool2D(dst Tensor, kernelSize, stride, padding []int) Tensor` (line 246) - "If dst is provided, writes result to dst and returns dst"
- `GlobalAvgPool2D(dst Tensor) Tensor` (line 250) - "If dst is provided, writes result to dst and returns dst"
- `AdaptiveAvgPool2D(dst Tensor, outputSize []int) Tensor` (line 251) - "If dst is provided, writes result to dst and returns dst"
- Backward operations also support dst: `MaxPool2DBackward(dst, ...)` (line 243), `AvgPool2DBackward(dst, ...)` (line 247)

**Action**: Pass pre-allocated output tensor as `dst` parameter to eliminate the `Copy()` operation.

**Files Affected**:
- `pooling.go`: Lines 114-121 (MaxPool2D), 282-289 (AvgPool2D), 393-400 (GlobalAvgPool2D)

---

### 2. Dense Layer - Bias Broadcasting Optimization

**Issue**: Bias addition for batch inputs creates intermediate tensors through Reshape and BroadcastTo operations. However, `AddScaled` (from SPEC.md) requires exact shape matching and doesn't support broadcasting.

**Current Pattern** (Inefficient):
```go
// dense.go lines 172-176
biasBroadcast := bias.Reshape(tensor.NewShape(1, outFeatures))  // View (zero-copy)
biasBroadcastFull, err := biasBroadcast.BroadcastTo(output.Shape())  // New tensor allocation
output.Add(output, biasBroadcastFull)  // Uses intermediate tensor
```

**Impact**:
- `BroadcastTo` creates a new tensor allocation
- Extra memory copy operation
- Fallback path (lines 181-187) uses `Slice` (zero-copy views) which is more efficient

**Optimization Using Existing API**:
According to SPEC.md, both `BroadcastTo` and `Slice` now support destination parameters:
- `BroadcastTo(dst Tensor, shape Shape) Tensor` (line 112) - "If dst is provided, writes result to dst and returns dst"
- `Slice(dst Tensor, dim int, start int, length int) Tensor` (line 107) - "If dst is provided, copies sliced data to dst and returns dst"

**Option 1 - Use BroadcastTo with dst** (Simpler):
- Pre-allocate a scratch tensor for broadcast result in `Init()`
- Use `biasBroadcast := bias.Reshape(nil, tensor.NewShape(1, outFeatures))` (view, zero-copy)
- Use `biasBroadcastFull := biasBroadcast.BroadcastTo(scratchTensor, output.Shape())` (writes to pre-allocated tensor)
- Use `output.Add(output, biasBroadcastFull)` (or use output as dst)

**Option 2 - Use slice-based approach** (More efficient):
- The fallback path using `Slice` + `AddScaled` is still more efficient:
  - Loop over batch, use `Slice` with pre-allocated slice tensor, then `AddScaled` directly to output slice
  - Avoids the intermediate broadcast tensor entirely

**Recommendation**: Use Option 2 (slice-based) as it eliminates the intermediate broadcast tensor.

**Files Affected**:
- `dense.go`: Lines 167-189 - Remove BroadcastTo path, use slice-based approach

---

### 3. Activation Layers - Unnecessary Cloning

**Issue**: Dropout layer clones input unnecessarily. According to SPEC.md, `DropoutForward(dst Tensor, mask Tensor)` already accepts a destination parameter.

**Current Pattern** (Inefficient):
```go
// activations.go lines 498, 502
output = input.Clone()  // Unnecessary clone
output = output.DropoutForward(output, d.mask)
```

**Impact**:
- Full tensor clone when `Copy` would be sufficient
- For inference mode (line 502), cloning is unnecessary since input is just passed through

**Optimization Using Existing API**:
- **Training mode**: Use `output.Copy(input)` then `output.DropoutForward(output, d.mask)` 
  - `Copy` copies data into existing tensor (per SPEC.md line 103), avoiding allocation
  - `DropoutForward(dst, mask)` accepts dst parameter (per SPEC.md line 257: "If dst is provided, writes result to dst and returns dst")
- **Inference mode**: Since output shape matches input, can use `output.Copy(input)` or even better, check if we can avoid the copy entirely if input can be used directly
- **Backward**: `DropoutBackward(dst, gradOutput, mask)` also accepts dst parameter (per SPEC.md line 259)

**Files Affected**:
- `activations.go`: Lines 498-502 (Forward), 532 (Backward - already uses Clone, could use Copy)

---

### 4. Softmax Backward - Multiple Intermediate Tensors

**Issue**: Softmax backward pass creates 5 intermediate tensors for a single gradient computation.

**Current Pattern** (Inefficient):
```go
// activations.go lines 357-377
prod := tensor.New(...)           // Intermediate 1
sumTerm := prod.Sum(...)           // Intermediate 2 (Sum with nil dst creates new tensor)
sumBroadcast, err := sumTerm.BroadcastTo(...)  // Intermediate 3
diff := tensor.New(...)            // Intermediate 4
gradInput := tensor.New(...)       // Intermediate 5
```

**Impact**:
- 5 tensor allocations for a single backward pass
- High memory pressure
- Multiple passes over data

**Optimization Using Existing API**:
According to SPEC.md, all operations now support destination parameters:
- `Sum(dst Tensor, dims []int)` (line 171) - "If dst is provided, writes result to dst and returns dst"
- `BroadcastTo(dst Tensor, shape Shape) Tensor` (line 112) - "If dst is provided, writes result to dst and returns dst"
- `Multiply(dst Tensor, other Tensor)` (line 126) - "If dst is provided, result is written to dst"
- `Subtract(dst Tensor, other Tensor)` (line 125) - "If dst is provided, result is written to dst"

**Action**:
- **Pre-allocate scratch tensors**: Store all intermediate tensors in layer's Base during `Init()`
  - `prod` tensor (for gradOutput * output)
  - `sumTerm` tensor (for Sum result)
  - `sumBroadcast` tensor (for broadcast result)
  - `diff` tensor (for gradOutput - sumBroadcast)
  - `gradInput` tensor (for final result)
- **Use destination parameters**: Pass pre-allocated tensors as `dst` to all operations
- **Reuse tensors**: All intermediate tensors can be pre-allocated and reused across backward passes

This completely eliminates all 5 tensor allocations in the backward pass.

**Files Affected**:
- `activations.go`: Lines 337-380 - Pre-allocate scratch tensors in `Init()`, use dst parameters

---

### 5. LSTM Layer - Excessive Cloning

**Issue**: LSTM forward pass clones output and cell state unnecessarily, and creates multiple clones for gate activations.

**Current Pattern** (Inefficient):
```go
// lstm.go lines 171-172
l.hiddenState = output.Clone()      // Clone 1
l.cellState = l.cellState.Clone()  // Clone 2

// lstm.go lines 258-270
iGateSigmoid := iGate.Clone().Sigmoid(nil)  // Clone 3 + in-place sigmoid
fGateSigmoid := fGate.Clone().Sigmoid(nil)  // Clone 4 + in-place sigmoid
gGateTanh := gGate.Clone().Tanh(nil)        // Clone 5 + in-place tanh
oGateSigmoid := oGate.Clone().Sigmoid(nil)  // Clone 6 + in-place sigmoid
```

**Impact**:
- 6+ clone operations per forward pass
- High memory allocation
- Activation functions already support in-place operations (when dst is nil)

**Optimization Using Existing API**:
- **Gate activations**: Pre-allocate gate activation tensors in `Init()` and reuse them
  - Use `Sigmoid(dst Tensor)` and `Tanh(dst Tensor)` with pre-allocated dst (per SPEC.md line 202: "All operations support destination tensor parameter")
  - This eliminates the Clone() calls before activation
- **State storage**: Consider if we need to clone or can store references
  - If output/cellState are modified in-place in computeForward, cloning may be necessary
  - However, if computeForward writes to separate tensors, we can store references
- **Use Copy instead of Clone**: If we need copies, `Copy` may be more efficient when destination exists
  - Per SPEC.md line 103: `Copy(src Tensor)` "Copies data from src tensor into this tensor" and "Returns self for method chaining"
  - This avoids allocation if the destination tensor already exists

**Files Affected**:
- `lstm.go`: Lines 171-172, 258-270, 320-321, 333 - Pre-allocate gate tensors, use dst parameters

---

### 6. Reshape Operations - Zero-Copy Views

**Issue**: Many operations use `Reshape()` which creates views. According to SPEC.md and implementation, `Reshape` creates zero-copy views when possible.

**Current Pattern** (Acceptable):
```go
// Multiple files
inputReshaped := input.Reshape(newShape)  // Creates zero-copy view
output.Copy(inputReshaped)               // Copies from view
```

**Impact**:
- Reshape creates a view tensor (zero-copy, minimal overhead)
- The Copy operation is necessary when writing to a different tensor
- This pattern is acceptable and efficient

**Note**: 
- `Reshape` is zero-copy (creates a view), so intermediate Reshape operations don't cause allocations
- The Copy operations are necessary when data needs to be in a different tensor
- No optimization needed for Reshape usage - it's already optimal

**Files Affected**:
- `utility.go`: Lines 91, 239, 395, 555 - Acceptable pattern
- `dense.go`: Lines 172, 184, 239, 240, 262, 267 - Acceptable pattern
- `conv2d.go`: Lines 232, 238, 282, 296 - Acceptable pattern
- `conv1d.go`: Lines 231, 234, 244 - Acceptable pattern

---

### 7. Conv Layers - Reshape Chains in Backward Pass

**Issue**: Conv2D backward pass creates multiple intermediate tensors. Some operations like `Reshape` are zero-copy views, but others allocate new tensors.

**Current Pattern**:
```go
// conv2d.go lines 274-296
inputCols := input.Im2Col(...)                    // New tensor (necessary)
gradOutputReshaped := gradOutput.Reshape(...)    // View (zero-copy)
gradOutputT := gradOutputReshaped.Transpose(dst, ...) // New tensor (but accepts dst!)
kernelGradMatrix := gradOutputT.MatMul(dst, ...)  // New tensor (but accepts dst!)
kernelGradReshaped := kernelGradMatrix.Reshape(...) // View (zero-copy)
```

**Impact**:
- `Im2Col` creates a new tensor (necessary, no alternative)
- `Transpose` and `MatMul` accept destination parameters but aren't being used
- `Reshape` operations are zero-copy views (acceptable)

**Optimization Using Existing API**:
- **Pre-allocate intermediate tensors**: Allocate `gradOutputT`, `kernelGradMatrix` in `Init()` as scratch tensors
- **Use destination parameters**: 
  - `Transpose(dst Tensor, dims []int)` accepts dst (per SPEC.md line 110: "If dst is provided, writes result to dst and returns dst")
  - `MatMul(dst Tensor, other Tensor)` accepts dst (per SPEC.md line 185: "If dst is provided, writes result to dst and returns dst")
  - Pre-allocate these tensors and pass them as dst parameters
- **Reuse tensors**: These intermediate tensors can be reused across backward passes
- **Note**: `Im2Col` returns a new tensor (per SPEC.md line 233) and has no dst parameter, so this allocation cannot be avoided

**Files Affected**:
- `conv2d.go`: Lines 273-299 - Pre-allocate intermediate tensors, use dst parameters
- `conv1d.go`: Lines 231-244 - Similar optimizations possible

---

## Summary of Recommendations (Using Existing API Only)

### High Priority (High Impact, Low-Medium Effort)

1. **Optimize Dense layer bias addition**
   - **Action**: Remove BroadcastTo path, always use slice-based approach
   - **Rationale**: `Slice` creates zero-copy views, `AddScaled` writes directly to slices
   - **Impact**: Eliminates 1 intermediate tensor allocation per forward pass
   - **Files**: `dense.go` lines 167-189

2. **Reduce LSTM cloning**
   - **Action**: Pre-allocate gate activation tensors in `Init()`, use `Sigmoid(dst)` and `Tanh(dst)` with pre-allocated dst
   - **Rationale**: Activation functions already support destination parameters
   - **Impact**: Eliminates 4 clone operations for gate activations per forward pass
   - **Files**: `lstm.go` lines 258-270

3. **Pre-allocate scratch tensors for Softmax backward**
   - **Action**: Allocate intermediate tensors in `Init()`, use `Sum(dst, dims)` with pre-allocated dst
   - **Rationale**: `Sum` accepts destination parameter, can reuse tensors
   - **Impact**: Reduces 5 tensor allocations to 0 (reuse existing) per backward pass
   - **Files**: `activations.go` lines 337-380

### Medium Priority (Medium Impact, Low Effort)

4. **Optimize Dropout cloning**
   - **Action**: Use `output.Copy(input)` instead of `input.Clone()` for training mode
   - **Rationale**: `Copy` may be more efficient when destination exists, `DropoutForward` already accepts dst
   - **Impact**: Potentially more efficient copy operation
   - **Files**: `activations.go` lines 498-502

5. **Pre-allocate intermediate tensors in Conv backward**
   - **Action**: Pre-allocate `gradOutputT`, `kernelGradMatrix` in `Init()`, use `Transpose(dst, ...)` and `MatMul(dst, ...)` with pre-allocated dst
   - **Rationale**: These operations already support destination parameters
   - **Impact**: Reduces intermediate tensor allocations in backward pass
   - **Files**: `conv2d.go` lines 273-299, `conv1d.go` lines 231-244

### Cannot Optimize (Requires API Changes)

6. **Pooling operations**
   - **Issue**: `MaxPool2D`, `AvgPool2D`, `GlobalAvgPool2D` don't support destination parameters per SPEC.md
   - **Current**: Must use `Copy()` to write to pre-allocated output
   - **Note**: This requires API changes to SPEC.md to optimize

---

## Implementation Notes

### Tensor API Capabilities (from SPEC.md)

**Operations with destination parameters** (can write to pre-allocated tensors):
According to SPEC.md, these operations follow the pattern: "If dst is nil, creates a new tensor. If dst is provided, writes result to dst and returns dst."

- **Element-wise binary**: `Add(dst, other)`, `Subtract(dst, other)`, `Multiply(dst, other)`, `Divide(dst, other)`
- **Element-wise scalar**: `ScalarMul(dst, scalar)`, `AddScalar(dst, scalar)`, `SubScalar(dst, scalar)`, `MulScalar(dst, scalar)`, `DivScalar(dst, scalar)`
- **Element-wise unary**: `Square(dst)`, `Sqrt(dst)`, `Exp(dst)`, `Log(dst)`, `Pow(dst, power)`, `Abs(dst)`, `Sign(dst)`, `Cos(dst)`, `Sin(dst)`, `Negative(dst)`
- **Reductions**: `Sum(dst, dims)`, `Mean(dst, dims)`, `Max(dst, dims)`, `Min(dst, dims)`, `ArgMax(dst, dim)`, `ArgMin(dst, dim)`
- **Linear algebra**: `MatMul(dst, other)`, `MatMulTransposed(dst, other, transposeA, transposeB)`, `MatVecMulTransposed(dst, matrix, vector, alpha, beta)`, `L2Normalize(dst, dim)`
- **Scaled operations**: `AddScaled(dst, other, alpha)` - Note: requires exact shape matching, panics if shapes don't match
- **Activations**: `ReLU(dst)`, `Sigmoid(dst)`, `Tanh(dst)`, `Softmax(dim, dst)`, `ReLU6(dst)`, `LeakyReLU(dst, alpha)`, `ELU(dst, alpha)`, `Softplus(dst)`, `Swish(dst)`, `GELU(dst)`
- **Convolutions**: `Conv1D(dst, kernel, bias, stride, padding)`, `Conv2D(dst, kernel, bias, stride, padding)`, `Conv2DTransposed(dst, kernel, bias, stride, padding)`
- **Transpose**: `Transpose(dst, dims)` - If dst is nil, creates new tensor; if provided, writes to dst
- **Filling**: `Fill(dst, value)`, `Pad(dst, padding, value)`
- **Dropout**: `DropoutForward(dst, mask)`, `DropoutBackward(dst, gradOutput, mask)`
- **Conditional**: `Where(dst, condition, a, b)`

**Operations that return new tensors** (require Copy to write to destination):
- **Pooling**: `MaxPool2D(...)`, `MaxPool2DWithIndices(...)`, `AvgPool2D(...)`, `GlobalAvgPool2D()`, `AdaptiveAvgPool2D(...)` - All return new tensors, no dst parameter
- **Gradient operations**: `MaxPool2DBackward(...)`, `AvgPool2DBackward(...)`, `Conv2DKernelGrad(...)`, `Conv1DKernelGrad(...)` - All return new tensors
- **Image/Column conversion**: `Im2Col(...)`, `Col2Im(...)` - Return new tensors
- **Broadcasting**: `BroadcastTo(shape)` - Returns new tensor (or error). "Currently creates a clone if shapes match exactly."
- **Permutation**: `Permute(dims)` - Returns new tensor (no dst parameter)
- **Unpadding**: `Unpad(padding)` - Returns new tensor
- **Comparison operations**: `Equal(other)`, `GreaterThan(other)`, `Less(other)`, etc. - All return new tensors (no dst parameter)

**Zero-Copy View Operations**:
- `Reshape(newShape)` - "Returns a new tensor with the same data but different shape (zero-copy when possible)"
- `Slice(dim, start, length)` - "Returns a new tensor view (zero-copy when possible)"

**In-Place Operations**:
- When `dst` is `nil`, many operations are in-place (modify the receiver tensor)
- Example: `Add(nil, other)` modifies tensor in-place
- However, some operations create new tensors even when dst is nil (e.g., pooling, comparisons)

**Important Notes from SPEC.md**:
- `AddScaled(dst, other, alpha)` requires exact shape matching - "Panics if shapes don't match" (line 195)
- `BroadcastTo` "Currently creates a clone if shapes match exactly" (line 112)
- All reduction operations support dst parameter (line 169)
- All activation functions support dst parameter (line 202)
- All forward convolution operations support dst parameter (line 218)

### Key Optimization Patterns

1. **Pre-allocate scratch tensors in `Init()`** - Reuse across forward/backward passes
   - Store intermediate computation tensors in layer Base
   - Reuse them across multiple forward/backward passes

2. **Use destination parameters** - Pass pre-allocated tensors to operations that support `dst`
   - Most operations follow pattern: "If dst is nil, creates new tensor. If dst is provided, writes to dst and returns dst"
   - Check SPEC.md to verify which operations support dst parameter

3. **Prefer `Copy` over `Clone`** - When destination exists, `Copy` avoids allocation
   - `Copy(src Tensor)` copies into existing tensor (SPEC.md line 103)
   - `Clone()` always creates new tensor (SPEC.md line 102)

4. **Use zero-copy views** - `Reshape` and `Slice` create views when possible
   - `Reshape`: "zero-copy when possible" (SPEC.md line 106)
   - `Slice`: "zero-copy when possible" (SPEC.md line 107)

5. **Avoid BroadcastTo when possible** - Creates new tensor; use slice-based approaches instead
   - `BroadcastTo` "Returns a new tensor" (SPEC.md line 112)
   - For bias addition, use `Slice` + `AddScaled` instead

6. **Understand operation semantics** - Some operations always create new tensors
   - Pooling operations: Always return new tensors (no dst parameter)
   - Comparison operations: Always return new tensors (no dst parameter)
   - Gradient operations: Always return new tensors (no dst parameter)

---

## Next Steps

1. **Implement high-priority optimizations** - Start with Dense bias, LSTM gates, and Softmax backward
2. **Add scratch tensor allocation to Base** - Extend layer Base to support pre-allocated scratch tensors
3. **Benchmark improvements** - Measure memory allocations and performance gains
4. **Consider API extensions** - For pooling operations, consider adding destination parameters to SPEC.md (future work)

