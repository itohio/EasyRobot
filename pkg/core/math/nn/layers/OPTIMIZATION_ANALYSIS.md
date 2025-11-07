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
According to SPEC.md:
- `Sigmoid(dst Tensor)` and `Tanh(dst Tensor)` accept dst (line 202: "All operations support destination tensor parameter")
- `Reshape(dst Tensor, newShape Shape) Tensor` (line 106) - "If dst is provided, copies reshaped data to dst and returns dst"
- `Copy(src Tensor)` (line 103) - Copies into existing tensor, avoids allocation

**Action**:
- **Gate activations**: Pre-allocate gate activation tensors in `Init()` and reuse them
  - Pre-allocate `iGateSigmoid`, `fGateSigmoid`, `gGateTanh`, `oGateSigmoid` tensors
  - Use `iGate.Sigmoid(iGateSigmoid)`, `fGate.Sigmoid(fGateSigmoid)`, etc. with pre-allocated dst
  - This eliminates all Clone() calls before activation
- **State storage**: Use `Copy` instead of `Clone` for state updates
  - Use `l.hiddenState.Copy(output)` instead of `output.Clone()` (if hiddenState is pre-allocated)
  - Or store references if computeForward doesn't modify output in-place
- **MatMul operations**: Pre-allocate intermediate tensors and use dst parameters
  - `MatMulTransposed(dst Tensor, other Tensor, ...)` (line 186) accepts dst

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
According to SPEC.md line 106, `Reshape(dst Tensor, newShape Shape)`:
- If dst is nil: "creates a new tensor view (zero-copy when possible)"
- If dst is provided: "copies reshaped data to dst and returns dst"

**Optimization Opportunity**:
- When using Reshape followed by Copy, consider using `Reshape` with dst parameter directly
- Example: Instead of `inputReshaped := input.Reshape(nil, newShape); output.Copy(inputReshaped)`
- Use: `input.Reshape(output, newShape)` (if output shape matches newShape)
- This eliminates the intermediate view tensor and the Copy operation

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
According to SPEC.md, operations now support destination parameters:
- `Transpose(dst Tensor, dims []int)` (line 110) - "If dst is provided, writes result to dst and returns dst"
- `MatMul(dst Tensor, other Tensor)` (line 185) - "If dst is provided, writes result to dst and returns dst"
- `Reshape(dst Tensor, newShape Shape)` (line 106) - "If dst is provided, copies reshaped data to dst and returns dst"
- `Permute(dst Tensor, dims []int)` (line 111) - "If dst is provided, writes permuted result to dst and returns dst"

**Action**:
- **Pre-allocate intermediate tensors**: Allocate all intermediate tensors in `Init()` as scratch tensors
  - `gradOutputReshaped` - can reuse gradOutput or use separate tensor
  - `gradOutputT` - for transpose result
  - `kernelGradMatrix` - for matrix multiplication result
  - `kernelGradReshaped` - can be a view or separate tensor
- **Use destination parameters**: Pass pre-allocated tensors as `dst` to all operations
- **Reuse tensors**: All intermediate tensors can be reused across backward passes
- **Note**: `Im2Col` returns a new tensor (line 233) and has no dst parameter, so this allocation cannot be avoided

**Files Affected**:
- `conv2d.go`: Lines 273-299 - Pre-allocate intermediate tensors, use dst parameters
- `conv1d.go`: Lines 231-244 - Similar optimizations possible

---

## Summary of Recommendations (Using Existing API Only)

### High Priority (High Impact, Low-Medium Effort)

1. **Optimize Pooling operations**
   - **Action**: Pass pre-allocated output tensor as `dst` parameter to pooling operations
   - **Rationale**: All pooling operations now support destination parameters (SPEC.md lines 241-251)
   - **Impact**: Eliminates temporary tensor allocation and Copy() operation per forward pass
   - **Files**: `pooling.go` lines 114-121 (MaxPool2D), 282-289 (AvgPool2D), 393-400 (GlobalAvgPool2D)

2. **Optimize Dense layer bias addition**
   - **Action**: Use slice-based approach or BroadcastTo with pre-allocated dst
   - **Rationale**: `BroadcastTo` and `Slice` now support dst parameters (SPEC.md lines 112, 107)
   - **Impact**: Eliminates intermediate tensor allocation per forward pass
   - **Files**: `dense.go` lines 167-189

3. **Pre-allocate scratch tensors for Softmax backward**
   - **Action**: Allocate all intermediate tensors in `Init()`, use dst parameters for all operations
   - **Rationale**: All operations (`Sum`, `BroadcastTo`, `Multiply`, `Subtract`) support dst parameters
   - **Impact**: Reduces 5 tensor allocations to 0 per backward pass
   - **Files**: `activations.go` lines 337-380

4. **Reduce LSTM cloning**
   - **Action**: Pre-allocate gate activation tensors in `Init()`, use `Sigmoid(dst)` and `Tanh(dst)` with pre-allocated dst
   - **Rationale**: Activation functions support destination parameters (SPEC.md line 202)
   - **Impact**: Eliminates 4 clone operations for gate activations per forward pass
   - **Files**: `lstm.go` lines 258-270

### Medium Priority (Medium Impact, Low Effort)

5. **Optimize Dropout cloning**
   - **Action**: Use `output.Copy(input)` instead of `input.Clone()` for training mode
   - **Rationale**: `Copy` avoids allocation when destination exists, `DropoutForward` accepts dst
   - **Impact**: More efficient copy operation
   - **Files**: `activations.go` lines 498-502

6. **Pre-allocate intermediate tensors in Conv backward**
   - **Action**: Pre-allocate all intermediate tensors in `Init()`, use `Transpose(dst, ...)`, `MatMul(dst, ...)`, `Reshape(dst, ...)` with pre-allocated dst
   - **Rationale**: All operations support destination parameters (SPEC.md lines 110, 185, 106)
   - **Impact**: Reduces intermediate tensor allocations in backward pass
   - **Files**: `conv2d.go` lines 273-299, `conv1d.go` lines 231-244

7. **Optimize Reshape usage**
   - **Action**: Use `Reshape(dst, newShape)` directly instead of `Reshape(nil, newShape)` followed by `Copy`
   - **Rationale**: `Reshape` supports dst parameter (SPEC.md line 106), eliminates intermediate view
   - **Impact**: Eliminates intermediate view tensor and Copy operation
   - **Files**: Multiple files using Reshape + Copy pattern

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

**Operations that now support destination parameters** (can write to pre-allocated tensors):
- **Pooling**: `MaxPool2D(dst, ...)`, `MaxPool2DWithIndices(dst, indicesDst, ...)`, `AvgPool2D(dst, ...)`, `GlobalAvgPool2D(dst)`, `AdaptiveAvgPool2D(dst, ...)` - All support dst (SPEC.md lines 241-251)
- **Backward pooling**: `MaxPool2DBackward(dst, ...)`, `AvgPool2DBackward(dst, ...)` - Support dst (lines 243, 247)
- **Shape manipulation**: `Reshape(dst, newShape)`, `Slice(dst, dim, start, length)`, `BroadcastTo(dst, shape)`, `Permute(dst, dims)`, `Unpad(dst, padding)` - All support dst (lines 106-112, 117)

**Operations that return new tensors** (no dst parameter):
- **Gradient operations**: `Conv2DKernelGrad(...)`, `Conv1DKernelGrad(...)` - Return new tensors (no dst parameter, lines 227-228)
- **Image/Column conversion**: `Im2Col(...)`, `Col2Im(...)` - Return new tensors (no dst parameter, lines 233-234)
- **Comparison operations**: `Equal(other)`, `GreaterThan(other)`, `Less(other)`, etc. - All return new tensors (no dst parameter, lines 155-161)
- **Mask creation**: `DropoutMask(...)` - Returns new tensor (no dst parameter, line 258)

**In-Place Operations**:
- When `dst` is `nil`, many operations are in-place (modify the receiver tensor)
- Example: `Add(nil, other)` modifies tensor in-place
- However, some operations create new tensors even when dst is nil (e.g., pooling, comparisons)

**Important Notes from SPEC.md**:
- `AddScaled(dst, other, alpha)` requires exact shape matching - "Panics if shapes don't match" (line 195)
- `Reshape(dst, newShape)` - When dst is nil, creates zero-copy view; when dst is provided, copies data to dst (line 106)
- `Slice(dst, dim, start, length)` - When dst is nil, creates new tensor with copied data; when dst is provided, copies to dst (line 107)
- `BroadcastTo(dst, shape)` - Now supports dst parameter (line 112)
- All pooling operations support dst parameter (lines 241-251)
- All reduction operations support dst parameter (line 169)
- All activation functions support dst parameter (line 202)
- All forward convolution operations support dst parameter (line 218)
- Gradient convolution operations (`Conv2DKernelGrad`, `Conv1DKernelGrad`) do NOT support dst parameter (lines 227-228)

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

4. **Use destination parameters for shape operations** - `Reshape`, `Slice`, `BroadcastTo` now support dst
   - `Reshape(dst, newShape)`: When dst provided, copies directly to dst (SPEC.md line 106)
   - `Slice(dst, dim, start, length)`: When dst provided, copies to dst (line 107)
   - `BroadcastTo(dst, shape)`: When dst provided, writes to dst (line 112)
   - Use dst parameters to avoid intermediate tensors

5. **Use pooling operations with dst** - All pooling operations now support destination parameters
   - `MaxPool2D(dst, ...)`, `AvgPool2D(dst, ...)`, `GlobalAvgPool2D(dst)` all accept dst (lines 241-251)
   - Pass pre-allocated output tensor directly to eliminate Copy() operations

6. **Understand operation limitations** - Some operations still don't support dst
   - Gradient convolution: `Conv2DKernelGrad`, `Conv1DKernelGrad` return new tensors (no dst, lines 227-228)
   - Image/Column conversion: `Im2Col`, `Col2Im` return new tensors (no dst, lines 233-234)
   - Comparison operations: Always return new tensors (no dst, lines 155-161)

---

### 8. LSTM Layer - Additional Intermediate Tensor Allocations

**Issue**: LSTM `computeForward` creates multiple intermediate tensors for gates computation and state updates that could be pre-allocated.

**Current Pattern** (Inefficient):
```go
// lstm.go lines 216, 233
gatesTmp := tensor.New(input.DataType(), gatesShape)  // Intermediate 1
hiddenContributionTmp := tensor.New(hiddenState.DataType(), hiddenContributionShape)  // Intermediate 2

// lstm.go line 250
biasFull := biasBroadcast.BroadcastTo(nil, gates.Shape())  // Intermediate 3 (allocates new tensor)

// lstm.go lines 289-298
cellNew := tensor.New(cellState.DataType(), cellState.Shape())  // Intermediate 4
iGateG := tensor.New(iGate.DataType(), iGate.Shape())  // Intermediate 5
cellNewTanhTmp := tensor.New(cellNew.DataType(), cellNew.Shape())  // Intermediate 6
outputNew := tensor.New(oGate.DataType(), oGate.Shape())  // Intermediate 7
```

**Impact**:
- 7+ tensor allocations per forward pass (in addition to gate activation tensors already optimized)
- High memory pressure for LSTM layers
- `BroadcastTo` creates new tensor even though it supports `dst` parameter

**Optimization Using Existing API**:
According to SPEC.md:
- `MatMulTransposed(dst Tensor, ...)` accepts dst parameter (line 186)
- `BroadcastTo(dst Tensor, shape Shape)` accepts dst parameter (line 112)
- `Multiply(dst Tensor, other Tensor)` accepts dst parameter (line 126)
- `Add(dst Tensor, other Tensor)` accepts dst parameter (line 124)
- `Tanh(dst Tensor)` accepts dst parameter (line 206)

**Action**:
- **Pre-allocate scratch tensors**: Store all intermediate tensors in `LSTM` struct during `Init()`
  - `gatesTmp` tensor (for input @ weight_ih.T result)
  - `hiddenContributionTmp` tensor (for hiddenState @ weight_hh.T result)
  - `biasFull` tensor (for broadcast bias result, shape: [batch_size, 4*hidden_size])
  - `cellNew` tensor (for updated cell state)
  - `iGateG` tensor (for iGate * gGate)
  - `cellNewTanhTmp` tensor (for tanh(cellNew))
  - `outputNew` tensor (for oGate * tanh(cellNew))
- **Use destination parameters**: Pass pre-allocated tensors as `dst` to all operations
- **Reuse tensors**: All intermediate tensors can be reused across forward passes
- **Handle dynamic shapes**: For non-batch case, some tensors have different shapes (need conditional allocation or separate tensors)

This reduces 7 tensor allocations to 0 per forward pass (after initial allocation in `Init()`).

**Files Affected**:
- `lstm.go`: Lines 209-298 - Pre-allocate intermediate tensors in `Init()`, use dst parameters

---

### 9. Dense Layer Backward - Unnecessary Tensor Allocations

**Issue**: Dense backward pass creates new tensors for gradInput that could use pre-allocated grad tensor from Base.

**Current Pattern** (Inefficient):
```go
// dense.go line 255 (single sample case)
gradInput2D := tensor.New(gradOutput.DataType(), tensor.NewShape(1, d.inFeatures))  // Intermediate
gradInput := gradInput2D.Reshape(nil, tensor.NewShape(d.inFeatures))  // View (zero-copy)

// dense.go line 303 (batch case)
gradInput := tensor.New(gradOutput.DataType(), tensor.NewShape(batchSize, d.inFeatures))  // New allocation
```

**Impact**:
- 1-2 tensor allocations per backward pass
- Base.Grad() is pre-allocated but not used for gradInput

**Optimization Using Existing API**:
- `Base.Grad()` returns pre-allocated grad tensor (if available)
- `Reshape(dst Tensor, newShape Shape)` accepts dst parameter (line 106)
- `MatMulTransposed(dst Tensor, ...)` accepts dst parameter (line 186)

**Action**:
- **Use Base.Grad()**: Check if Base.Grad() exists and has correct shape, use it instead of creating new tensor
- **Pre-allocate gradInput2D**: For single sample case, pre-allocate 2D intermediate tensor in `Init()` if needed
- **Reshape with dst**: Use `Reshape(dst, newShape)` to write directly to final gradInput tensor

**Files Affected**:
- `dense.go`: Lines 250-307 - Use Base.Grad() and pre-allocated intermediate tensors

---

### 10. ReLU Backward - Mask and Intermediate Tensor Allocations

**Issue**: ReLU backward pass creates multiple intermediate tensors including zeros tensor and mask tensor.

**Current Pattern** (Inefficient):
```go
// activations.go lines 80-84
zeros := tensor.ZerosLike(input)  // Creates new tensor
mask := input.GreaterThan(nil, zeros)  // Creates new tensor (comparison always returns new)
gradInput := tensor.New(gradOutput.DataType(), gradOutput.Shape())  // New allocation
```

**Impact**:
- 3 tensor allocations per backward pass
- `ZerosLike` creates new tensor every time
- Comparison operations always return new tensors (no dst parameter)

**Optimization Using Existing API**:
According to SPEC.md:
- `ZerosLike(t Tensor)` creates new tensor (no alternative, but can be pre-allocated)
- `GreaterThan(other Tensor)` always returns new tensor (line 156 - no dst parameter)
- `Multiply(dst Tensor, other Tensor)` accepts dst parameter (line 126)

**Action**:
- **Pre-allocate scratch tensors**: Store in `ReLU` struct during `Init()`
  - `zeros` tensor (same shape as input/output, can be reused)
  - `mask` tensor (same shape as input/output, created by comparison operation)
  - `gradInput` tensor (use Base.Grad() if available, otherwise pre-allocate)
- **Use ZerosLike once**: Fill zeros tensor once in `Init()`, reuse across backward passes
- **Handle mask allocation**: Since `GreaterThan` always returns new tensor, we need to allocate it each time, but can reuse the zeros tensor
- **Note**: Comparison operations don't support dst, so mask tensor must be allocated. However, we can still pre-allocate zeros tensor and use Base.Grad() for gradInput.

This reduces 3 allocations to 1 per backward pass (mask allocation cannot be avoided due to comparison operation limitation).

**Files Affected**:
- `activations.go`: Lines 64-89 - Pre-allocate zeros tensor, use Base.Grad()

---

### 11. Sigmoid Backward - Multiple Intermediate Tensor Allocations

**Issue**: Sigmoid backward pass creates 4 intermediate tensors for gradient computation.

**Current Pattern** (Inefficient):
```go
// activations.go lines 171-177
ones := tensor.OnesLike(output)  // Creates new tensor
term1 := tensor.New(output.DataType(), output.Shape())  // Intermediate 1
term2 := tensor.New(output.DataType(), output.Shape())  // Intermediate 2
gradInput := tensor.New(gradOutput.DataType(), gradOutput.Shape())  // Intermediate 3
```

**Impact**:
- 4 tensor allocations per backward pass
- `OnesLike` creates new tensor every time
- All intermediate computations allocate new tensors

**Optimization Using Existing API**:
According to SPEC.md:
- `OnesLike(t Tensor)` creates new tensor (no alternative, but can be pre-allocated)
- `Subtract(dst Tensor, other Tensor)` accepts dst parameter (line 125)
- `Multiply(dst Tensor, other Tensor)` accepts dst parameter (line 126)

**Action**:
- **Pre-allocate scratch tensors**: Store in `Sigmoid` struct during `Init()`
  - `ones` tensor (same shape as output, fill once in `Init()`, reuse across backward passes)
  - `term1` tensor (for ones - output)
  - `term2` tensor (for output * term1)
  - `gradInput` tensor (use Base.Grad() if available, otherwise pre-allocate)
- **Use destination parameters**: Pass pre-allocated tensors as `dst` to `Subtract` and `Multiply`
- **Reuse tensors**: All intermediate tensors can be reused across backward passes

This reduces 4 tensor allocations to 0 per backward pass (after initial allocation in `Init()`).

**Files Affected**:
- `activations.go`: Lines 151-181 - Pre-allocate scratch tensors in `Init()`, use dst parameters

---

### 12. Tanh Backward - Multiple Intermediate Tensor Allocations

**Issue**: Tanh backward pass creates 4 intermediate tensors for gradient computation.

**Current Pattern** (Inefficient):
```go
// activations.go lines 263-269
squared := tensor.New(output.DataType(), output.Shape())  // Intermediate 1
ones := tensor.OnesLike(output)  // Creates new tensor
term := tensor.New(ones.DataType(), ones.Shape())  // Intermediate 2
gradInput := tensor.New(gradOutput.DataType(), gradOutput.Shape())  // Intermediate 3
```

**Impact**:
- 4 tensor allocations per backward pass
- Similar pattern to Sigmoid backward

**Optimization Using Existing API**:
According to SPEC.md:
- `OnesLike(t Tensor)` creates new tensor (can be pre-allocated)
- `Multiply(dst Tensor, other Tensor)` accepts dst parameter (line 126)
- `Subtract(dst Tensor, other Tensor)` accepts dst parameter (line 125)

**Action**:
- **Pre-allocate scratch tensors**: Store in `Tanh` struct during `Init()`
  - `ones` tensor (same shape as output, fill once in `Init()`, reuse)
  - `squared` tensor (for output * output)
  - `term` tensor (for ones - squared)
  - `gradInput` tensor (use Base.Grad() if available, otherwise pre-allocate)
- **Use destination parameters**: Pass pre-allocated tensors as `dst` to all operations
- **Reuse tensors**: All intermediate tensors can be reused across backward passes

This reduces 4 tensor allocations to 0 per backward pass (after initial allocation in `Init()`).

**Files Affected**:
- `activations.go`: Lines 243-273 - Pre-allocate scratch tensors in `Init()`, use dst parameters

---

### 13. Reshape Layer - Reshape + Copy Pattern

**Issue**: Reshape layer uses `Reshape(nil, ...)` followed by `Copy()`, which could use `Reshape(dst, ...)` directly.

**Current Pattern** (Acceptable but can be optimized):
```go
// utility.go lines 239, 271
inputReshaped := input.Reshape(nil, output.Shape())  // Creates view (zero-copy)
output.Copy(inputReshaped)  // Copies from view
```

**Impact**:
- Creates intermediate view tensor (zero-copy, minimal overhead)
- Extra Copy operation

**Optimization Using Existing API**:
According to SPEC.md line 106, `Reshape(dst Tensor, newShape Shape)`:
- If dst is nil: "creates a new tensor view (zero-copy when possible)"
- If dst is provided: "copies reshaped data to dst and returns dst"

**Action**:
- **Use Reshape with dst**: Replace `input.Reshaped := input.Reshape(nil, output.Shape()); output.Copy(inputReshaped)` with `input.Reshape(output, output.Shape())`
- This eliminates the intermediate view tensor and the Copy operation
- **Backward pass**: Similarly, use `gradOutput.Reshape(gradInput, input.Shape())` instead of Reshape + Copy

**Files Affected**:
- `utility.go`: Lines 238-240 (Forward), 271-272 (Backward) - Use Reshape with dst parameter

---

### 14. Conv2D Backward - Input Gradient Tensor Allocation

**Issue**: Conv2D backward pass creates `inputGradTmpTensor` and `inputGrad` tensors that could potentially reuse Base.Grad().

**Current Pattern** (Partially optimized):
```go
// conv2d.go lines 237-242
inputGradTmpTensor := tensor.New(gradOutput.DataType(), inputShape)  // Intermediate
inputGradTmp := gradOutput.Conv2DTransposed(inputGradTmpTensor, ...)
inputGrad := tensor.New(input.DataType(), inputShape)  // Final gradInput
```

**Impact**:
- 2 tensor allocations per backward pass
- Base.Grad() might be available but not used

**Optimization Using Existing API**:
- `Conv2DTransposed(dst Tensor, ...)` accepts dst parameter (line 222)
- Base.Grad() returns pre-allocated grad tensor (if available)

**Action**:
- **Pre-allocate inputGradTmpTensor**: Store in `Conv2D` struct during `Init()`
- **Use Base.Grad()**: Check if Base.Grad() exists and has correct shape/type, use it for `inputGrad` if possible
- **Note**: `inputGradTmpTensor` needs to match `gradOutput.DataType()`, while `inputGrad` needs to match `input.DataType()`, so they might need to be separate tensors

This reduces 2 allocations to 0 per backward pass (after initial allocation in `Init()`).

**Files Affected**:
- `conv2d.go`: Lines 227-251 - Pre-allocate inputGradTmpTensor, use Base.Grad() for inputGrad

---

## Summary of Additional Recommendations

### High Priority (High Impact, Medium Effort)

8. **Pre-allocate LSTM intermediate tensors**
   - **Action**: Pre-allocate gatesTmp, hiddenContributionTmp, biasFull, cellNew, iGateG, cellNewTanhTmp, outputNew in `Init()`
   - **Rationale**: All operations support destination parameters
   - **Impact**: Eliminates 7 tensor allocations per forward pass
   - **Files**: `lstm.go` lines 209-298

9. **Use Base.Grad() in Dense backward**
   - **Action**: Use Base.Grad() for gradInput, pre-allocate gradInput2D for single sample case
   - **Rationale**: Base.Grad() is pre-allocated, Reshape supports dst
   - **Impact**: Eliminates 1-2 tensor allocations per backward pass
   - **Files**: `dense.go` lines 250-307

10. **Pre-allocate Sigmoid backward scratch tensors**
    - **Action**: Pre-allocate ones, term1, term2, gradInput in `Init()`
    - **Rationale**: All operations support destination parameters
    - **Impact**: Eliminates 4 tensor allocations per backward pass
    - **Files**: `activations.go` lines 151-181

11. **Pre-allocate Tanh backward scratch tensors**
    - **Action**: Pre-allocate ones, squared, term, gradInput in `Init()`
    - **Rationale**: All operations support destination parameters
    - **Impact**: Eliminates 4 tensor allocations per backward pass
    - **Files**: `activations.go` lines 243-273

### Medium Priority (Medium Impact, Low Effort)

12. **Optimize ReLU backward**
    - **Action**: Pre-allocate zeros tensor, use Base.Grad() for gradInput
    - **Rationale**: ZerosLike can be pre-allocated and filled once, comparison operations don't support dst (limitation)
    - **Impact**: Reduces 3 allocations to 1 per backward pass (mask allocation cannot be avoided)
    - **Files**: `activations.go` lines 64-89

13. **Optimize Reshape layer**
    - **Action**: Use `Reshape(dst, newShape)` directly instead of Reshape + Copy
    - **Rationale**: Reshape supports dst parameter, eliminates intermediate view
    - **Impact**: Eliminates intermediate view tensor and Copy operation
    - **Files**: `utility.go` lines 238-240, 271-272

14. **Pre-allocate Conv2D backward input gradient tensors**
    - **Action**: Pre-allocate inputGradTmpTensor in `Init()`, use Base.Grad() for inputGrad
    - **Rationale**: Conv2DTransposed supports dst, Base.Grad() is pre-allocated
    - **Impact**: Eliminates 2 tensor allocations per backward pass
    - **Files**: `conv2d.go` lines 227-251

---

## Additional Implementation Notes

### Operations That Always Allocate (No dst Parameter)
Some operations in the tensor API always return new tensors and don't support destination parameters:
- **Comparison operations**: `Equal`, `GreaterThan`, `Less`, `NotEqual`, `GreaterEqual`, `LessEqual` (SPEC.md lines 155-161)
  - These always return new tensors (no dst parameter)
  - Workaround: Pre-allocate result tensor and copy, or accept the allocation
- **Helper functions**: `ZerosLike`, `OnesLike`, `FullLike` (create new tensors)
  - Workaround: Pre-allocate tensors in `Init()` and fill them once, reuse across passes
- **Gradient operations**: `Conv2DKernelGrad`, `Conv1DKernelGrad` (SPEC.md lines 227-228)
  - These return new tensors (no dst parameter)
- **Image/Column conversion**: `Im2Col`, `Col2Im` (SPEC.md lines 233-234)
  - These return new tensors (no dst parameter)

### Base.Grad() Usage Pattern
Many layers create new tensors for `gradInput` even though `Base.Grad()` provides a pre-allocated grad tensor:
- **Pattern**: Check if `Base.Grad()` exists and has correct shape/type
- **Use**: If available and compatible, use `Base.Grad()` instead of `tensor.New()`
- **Fallback**: If not available or incompatible, create new tensor and store it with `Base.StoreGrad()`
- **Layers to optimize**: Dense, Conv2D, ReLU, Sigmoid, Tanh, Reshape, Flatten, and other utility layers

### Pre-allocation Strategy
When pre-allocating scratch tensors:
1. **Allocate in Init()**: All scratch tensors should be allocated in `Init()` based on input/output shapes
2. **Store in struct**: Add scratch tensor fields to layer struct (similar to Softmax, LSTM patterns)
3. **Handle dynamic shapes**: Some layers may have variable batch sizes - check if tensors need resizing
4. **Reuse across passes**: All scratch tensors are reused across forward/backward passes
5. **Zero-initialization**: Some tensors (like zeros, ones) can be filled once in `Init()` and reused

---

## Next Steps

1. **Implement high-priority optimizations** - Start with Pooling operations (highest impact, easiest fix)
2. **Add scratch tensor allocation to Base** - Extend layer Base to support pre-allocated scratch tensors
3. **Optimize remaining operations** - Dense bias, LSTM gates, Softmax backward, Conv backward
4. **Implement additional optimizations** - LSTM intermediate tensors, activation backward passes, Reshape layer
5. **Benchmark improvements** - Measure memory allocations and performance gains
6. **Update layer implementations** - Ensure all layers use dst parameters where available and Base.Grad() where possible

