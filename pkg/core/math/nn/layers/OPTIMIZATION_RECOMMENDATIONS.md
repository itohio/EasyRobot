# Layers Code Optimization Recommendations

This document identifies optimization opportunities in the `pkg/core/math/nn/layers` package. All recommendations are scoped to the layers code only.

## Table of Contents

1. [Excessive Cloning](#excessive-cloning)
2. [Unnecessary Copies](#unnecessary-copies)
3. [Intermediate Tensor Creation](#intermediate-tensor-creation)
4. [Reshape Overhead](#reshape-overhead)
5. [Parameter Access Patterns](#parameter-access-patterns)
6. [Shape Computations](#shape-computations)
7. [Non-Idiomatic Go Code](#non-idiomatic-go-code)
8. [Weird Code Patterns](#weird-code-patterns)

---

## 1. Excessive Cloning

### Problem
Many operations clone tensors unnecessarily, creating new memory allocations when in-place operations could be used.

### Locations

#### activations.go

**ReLU.Backward (line 84):**
```go
gradInput := gradOutput.Clone().Mul(mask)
```
**Issue**: Clones `gradOutput` before multiplying. If `gradOutput` is not needed after this operation, we could use it directly.

**Recommendation**: 
- If `gradOutput` is not needed after backward: Use `gradOutput.Mul(mask)` directly (if mask shape matches)
- If `gradOutput` must be preserved: Consider using a pre-allocated gradInput tensor from Base

**Sigmoid.Backward (lines 171-173):**
```go
term1 := ones.Clone().Sub(output)          // (1 - output)
term2 := output.Clone().Mul(term1)         // output * (1 - output)
gradInput := gradOutput.Clone().Mul(term2) // gradOutput * output * (1 - output)
```
**Issues**: 
- Creates 3 clones unnecessarily
- `ones.Clone().Sub(output)` creates a new tensor when we could reuse pre-allocated space
- `output.Clone()` is unnecessary if output is not needed after backward

**Recommendation**:
```go
// Pre-allocate intermediate tensors in Base during Init
// Use in-place operations where possible
term1 := preallocatedIntermediate1.Copy(output).Scale(-1).AddScaled(ones, 1.0)
// Or use a more efficient approach:
gradInput := r.Base.Grad() // Get pre-allocated grad
gradInput.Copy(gradOutput)
gradInput.Mul(output).Mul(ones.Clone().Sub(output)) // Still need ones manipulation
```

**Tanh.Backward (lines 259-262):**
```go
squared := output.Clone().Mul(output) // output^2
ones := tensor.OnesLike(output)
term := ones.Clone().Sub(squared)         // (1 - output^2)
gradInput := gradOutput.Clone().Mul(term) // gradOutput * (1 - output^2)
```
**Issues**: Similar to Sigmoid - multiple clones

**Recommendation**: Pre-allocate intermediate tensors in Base and reuse them

**Softmax.Backward (lines 351-366):**
```go
prod := gradOutput.Clone().Mul(output)
sumTerm := prod.Sum(s.dim)
sumBroadcast, err := sumTerm.BroadcastTo(output.Shape())
diff := gradOutput.Clone().Sub(sumBroadcast)
gradInput := output.Clone().Mul(diff)
```
**Issues**: 
- Two clones of `gradOutput` (lines 351 and 363)
- Clone of `output` when we already have it stored

**Recommendation**: 
- Reuse `gradOutput` for the first computation if possible
- Pre-allocate `gradInput` and use it directly

**Dropout.Forward (lines 487, 491):**
```go
output = input.Clone()
output = output.DropoutForward(d.mask)
```
and
```go
output = input.Clone()
```
**Issue**: Always clones input even in inference mode (line 491)

**Recommendation**: 
- In training mode: Clone is necessary for dropout
- In inference mode: Could potentially use input directly if it's safe (but this depends on layer contract)

#### lstm.go

**computeForward (lines 254-277):**
```go
iGateSigmoid := iGate.Clone().Sigmoid(nil)
fGateSigmoid := fGate.Clone().Sigmoid(nil)
gGateTanh := gGate.Clone().Tanh(nil)
oGateSigmoid := oGate.Clone().Sigmoid(nil)
cellNew := cellState.Clone()
iGateG := iGate.Clone().Mul(gGate)
cellNewTanh := cellNew.Clone().Tanh(nil)
outputNew := oGate.Clone().Mul(cellNewTanh)
```
**Issues**: 
- 7 clone operations in a single forward pass
- Each activation clones before applying, then we clone again for operations
- `cellNew.Clone().Tanh(nil)` creates unnecessary intermediate

**Recommendation**:
- Pre-allocate gate tensors in LSTM struct during Init
- Use in-place operations where gates are intermediate results
- Consider: `cellNew.Tanh(nil)` instead of `cellNew.Clone().Tanh(nil)` if `cellNew` is already a copy

**Forward (lines 171-172):**
```go
l.hiddenState = output.Clone()
l.cellState = l.cellState.Clone() // cellState is updated in computeForward
```
**Issue**: Clones state even though `cellState` is already updated in `computeForward` via `Copy`

**Recommendation**: 
- `cellState` is already updated in `computeForward` via `cellState.Copy(cellNew)`, so the clone on line 172 is unnecessary
- For `hiddenState`, we could potentially reuse `output` if it's safe (but this depends on ownership semantics)

**SetState (lines 313-314):**
```go
l.hiddenState = hiddenState.Clone()
l.cellState = cellState.Clone()
```
**Issue**: Always clones even if caller wants to transfer ownership

**Recommendation**: Add a boolean parameter `transferOwnership` or use a different method

**GetState (line 326):**
```go
return l.hiddenState.Clone(), l.cellState.Clone()
```
**Issue**: Always clones, but this might be necessary for safety

**Recommendation**: Consider if callers need copies or if they can work with references

---

## 2. Unnecessary Copies

### Problem
Using `.Copy()` when operations could be done in-place or when tensors are already in the correct format.

### Locations

#### pooling.go

**MaxPool2D.Forward (line 121):**
```go
result, indices := input.MaxPool2DWithIndices(...)
output.Copy(result)
```
**Issue**: `MaxPool2DWithIndices` returns a new tensor, then we copy it to pre-allocated output

**Recommendation**: 
- Add `MaxPool2DWithIndicesTo` method that writes directly to output tensor
- Or check if `result` can be used directly (if it's the pre-allocated output)

**AvgPool2D.Forward (line 289):**
```go
result := input.AvgPool2D(...)
output.Copy(result)
```
**Issue**: Similar to MaxPool2D

**Recommendation**: Add `AvgPool2DTo` method

**GlobalAvgPool2D.Forward (line 400):**
```go
result := input.GlobalAvgPool2D()
output.Copy(result)
```
**Issue**: Similar pattern

**Recommendation**: Add `GlobalAvgPool2DTo` method

#### utility.go

**Flatten.Forward (lines 91-92):**
```go
inputReshaped := input.Reshape(output.Shape())
output.Copy(inputReshaped)
```
**Issue**: Reshape creates a view, then we copy. For contiguous tensors, this might be unnecessary.

**Recommendation**: 
- Check if input is contiguous and shapes are compatible
- If so, use direct copy or memcpy
- Otherwise, the copy is necessary

**Similar patterns in**: Reshape.Forward (line 240), Unsqueeze.Forward (line 396), Squeeze.Forward (line 556), Transpose.Forward (line 682)

#### conv2d.go

**Backward (lines 231-238):**
```go
inputGradTmpReshaped := inputGradTmp.Reshape(inputShape)
inputGrad.Copy(inputGradTmpReshaped)
```
**Issue**: Reshape + Copy when we could potentially reshape directly into `inputGrad`

**Recommendation**: Check if `Conv2DTransposed` can write directly to a pre-allocated tensor with correct shape

#### lstm.go

**computeForward (lines 280-281):**
```go
output.Copy(outputNew)
cellState.Copy(cellNew)
```
**Issue**: We already have `output` and `cellState` as parameters, but we're copying into them

**Recommendation**: 
- If `outputNew` and `cellNew` are temporary tensors, this is necessary
- Consider pre-allocating these in LSTM struct and reusing them

---

## 3. Intermediate Tensor Creation

### Problem
Creating temporary tensors that could be pre-allocated and reused.

### Locations

#### activations.go

**ReLU.Backward (line 80):**
```go
zeros := tensor.ZerosLike(input)
```
**Issue**: Creates a new tensor every backward pass just to compare with zero

**Recommendation**: 
- Pre-allocate `zeros` tensor in Base during Init
- Or use a more efficient comparison: `input.GreaterThanScalar(0.0)` if such method exists

**Sigmoid/Tanh.Backward (lines 170, 260):**
```go
ones := tensor.OnesLike(output)
```
**Issue**: Creates new tensor every backward pass

**Recommendation**: Pre-allocate `ones` tensor in Base during Init (if output shape is constant)

#### dense.go

**computeLinear (lines 172-188):**
```go
biasBroadcast := bias.Reshape(tensor.NewShape(1, outFeatures))
biasBroadcastFull, err := biasBroadcast.BroadcastTo(output.Shape())
if err == nil {
    output.Add(biasBroadcastFull)
} else {
    // Fallback: element-wise iteration
    // ...
}
```
**Issue**: Creates intermediate tensors for broadcasting that might fail

**Recommendation**: 
- Pre-allocate broadcast tensor if shape is known
- Or use a more efficient bias addition method: `AddBias` that handles broadcasting internally

**Backward (lines 239-241, 262-264):**
```go
inputReshaped := input.Reshape(tensor.NewShape(1, d.inFeatures))
gradReshaped := gradOutput.Reshape(tensor.NewShape(1, d.outFeatures))
// ...
gradInput2D := tensor.New(gradOutput.DataType(), tensor.NewShape(1, d.inFeatures))
```
**Issue**: Creates multiple reshaped views and new tensors

**Recommendation**: 
- Pre-allocate `gradInput2D` in Base during Init
- Use operations that work directly with 1D tensors without reshaping

#### lstm.go

**computeForward (lines 202-218):**
```go
inputReshaped := input.Reshape(tensor.NewShape(1, l.inputSize))
gatesTemp := inputReshaped.MatMulTransposed(weightIH, false, true, nil)
gates = gatesTemp.Reshape(tensor.NewShape(4 * l.hiddenSize))
// Similar for hiddenState
```
**Issue**: Multiple reshape operations creating intermediate tensors

**Recommendation**: 
- Pre-allocate reshaped tensors in LSTM struct
- Or use operations that handle both 1D and 2D inputs

**computeForward (lines 228-233):**
```go
biasBroadcast := bias.Reshape(tensor.NewShape(1, 4*l.hiddenSize))
biasFull, err := biasBroadcast.BroadcastTo(gates.Shape())
```
**Issue**: Creates intermediate broadcast tensor

**Recommendation**: Pre-allocate broadcast tensor if batch size is known

#### conv1d.go

**Backward (lines 229-241):**
```go
gradOutput4D := gradOutput.Reshape(tensor.NewShape(batchSize, c.outChannels, gradOutputShape[2], 1))
kernel4D := kernelParam.Data.Reshape(tensor.NewShape(c.outChannels, c.inChannels, c.kernelLen, 1))
// ...
gradInput := gradInput4D.Reshape(tensor.NewShape(gradInputShape[0], gradInputShape[1], gradInputShape[2]))
```
**Issue**: Multiple reshape operations

**Recommendation**: 
- Pre-allocate 4D reshaped views if shapes are constant
- Or add `Conv1DBackward` method that works directly with 3D tensors

#### conv2d.go

**Backward (lines 272-289):**
```go
inputCols := input.Im2Col(...)
gradOutputReshaped := gradOutput.Reshape(...)
kernelGradMatrix := gradOutputReshaped.Transpose().MatMul(inputCols)
kernelGradReshaped := kernelGradMatrix.Reshape(...)
```
**Issue**: Multiple intermediate tensors (inputCols, gradOutputReshaped, kernelGradMatrix)

**Recommendation**: 
- `inputCols` might be large - consider if it can be computed on-the-fly
- Pre-allocate reshaped tensors if sizes are known
- Consider fused operation: `Conv2DKernelGrad` that computes gradient directly

---

## 4. Reshape Overhead

### Problem
Excessive use of `Reshape()` which may create views or copies. Many reshapes are done just to match shapes for operations.

### Locations

**Pattern**: Reshape → Operation → Reshape back

**dense.go:**
- Lines 172, 183-184, 239-240, 262-264, 270: Multiple reshapes for batch/single sample handling

**lstm.go:**
- Lines 202-205, 216-218, 228: Reshapes for 1D/2D compatibility

**conv1d.go:**
- Lines 230, 233, 241: Reshapes for Conv1D → Conv2D conversion

**Recommendation**: 
- Add operations that handle both 1D and 2D inputs natively
- Pre-allocate reshaped views and reuse them
- Consider if operations can work with original shapes using strides

---

## 5. Parameter Access Patterns

### Problem
Inefficient access to parameters stored in Base.

### Locations

**base.go:**

**ZeroGrad (lines 361-368):**
```go
for idx := range b.params {
    param := b.params[idx]
    paramPtr := &param
    paramPtr.ZeroGrad()
    b.params[idx] = param
}
```
**Issue**: 
- Copies parameter from map
- Creates pointer to copy
- Modifies copy, then writes back
- This is inefficient - parameters are stored by value

**Recommendation**: 
- Store parameters as pointers in map: `map[ParamIndex]*Parameter`
- Or use a method that takes parameter by pointer: `param.ZeroGrad()` should work on the parameter directly

**Update (lines 436-451):**
```go
for idx := range b.params {
    param := b.params[idx]
    // ...
    if err := optimizer.Update(param); err != nil {
        // ...
    }
}
```
**Issue**: Copies parameter for each update

**Recommendation**: Same as ZeroGrad - use pointers in map

**Parameter access pattern:**
```go
weightParam, ok := d.Base.Parameter(types.ParamWeights)
if !ok || tensor.IsNil(weightParam.Data) {
    return nil, fmt.Errorf("...")
}
```
**Issue**: This pattern is repeated in every layer's Forward/Backward

**Recommendation**: 
- Add helper method: `Base.RequireParameter(idx)` that returns parameter or panics
- Cache parameter access in layer struct during Init

---

## 6. Shape Computations

### Problem
Redundant shape calculations and validations.

### Locations

**All layers' Init methods:**
```go
outputSize := 1
for _, dim := range outputShape {
    outputSize *= dim
}
```
**Issue**: This calculation is done in every Init, but `Shape` should have a `Size()` method

**Recommendation**: Use `tensor.NewShape(...).Size()` if available, or add it

**dense.go - computeLinear:**
```go
biasShape := bias.Shape()
if len(biasShape) == 1 && biasShape[0] == outFeatures {
    // ...
}
```
**Issue**: Shape is accessed multiple times, and we check `bias.Shape().Rank() > 0` separately

**Recommendation**: 
- Cache shape: `biasShape := bias.Shape()`
- Use `biasShape.Rank()` instead of `len(biasShape)`

**conv2d.go - Backward:**
```go
inputShape := input.Shape()
// ... later ...
inputGrad := tensor.New(input.DataType(), inputShape)
```
**Issue**: Shape is accessed once, which is fine, but pattern is inconsistent

**Recommendation**: Consistent shape caching pattern

---

## 7. Non-Idiomatic Go Code

### Problem
Code that doesn't follow Go best practices or repository conventions.

### Locations

**base.go:**

**ParametersByIndex (lines 335-342):**
```go
result := make(map[types.ParamIndex]types.Parameter)
for idx, param := range b.params {
    result[idx] = param
}
if len(result) == 0 {
    return nil
}
return result
```
**Issue**: 
- Creates copy of entire map
- Checks length after copying
- Returns `nil` for empty map, but caller might expect empty map

**Recommendation**: 
- Return empty map instead of nil (Go convention)
- Or check `len(b.params) == 0` before copying

**ZeroGrad (lines 361-368):**
```go
param := b.params[idx]
paramPtr := &param
paramPtr.ZeroGrad()
b.params[idx] = param
```
**Issue**: 
- Non-idiomatic: creates pointer to copy
- Should work directly with parameter value if ZeroGrad has pointer receiver, or store pointers in map

**Recommendation**: Fix parameter storage to use pointers or fix ZeroGrad signature

**dense.go:**

**computeLinear (lines 180-188):**
```go
for b := 0; b < batchSize; b++ {
    outputBatch := output.Reshape(tensor.NewShape(batchSize, outFeatures))
    outputBatchSlice := outputBatch.Reshape(tensor.NewShape(outFeatures))
    outputBatchSlice.AddScaled(bias, 1.0)
}
```
**Issue**: 
- Reshapes inside loop - very inefficient
- Logic seems wrong - `outputBatchSlice` is same for all iterations
- This fallback code path looks broken

**Recommendation**: 
- Fix the logic - should use `Slice` to get batch element
- Or remove this fallback if it's not needed

**lstm.go:**

**computeForward parameter list (line 187):**
```go
func (l *LSTM) computeForward(input, weightIH, weightHH, bias,
    hiddenState, cellState, output tensorTypes.Tensor) error {
```
**Issue**: Too many parameters (7) - violates "Keep number of arguments low" rule

**Recommendation**: 
- Group into struct: `type LSTMForwardParams struct { ... }`
- Or use Options pattern

**utility.go:**

**Pad.applyPadding and Pad.extractGradient:**
- Recursive functions with many parameters
- Complex nested logic

**Recommendation**: 
- Consider iterative approach for better performance
- Extract helper functions
- Use tensor operations instead of element-wise access

---

## 8. Weird Code Patterns

### Problem
Code that works but is confusing or inefficient.

### Locations

**conv2d.go - Backward (lines 228-238):**
```go
inputGrad := tensor.New(input.DataType(), inputShape)
if inputGradTmp.Size() == inputGrad.Size() {
    inputGradTmpReshaped := inputGradTmp.Reshape(inputShape)
    inputGrad.Copy(inputGradTmpReshaped)
} else {
    inputGradTmpReshaped := inputGradTmp.Reshape(inputShape)
    inputGrad.Copy(inputGradTmpReshaped)
}
```
**Issue**: Both branches do the same thing! The `if` check is useless.

**Recommendation**: Remove the if statement and always do the reshape + copy

**dense.go - SetWeight (lines 398-405):**
```go
if d.hasBias {
    biasParam, ok := d.Base.Parameter(types.ParamBiases)
    if ok && !tensor.IsNil(biasParam.Data) && biasParam.Data.Shape() != nil && biasParam.Data.DataType() != weight.DataType() {
        biasShape := biasParam.Data.Shape()
        biasParam.Data = tensor.XavierUniform(weight.DataType(), biasShape, 1, d.outFeatures, d.Base.rng)
        d.Base.SetParam(types.ParamBiases, biasParam)
    }
}
```
**Issue**: 
- Side effect in SetWeight - recreates bias if data type doesn't match
- This is unexpected behavior
- `biasParam.Data.Shape() != nil` check is redundant (Shape() never returns nil)

**Recommendation**: 
- Make this explicit: `SetWeightAndSyncBiasType` or separate method
- Remove redundant nil check

**lstm.go - computeForward (lines 254-277):**
```go
iGateSigmoid := iGate.Clone().Sigmoid(nil)
iGate = iGateSigmoid
```
**Issue**: Clones, applies sigmoid, then reassigns. The clone seems unnecessary if we're reassigning.

**Recommendation**: 
- If `iGate` is a slice/view, the clone is necessary
- Otherwise, use `iGate.Sigmoid(nil)` directly if it modifies in-place
- Or pre-allocate `iGateSigmoid` and use `iGate.Sigmoid(iGateSigmoid)`

**pooling.go - GlobalAvgPool2D.Backward (lines 427-436):**
```go
if g.Base.CanLearn() {
    return nil, fmt.Errorf("GlobalAvgPool2D.Backward: backward pass not yet implemented")
}
gradInput := tensor.New(gradOutput.DataType(), gradOutput.Shape())
```
**Issue**: 
- Creates new tensor with wrong shape (`gradOutput.Shape()` instead of `input.Shape()`)
- Returns error if CanLearn, but then creates tensor anyway
- Logic is inverted

**Recommendation**: Fix the logic - should create tensor with `input.Shape()`

---

## Summary of High-Priority Optimizations

1. **Pre-allocate intermediate tensors** in Base.Init or layer-specific Init
2. **Reduce cloning** - use in-place operations where safe
3. **Add `*To` variants** for operations that write to pre-allocated tensors
4. **Fix parameter storage** - use pointers in map to avoid copying
5. **Cache frequently accessed values** (shapes, parameters)
6. **Remove redundant operations** (unnecessary clones, copies, reshapes)
7. **Fix broken code paths** (dense computeLinear fallback, conv2d backward if/else)

---

## Implementation Priority

1. **High Priority** (Easy wins, high impact):
   - Pre-allocate intermediate tensors (ones, zeros, broadcast tensors)
   - Fix parameter storage to use pointers
   - Remove redundant clones in activations

2. **Medium Priority** (Moderate effort, good impact):
   - Add `*To` variants for pooling operations
   - Reduce reshape overhead in LSTM and conv layers
   - Cache parameter access

3. **Low Priority** (Requires more design work):
   - Refactor LSTM computeForward to use struct parameters
   - Optimize Pad operations
   - Add fused operations for common patterns

