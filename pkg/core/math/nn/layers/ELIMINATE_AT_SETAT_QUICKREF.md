# Quick Reference: Eliminating At/SetAt/Elements Usage

## Current Status

**Total Production Code Lines Using At/SetAt/Elements**: ~109 lines

| File | Lines | Operation | Priority |
|------|-------|-----------|----------|
| `pooling.go` | 67 | MaxPool2D.Backward | P1 |
| `conv1d.go` | 27 | Conv1D.Backward kernel grad | P1 |
| `conv2d.go` | 12 | Conv2D.Backward transpose/copy | P1 |
| `utility.go` | 3 | Pad, Fill operations | P2 |

## Required Operations (Phase 1 - Critical)

### 1. MaxPool2DWithIndices + MaxPool2DBackward
```go
// Forward: Store indices
output, indices := input.MaxPool2DWithIndices(kernelSize, stride, padding)

// Backward: Use indices
gradInput := gradOutput.MaxPool2DBackward(indices, kernelSize, stride, padding)
```
**Impact**: Eliminates 67 lines in MaxPool2D.Backward

### 2. Conv1DKernelGrad
```go
kernelGrad := gradOutput.Conv1DKernelGrad(input, stride, padding)
```
**Impact**: Eliminates 27 lines in Conv1D.Backward

### 3. Transpose/Permute for 4D+
```go
// Option 1: Extend Transpose
kernelTransposed := kernel.Transpose(1, 0, 2, 3)

// Option 2: Add Permute
kernelTransposed := kernel.Permute([]int{1, 0, 2, 3})
```
**Impact**: Eliminates 10 lines in Conv2D.Backward

### 4. AvgPool2DBackward
```go
gradInput := gradOutput.AvgPool2DBackward(kernelSize, stride, padding)
```
**Impact**: Enables AvgPool2D.Backward (currently not implemented)

## Required Operations (Phase 2 - Complete)

### 5. ScatterAdd
```go
gradInput := gradOutput.ScatterAdd(indices, dim, gradInput)
```
**Impact**: Alternative to MaxPool2DBackward if indices not stored

### 6. Unpad
```go
gradInput := gradOutput.Unpad(padding)
```
**Impact**: Eliminates 1 line in Pad.Backward

### 7. Fill/FillValue
```go
output := input.Clone().Fill(value)
```
**Impact**: Eliminates 4 lines in Fill layer

## Implementation Checklist

### Phase 1 (Critical) - 95% Elimination
- [ ] MaxPool2DWithIndices (forward pass modification)
- [ ] MaxPool2DBackward
- [ ] Conv1DKernelGrad
- [ ] Transpose/Permute for 4D+ tensors
- [ ] AvgPool2DBackward

### Phase 2 (Complete) - 100% Elimination
- [ ] ScatterAdd
- [ ] Unpad
- [ ] Fill/FillValue

## After Implementation

**Before**: ~109 lines using At/SetAt/Elements
**After Phase 1**: ~5 lines remaining (95% eliminated)
**After Phase 2**: 0 lines remaining (100% eliminated)

## Notes

- Test files using At/SetAt are acceptable (not production code)
- Gradient test helpers using At/SetAt are acceptable (numerical gradient computation)
- Focus on production layer code elimination

