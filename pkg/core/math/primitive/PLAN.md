# Primitive Pooling Test Plan

## Goal
Add deterministic unit tests for fp32 pooling primitives to validate both forward and backward computations, especially when padding or stride introduces partial windows.

## Scope
- `fp32.AvgPool2D` / `fp32.AvgPool2DBackward`
- `fp32.MaxPool2D` / `fp32.MaxPool2DBackward`

## Approach
- Construct small synthetic inputs (e.g., 1×1×3×3 tensors) where exact averages and maxima can be computed by hand.
- Cover edge cases with padding so only a subset of the kernel overlaps the input.
- Compare primitive outputs and gradients with expected values using tight tolerances.

## Files
- New test file under `pkg/core/math/primitive/fp32/` (e.g., `pooling_test.go`).
- No production code changes unless tests reveal bugs.
