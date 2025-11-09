# Multi-dimensional Softmax Support Plan

## Goal
Extend `Tensor.Softmax` and `Tensor.SoftmaxGrad` so they operate on tensors of arbitrary rank, not just rank-1 or rank-2, while retaining compatible behaviour with existing call sites.

## Constraints & Considerations
- Data lives in row-major layout; we must iterate over strided slices when the softmax dimension is not the innermost axis.
- Existing fp32 helpers only cover 1D/2D fast paths; we should reuse them when applicable for performance, but add a generic fallback without introducing heavy allocations.
- API contract remains unchanged: in-place when `dst` is nil/empty; otherwise write into `dst` with shape validation.
- Gradients must mirror forward softmax semantics to keep autodiff expectations.

## Approach
1. **Axis Decomposition**: Decompose the tensor into `(outer, axis, inner)` sizes where `outer = prod(shape[:dim])`, `axis = shape[dim]`, and `inner = prod(shape[dim+1:])`. Each softmax vector then consists of `axis` elements spaced by `inner` in the flattened buffer.
2. **Scratch Buffer**: Allocate a reusable temporary slice of length `axis` per invocation (stack-allocated via `make([]float32, axis)` once). For each vector, copy the axis slice into the scratch buffer before computing max/sum to support in-place updates safely.
3. **Two-pass Computation**: Use the scratch data to compute the numerically stable softmax (subtract max, exponentiate, normalise) and write results back into the destination buffer using the calculated stride offsets.
4. **Gradient Update**: Apply the same iteration pattern for `SoftmaxGrad`, matching the softmax derivative formula using the scratch buffer to read original `output` and `gradOutput` values without aliasing issues.
5. **Optimised Fast Paths**: Retain existing specialised 1D/2D kernels by detecting rank/axis combinations and delegating to fp32 helpers before falling back to the generic routine.
6. **Validation & Tests**: Add unit tests covering 3D and 4D tensors across different dimensions for both forward softmax and gradient. Include cases for in-place vs out-of-place operation, ensuring normalisation and gradient correctness.

## Risks & Mitigations
- **Memory Overhead**: Scratch buffer per vector could be large for very wide axes. We mitigate by allocating a single buffer reused for all vectors instead of per vector allocations.
- **Numerical Stability**: Follow existing subtract-max pattern to avoid overflow; reuse math.Exp and keep float32 precision consistent with existing implementation.
- **Strided Tensors**: Copy from source to destination using existing generic strided copy before running softmax when destination is distinct; this preserves behaviour for views.

## Deliverables
- Updated implementations of `Tensor.Softmax` and `Tensor.SoftmaxGrad` supporting arbitrary rank.
- Additional helper routines (if needed) scoped within `activations.go`.
- Expanded tests in `activations_test.go` covering higher-dimensional tensors.
- Documentation/spec updates noting multi-dimensional support.
