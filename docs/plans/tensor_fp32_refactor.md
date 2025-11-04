## Goal
- decouple float32-specific tensor math from `pkg/core/math/tensor` so data-type agnostic tensor wrappers can delegate to primitives.
- prepare ground for future multi-dtype tensor support (e.g. TF Lite layers) by consolidating fp32 kernels under `pkg/core/math/primitive/fp32`.
- encapsulate tensor state behind a typed API so callers cannot mutate shape/data directly.

## Current Pain Points
- `tensor` package mixes shape/stride management with low-level loop kernels, making dtype specialization hard.
- Repeated recursive implementations for strided elementwise ops (`addStrided`, `sumDims`, etc.) duplicate logic that should live in primitives.
- Reductions (sum/max/min/argmax) and broadcasting helpers rely on ad-hoc code paths that are difficult to share across future dtype backends.

## Existing fp32 Tensor Kernels (baseline audit)
- **core utilities**: `ComputeStrides`, `SizeFromShape`, `SwapRows`, `GetElem` + linear algebra helpers in `la.go` (used by GEMM/SVD paths).
- **level1 BLAS** (`level1.go`): `Axpy`, `Scal`, `Copy`, `Dot`, `Iamax`, `Asum`, etc.—used for contiguous vector-style operations.
- **array helpers** (`array.go`): `SumArr`, `DiffArr`, `MulArr`, `DivArr`, `Sum`, `SqrSum`, `StatsArr`, `PercentileArr`, plus in-place variants.
- **level2 BLAS** (`level2.go`): `Gemv_N`, `Gemv_T`, `Ger`, etc.—support matrix-vector style tensor ops.
- **level3 BLAS** (`level3.go`): `Gemm_NN`, `Gemm_NT`, `Gemm_TN`, `Gemm_TT`—power `MatMulTransposed` and convolution reshapes.
- **convolution kernels** (`conv.go`): `Im2Col`, `Col2Im`, `Conv2D`, `Conv2DTransposed`—already pure fp32 implementations.
- **tensor utilities** (`tensor.go`): currently only `Im2Col`/`Col2Im`/`Conv2D`, repurposed by tensor package.
- **linear algebra** (`la.go`): `Gesvd`, `Gepseu`, `Gnnls`, etc.—back the matrix pseudo-inverse/SVD logic.
- **batched helpers** (`batched.go`): batched GEMM/GEMV variants for multi-matrix operations.

This inventory will guide which primitives we can reuse versus new APIs we must add (e.g., axis-aware reductions, generic strided elementwise kernels).
## Target Responsibilities Split
- `primitive/fp32` should expose reusable kernels operating on flat slices with explicit shape/stride metadata (no Tensor coupling).
  - elementwise ops: add/sub/mul/div/scale with optional strides.
  - copy / broadcast expansion helpers.
  - reductions: sum/mean/max/min/argmax along arbitrary axes.
- `tensor` should focus on:
  - validating shapes, managing metadata, allocating outputs.
  - delegating actual math to the primitives.

## Step-by-Step Plan
1. **Refactor tensor core** *(in progress)*
   - Introduce `DataType` enum (starting with `DTFP32`).
   - Encapsulate tensor state (`dtype`, `shape`, `data`) with private fields.
   - Add constructors `New(dtype DataType, shape ...int)` and `FromFloat32(shape []int, data []float32)` (no-copy wrapping when `data` is non-nil, allocation otherwise).
   - Ensure `Shape`, `Dims`, `Data`, `Rank`, `Size`, `Clone`, `Reshape`, etc. operate on the new structure.
   - Keep zero-value tensors usable.

2. **Design fp32 tensor-kernel API** *(partial)*
   - Define `tensor.Shape` helpers, stride utilities, and keep primitives operating on raw slices (no cyclic deps). *(done)*
   - Provide elementwise/broadcast/reduction primitives as previously outlined. *(done)*
   - `tensor` must use kernels from `fp32` *(partial)*

3. **Refactor tensor internals** *(todo)*
   - Update tensor math, linalg, convolution code to rely on the encapsulated tensor API (no struct literals or direct field use).

4. **Refactor tensor consumers** *(todo)*
   - Sweep other packages (NN layers, tests, specs, etc.) to construct tensors via `New`/`FromFloat32` and access data through accessors.
     - pkg/core/math/learn/datasets/mnist/loader.go
     - pkg/core/math/nn/layers
     - pkg/core/math/nn
     - pkg/core/math/learn
   - Replace raw `.Dim`/`.Data` usage with `Dims()`/`Data()` or higher-level helpers.

5. **Implement parallel fp32 kernels** *(todo)*
   - Add optional parallel execution paths (e.g. chunked workers) for heavy fp32 kernels where it improves performance.
   - Gate with sensible heuristics to avoid overhead on small workloads.

6. **Tests & benchmarks** *(todo)*
   - Re-run targeted packages (`tensor`, `primitive/fp32`, NN layers) and existing benchmarks to confirm no regressions.

7. **Eliminate panics** *(todo)*
   - After the API is in place and consumers updated, replace remaining runtime panics inside tensor methods with error returns or documented invariants.

## Out of Scope (Future Work)
- Introducing non-fp32 tensor data types (bf16/int8) — planned once fp32 kernels are centralized.
- GPU / SIMD acceleration; current goal is clean CPU reference path.

