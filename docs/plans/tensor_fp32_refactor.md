## Goal
- decouple float32-specific tensor math from `pkg/core/math/tensor` so data-type agnostic tensor wrappers can delegate to primitives.
- prepare ground for future multi-dtype tensor support (e.g. TF Lite layers) by consolidating fp32 kernels under `pkg/core/math/primitive/fp32`.

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
1. **Inventory existing kernels**
   - catalogue tensor methods that already call `fp32` (`Add`, `Scale`, MatMul wrappers) vs those with bespoke loops.
   - document required signatures for each candidate primitive (inputs: data slice, shape, strides, axes, etc.).

2. **Design fp32 tensor-kernel API**
   - define a `tensor.Shape` type (`type Shape []int`) hosting shape/stride helpers so they can be reused across dtypes; fp32 kernels will accept the derived `[]int` data.
   - add lightweight helpers in `tensor` (e.g. `Shape.Strides()`, `Shape.Size()`, `Shape.IsContiguous(strides []int)`) and make fp32 utilities operate on raw slices to avoid duplication.
   - define core elementwise APIs accepting `dst`, `a`, `b`, `shape`, `strides`.
   - design reduction APIs returning scalars or writing into preallocated buffers, supporting axis lists.
   - **Proposed API surface** (all under `primitive/fp32`):
     - Shape utilities move under `tensor.Shape`, so fp32 only needs helper shims for safety when called directly.
     - Elementwise (contiguous+strided):
       - `ElemAdd(dst, a, b []float32, shape []int, stridesDst, stridesA, stridesB []int)`
       - `ElemSub(...)`, `ElemMul(...)`, `ElemDiv(...)` (same signature).
       - `ElemScale(dst []float32, scalar float32, shape []int, stridesDst []int)`.
       - `ElemCopy(dst, src []float32, shape []int, stridesDst, stridesSrc []int)`.
     - Broadcast helpers:
       - `BroadcastStrides(shape, broadcastShape []int) ([]int, error)` to compute effective strides.
       - `ExpandTo(dst, src []float32, dstShape, srcShape []int, dstStrides, srcStrides []int)` for materializing views when needed.
     - Reductions:
       - `ReduceSum(dst []float32, src []float32, srcShape []int, axes []int, keepDims bool)`.
       - `ReduceMax(...)`, `ReduceMin(...)`, `ReduceMean(...)` implemented on top of sum/count.
       - `Argmax(dst []int, src []float32, srcShape []int, axis int)` returning indices (int slice) while optionally writing float32 for compatibility.
     - Utilities:
       - `IndexLinear(indices []int, strides []int) int` (shared with tensor for validation loops).
       - `ValidateAxes(shape []int, axes []int) error` to mirror current panic messages before delegating.
   - All functions accept explicit shape/stride metadata so they can perform contiguous fast-paths internally and fall back to generic loops otherwise.
   - **Usage & dependency boundary**: `tensor.Shape` lives in `pkg/core/math/tensor` and is only consumed by higher-level tensor code (and future dtype-aware packages). The fp32 primitives stay decoupled by continuing to work with raw `[]int` shapes/strides, so there is no `primitive ↔ tensor` import cycle. Callers convert between the two (`Shape(t.Dim)` or `shape.Slice()`) when crossing the boundary.

3. **Implement fps32 kernels incrementally**
   - start with contiguous fast paths using existing BLAS-like routines (`Axpy`, `Scal`, `Dot`).
   - add generic stride-aware loops reused across operations (single-pass walkers, no callbacks, minimal allocations).
   - unit-test kernels in `pkg/core/math/primitive/fp32/tensor_test.go` mirroring current tensor tests.

4. **Refactor tensor package**
   - replace recursive helpers with calls into new primitives.
   - keep tensor-specific shape checks and higher-level orchestration.
   - ensure behavior/edge cases remain unchanged (panic messages, broadcasting semantics).

5. **Validation**
   - extend existing tensor tests to cover both contiguous and non-contiguous (reshaped) cases.
   - run full `go test ./pkg/core/math/tensor ./pkg/core/math/primitive/fp32` and targeted benchmarks once walkers are stabilized.

## Out of Scope (Future Work)
- Introducing non-fp32 tensor data types (bf16/int8) — planned once fp32 kernels are centralized.
- GPU / SIMD acceleration; current goal is clean CPU reference path.

